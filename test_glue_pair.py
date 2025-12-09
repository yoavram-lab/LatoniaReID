IMAGE_A_PATH = "rotated/2013-11/14/IMGP0969.JPG"
IMAGE_B_PATH = "rotated/2013-12/14/IMGP1235.JPG"

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from lightglue import LightGlue, ALIKED
from lightglue.utils import rbd

# --- CONFIGURATION ---
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
DEVICE = torch.device("cpu")

def resize_image(image, max_dim=1024):
    """Resizes image maintaining aspect ratio so longest side is max_dim."""
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1: return image # Don't upscale
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def get_masked_input(predictor, image_bgr):
    """
    1. Prompts SAM with the Center Point.
    2. Returns the Mask and the Image with Background removed.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    
    h, w = image_bgr.shape[:2]
    # Prompt: Center point
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    
    # We take index 0 (usually the most concise object mask)
    best_mask = masks[0]
    
    # Create the "Blacked Out" version for the AI
    mask_3ch = np.stack([best_mask]*3, axis=-1)
    masked_img = (image_bgr * mask_3ch).astype(np.uint8)
    
    return best_mask, masked_img

def plot_matches(img1, img2, kpts0, kpts1, matches):
    """
    Custom clean plotting function.
    """
    # Create a new combined image for plotting
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    new_h = max(h1, h2)
    new_w = w1 + w2
    
    out_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    out_img[:h1, :w1] = img1
    out_img[:h2, w1:w1+w2] = img2
    
    # Shift keypoints of second image
    kpts1_shifted = kpts1.copy()
    kpts1_shifted[:, 0] += w1
    
    plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    
    # Draw lines
    for (x0, y0), (x1, y1) in zip(kpts0, kpts1_shifted):
        plt.plot([x0, x1], [y0, y1], c="lime", lw=0.5, alpha=0.7)
        plt.scatter(x0, y0, c="lime", s=3)
        plt.scatter(x1, y1, c="lime", s=3)
    
    plt.axis('off')

# --- MAIN ---
def main():
    # 1. Load Models
    print("Loading Models...")
    # sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(DEVICE)
    # sam_predictor = SamPredictor(sam)
    
    # ALIKED + LightGlue
    extractor = ALIKED(max_num_keypoints=2048, detection_threshold=0.01).eval().to(DEVICE)
    matcher = LightGlue(features='aliked').eval().to(DEVICE)

    # 2. Load & Resize Images
    print("Loading Images...")
    img1_raw = cv2.imread(IMAGE_A_PATH)
    img2_raw = cv2.imread(IMAGE_B_PATH)
    
    if img1_raw is None or img2_raw is None:
        print("❌ Error: Images not found.")
        return

    # RESIZE STEP (Critical for visualization)
    img1_raw = resize_image(img1_raw)
    img2_raw = resize_image(img2_raw)

    # 3. Segment (Get Masks)
    # print("Segmenting...")
    # mask1, img1_masked = get_masked_input(sam_predictor, img1_raw)
    img1_masked = img1_raw
    # mask2, img2_masked = get_masked_input(sam_predictor, img2_raw)
    img2_masked = img2_raw

    # 4. Extract & Match
    print("Matching...")
    def prep_tensor(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(gray / 255.0).float()[None, None].to(DEVICE)

    with torch.no_grad():
        feats1 = extractor.extract(prep_tensor(img1_masked))
        feats2 = extractor.extract(prep_tensor(img2_masked))
        matches01 = matcher({'image0': feats1, 'image1': feats2})
        feats1, feats2, matches01 = [rbd(x) for x in [feats1, feats2, matches01]]
        
        kpts0, kpts1, matches = feats1['keypoints'], feats2['keypoints'], matches01['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # 5. VISUALIZATION (One Clean Figure)
    print(f"✅ Found {len(m_kpts0)} matches.")
    
    plt.figure(figsize=(12, 10))
    
    # Row 1: Mask Quality Check
    # We overlay the mask on the ORIGINAL image in semi-transparent blue
    # This helps you see if SAM missed the legs or head
    def overlay(img, mask):
        overlay = img.copy()
        overlay[mask == 1] = (0, 255, 0) # Green tint on frog
        return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    plt.subplot(2, 2, 1)
    # plt.imshow(cv2.cvtColor(overlay(img1_raw, mask1), cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(img1_raw, cv2.COLOR_BGR2RGB))
    plt.title(f"Segmentation Check {IMAGE_A_PATH}")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    # plt.imshow(cv2.cvtColor(overlay(img2_raw, mask2), cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(img2_raw, cv2.COLOR_BGR2RGB))
    plt.title(f"Segmentation Check {IMAGE_B_PATH}")
    plt.axis('off')

    # Row 2: The Matches
    plt.subplot(2, 1, 2)
    plot_matches(img1_masked, img2_masked, m_kpts0.cpu().numpy(), m_kpts1.cpu().numpy(), matches)
    plt.title(f"LightGlue Matches: {len(m_kpts0)}")
    
    plt.tight_layout()
    plt.savefig("clean_output.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()