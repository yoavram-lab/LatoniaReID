import torch
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm # Install with: pip install tqdm
from segment_anything import sam_model_registry, SamPredictor

# --- CONFIGURATION ---
INPUT_ROOT = "rotated"       # Folder containing your images
OUTPUT_ROOT = "rotated_sam"  # Where masks will be saved
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Supported image extensions
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def initialize_sam():
    if not os.path.exists(SAM_CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {SAM_CHECKPOINT}")
    
    print(f"Loading {MODEL_TYPE} model on {DEVICE}...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    return SamPredictor(sam)

def process_dataset():
    # 1. Setup
    predictor = initialize_sam()
    input_path = Path(INPUT_ROOT)
    output_path = Path(OUTPUT_ROOT)

    # 2. Collect all image files first (for progress bar)
    all_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if Path(file).suffix.lower() in VALID_EXTS:
                all_files.append(Path(root) / file)
    
    print(f"Found {len(all_files)} images. Starting processing...")

    # 3. Iterate and Process
    counter = 0
    for img_path in tqdm(all_files, desc="Segmenting"):
        # if counter == 30: break
        try:
            # --- A. Read Image ---
            # (cv2.imread doesn't handle special characters/paths well, numpy fix used)
            image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: Could not read {img_path}")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # --- B. Set Image in SAM ---
            predictor.set_image(image_rgb)
            
            # --- C. Center Point Prompt ---
            h, w = image.shape[:2]
            input_point = np.array([[w // 2, h // 2]])
            input_label = np.array([1]) # Foreground
            
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            # --- D. Select Mask 1 (Index 0) ---
            # User observation: Mask 1 is consistently best
            selected_mask = masks[0] 
            
            # --- E. Save Result ---
            # Construct output path preserving structure
            # e.g. input/frog1/A.jpg -> output/frog1/A.png
            relative_path = img_path.relative_to(input_path)
            save_path = output_path / relative_path.with_suffix('.png')
            
            # Ensure folder exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to uint8 (0=Black, 255=White)
            mask_uint8 = (selected_mask * 255).astype(np.uint8)
            
            # Use imencode/tofile to handle special chars in paths if needed
            is_success, buffer = cv2.imencode(".png", mask_uint8)
            if is_success:
                with open(str(save_path), "wb") as f:
                    f.write(buffer)
                counter += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"\n✅ Processing Complete. Masks saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ Error: Input folder '{INPUT_ROOT}' does not exist.")
    else:
        process_dataset()