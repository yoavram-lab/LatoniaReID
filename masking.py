from __future__ import annotations
import argparse
import torch
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm # Install with: pip install tqdm
from segment_anything import sam_model_registry, SamPredictor

# Supported image extensions
VALID_EXTS = {'.jpg', '.jpeg'}

def initialize_sam(sam_checkpoint, sam_type, device):
    if not os.path.exists(sam_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {sam_checkpoint}")
    
    print(f"Loading {sam_type} model on {device}...")
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)

def process_folder(predictor, data_root, mask_root):
    input_path = Path(data_root)
    output_path = Path(mask_root)

    # 2. Collect all image files first (for progress bar)
    all_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if Path(file).suffix.lower() in VALID_EXTS:
                all_files.append(Path(root) / file)
    
    print(f"Found {len(all_files)} images. Starting processing...")

    # 3. Iterate and Process
    for img_path in tqdm(all_files, desc="Masking"):
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
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"\n✅ Processing Complete. Masks saved to: {mask_root}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Masking images with SegmentAnythingModel.")
    parser.add_argument("--data-root", type=Path, default=Path("rotated"))
    parser.add_argument("--mask-root", type=Path, default=Path("rotated_sam"))
    parser.add_argument("--device", type=str, default="cpu", help="Force device (cuda or cpu).")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    sam_checkpoint = "sam_vit_b_01ec64.pth"
    sam_type = "vit_b"

    predictor = initialize_sam(sam_checkpoint, sam_type, device)

    process_folder(predictor, args.data_root, args.mask_root)




if __name__ == "__main__":
    main()