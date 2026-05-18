#!/bin/bash
# Preprocessing pipeline for Latonia ReID
#
# Prerequisites:
#   - labeled.csv exists (paths point to data/labeled/)
#   - data symlink is set up
#
# Inputs: labeled.csv
# Outputs:
#   - labeled_crop.csv (paths point to data/labeled_bbox/)
#   - labeled_mask.csv (paths point to data/labeled_mask/)
#   - data/labeled_bbox/ (cropped images)
#   - data/labeled_mask/ (SAM-masked images)
#
# All paths in CSVs are relative to repo root, making scripts portable.

set -e

echo "Latonia ReID — Preprocessing"
echo ""


# Path 1: bbox crop (for global models)
echo "Part 1: MegaDetector bbox cropping..."
python3 crop.py labeled.csv \
  --output_root data/labeled_bbox \
  --no-pad

echo "Saved cropped images to data/labeled_bbox/"
echo "Wrote labeled_bbox.csv"
echo ""

# Path 2: SAM masking (for local models)
echo "Part 2: SAM masking..."
python3 masking.py labeled.csv \
  --output_root data/labeled_mask \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --sam_type vit_b \
  --device cuda

echo "Saved masked images to data/labeled_mask/"
echo "Wrote labeled_mask.csv"
echo ""

echo "Preprocessing complete"
echo ""
echo "Output CSVs:"
echo "  - data/labeled_bbox.csv (paths to data/labeled_bbox/)"
echo "  - data/labeled_mask.csv (paths to data/labeled_mask/)"
echo ""
echo "Next: ./run_experiments.sh"
