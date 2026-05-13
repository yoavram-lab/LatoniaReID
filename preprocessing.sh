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

# Verify inputs exist
if [ ! -f labeled.csv ]; then
  echo "Error: labeled.csv not found"
  echo "   Please ensure labeled.csv exists in the repo root"
  exit 1
fi

if [ ! -d data/labeled ]; then
  echo "Error: data/labeled symlink not set up"
  echo "   Please create symlink: ln -s /path/to/zenodo data"
  exit 1
fi

# Path 1: bbox crop (for global models)
echo "Step 1: MegaDetector bbox cropping..."
python3 crop.py labeled.csv \
  --output_root data/labeled_bbox \
  --no-pad

echo "Saved cropped images to data/labeled_bbox/"
echo "Wrote labeled_bbox.csv"
echo ""

# Path 2: SAM masking (for local models)
echo "Step 2: SAM masking..."
python3 masking.py labeled_bbox.csv \
  --output_root data/labeled_bbox_mask \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --sam_type vit_b \
  --device cuda

echo "Saved masked images to data/labeled_bbox_mask/"
echo "Wrote labeled_bbox_mask.csv"
echo ""

echo "Preprocessing complete"
echo ""
echo "Output CSVs:"
echo "  - data/labeled_bbox.csv (paths to data/labeled_bbox/)"
echo "  - data/labeled_bbox_mask.csv (paths to data/labeled_bbox_mask/)"
echo ""
echo "Next: ./run_experiments.sh"
