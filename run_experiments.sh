#!/bin/bash
# Reproduce all paper evaluation results
# Produces: evaluation_results.md with all model metrics
# Requires: labeled_crop.csv, labeled_mask.csv (from preprocessing.sh)

set -e

DEVICE=${DEVICE:-cuda}

echo "Latonia ReID — Running all evaluations"
echo ""

# Create results directory
mkdir -p results

# Initialize evaluation log
> evaluation_results.md

echo "Running evaluations..."
echo ""

# ====================== GLOBAL MODELS ======================
echo "## Global Models (MiewID, MegaDescriptor)" >> evaluation_results.md
echo "" >> evaluation_results.md

echo "  MiewID-msv3 (zero-shot)..."
python evaluate.py miewid-msv3 cosine \
  --val_csv data/labeled_bbox.csv --device $DEVICE --ignore_cache >> evaluation_results.md

echo "  MiewID-msv3 (finetuned)..."
python evaluate.py miewid-msv3 cosine \
  --checkpoint checkpoints/miewid-msv3_20260510-174925/final_model.ckpt \
  --val_csv validation_set.csv --device $DEVICE --ignore_cache >> evaluation_results.md

echo "  MegaDescriptor-L-224..."
python evaluate.py MegaDescriptor-L-224 cosine \
  --val_csv data/labeled_bbox.csv --device $DEVICE --ignore_cache >> evaluation_results.md

echo "  MegaDescriptor-L-384..."
python evaluate.py MegaDescriptor-L-384 cosine \
  --val_csv data/labeled_bbox.csv --device $DEVICE --ignore_cache >> evaluation_results.md

# ====================== LOCAL MODELS ======================
echo "## Local Models (ALIKED, SIFT)" >> evaluation_results.md
echo "" >> evaluation_results.md

echo "  ALIKED+LightGlue..."
python evaluate.py aliked lightglue \
  --val_csv data/labeled_mask_crop.csv --device $DEVICE --ignore_cache >> evaluation_results.md

echo "  ALIKED+Classical..."
python evaluate.py aliked classical \
  --val_csv data/labeled_mask_crop.csv --device $DEVICE --ignore_cache >> evaluation_results.md

echo "  SIFT+LightGlue..."
python evaluate.py sift lightglue \
  --val_csv data/labeled_mask_crop.csv --device $DEVICE --ignore_cache >> evaluation_results.md

echo "  SIFT+Classical..."
python evaluate.py sift classical \
  --val_csv data/labeled_mask_crop.csv --device cpu --ignore_cache >> evaluation_results.md

# # ====================== TWO-STAGE PIPELINE ======================
echo "## Two-Stage Pipeline" >> evaluation_results.md
echo "" >> evaluation_results.md

echo "  Two-stage (MiewID-FT + ALIKED+LG, k=200)..."
python evaluate_twostage.py miewid-msv3 aliked \
  --stage1_csv data/labeled_bbox.csv \
  --stage2_csv data/labeled_mask_crop.csv \
  --checkpoint1 checkpoints/miewid-msv3_20260510-174925/final_model.ckpt \
  --device $DEVICE --top_k 200 --ignore_cache >> evaluation_results.md


python evaluate_twostage.py miewid-msv3 aliked \
  --stage1_csv data/validation_set.csv \
  --stage2_csv data/validation_set_mask.csv \
  --checkpoint1 checkpoints/miewid-msv3_20260510-174925/final_model.ckpt \
  --device $DEVICE --top_k 200 --ignore_cache >> evaluation_results.md

# # ====================== KEYPOINT SWEEP (Figure 3B) ======================
# echo "## ALIKED Keypoint Sweep" >> evaluation_results.md
# echo "" >> evaluation_results.md

# echo "  Running keypoint sweep M=200..1432..."
# for M in 200 400 600 800 0 1000 1200; do
#   python evaluate.py aliked-$M lightglue \
#     --val_csv data/labeled_mask_crop.csv --device $DEVICE --ignore_cache >> evaluation_results.md
# done

# ====================== OPEN-SET ANALYSIS (Figure 4) ======================
echo "## Open-Set Analysis" >> evaluation_results.md
echo "" >> evaluation_results.md

echo "  ALIKED+LightGlue histograms and PR curve..."
python openset.py \
  --sim-path results/aliked_labeled_mask_crop_lightglue_similarity.pt \
  --csv-path data/labeled_mask_crop.csv \
  --out results/fig4A_hist_aliked.pdf \
  --pr-out results/fig4C_pr_aliked.pdf \
  --threshold 344



echo "  MiewID-FT histograms and PR curve..."
python openset.py \
  --sim-path results/miewid-msv3_20260510-174925_labeled_bbox_cosine_similarity.pt \
  --csv-path data/labeled_bbox.csv \
  --out results/fig4B_hist_miewid.pdf \
  --pr-out results/fig4D_pr_miewid.pdf

# ====================== GENERATE FIGURES ======================
echo "## Generating figures" >> evaluation_results.md
echo "" >> evaluation_results.md

echo "  Producing Table 1 and Figure 3..."
python compare_performance.py --plot figures/figure3A.pdf
python compare_performance.py --plot figures/figure3B.pdf --max_num_keypoints

echo ""
echo "All evaluations complete"
echo "Results saved to evaluation_results.md"
echo "Figures saved to figures/"
echo ""
echo "Next: Review evaluation_results.md and compare with paper numbers"
