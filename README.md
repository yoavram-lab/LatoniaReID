# Latonia ReID: Re-identification of Latonia frogs

This repository implements a **Person Re-ID system for Latonia frogs**, using global and local feature matching approaches. The paper evaluates multiple models (MiewID, MegaDescriptor, ALIKED+LightGlue, SIFT) and provides a two-stage pipeline for improved accuracy.

## Quick Start

### 1. Setup Environment

```bash
# Create conda environments (if not already done)
conda env create -f requirements.txt -n glue

# Activate environment for preprocessing and evaluation
conda activate glue
```

### 2. Download Data

The repo requires Zenodo data (labeled and unlabeled images). Set it up as a symlink:

```bash
# Create symlink to Zenodo data directory
ln -s /path/to/zenodo/data data

# Verify structure
ls data/labeled/          # Raw labeled images
ls data/unlabeled/        # Raw unlabeled images
```

### 3. Run Preprocessing

Preprocess images using MegaDetector (bbox cropping) and SAM (masking):

```bash
conda activate glue
./preprocessing.sh
```

**Output CSVs:**
- `data/labeled_crop.csv` — paths to bbox-cropped images
- `data/labeled_mask.csv` — paths to SAM-masked images

### 4. Reproduce Paper Results

Run all evaluations (takes ~2–4 hours on GPU):

```bash
conda activate glue
./run_experiments.sh
```

**Output:** `evaluation_results.md` with all model metrics, figures saved to `results/`

## File Structure

```
LatoniaReIDpaper/
├── README.md                          # This file
├── preprocessing.sh                   # Bbox cropping + masking pipeline
├── run_experiments.sh                 # Full evaluation pipeline
├── crop.py                            # MegaDetector bbox cropping (CSV-based)
├── masking.py                         # SAM masking (CSV-based)
├── train_set.csv                      # Training split (1,000 labeled images)
├── validation_set.csv                 # Validation split (232 labeled images)
├── labeled.csv                        # Full labeled dataset reference
│
├── evaluate.py                        # Single-model evaluation
├── evaluate_twostage.py               # Two-stage evaluation
├── compare_performance.py             # Generate result tables/plots
├── openset.py                         # Open-set analysis (PR curves)
│
├── batch_prediction/
│   ├── batch_predict.py               # Generate predictions for unlabeled data
│   └── batch_predictions_stats.py     # Analyze prediction results
│
├── apps/
│   └── batch_prediction_app.py        # Gradio UI for expert review
│
├── data/                              # Local data (created by preprocessing)
│   ├── labeled/                       # Raw labeled images (symlink to Zenodo)
│   ├── labeled_bbox/                  # Bbox-cropped labeled images
│   ├── labeled_mask.csv               # CSV pointing to masked images
│   ├── unlabeled/                     # Raw unlabeled images (symlink)
│   └── unlabeled_bbox/, unlabeled_mask/  # Preprocessed unlabeled images
│
└── results/                           # Evaluation outputs
    ├── evaluation_results.md          # Metric summary
    └── *.pdf                          # Figures and plots
```

## Usage

### Single-Model Evaluation

```bash
# Global models (use bbox-cropped images)
python evaluate.py MegaDescriptor-L-224 cosine --val_csv data/labeled_bbox.csv --device cuda

# Local models (use SAM-masked images)
python evaluate.py aliked lightglue --val_csv data/labeled_mask.csv --device cuda

# With checkpoint
python evaluate.py miewid-msv3 cosine \
  --checkpoint checkpoints/miewid-msv3_20250808-143106/ckpt \
  --val_csv validation_set.csv --device cuda
```

### Two-Stage Evaluation

Combine a global model (stage 1) with local matching (stage 2):

```bash
python evaluate_twostage.py miewid-msv3 aliked \
  --stage1_csv validation_set.csv \
  --stage2_csv data/labeled_mask.csv \
  --checkpoint1 checkpoints/miewid-msv3_20250808-143106/ckpt \
  --device cuda --top_k 100
```

### Batch Prediction (Unlabeled Data)

Generate predictions for unlabeled images and review them interactively:

```bash
# Generate predictions
python batch_prediction/batch_predict.py \
  --unlabeled_csv data/unlabeled_mask.csv \
  --labeled_csv data/labeled_mask.csv \
  --output batch_predictions.json

# Review predictions in Gradio UI
python apps/batch_prediction_app.py
# Open http://localhost:7860 in browser
```

## Expected Results

From the paper (with best configuration: s=30, α=0.4, m=1, k=1):

| Model | Top-1 ID | Top-3 ID | Top-10 ID |
|-------|----------|----------|-----------|
| MiewID-msv3 (zero-shot) | 10.5% | 20.7% | — |
| MegaDescriptor-L-224 (zero-shot) | ? | ? | — |
| MegaDescriptor-L-384 (zero-shot) | ? | ? | — |
| MiewID-msv3 (finetuned) | 61.2% | — | 74.8% |
| ALIKED+LightGlue (local) | 97.8% | — | — |
| SIFT+LightGlue (local) | 79.0% | — | — |

## Training

To finetune MiewID on your data:

```bash
conda activate Latonia  # Different environment for training

python train.py \
  --train_csv train_set.csv \
  --val_csv validation_set.csv \
  --checkpoint path/to/miewid-msv3 \
  --margin 0.4 --scale 30 \
  --batch_size 32 --epochs 100
```

## Citation

If you use this work, please cite the paper:

```bibtex
[Citation info from paper]
```

## License

[Project-specific license]
