# Near-perfect photo-ID of the Hula painted frog with zero-shot deep local-feature matching

This repository implements a **non-invasive photo-identification system for individual Hula painted frogs** (an endangered species), using deep learning-based local and global feature matching. The system achieves **98% accuracy** in closed-set individual frog identification using zero-shot deep local-feature matching (ALIKED+LightGlue).

**Paper:** [arXiv:2601.08798](https://arxiv.org/abs/2601.08798)  
**Key finding:** Zero-shot local feature matching significantly outperforms fine-tuned global embedding models for individual frog re-identification.

## System Overview

- **Dataset:** 1,232 ventral images from 191 endangered Hula painted frogs
- **Best approach:** Two-stage pipeline (MiewID global + ALIKED+LightGlue local)
- **Key models evaluated:** MiewID, MegaDescriptor, ALIKED+LightGlue, SIFT+LightGlue
- **Application:** Practical, non-invasive field monitoring for conservation

## Quick Start

### 1. Setup Environment

```bash
# Create conda environments (if not already done)
conda env create -f requirements.txt -n glue

# Activate environment for preprocessing and evaluation
conda activate glue
```

### 2. Download Data

**Dataset:** [Zenodo: 20026776](https://zenodo.org/records/20026776)  
1,232 ventral photographs from 191 Hula painted frogs (labeled/unlabeled splits)

```bash
# Create symlink to Zenodo data directory
ln -s /path/to/zenodo/zenodo_data data
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
│
├── evaluate.py                        # Single-model evaluation
├── evaluate_twostage.py               # Two-stage evaluation
├── compare_performance.py             # Generate result tables/plots
├── openset.py                         # Open-set analysis (PR curves)
│
├── batch_prediction/
│   ├── batch_predict.py               # Generate predictions for unlabeled data
│   ├── batch_predictions_stats.py     # Analyze expert review statistics
│   └── batch_prediction_app.py        # Gradio UI for expert review
│
└── results/                           # Evaluation outputs (generated)
    ├── evaluation_results.md          # Metric summary
    └── *.pdf                          # Figures and plots
```

**Note:** Data directory (`data/`) and preprocessed results are created locally via `preprocessing.sh` and are not committed. Download the dataset from [Zenodo](https://zenodo.org/records/20026776) and set up as a symlink.

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

Generate predictions for unlabeled images and review results:

```bash
# Generate predictions
python batch_prediction/batch_predict.py \
  --unlabeled_csv data/unlabeled_mask.csv \
  --labeled_csv data/labeled_mask.csv \
  --output batch_predictions.json

# Analyze expert review statistics
python batch_prediction/batch_predictions_stats.py --input batch_predictions.json

# Review predictions interactively with Gradio UI
python batch_prediction/batch_prediction_app.py
# Open http://localhost:7860 in browser
```

## Expected Results

From the paper (closed-set evaluation on 191 frogs):

| Model | Method | Top-1 Accuracy |
|-------|--------|---|
| **ALIKED+LightGlue** | **Zero-shot local feature matching** | **98.0%** ✓ |
| SIFT+LightGlue | Local feature matching | 79.0% |
| MiewID-msv3 (finetuned) | Global embedding (trained) | 61.2% |
| MiewID-msv3 (zero-shot) | Global embedding (zero-shot) | 10.5% |
| MegaDescriptor-L-224 | Global embedding (zero-shot) | 4.1% |
| MegaDescriptor-L-384 | Global embedding (zero-shot) | 2.4% |

**Key insight:** Zero-shot local feature matching (ALIKED+LightGlue) **substantially outperforms** fine-tuned global models, achieving near-perfect individual frog identification without any species-specific training data.

## Training

To finetune MiewID on Hula painted frog data:

```bash
conda activate Latonia  # Different environment for training

python train.py \
  --train_csv train_set.csv \
  --val_csv validation_set.csv \
  --checkpoint path/to/miewid-msv3 \
  --margin 0.4 --scale 30 \
  --batch_size 24 --epochs 100
```

**Note:** Global embedding models (like MiewID) are less effective than zero-shot local matching for this task. The paper demonstrates that **ALIKED+LightGlue achieves 98% accuracy without any frog-specific training**, making finetuning unnecessary.

## Citation

If you use this work, please cite the paper:

```bibtex
@article{Yesharim2026,
  title={Near-perfect photo-ID of the Hula painted frog with zero-shot deep local-feature matching},
  author={Maayan Yesharim and R. G. Bina Perl and Uri Roll and Sarig Gafny and Eli Geffen and Yoav Ram},
  journal={arXiv preprint arXiv:2601.08798},
  year={2026}
}
```

**arXiv:** https://arxiv.org/abs/2601.08798

## License

[Project-specific license]
