# LatoniaDetector

Re-ID Latonia frogs from images.

## Checkpoints
 - `miewid-msv3_20250720-154043` 2152, no proj in head, backbone finetuning
 - `miewid-msv3_20250719-150002` 1024D
 - `miewid-msv3_20250719-084418` 512D
 - `miewid-msv3_20250718-225339 ` 256D
 - `miewid-msv3_20250714-184156` 128D
 - `miewid_final_07JUL25` 128D, trained on all bina
- `miewid-msv3_20250710-234908` 128D, ???

```
# bbox images
python detect_bbox.py data bbox.json --crop
mv data_bbox bbox

# finetune model
python training_miewid.py --train_csv bina_photos_train_bbox.csv --val_csv bina_photos_validation_bbox.csv
# see output for the filename of the checkpoint

# embed images
python embedding.py miewid-msv3 bbox
python embedding.py miewid-msv3_DATETIME bbox # repalce DATETIME with checkpoint name

# run validation
python validation.py miewid-msv3 bbox
python validation.py miewid-msv3_DATETIME bbox # repalce DATETIME with checkpoint name
```

## Evaluation and comparison

- Single-stage evaluation:
  - `python evaluate.py <model> [similarity] [--val_csv <csv>]`
  - Similarities include `lightglue`, `classical`, and `cosine` (as supported by the model/matcher).
- Two-stage evaluation:
  - `python evaluate_twostage.py <model_stage1> <model_stage2> --similarity1 cosine --similarity2 lightglue`
  - Use `--stage1_csv` / `--stage2_csv` for validation sets.
- Summarize `evaluation_results.md` into tables/plots:
  - `python compare_performance.py --format tsv`
  - `python compare_performance.py --plot results/evaluation_results.pdf`
  - `python compare_performance.py --max_num_keypoints --plot results/evaluation_results_max_kpts.pdf`

## Open-set analysis

Generate histograms and precision-recall curves from a similarity matrix:
`python openset.py --sim-path results/<similarity>.pt --csv-path <csv> --out openset_hist.png --pr-out openset_pr.png`
