# LatoniaDetector
Re-ID Lationa frogs from images

- Checkpoints:
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