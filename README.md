# LatoniaDetector
Re-ID Lationa frogs from images

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