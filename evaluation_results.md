## Global Models (MiewID, MegaDescriptor)

Building Model Backbone for efficientnetv2_rw_m model
config.model_name efficientnetv2_rw_m
model_name efficientnetv2_rw_m
final_in_features 2152
python evaluate.py miewid-msv3 cosine --val_csv data/labeled_bbox.csv
Evaluating miewid-msv3-cosine on data/labeled_bbox.csv with device cuda...
Saved embeddings to results/miewid-msv3_labeled_bbox_embeddings.pt
Saved similarity matrix to results/miewid-msv3_labeled_bbox_cosine_similarity.pt
miewid-msv3 | data/labeled_bbox.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.105              0.205              0.373              0.105              0.181              0.0                0.612                                     
Wall-clock 45.04s
Building Model Backbone for efficientnetv2_rw_m model
config.model_name efficientnetv2_rw_m
model_name efficientnetv2_rw_m
final_in_features 2152
python evaluate.py miewid-msv3 cosine --val_csv validation_set.csv
Evaluating miewid-msv3-cosine on validation_set.csv with device cuda...
Loading checkpoint from checkpoints/miewid-msv3_20260510-174925/final_model.ckpt...
Saved embeddings to results/miewid-msv3_20260510-174925_validation_set_embeddings.pt                                                                                                                         
Saved similarity matrix to results/miewid-msv3_20260510-174925_validation_set_cosine_similarity.pt
miewid-msv3_20260510-174925 | validation_set.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.621             0.738             0.922             0.621             0.680             0.0               0.990
Wall-clock 11.00s
python evaluate.py MegaDescriptor-L-224 cosine --val_csv data/labeled_bbox.csv
Evaluating MegaDescriptor-L-224-cosine on data/labeled_bbox.csv with device cuda...
Saved embeddings to results/MegaDescriptor-L-224_labeled_bbox_embeddings.pt
Saved similarity matrix to results/MegaDescriptor-L-224_labeled_bbox_cosine_similarity.pt
MegaDescriptor-L-224 | data/labeled_bbox.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.041             0.128             0.292             0.041             0.104             0.0               0.612
Wall-clock 51.15s
python evaluate.py MegaDescriptor-L-384 cosine --val_csv data/labeled_bbox.csv
Evaluating MegaDescriptor-L-384-cosine on data/labeled_bbox.csv with device cuda...
Saved embeddings to results/MegaDescriptor-L-384_labeled_bbox_embeddings.pt
Saved similarity matrix to results/MegaDescriptor-L-384_labeled_bbox_cosine_similarity.pt
MegaDescriptor-L-384 | data/labeled_bbox.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.024             0.089             0.226             0.024             0.070             0.0               0.530
Wall-clock 51.67s
## Local Models (ALIKED, SIFT)

Creating ALIKED model with max_num_keypoints=1432
python evaluate.py aliked lightglue --val_csv data/labeled_mask_crop.csv
Evaluating aliked-lightglue on data/labeled_mask_crop.csv with device cuda...
Saved embeddings to results/aliked_labeled_mask_crop_embeddings.pt
Saved similarity matrix to results/aliked_labeled_mask_crop_lightglue_similarity.pt
aliked | data/labeled_mask_crop.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.998             0.998             0.998             0.998             0.998             0.0               0.998
Wall-clock 22760.22s
Creating ALIKED model with max_num_keypoints=1432
python evaluate.py aliked classical --val_csv data/labeled_mask_crop.csv
Evaluating aliked-classical on data/labeled_mask_crop.csv with device cuda...
Saved embeddings to results/aliked_labeled_mask_crop_embeddings.pt
Saved similarity matrix to results/aliked_labeled_mask_crop_classical_similarity.pt
aliked | data/labeled_mask_crop.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.933             0.950             0.965             0.933             0.946             0.0               0.989
Wall-clock 4249.07s
Creating SIFT model with max_num_keypoints=1432
python evaluate.py sift lightglue --val_csv data/labeled_mask_crop.csv
Evaluating sift-lightglue on data/labeled_mask_crop.csv with device cuda...
Saved embeddings to results/sift_labeled_mask_crop_embeddings.pt
Saved similarity matrix to results/sift_labeled_mask_crop_lightglue_similarity.pt
sift | data/labeled_mask_crop.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.808             0.843             0.861             0.808             0.843             0.0               0.904
Wall-clock 22901.17s
Creating SIFT model with max_num_keypoints=1432
python evaluate.py sift classical --val_csv data/labeled_mask_crop.csv
Evaluating sift-classical on data/labeled_mask_crop.csv with device cpu...
Saved embeddings to results/sift_labeled_mask_crop_embeddings.pt
Saved similarity matrix to results/sift_labeled_mask_crop_classical_similarity.pt
sift | data/labeled_mask_crop.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.407             0.488             0.601             0.407             0.488             0.0               0.821
Wall-clock 8589.43s

## Two-Stage Pipeline

Building Model Backbone for efficientnetv2_rw_m model
config.model_name efficientnetv2_rw_m
model_name efficientnetv2_rw_m
final_in_features 2152
python evaluate_twostage.py miewid-msv3 aliked --similarity1 cosine --similarity2 lightglue
Loading checkpoint for model1 from checkpoints/miewid-msv3_20260510-174925/final_model.ckpt
Saved embeddings to results/miewid-msv3_20260510-174925_labeled_bbox_embeddings.pt
Saved similarity matrix to results/miewid-msv3_20260510-174925_labeled_bbox_cosine_similarity.pt
Creating ALIKED model with max_num_keypoints=1432
Saved embeddings to results/aliked_labeled_mask_crop_embeddings.pt

Stage 1 (full ranking) metrics:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.556              0.689              0.821              0.556              0.643              0.900              0.933             

Stage 2 (top-100 reranked with aliked+LightGlue) metrics:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.930              0.930              0.930              0.930              0.930              0.933              0.933             

Wall-clock 2434.89s

Building Model Backbone for efficientnetv2_rw_m model
config.model_name efficientnetv2_rw_m
model_name efficientnetv2_rw_m
final_in_features 2152
python evaluate_twostage.py miewid-msv3 aliked --similarity1 cosine --similarity2 lightglue
Loading checkpoint for model1 from checkpoints/miewid-msv3_20260510-174925/final_model.ckpt
Saved embeddings to results/miewid-msv3_20260510-174925_validation_set_embeddings.pt
Saved similarity matrix to results/miewid-msv3_20260510-174925_validation_set_cosine_similarity.pt
Creating ALIKED model with max_num_keypoints=1432
Saved embeddings to results/aliked_validation_set_mask_embeddings.pt

Stage 1 (full ranking) metrics:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.621              0.738              0.922              0.621              0.680              0.971              0.990             

Stage 2 (top-100 reranked with aliked+LightGlue) metrics:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.990              0.990              0.990              0.990              0.990              0.990              0.990             

Wall-clock 460.26s

## Open-Set Analysis

[ID-level] Collected 541 same-id/different-date scores and 97561 different-id/different-date scores
Saved histogram to results/fig4A_hist_aliked.png
Saved precision-recall curve to results/fig4C_pr_aliked.png
Recall 0.95 -> Precision 0.998 at threshold 355.000
Precision 0.95 -> Recall 0.996 at threshold 280.000
Threshold 342.000 -> Precision 0.996, Recall 0.987

[ID-level] Collected 541 same-id/different-date scores and 97561 different-id/different-date scores
Saved histogram to results/fig4B_hist_miewid.png
Saved precision-recall curve to results/fig4D_pr_miewid.png
Recall 0.95 -> Precision 0.018 at threshold 0.127
Precision 0.95 -> Recall 0.006 at threshold 0.670

## ALIKED Keypoint Sweep

Creating ALIKED model with max_num_keypoints=200
python evaluate.py aliked lightglue --val_csv data/labeled_mask_crop.csv
Evaluating aliked-lightglue on data/labeled_mask_crop.csv with device cuda:1...
Saved embeddings to results/aliked_labeled_mask_crop_embeddings.pt
Saved similarity matrix to results/aliked_labeled_mask_crop_lightglue_similarity.pt
aliked | data/labeled_mask_crop.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.876             0.922             0.954             0.876             0.919             0.0               0.980
Wall-clock 8744.75s
Creating ALIKED model with max_num_keypoints=300
python evaluate.py aliked lightglue --val_csv data/labeled_mask_crop.csv
Evaluating aliked-lightglue on data/labeled_mask_crop.csv with device cuda:1...
Saved embeddings to results/aliked_labeled_mask_crop_embeddings.pt
Saved similarity matrix to results/aliked_labeled_mask_crop_lightglue_similarity.pt
aliked | data/labeled_mask_crop.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.959             0.978             0.980             0.959             0.978             0.0               0.994
Wall-clock 12383.00s
Creating ALIKED model with max_num_keypoints=400
python evaluate.py aliked lightglue --val_csv data/labeled_mask_crop.csv
Evaluating aliked-lightglue on data/labeled_mask_crop.csv with device cuda:1...
Saved embeddings to results/aliked_labeled_mask_crop_embeddings.pt
Saved similarity matrix to results/aliked_labeled_mask_crop_lightglue_similarity.pt
aliked | data/labeled_mask_crop.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.991             0.993             0.996             0.991             0.993             0.0               0.996
Wall-clock 11904.07s
Creating ALIKED model with max_num_keypoints=500
python evaluate.py aliked lightglue --val_csv data/labeled_mask_crop.csv
Evaluating aliked-lightglue on data/labeled_mask_crop.csv with device cuda:1...
Saved embeddings to results/aliked_labeled_mask_crop_embeddings.pt
Saved similarity matrix to results/aliked_labeled_mask_crop_lightglue_similarity.pt
aliked | data/labeled_mask_crop.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.989             0.994             0.996             0.989             0.994             0.0               0.998
Wall-clock 11474.46s
Creating ALIKED model with max_num_keypoints=600
python evaluate.py aliked lightglue --val_csv data/labeled_mask_crop.csv
Evaluating aliked-lightglue on data/labeled_mask_crop.csv with device cuda:1...
Saved embeddings to results/aliked_labeled_mask_crop_embeddings.pt
Saved similarity matrix to results/aliked_labeled_mask_crop_lightglue_similarity.pt
aliked | data/labeled_mask_crop.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.994             0.994             0.994             0.994             0.994             0.0               0.998
Wall-clock 14144.66s
Creating ALIKED model with max_num_keypoints=700
python evaluate.py aliked lightglue --val_csv data/labeled_mask_crop.csv
Evaluating aliked-lightglue on data/labeled_mask_crop.csv with device cuda:1...
Saved embeddings to results/aliked_labeled_mask_crop_embeddings.pt
Saved similarity matrix to results/aliked_labeled_mask_crop_lightglue_similarity.pt
aliked | data/labeled_mask_crop.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.993             0.996             0.996             0.993             0.996             0.0               0.998
Wall-clock 13700.03s
Creating ALIKED model with max_num_keypoints=800
python evaluate.py aliked lightglue --val_csv data/labeled_mask_crop.csv
Evaluating aliked-lightglue on data/labeled_mask_crop.csv with device cuda:1...
Saved embeddings to results/aliked_labeled_mask_crop_embeddings.pt
Saved similarity matrix to results/aliked_labeled_mask_crop_lightglue_similarity.pt
aliked | data/labeled_mask_crop.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.998             0.998             0.998             0.998             0.998             0.0               0.998
Wall-clock 17041.35s

Creating ALIKED model with max_num_keypoints=900
python evaluate.py aliked lightglue --val_csv data/labeled_mask_crop.csv
Evaluating aliked-lightglue on data/labeled_mask_crop.csv with device cuda:1...
Saved embeddings to results/aliked_labeled_mask_crop_embeddings.pt
Saved similarity matrix to results/aliked_labeled_mask_crop_lightglue_similarity.pt
aliked | data/labeled_mask_crop.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy
0.996             0.998             0.998             0.996             0.998             0.0               0.998
Wall-clock 16978.83s
Creating ALIKED model with max_num_keypoints=1000
python evaluate.py aliked lightglue --val_csv data/labeled_mask_crop.csv
Evaluating aliked-lightglue on data/labeled_mask_crop.csv with device cuda:1...
Saved embeddings to results/aliked_labeled_mask_crop_embeddings.pt
Saved similarity matrix to results/aliked_labeled_mask_crop_lightglue_similarity.pt
aliked | data/labeled_mask_crop.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-100 accuracy   Top-200 accuracy   mAP@R             
0.998              0.998              0.998              0.998              0.998              0.998              0.998              0.971             
Wall-clock 16749.37s