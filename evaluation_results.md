## Global Models (MiewID, MegaDescriptor)

Building Model Backbone for efficientnetv2_rw_m model
config.model_name efficientnetv2_rw_m
model_name efficientnetv2_rw_m
final_in_features 2152
Evaluating miewid-msv3-cosine on data/labeled_bbox.csv with device cuda...
Saved embeddings to results/miewid-msv3_labeled_bbox_embeddings.pt
Saved similarity matrix to results/miewid-msv3_labeled_bbox_cosine_similarity.pt
miewid-msv3 | data/labeled_bbox.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy  
0.105              0.205              0.373              0.105              0.181              0.527              0.612             
Wall-clock 42.38s

Building Model Backbone for efficientnetv2_rw_m model
config.model_name efficientnetv2_rw_m
model_name efficientnetv2_rw_m
final_in_features 2152
Evaluating miewid-msv3-cosine on validation_set.csv with device cuda...
Loading checkpoint from checkpoints/miewid-msv3_20250808-143106/final_model.ckpt...
Saved embeddings to results/miewid-msv3_20250808-143106_validation_set_embeddings.pt
Saved similarity matrix to results/miewid-msv3_20250808-143106_validation_set_cosine_similarity.pt
miewid-msv3_20250808-143106 | validation_set.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy  
0.602              0.738              0.913              0.602              0.670              0.961              0.990             
Wall-clock 8.75s
Evaluating MegaDescriptor-L-224-cosine on data/labeled_bbox.csv with device cuda:1...
Saved embeddings to results/MegaDescriptor-L-224_labeled_bbox_embeddings.pt
Saved similarity matrix to results/MegaDescriptor-L-224_labeled_bbox_cosine_similarity.pt
MegaDescriptor-L-224 | data/labeled_bbox.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy  
0.041              0.128              0.292              0.041              0.104              0.453              0.612             
Wall-clock 48.83s

Evaluating MegaDescriptor-L-384-cosine on data/labeled_bbox.csv with device cuda:1...
Saved embeddings to results/MegaDescriptor-L-384_labeled_bbox_embeddings.pt
Saved similarity matrix to results/MegaDescriptor-L-384_labeled_bbox_cosine_similarity.pt
MegaDescriptor-L-384 | data/labeled_bbox.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy  
0.024              0.089              0.226              0.024              0.070              0.401              0.530             
Wall-clock 49.52s

Creating ALIKED model with max_num_keypoints=1432
Evaluating aliked-classical on data/labeled_bbox.csv with device cuda:1...
Saved embeddings to results/aliked_labeled_bbox_embeddings.pt
Saved similarity matrix to results/aliked_labeled_bbox_classical_similarity.pt
aliked | data/labeled_bbox.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy  
0.900              0.926              0.954              0.900              0.926              0.969              0.976             
Wall-clock 3976.53s

Creating ALIKED model with max_num_keypoints=1432
Evaluating aliked-lightglue on data/labeled_mask.csv with device cuda...Saved embeddings to results/aliked_labeled_mask_embeddings.pt
Saved similarity matrix to results/aliked_labeled_mask_lightglue_similarity.pt
aliked | data/labeled_mask.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy  
0.991              0.993              0.996              0.991              0.993              0.996              0.996             
Wall-clock 26593.20s

Creating ALIKED model with max_num_keypoints=1432
Evaluating aliked-classical on data/labeled_mask.csv with device cuda:1...
Saved embeddings to results/aliked_labeled_mask_embeddings.pt
Saved similarity matrix to results/aliked_labeled_mask_classical_similarity.pt
aliked | data/labeled_mask.csv:
Top-1 ID accuracy  Top-3 ID accuracy  Top-10 ID accuracy Top-1 accuracy     Top-3 accuracy     Top-50 accuracy    Top-100 accuracy  
0.908              0.930              0.945              0.908              0.930              0.967              0.976             
Wall-clock 4434.30s
