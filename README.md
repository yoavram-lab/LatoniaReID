# LatoniaDetector
Identify Lationa frog from previous images



```
# segment images
python app.py segment

# train ArcFace models
python training.py --data-folder=data
python training.py --data-folder=segmented_white
# use training_history.ipynb to generate training plot

# check available models
python app.py list-models

# generate all embeddings after updating checkpoint names in embedding.sh
source embedding.sh 

# run validation after updating checkpoint names in validation.sh
source validation.sh > validation.log
# use tpr-fpr.ipynb to generate ROC curve
# use umap.ipynb to generate UMAP plots
```