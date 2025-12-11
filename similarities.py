import torch
import numpy as np


def get_similarity_function(similarity_name, features=None):
    if similarity_name.lower() == 'cosine':
        try:
            from pytorch_metric_learning.distances import CosineSimilarity
        except ImportError:
            from torchmetrics import CosineSimilarity
        return CosineSimilarity()
    elif similarity_name.lower() == 'lightglue':        
        return LightGlueSimilarity(features=features or 'aliked')
    else:
        raise ValueError(f"Unknown similarity function: {similarity_name}")

class LightGlueSimilarity():
    def __init__(self, features='aliked'):
        from lightglue import LightGlue
        from lightglue.utils import rbd
        self.rbd = rbd
        model = LightGlue(features=features)
        self.model = model.eval()
        self.model.compile(mode='reduce-overhead') #

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def __call__(self, query_emb, ref_emb):
        n, m = len(query_emb), len(ref_emb)
        M = np.zeros((n, m), dtype=np.int32)
        for i, q in enumerate(query_emb):
            for j, r in enumerate(ref_emb):
                with torch.inference_mode():
                    matches = self.model({"image0": q, "image1": r})
                    matches = self.rbd(matches)
                match_idx = matches["matches"]
                if isinstance(match_idx, torch.Tensor) and match_idx.ndim == 3:
                    match_idx = match_idx[0]
                M[i,j] = int(match_idx.shape[0])
        return M
