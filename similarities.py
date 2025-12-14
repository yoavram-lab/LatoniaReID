import torch
import numpy as np
import cv2


def get_similarity_function(similarity_name, features=None):
    if similarity_name.lower() == 'cosine':
        try:
            from pytorch_metric_learning.distances import CosineSimilarity
        except ImportError:
            from torchmetrics import CosineSimilarity
        return CosineSimilarity()
    elif similarity_name.lower() == 'lightglue':        
        return LightGlueSimilarity(features=features or 'aliked')
    elif similarity_name.lower() == 'classical':
        return ClassicalSimilarity()
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


class ClassicalSimilarity():
    """
    BF matcher with Lowe ratio test + RANSAC homography inlier counting.
    Expects feature dicts containing 'keypoints' (N, 2) and 'descriptors' (N, D).
    """
    def __init__(self, ratio_thresh: float = 0.75, ransac_thresh: float = 5.0):
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh = ransac_thresh
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def to(self, device):
        # CPU-only, keep API compatibility.
        return self

    def _to_numpy(self, arr):
        if torch.is_tensor(arr):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _extract(self, feats):
        if not isinstance(feats, dict):
            return None, None
        kpts = feats.get("keypoints")
        if "descriptors" in feats:
            desc = feats.get("descriptors")
        elif "descriptors0" in feats:
            desc = feats.get("descriptors0")
        elif "descriptors1" in feats:
            desc = feats.get("descriptors1")
        else:
            desc = None
        if kpts is None or desc is None:
            return None, None
        kpts = self._to_numpy(kpts)
        desc = self._to_numpy(desc)
        if kpts.ndim == 3 and kpts.shape[0] == 1:
            kpts = kpts[0]
        if desc.ndim == 3 and desc.shape[0] == 1:
            desc = desc[0]
        kpts = kpts.astype(np.float32)
        desc = desc.astype(np.float32)
        return kpts, desc

    def _match_pair(self, q, r):
        qk, qd = self._extract(q)
        rk, rd = self._extract(r)
        if qk is None or rk is None or len(qk) == 0 or len(rk) == 0:
            return 0
        matches_knn = self.matcher.knnMatch(qd, rd, k=2)
        good = []
        for mn in matches_knn:
            if len(mn) < 2:
                continue
            m, n = mn
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)
        if len(good) < 4:
            return len(good)
        src_pts = np.float32([qk[m.queryIdx] for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([rk[m.trainIdx] for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
        if mask is None:
            return len(good)
        return int(mask.ravel().sum())

    def __call__(self, query_emb, ref_emb):
        n, m = len(query_emb), len(ref_emb)
        M = np.zeros((n, m), dtype=np.int32)
        for i, q in enumerate(query_emb):
            for j, r in enumerate(ref_emb):
                M[i, j] = self._match_pair(q, r)
        return M
