import os
import click
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from pytorch_metric_learning.distances import CosineSimilarity
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from image_transform import ZoomCenterCrop
from models import get_model, load_checkpoint
from similarities import get_similarity_function
from datasets import DataFrameDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = 'cpu'

def embed(model, dataset, device=device, batch_size=32, num_workers=4):
    model.eval()
    model = model.to(device)
    embeddings = []
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,   # any size that fits GPU
        shuffle=False,           # keep deterministic order
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True          # for faster data transfer to GPU
    )
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Embedding on {device}", leave=False, total=(len(dataset) + batch_size - 1) // batch_size)
        for item in pbar:
            if isinstance(item, (tuple, list)):
                data = item[0]
            else:
                data = item
            data = data.to(device)
            if device.startswith('cuda'):
                with torch.amp.autocast(device):
                    emb = model(data)
            else:
                emb = model(data)
            embeddings.append(emb)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def labels_and_scores(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates):
    n = len(query_labels)
    m = len(ref_labels)
    assert similarity_matrix.shape == (n, m), f"Shape mismatch: {similarity_matrix.shape} != ({n}, {m})"

    labels = []
    scores = []
    for i in range(n):
        qi_label = query_labels[i]
        qi_date = query_dates[i]
        sim_row = similarity_matrix[i]
        for j in range(m):
            if qi_date == ref_dates[j]: # ignore pairs from same date
                continue
            labels.append(qi_label == ref_labels[j])  # 1 if same individual, 0 if different
            scores.append(sim_row[j])

    return torch.tensor(labels, dtype=int), torch.tensor(scores, dtype=float)


def recall_at_k(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates, k):
    recalls = []
    for i in range(len(query_labels)):
        valid = [j for j in similarity_matrix[i].argsort(descending=True)
                 if query_dates[i] != ref_dates[j]]
        relevant = [j for j in range(len(ref_labels))
                    if query_labels[i] == ref_labels[j] and query_dates[i] != ref_dates[j]]
        R = len(relevant)
        if R:
            hits = sum(query_labels[i] == ref_labels[j] for j in valid[:k])
            recalls.append(hits / R)
    return float(np.mean(recalls)) if recalls else 0.0

def precision_at_k(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates, k):
    precisions = []
    for i in range(len(query_labels)):
        valid = [j for j in similarity_matrix[i].argsort(descending=True)
                 if query_dates[i] != ref_dates[j]]
        k_eff = min(k, len(valid))
        if k_eff:
            hits = sum(query_labels[i] == ref_labels[j] for j in valid[:k_eff])
            precisions.append(hits / k_eff)
    return float(np.mean(precisions)) if precisions else 0.0

def top_k_accuracy(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates, k):
    n = len(query_labels)
    m = len(ref_labels)
    assert similarity_matrix.shape == (n, m), f"Shape mismatch: {similarity_matrix.shape} != ({n}, {m})"
    correct = 0
    total = 0

    ref_labels_set = set(ref_labels)
    for i in range(n):
        if query_labels[i] not in ref_labels_set:  # skip if no matching identity in ref
            continue
        sorted_idx = similarity_matrix[i].argsort(descending=True)  # descending order
        valid_candidates = [j for j in sorted_idx if query_dates[i] != ref_dates[j]] # ignore pairs from same date
        total += 1  # count this query even if valid_candidates has fewer than k elements
        # Check if any of the top-k valid candidates matches the query identity
        if any(query_labels[i] == ref_labels[j] for j in valid_candidates[:k]):
            correct += 1
    return correct / total if total > 0 else 0.0

def micro_precision_at_k(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates, k):
    n = len(query_labels)
    m = len(ref_labels)
    assert similarity_matrix.shape == (n, m), f"Shape mismatch: {similarity_matrix.shape} != ({n}, {m})"
    correct = 0
    total = 0

    for i in range(n):
        sorted_idx = similarity_matrix[i].argsort(descending=True)#[::-1]  # descending order
        valid_candidates = [j for j in sorted_idx if query_dates[i] != ref_dates[j]] # ignore pairs from same date
        total += min(k, len(valid_candidates))  # count only up to k candidates
        # Check if any of the top-k valid candidates matches the query identity
        correct += sum(query_labels[i] == ref_labels[j] for j in valid_candidates[:k])
    
    return correct / total if total > 0 else 0.0


def R_precision(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates):
    n = len(query_labels)
    m = len(ref_labels)
    assert similarity_matrix.shape == (n, m), f"Shape mismatch: {similarity_matrix.shape} != ({n}, {m})"
    precisions = []
    
    for i in range(n):
        sorted_idx = similarity_matrix[i].argsort(descending=True)#[::-1]  # descending order
        valid_candidates = [j for j in sorted_idx if query_dates[i] != ref_dates[j]] # ignore pairs from same date
        is_relevant = [query_labels[i] == ref_labels[j] for j in valid_candidates]
        R = np.sum(is_relevant) # number of relevant items
        if R > 0: # skip i if no relevant items
            r = np.sum(is_relevant[:R]) # number of relevant items in top-R
            precisions.append(r / R)
    return float(np.mean(precisions)) if precisions else 0.0

def mean_average_precision_at_R(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates):
    n = len(query_labels)
    m = len(ref_labels)
    assert similarity_matrix.shape == (n, m), f"Shape mismatch: {similarity_matrix.shape} != ({n}, {m})"
    average_precisions = []
    
    for i in range(n):
        sorted_idx = similarity_matrix[i].argsort(descending=True)#[::-1]  # descending order
        valid_candidates = [j for j in sorted_idx if query_dates[i] != ref_dates[j]] # ignore pairs from same date
        is_relevant = np.array([query_labels[i] == ref_labels[j] for j in valid_candidates], dtype=np.int32)
        R = np.sum(is_relevant) # number of relevant items
        if R > 0: # skip i if no relevant items
            cumsum_relevant = np.cumsum(is_relevant)
            precision_at_r = cumsum_relevant[:R] / (np.arange(1, R + 1))
            average_precisions.append(np.mean(precision_at_r))

    return np.mean(average_precisions) if average_precisions else 0.0

def evaluate(similarity_matrix, dataset):
    labels, scores = labels_and_scores(
        similarity_matrix, 
        dataset.labels, dataset.labels, 
        dataset.dates, dataset.dates
    )
    return {
        "AUC": roc_auc_score(labels.cpu(), scores.cpu()),
        "AP": average_precision_score(labels.cpu(), scores.cpu()),
        "Top-1 accuracy": top_k_accuracy(
            similarity_matrix, dataset.labels, dataset.labels, 
            dataset.dates, dataset.dates, k=1),
        "Precision@3": precision_at_k(
            similarity_matrix, dataset.labels, dataset.labels, 
            dataset.dates, dataset.dates, k=3),
        "Recall@3": recall_at_k(
            similarity_matrix, dataset.labels, dataset.labels, 
            dataset.dates, dataset.dates, k=3),
        "R-Precision": R_precision(
            similarity_matrix, dataset.labels, dataset.labels, 
            dataset.dates, dataset.dates),
        "mAP@R": mean_average_precision_at_R(
            similarity_matrix, dataset.labels, dataset.labels, 
            dataset.dates, dataset.dates)
    }

@click.command()
@click.argument('model_name', type=str)
@click.option('--similarity_name', type=str, default='cosine', help='Similarity function to use')
@click.option('--val_csv', type=str, default='bina_photos_validation.csv')
@click.option('--checkpoint', type=str, default=None, help='Path to the model checkpoint')
@click.option('--device', type=str, default='cpu', help='Device to run the model on (e.g., cpu, cuda)')
@click.option('--zoomcentercrop', type=bool, default=False, help='Should we apply ZoomCenterCrop before embedding?')
def main(model_name, similarity_name, val_csv, checkpoint, device, zoomcentercrop):
    print(f"Evaluating {model_name} on {val_csv}...")
    
    model, preprocess, model_name = get_model(model_name)
    similarity_func = get_similarity_function(similarity_name)

    df = pd.read_csv(val_csv) 
    if zoomcentercrop:   
        preprocess.transforms.insert(0, ZoomCenterCrop(zoom=2.0))
    ds = DataFrameDataset(df, transform=preprocess)

    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}...")
        load_checkpoint(checkpoint, model, map_location=device)
        model_name = checkpoint.split('/')[1]

    cache_file = f"results/{model_name}_{Path(val_csv).stem}.npz"
    if os.path.exists(cache_file):
        embeddings = np.load(cache_file)["embeddings"]
        print(f"Loaded embeddings from {cache_file}")
    else:
        embeddings = embed(model, ds, device)
        np.savez_compressed(cache_file, embeddings=embeddings.to('cpu').numpy())
        print(f"Saved embeddings to {cache_file}")
    if device != 'cpu':
        embeddings = torch.tensor(embeddings).to(device)
    else:
        embeddings = torch.tensor(embeddings)

    similarity_matrix = similarity_func(embeddings, embeddings)
    metrics = evaluate(similarity_matrix, ds)

    print(f"{model_name} | {val_csv}:")
    print("{}".format(" ".join([f"{k:<15}" for k in metrics.keys()])))
    print("{}".format(" ".join([f"{v:<15.3f}" for v in metrics.values()])), flush=True)

if __name__ == "__main__":
    main()
