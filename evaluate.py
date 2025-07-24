import os
import click
import pandas as pd
import numpy as np

import torch
from pytorch_metric_learning.distances import CosineSimilarity

from sklearn.metrics import roc_auc_score, average_precision_score

from models import get_model
from train import embed, DataFrameDataset, load_ckpt

device = 'cpu'

def get_embeddings(val_csv, model_name, model, val_dataset, device):
    cache_file = f"results/{model_name}_{os.path.basename(val_csv)}.npz"
    if os.path.exists(cache_file):
        embeddings = np.load(cache_file)["embeddings"]
        print(f"Loaded embeddings from {cache_file}")
    else:
        embeddings = embed(model.to(device), val_dataset, device=device)
        np.savez_compressed(cache_file, embeddings=embeddings.to('cpu').numpy())
        print(f"Saved embeddings to {cache_file}")
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

def recall_at_k(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates, k=1):
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

def precision_at_k(similarity_matrix, query_labels, ref_labels, query_dates, ref_dates, k=1):
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


def evaluate(embeddings, dataset, similarity_func=CosineSimilarity()):
    similarity_matrix = similarity_func(embeddings, embeddings)
    labels, scores = labels_and_scores(
        similarity_matrix, 
        dataset.labels, dataset.labels, 
        dataset.dates, dataset.dates
    )
    return {
        "AUC": roc_auc_score(labels.cpu(), scores.cpu()),
        "AP": average_precision_score(labels.cpu(), scores.cpu()),
        "Precision@1": precision_at_k(similarity_matrix, dataset.labels, dataset.labels, 
                                     dataset.dates, dataset.dates, k=1),
        "Recall@1": recall_at_k(similarity_matrix, dataset.labels, dataset.labels, 
                              dataset.dates, dataset.dates, k=1),
        "R-Precision": R_precision(similarity_matrix, dataset.labels, dataset.labels, 
                                     dataset.dates, dataset.dates),
        "mAP@R": mean_average_precision_at_R(similarity_matrix, dataset.labels, dataset.labels, 
                                              dataset.dates, dataset.dates)
    }

@click.command()
@click.argument('model_name', type=str)
@click.option('--val_csv', type=str, default='bina_photos_validation.csv')
@click.option('--checkpoint', type=str, default=None, help='Path to the model checkpoint')
@click.option('--device', type=str, default='cpu', help='Device to run the model on (e.g., cpu, cuda)')
def main(model_name, val_csv, checkpoint, device):
    print(f"Evaluating {model_name} on {val_csv}...")
    
    model_name, model, preprocess = get_model(model_name)

    val_df = pd.read_csv(val_csv)
    val_dataset = DataFrameDataset(val_df, transform=preprocess)

    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}...")
        load_ckpt(checkpoint, model, map_location=device)
        model_name += '_' + checkpoint.split('/')[1]
    
    embeddings = get_embeddings(val_csv, model_name, model, val_dataset, device)
    if device != 'cpu':
        embeddings = torch.tensor(embeddings).to(device)
    metrics = evaluate(embeddings, val_dataset)
     
    print("{}".format(" ".join([f"{k:<15}" for k in metrics.keys()])))
    print("{}".format(" ".join([f"{v:<15.6f}" for v in metrics.values()])), flush=True)

if __name__ == "__main__":
    main()
