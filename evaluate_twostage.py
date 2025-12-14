import click
import pandas as pd
import numpy as np
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from image_transform import ZoomCenterCrop
from models import get_model, load_checkpoint
from similarities import get_similarity_function
from datasets import DataFrameDataset
from metrics import (
    labels_and_scores,
    recall_at_k,
    precision_at_k,
    top_k_accuracy,
    R_precision,
    mean_average_precision_at_R,
    top_k_id_accuracy,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def embed(model, dataset, device: torch.device, batch_size, num_workers):
    model = model.to(device)
    model.eval()

    embeddings = []
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    with torch.no_grad():
        for item in dataloader:
            data = item[0] if isinstance(item, (tuple, list)) else item
            data = data.to(device)
            if device.type == "cuda":
                with torch.amp.autocast(device.type):
                    emb = model(data)
            else:
                emb = model(data)

            if isinstance(emb, dict):
                emb = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in emb.items()}
                emb["image_shape"] = data.shape[:2]
            elif torch.is_tensor(emb):
                emb = emb.detach()
            embeddings.append(emb)
    if embeddings and torch.is_tensor(embeddings[0]):
        return torch.cat(embeddings, dim=0)
    return embeddings


def move_to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def compute_metrics(similarity_matrix, dataset):
    labels, scores = labels_and_scores(
        similarity_matrix,
        dataset.labels,
        dataset.labels,
        dataset.dates,
        dataset.dates,
    )
    return {
        "AUC": roc_auc_score(labels.cpu(), scores.cpu()),
        "AP": average_precision_score(labels.cpu(), scores.cpu()),
        "Top-1 ID accuracy": top_k_id_accuracy(
            similarity_matrix, dataset.labels, dataset.labels, dataset.dates, dataset.dates, k=1
        ),
        "Top-3 ID accuracy": top_k_id_accuracy(
            similarity_matrix, dataset.labels, dataset.labels, dataset.dates, dataset.dates, k=3
        ),
        "Top-10 ID accuracy": top_k_id_accuracy(
            similarity_matrix, dataset.labels, dataset.labels, dataset.dates, dataset.dates, k=10
        ),
        "Top-1 accuracy": top_k_accuracy(
            similarity_matrix, dataset.labels, dataset.labels, dataset.dates, dataset.dates, k=1
        ),
        "Top-3 accuracy": top_k_accuracy(
            similarity_matrix, dataset.labels, dataset.labels, dataset.dates, dataset.dates, k=3
        ),
        "Top-50 accuracy": top_k_accuracy(
            similarity_matrix, dataset.labels, dataset.labels, dataset.dates, dataset.dates, k=50
        ),
        "Top-100 accuracy": top_k_accuracy(
            similarity_matrix, dataset.labels, dataset.labels, dataset.dates, dataset.dates, k=100
        ),
        "Precision@3": precision_at_k(
            similarity_matrix, dataset.labels, dataset.labels, dataset.dates, dataset.dates, k=3
        ),
        "Recall@3": recall_at_k(
            similarity_matrix, dataset.labels, dataset.labels, dataset.dates, dataset.dates, k=3
        ),
        "R-Precision": R_precision(
            similarity_matrix, dataset.labels, dataset.labels, dataset.dates, dataset.dates
        ),
        "mAP@R": mean_average_precision_at_R(
            similarity_matrix, dataset.labels, dataset.labels, dataset.dates, dataset.dates
        ),
    }


def load_or_embed(model, preprocess, model_name, dataset, device, batch_size, num_workers, val_csv, ignore_cache):
    if preprocess is not None and hasattr(preprocess, "transforms"):
        transforms = preprocess.transforms
        dataset.transform = preprocess
    else:
        dataset.transform = preprocess

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    suffix = f"{model_name}_{Path(val_csv).stem}"

    emb_cache = results_dir / f"{suffix}_embeddings.pt"
    if emb_cache.exists() and not ignore_cache:
        embeddings = torch.load(emb_cache)
        click.echo(f"Loaded embeddings from {emb_cache}")
    else:
        embeddings = embed(model, dataset, device, batch_size=batch_size, num_workers=num_workers)
        torch.save(move_to_device(embeddings, torch.device("cpu")), emb_cache)
        click.echo(f"Saved embeddings to {emb_cache}")
    embeddings = move_to_device(embeddings, device)
    return embeddings, suffix


@click.command()
@click.argument("model1", type=str)
@click.argument("model2", type=str)
@click.option("--val_csv", default="bina_photos_validation.csv", show_default=True)
@click.option("--checkpoint1", default=None, help="Checkpoint path for model1", show_default=True)
@click.option("--checkpoint2", default=None, help="Checkpoint path for model2", show_default=True)
@click.option("--device", default="cpu", show_default=True, help="Device to run on (e.g., cpu, cuda)")
@click.option("--batch_size1", default=32, show_default=True, help="Batch size for model1 embeddings")
@click.option("--batch_size2", default=1, show_default=True, help="Batch size for model2 embeddings")
@click.option("--num_workers", default=4, show_default=True, help="DataLoader workers for embeddings")
@click.option("--top_k", default=100, show_default=True, help="Top-K candidates to re-rank")
@click.option("--ignore_cache", is_flag=True, default=False, help="Recompute embeddings/similarity instead of using cached files")
def main(model1, model2, val_csv, checkpoint1, checkpoint2, device, batch_size1, batch_size2, num_workers, top_k, ignore_cache):
    """Evaluate two-stage re-identification: stage1 ranks all, stage2 reranks top-K."""
    start_time = time.perf_counter()
    device = torch.device(device)
    df = pd.read_csv(val_csv)

    # Stage 1 setup
    model_1, preprocess_1, name_1 = get_model(model1)
    if name_1.lower() == "aliked":
        batch_size1 = 1
        num_workers1 = 0
        similarity_name_1 = "lightglue"
        zoomcentercrop1 = False
    else:
        similarity_name_1 = "cosine"
        num_workers1 = num_workers
        zoomcentercrop1 = True
        num_workers1 = num_workers

    if checkpoint1:
        click.echo(f"Loading checkpoint for model1 from {checkpoint1}")
        load_checkpoint(checkpoint1, model_1, map_location=device)
        name_1 = checkpoint1.split("/")[1]
    if zoomcentercrop1 and hasattr(preprocess_1, "transforms"):
        preprocess_1.transforms.insert(0, ZoomCenterCrop(zoom=2.0))
    ds = DataFrameDataset(df, transform=preprocess_1)

    embeddings1, suffix1 = load_or_embed(
        model_1, preprocess_1, name_1, ds, device, batch_size1, num_workers1, val_csv, ignore_cache
    )

    similarity1 = get_similarity_function(similarity_name_1).to(device)
    sim_cache1 = Path("results") / f"{suffix1}_{similarity_name_1}_similarity.pt"
    if sim_cache1.exists() and not ignore_cache:
        stage1_matrix = torch.load(sim_cache1, map_location="cpu")
        click.echo(f"Loaded similarity matrix from {sim_cache1}")
    else:
        stage1_matrix = similarity1(embeddings1, embeddings1)
        stage1_matrix = torch.as_tensor(stage1_matrix).cpu()
        torch.save(stage1_matrix, sim_cache1)
        click.echo(f"Saved similarity matrix to {sim_cache1}")

    # Stage 2 setup (LightGlue + ALIKED expected)
    model_2, preprocess_2, name_2 = get_model(model2)
    if name_2.lower() != "aliked":
        raise click.ClickException("Stage 2 must be ALIKED for LightGlue reranking.")
    batch_size2 = 1
    num_workers2 = 0
    if checkpoint2:
        click.echo(f"Loading checkpoint for model2 from {checkpoint2}")
        load_checkpoint(checkpoint2, model_2, map_location=device)
        name_2 = checkpoint2.split("/")[1]
    ds2 = DataFrameDataset(df, transform=preprocess_2)
    embeddings2, suffix2 = load_or_embed(
        model_2, preprocess_2, name_2, ds2, device, batch_size2, num_workers2, val_csv, ignore_cache
    )
    similarity2 = get_similarity_function("lightglue").to(device)

    # Two-stage reranking
    n = len(ds)
    stage2_matrix = torch.full_like(stage1_matrix, float("-inf"))
    for i in range(n):
        sorted_idx = torch.argsort(stage1_matrix[i], descending=True)
        candidates = [j for j in sorted_idx.tolist() if ds.dates[i] != ds.dates[j]]
        candidates = candidates[:top_k]
        if not candidates:
            continue
        query_emb = [embeddings2[i]]
        ref_embs = [embeddings2[j] for j in candidates]
        scores = similarity2(query_emb, ref_embs)[0]
        stage2_matrix[i, candidates] = torch.as_tensor(scores, dtype=torch.float32)

    metrics_stage1 = compute_metrics(stage1_matrix, ds)
    metrics_stage2 = compute_metrics(stage2_matrix, ds)

    def fmt(metrics):
        keys = [
            "Top-1 ID accuracy",
            "Top-3 ID accuracy",
            "Top-10 ID accuracy",
            "Top-1 accuracy",
            "Top-3 accuracy",
            "Top-50 accuracy",
            "Top-100 accuracy",
            "AUC",
            "AP",
            "Precision@3",
            "Recall@3",
            "R-Precision",
            "mAP@R",
        ]
        header = " ".join([f"{k:<18}" for k in keys])
        values = " ".join([f"{metrics[k]:<18.3f}" for k in keys])
        return header, values

    click.echo("\nStage 1 (full ranking) metrics:")
    h1, v1 = fmt(metrics_stage1)
    click.echo(h1)
    click.echo(v1)

    click.echo(f"\nStage 2 (top-{top_k} reranked with {name_2}+LightGlue) metrics:")
    h2, v2 = fmt(metrics_stage2)
    click.echo(h2)
    click.echo(v2)

    elapsed = time.perf_counter() - start_time
    click.echo(f"\nWall-clock {elapsed:.2f}s")


if __name__ == "__main__":
    main()
