import click
import pandas as pd
import numpy as np
from tqdm import tqdm
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
        batch_size=batch_size,  # any size that fits GPU
        shuffle=False,  # keep deterministic order
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,  # for faster data transfer to GPU
    )
    with torch.no_grad():
        pbar = tqdm(
            dataloader,
            desc=f"Embedding on {device}",
            leave=False,
            total=(len(dataset) + batch_size - 1) // batch_size,
        )
        for item in pbar:
            if isinstance(item, (tuple, list)):
                data = item[0]
            else:
                data = item
            data = data.to(device)
            if device.type == "cuda":
                with torch.amp.autocast(device.type):
                    emb = model(data)
            else:
                emb = model(data)
            if isinstance(emb, dict):
                emb = {
                    k: (v.detach() if torch.is_tensor(v) else v) for k, v in emb.items()
                }
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


def evaluate(similarity_matrix, dataset):
    labels, scores = labels_and_scores(
        similarity_matrix, dataset.labels, dataset.labels, dataset.dates, dataset.dates
    )
    return {
        # "AUC": roc_auc_score(
        #     labels.cpu(),
        #     scores.cpu()),
        # "AP": average_precision_score(
        #     labels.cpu(),
        #     scores.cpu()),
        "Top-1 ID accuracy": top_k_id_accuracy(
            similarity_matrix,
            dataset.labels,
            dataset.labels,
            dataset.dates,
            dataset.dates,
            k=1,
        ),
        "Top-3 ID accuracy": top_k_id_accuracy(
            similarity_matrix,
            dataset.labels,
            dataset.labels,
            dataset.dates,
            dataset.dates,
            k=3,
        ),
        "Top-10 ID accuracy": top_k_id_accuracy(
            similarity_matrix,
            dataset.labels,
            dataset.labels,
            dataset.dates,
            dataset.dates,
            k=10,
        ),
        "Top-10 accuracy": top_k_accuracy(
            similarity_matrix,
            dataset.labels,
            dataset.labels,
            dataset.dates,
            dataset.dates,
            k=10,
        ),
        "Top-1 accuracy": top_k_accuracy(
            similarity_matrix,
            dataset.labels,
            dataset.labels,
            dataset.dates,
            dataset.dates,
            k=1,
        ),
        "Top-3 accuracy": top_k_accuracy(
            similarity_matrix,
            dataset.labels,
            dataset.labels,
            dataset.dates,
            dataset.dates,
            k=3,
        ),
        "Top-50 accuracy": top_k_accuracy(
            similarity_matrix,
            dataset.labels,
            dataset.labels,
            dataset.dates,
            dataset.dates,
            k=50,
        ),
        "Top-100 accuracy": top_k_accuracy(
            similarity_matrix,
            dataset.labels,
            dataset.labels,
            dataset.dates,
            dataset.dates,
            k=100,
        ),
        # "Precision@3": precision_at_k(
        #     similarity_matrix,
        #     dataset.labels,
        #     dataset.labels,
        #     dataset.dates,
        #     dataset.dates, k=3),
        # "Recall@3": recall_at_k(
        #     similarity_matrix,
        #     dataset.labels,
        #     dataset.labels,
        #     dataset.dates,
        #     dataset.dates, k=3),
        # "R-Precision": R_precision(
        #     similarity_matrix,
        #     dataset.labels,
        #     dataset.labels,
        #     dataset.dates,
        #     dataset.dates),
        # "mAP@R": mean_average_precision_at_R(
        #     similarity_matrix,
        #     dataset.labels,
        #     dataset.labels,
        #     dataset.dates,
        #     dataset.dates,
        # ),
    }


@click.command()
@click.argument("model_name", type=str)
@click.argument("similarity", type=str)
@click.option("--val_csv", type=str, default="bina_photos_validation.csv")
@click.option(
    "--checkpoint", type=str, default=None, help="Path to the model checkpoint"
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help="Device to run the model on (e.g., cpu, cuda)",
)
@click.option("--batch_size", type=int, default=32, help="Batch size for embedding")
@click.option("--num_workers", type=int, default=4, help="Number of DataLoader workers")
@click.option(
    "--ignore_cache",
    is_flag=True,
    default=False,
    help="Recompute embeddings and similarity instead of using cached files",
)
def main(
    model_name,
    similarity,
    val_csv,
    checkpoint,
    device,
    batch_size,
    num_workers,
    ignore_cache,
):
    start_time = time.perf_counter()
    device = torch.device(device)
    model, preprocess, model_name = get_model(model_name)

    if model_name.lower() in ("aliked", "sift"):
        batch_size = 1  # variable keypoints; use single-image batches
        num_workers = 0  # avoid multiprocessing issues with keypoint models
        similarity_name = similarity or "lightglue"
        zoomcentercrop = False
    else:
        similarity_name = similarity or "cosine"
        zoomcentercrop = True
    similarity_fn = get_similarity_function(
        similarity_name,
        features=model_name.lower() if similarity_name == "lightglue" else None,
    )
    similarity_fn = similarity_fn.to(device)
    print(
        f"Evaluating {model_name}-{similarity_name} on {val_csv} with device {device}..."
    )

    df = pd.read_csv(val_csv)
    if zoomcentercrop:
        preprocess.transforms.insert(0, ZoomCenterCrop(zoom=2.0))
    ds = DataFrameDataset(df, transform=preprocess)

    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}...")
        load_checkpoint(checkpoint, model, map_location=device)
        model_name = checkpoint.split("/")[1]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    suffix = f"{model_name}_{Path(val_csv).stem}"

    emb_cache = results_dir / f"{suffix}_embeddings.pt"
    if emb_cache.exists() and not ignore_cache:
        embeddings = torch.load(emb_cache)
        print(f"Loaded embeddings from {emb_cache}")
    else:
        embeddings = embed(
            model, ds, device, batch_size=batch_size, num_workers=num_workers
        )
        torch.save(move_to_device(embeddings, torch.device("cpu")), emb_cache)
        print(f"Saved embeddings to {emb_cache}")
    embeddings = move_to_device(embeddings, device)

    sim_cache = results_dir / f"{suffix}_{similarity_name}_similarity.pt"
    if sim_cache.exists() and not ignore_cache:
        similarity_matrix = torch.load(sim_cache, map_location="cpu")
        print(f"Loaded similarity matrix from {sim_cache}")
    else:
        similarity_matrix = similarity_fn(embeddings, embeddings)
        similarity_matrix = torch.as_tensor(similarity_matrix).cpu()
        torch.save(similarity_matrix, sim_cache)
        print(f"Saved similarity matrix to {sim_cache}")

    metrics = evaluate(similarity_matrix, ds)

    print(f"{model_name} | {val_csv}:")
    print("{}".format(" ".join([f"{k:<18}" for k in metrics.keys()])))
    print("{}".format(" ".join([f"{v:<18.3f}" for v in metrics.values()])), flush=True)
    elapsed = time.perf_counter() - start_time
    print(f"Wall-clock {elapsed:.2f}s")


if __name__ == "__main__":
    main()
