from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve


def parse_date_id(rel_path: str) -> tuple[str, str]:
    """Extract date and id from a path like <data>/<date>/<id>/<image>."""
    parts = Path(rel_path).parts
    if len(parts) < 3:
        raise ValueError(f"Expected path format <data>/<date>/<id>/<image>, got: {rel_path}")
    date = parts[-3]
    frog_id = parts[-2]
    return date, frog_id


def collect_pairs(similarity, dates, ids):
    same_id_diff_date = []
    diff_id_diff_date = []
    n = len(dates)
    for i in range(n):
        for j in range(i + 1, n):
            if dates[i] == dates[j]:
                continue  # skip same-date comparisons
            sim = float(similarity[i, j])
            if ids[i] == ids[j]:
                same_id_diff_date.append(sim)
            else:
                diff_id_diff_date.append(sim)
    return same_id_diff_date, diff_id_diff_date


def collect_id_level_scores(similarity, dates, ids):
    """Aggregate scores at identity level (max over images per identity, excluding same-date)."""
    sim = np.asarray(similarity)
    id_to_indices = {}
    for idx, fid in enumerate(ids):
        id_to_indices.setdefault(fid, []).append(idx)

    pos_scores = []  # same ID, different date (max over that ID's valid images)
    neg_scores = []  # different ID, different date (max over each other ID's valid images)

    for i, (fid, date) in enumerate(zip(ids, dates)):
        # positives: same ID, different date
        same_idxs = [j for j in id_to_indices[fid] if dates[j] != date]
        if not same_idxs:
            continue  # no cross-date mate for this query
        pos_scores.append(float(sim[i, same_idxs].max()))

        # negatives: other IDs, take max score per ID (excluding same-date)
        for other_id, idxs in id_to_indices.items():
            if other_id == fid:
                continue
            valid = [j for j in idxs if dates[j] != date]
            if not valid:
                continue
            neg_scores.append(float(sim[i, valid].max()))

    return pos_scores, neg_scores


def top1_id_accuracy_from_scores(similarity, dates, ids):
    """Compute top-1 ID accuracy using identity-level aggregation (matches top_k_id_accuracy logic)."""
    sim = np.asarray(similarity)
    id_to_indices = {}
    for idx, fid in enumerate(ids):
        id_to_indices.setdefault(fid, []).append(idx)

    correct = 0
    total = 0
    for i, (fid, date) in enumerate(zip(ids, dates)):
        # Build per-ID max scores excluding same-date pairs
        id_scores = {}
        for other_id, idxs in id_to_indices.items():
            valid = [j for j in idxs if dates[j] != date]
            if not valid:
                continue
            id_scores[other_id] = float(sim[i, valid].max())

        if fid not in id_scores:
            continue  # no cross-date positive for this query

        total += 1
        pred_id = max(id_scores.items(), key=lambda x: x[1])[0]
        if pred_id == fid:
            correct += 1

    return correct / total if total else 0.0


def plot_histograms(same, different, out_path: Path, x_lines=None):
    plt.figure(figsize=(6.6, 4.4))
    bins = 50
    plt.hist(
        different,
        bins=bins,
        alpha=0.6,
        label="Different ID, different date",
        density=True,
    )
    plt.hist(same, bins=bins, alpha=0.6, label="Same ID, different date", density=True)
    if x_lines:
        for x in x_lines:
            plt.axvline(x, color="k", linestyle="--", linewidth=1, alpha=0.7)
    plt.xlabel("Similarity score", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved histogram to {out_path}")


def plot_precision_recall(labels, scores, out_path: Path, highlights=None):
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(6.6, 4.4))
    plt.plot(recall, precision)
    if highlights:
        for r, p in highlights.items():
            if p is None:
                continue
            plt.scatter(r, p, s=40, color="black")
    plt.xlabel("Recall", fontsize=13)
    plt.ylabel("Precision", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved precision-recall curve to {out_path}")


def precision_at_recalls(precision, recall, targets):
    """Return precision values at or above target recalls (using max precision where recall>=target)."""
    results = {}
    for r_target in targets:
        mask = recall >= r_target
        if np.any(mask):
            results[r_target] = float(np.max(precision[mask]))
        else:
            results[r_target] = None
    return results


def recall_at_precision(precision, recall, target_precision):
    """Return max recall achievable at or above a target precision."""
    mask = precision >= target_precision
    if np.any(mask):
        return float(np.max(recall[mask]))
    return None


@click.command(help="Plot similarity histograms for same-ID/different-ID pairs across dates.")
@click.option(
    "--sim-path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("results/aliked_bina_photos_mask_lightglue_similarity.pt"),
    show_default=True,
    help="Path to the similarity matrix .pt file.",
)
@click.option(
    "--csv-path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("bina_photos_mask.csv"),
    show_default=True,
    help="Path to the CSV with rel_path and labels.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=Path("openset_hist.png"),
    show_default=True,
    help="Where to save the histogram figure.",
)
@click.option(
    "--pr-out",
    "pr_out_path",
    type=click.Path(path_type=Path),
    default=Path("openset_pr.png"),
    show_default=True,
    help="Where to save the precision-recall curve.",
)
@click.option(
    "--x",
    "x_lines_raw",
    multiple=True,
    help="Comma-separated x values for vertical lines, e.g. '--x 15,189,666' or repeated '--x 15 --x 189'.",
)
def main(sim_path, csv_path, out_path, pr_out_path, x_lines_raw):
    sim = torch.load(sim_path, map_location="cpu")
    if hasattr(sim, "numpy"):
        sim = sim.numpy()

    df = pd.read_csv(csv_path)
    rel_paths = df["rel_path"].tolist()
    dates, ids = zip(*(parse_date_id(p) for p in rel_paths))

    # Identity-level aggregation: max score per identity (excluding same-date pairs).
    same, different = collect_id_level_scores(sim, dates, ids)
    print(f"[ID-level] Collected {len(same)} same-id/different-date scores and {len(different)} different-id/different-date scores")
    x_lines = None
    if x_lines_raw:
        try:
            x_lines = [
                float(v)
                for chunk in x_lines_raw
                for v in chunk.split(",")
                if v.strip() != ""
            ]
        except ValueError as exc:
            raise SystemExit(f"Could not parse --x values '{x_lines_raw}': {exc}")

    plot_histograms(same, different, out_path, x_lines=x_lines)

    # Precision-Recall curve
    labels = np.concatenate([np.ones(len(same)), np.zeros(len(different))])
    scores = np.concatenate([np.array(same), np.array(different)])
    precision, recall, _ = precision_recall_curve(labels, scores)
    highlights = precision_at_recalls(precision, recall, targets=[0.95])
    prec_target = 0.95
    rec_at_prec = recall_at_precision(precision, recall, prec_target)
    if rec_at_prec is not None:
        highlights[rec_at_prec] = prec_target  # for plotting, keyed by recall value
    plot_precision_recall(labels, scores, pr_out_path, highlights=highlights)

    for r_target, p_val in precision_at_recalls(precision, recall, targets=[0.95]).items():
        if p_val is None:
            print(f"Recall {r_target:.2f}: not achievable")
        else:
            print(f"Recall {r_target:.2f} -> Precision {p_val:.3f}")
    if rec_at_prec is None:
        print(f"Precision {prec_target:.2f}: not achievable")
    else:
        print(f"Precision {prec_target:.2f} -> Recall {rec_at_prec:.3f}")


if __name__ == "__main__":
    main()
