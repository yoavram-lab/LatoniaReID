"""Batch Prediction Script for Latonia ReID

Generates predictions for unlabeled images by matching against a labeled registry.
Uses CSV-based grouping (session_id) and labeling (identity column).

=== WORKFLOW ===

INPUT:
    - unlabeled_mask.csv: CSV with columns [rel_path, session_id, date]
      Images to identify, grouped by session_id (capture session)
    - labeled_mask.csv: CSV with columns [rel_path, label, date]
      Reference registry of known individuals

MATCHING:
    1. Group unlabeled images by session_id
    2. For each session, extract features using ALIKED
    3. Match against all labeled images using LightGlue
    4. Score each identity by aggregating match counts
    5. Return top-3 predicted identities per session

OUTPUT:
    - batch_predictions.json: Top-3 identity predictions per session
      Format: {session_id: {'images': [...], 'top3': [id1, id2, id3], 'scores': [...]}, ...}

=== USAGE ===
    # Basic: match against all labeled images
    python batch_predict.py \\
      --unlabeled_csv data/unlabeled_mask.csv \\
      --labeled_csv data/labeled_mask.csv \\
      --output batch_predictions.json

    # With temporal filtering: only match against labeled images from earlier dates
    python batch_predict.py \\
      --unlabeled_csv data/unlabeled_mask.csv \\
      --labeled_csv data/labeled_mask.csv \\
      --output batch_predictions.json \\
      --temporal_filter
"""

import argparse
import csv
import json
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from lightglue import ALIKED, LightGlue
from lightglue.utils import rbd
from lib.feature_cache import get_or_cache_features
from lib.lightglue_utils import move_to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_date_to_comparable(date_str: str) -> tuple:
    """Parse date string 'YYYY-M' to comparable tuple (year, month, sub_visit).

    Handles letter suffixes to distinguish multiple visits in the same month.
    'a' is treated as earlier than 'b', etc.

    Examples:
        '2024-3' -> (2024, 3, 0)
        '2024-12' -> (2024, 12, 0)
        '2015-2a' -> (2015, 2, 1)  # 'a' -> 1 (first visit)
        '2015-2b' -> (2015, 2, 2)  # 'b' -> 2 (second visit)
        '2015-5b' -> (2015, 5, 2)
    """
    year, month_str = date_str.split("-")

    # Extract numeric part and optional letter suffix
    month_numeric = "".join(c for c in month_str if c.isdigit())
    letter_suffix = "".join(c for c in month_str if c.isalpha())

    # Convert letter to number: 'a' -> 1, 'b' -> 2, etc.
    # If no letter, use 0
    sub_visit = 0
    if letter_suffix:
        sub_visit = ord(letter_suffix.lower()) - ord("a") + 1

    return (int(year), int(month_numeric), sub_visit)


def load_unlabeled_batches(csv_path: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Load unlabeled images from CSV and group by session_id.

    Expected columns: rel_path, session_id, [date, Inferred, ...]
    Returns: {session_id: [row_dict1, row_dict2, ...], ...}
    """
    batches = defaultdict(list)

    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            session_id = row["session_id"]  # e.g., '2021-5/1'
            batches[session_id].append(row)  # Keep full row for access to date column

    return dict(batches)


def build_labeled_registry(
    csv_path: str, embeddings_dict: Dict[str, np.ndarray] = None
) -> Dict[int, List[str]]:
    """
    Build registry: identity → list of image paths from labeled data.

    Expected columns: rel_path, label, [date, ...]
    embeddings_dict: {rel_path: embedding_vector, ...} (optional filter)
    Returns: {identity: [rel_path1, rel_path2, ...], ...}
    """
    registry = defaultdict(list)

    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            identity = int(row["label"])
            rel_path = row["rel_path"]
            # Only include if we have embedding (or if no embeddings provided)
            if embeddings_dict is None or rel_path in embeddings_dict:
                registry[identity].append(rel_path)

    return dict(registry)


def batch_predict_session(
    session_rows: List[Dict[str, str]],
    registry: Dict[int, List[str]],
    registry_dates: Dict[str, str],
    embeddings_dict: Dict[str, np.ndarray],
    matcher: Any,
    extractor: Any,
    top_k: int = 3,
    apply_temporal_filter: bool = False,
) -> Dict[str, Any]:
    """
    Predict identities for a session group.

    Args:
        session_rows: List of row dicts [{rel_path, session_id, date, ...}, ...] for one session
        registry: {identity: [rel_paths], ...}
        registry_dates: {rel_path: date_str, ...} for temporal filtering
        embeddings_dict: {rel_path: embedding_vector, ...}
        matcher: Configured matcher (LightGlue)
        extractor: Feature extractor (ALIKED)
        top_k: Return top-K predictions
        apply_temporal_filter: If True, only match against labeled images from earlier dates

    Returns:
        {
            'images': ['DSC_7206.JPG', ...],
            'top3': [300, 301, 302],
            'scores': [0.95, 0.87, 0.75],
            'details': {identity: [match_count, ...], ...}
        }
    """
    # Extract image paths and date from rows
    session_images = [row["rel_path"] for row in session_rows]
    query_date = session_rows[0].get("date") if session_rows else None

    # Parse query date for temporal comparison
    query_date_tuple = None
    if apply_temporal_filter and query_date:
        try:
            query_date_tuple = parse_date_to_comparable(query_date)
        except (ValueError, IndexError):
            pass  # If date parsing fails, skip temporal filtering

    results = {"images": session_images, "top3": [], "scores": [], "details": {}}

    # Score each identity by aggregating matches across all images in session
    identity_scores = defaultdict(list)

    for img_path in session_images:
        # Get or cache features for query image
        query_features = get_or_cache_features(img_path, extractor, device)
        query_features = move_to_device(query_features, device)

        # Match against all labeled identities
        for identity, labeled_paths in registry.items():
            for labeled_path in labeled_paths:
                # Apply temporal filter if enabled
                if apply_temporal_filter and query_date_tuple:
                    labeled_date = registry_dates.get(labeled_path)
                    if labeled_date:
                        try:
                            labeled_date_tuple = parse_date_to_comparable(labeled_date)
                            # ONLY match against EARLIER dates
                            if labeled_date_tuple >= query_date_tuple:
                                continue
                        except (ValueError, IndexError):
                            continue

                # Get or cache features for labeled image
                labeled_features = get_or_cache_features(
                    labeled_path, extractor, device
                )
                labeled_features = move_to_device(labeled_features, device)

                # Compute match score using LightGlue
                with torch.inference_mode():
                    matches = matcher(
                        {"image0": query_features, "image1": labeled_features}
                    )
                    matches = rbd(matches)

                # Extract match count
                match_idx = matches["matches"]
                if isinstance(match_idx, torch.Tensor) and match_idx.ndim == 3:
                    match_idx = match_idx[0]
                match_count = int(match_idx.shape[0])

                identity_scores[identity].append(match_count)

    # Aggregate scores per identity
    identity_agg_scores = {}
    for identity, scores in identity_scores.items():
        identity_agg_scores[identity] = {
            "mean": np.mean(scores),
            "max": np.max(scores),
            "count": len(scores),
        }

    # Get top-K identities
    sorted_identities = sorted(
        identity_agg_scores.items(),
        key=lambda x: x[1]["mean"],
        reverse=True,
    )

    for identity, score_info in sorted_identities[:top_k]:
        results["top3"].append(int(identity))
        results["scores"].append(float(score_info["mean"]))
        results["details"][str(identity)] = {
            "mean_score": float(score_info["mean"]),
            "max_score": float(score_info["max"]),
            "match_count": score_info["count"],
        }

    return results


def run_batch_prediction(
    unlabeled_csv: str,
    labeled_csv: str,
    embeddings_dict: Dict[str, np.ndarray],
    matcher: Any,
    extractor: Any,
    output_json: str = "batch_predictions.json",
    apply_temporal_filter: bool = False,
) -> None:
    """
    Main batch prediction pipeline.

    Steps:
        1. Load unlabeled images grouped by session_id
        2. Build labeled registry from CSV
        3. For each session, predict top-3 identities
        4. Write JSON output with predictions

    Output JSON format:
        {
            'session_id_1': {'images': [...], 'top3': [300, 301, 302], 'scores': [...]},
            'session_id_2': {'images': [...], 'top3': [...]},
            ...
        }
    """
    # Load data
    print(f"Loading unlabeled images from {unlabeled_csv}...")
    batches = load_unlabeled_batches(unlabeled_csv)
    print(f"✓ Loaded {len(batches)} sessions")

    print(f"Building labeled registry from {labeled_csv}...")
    registry = build_labeled_registry(labeled_csv, embeddings_dict)
    print(f"✓ Built registry with {len(registry)} identities")

    # Build registry_dates mapping for temporal filtering
    registry_dates = {}
    if apply_temporal_filter:
        print("Building date index for temporal filtering...")
        with open(labeled_csv, "r") as f:
            for row in csv.DictReader(f):
                if "date" in row:
                    registry_dates[row["rel_path"]] = row["date"]
        print(f"✓ Indexed {len(registry_dates)} labeled images with dates")

    if apply_temporal_filter:
        print("⏰ Temporal filtering: will only match against earlier dates")
    print()

    # Predict for each session
    print("Running batch predictions...")
    all_predictions = {}
    for session_id, session_rows in tqdm(batches.items(), desc="Sessions"):
        prediction = batch_predict_session(
            session_rows,
            registry,
            registry_dates,
            embeddings_dict,
            matcher,
            extractor,
            apply_temporal_filter=apply_temporal_filter,
        )
        prediction["session_id"] = session_id
        all_predictions[session_id] = prediction

    # Write output
    print(f"Saving predictions to {output_json}...")
    with open(output_json, "w") as f:
        json.dump(all_predictions, f, indent=2)

    print(f"✓ Saved {len(all_predictions)} predictions")
    print()
    print("📊 Summary:")
    print(f"  Sessions: {len(all_predictions)}")
    print(f"  Identities in registry: {len(registry)}")
    total_query_images = sum(len(p["images"]) for p in all_predictions.values())
    print(f"  Total query images: {total_query_images}")
    if apply_temporal_filter:
        print(f"  Temporal filter: ENABLED (earlier dates only)")
    print()
    print("✅ Ready for expert review")


def main():
    parser = argparse.ArgumentParser(
        description="Batch prediction for Latonia ReID - CSV-based grouping and matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python batch_predict.py \\
    --unlabeled_csv data/unlabeled_mask.csv \\
    --labeled_csv data/labeled_mask.csv \\
    --output batch_predictions.json

Notes:
  - Unlabeled CSV should have columns: rel_path, session_id, [date, ...]
  - Labeled CSV should have columns: rel_path, label, [date, ...]
  - Returns top-3 identity predictions per session as JSON
        """,
    )
    parser.add_argument(
        "--unlabeled_csv",
        required=True,
        help="CSV with unlabeled masked images (columns: rel_path, session_id, date)",
    )
    parser.add_argument(
        "--labeled_csv",
        required=True,
        help="CSV with labeled masked images (columns: rel_path, label, date)",
    )
    parser.add_argument(
        "--output",
        default="batch_predictions.json",
        help="Output JSON file for predictions (default: batch_predictions.json)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (cuda or cpu)",
    )
    parser.add_argument(
        "--temporal_filter",
        action="store_true",
        help="If enabled, only match query images against labeled images from earlier dates",
    )

    args = parser.parse_args()

    print("Latonia ReID - Batch Prediction")
    print(f"Device: {args.device}")
    print()

    # Load models
    print("Loading ALIKED + LightGlue models...")
    extractor = ALIKED(max_num_keypoints=1432, detection_threshold=0.01).to(
        args.device
    ).eval()
    matcher = LightGlue(features="aliked").to(args.device).eval()
    print("✓ Models loaded")
    print()

    # Run batch prediction (without embeddings_dict for now)
    run_batch_prediction(
        args.unlabeled_csv,
        args.labeled_csv,
        embeddings_dict=None,
        matcher=matcher,
        extractor=extractor,
        output_json=args.output,
        apply_temporal_filter=args.temporal_filter,
    )


if __name__ == "__main__":
    main()
