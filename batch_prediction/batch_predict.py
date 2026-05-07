"""Batch Prediction Script for Latonia ReID

This script matches query images (already masked) against both Bina and URI registries
to generate batch predictions for review.

=== WORKFLOW OVERVIEW ===

QUERY IMAGES (to identify):
    - New/unidentified individuals you want to match
    - Loaded from CSV file with MASKED images
    - Grouped by date/temp_id
    - The image with highest match count is auto-selected as "best query"

REFERENCE REGISTRIES:
    1. Bina Registry (bina_photos_mask.csv):
       - Matched against all dates
    2. URI Registry (uri_photos_mask.csv):
       - Matched only against EARLIER dates (temporal filtering)

MATCHING PROCESS:
    1. Extract features from query image using ALIKED
    2. Match against ALL images in both registries using LightGlue
    3. Keep best match per registry ID
    4. Return top-3 predictions sorted by match count
    5. Each prediction includes "dataset" key ("Bina" or "Uri")

=== REQUIREMENTS ===
    - CSV must contain paths to MASKED images
    - Expected paths: data/mask1/<date>/<temp_id>/<image>.png (Bina)
                      data_uri/mask1/<date>/<temp_id>/<image>.png (URI)
    - Images grouped by date/temp_id

=== USAGE ===
    python -m scripts.batch_predict  --csv <query_csv> --output batch_predictions.json
    python -m scripts.batch_predict --csv data_uri/uri_photos_mask.csv --output data_uri/batch_predictions.json


=== OUTPUT ===
    - Predictions JSON: Contains top-3 matches per query (ready for Gradio review app)
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from lightglue import ALIKED, LightGlue
from lightglue.utils import rbd
from lib.lightglue_utils import (
    image_to_tensor,
    move_to_device,
    resize_pair,
)
from lib.feature_cache import get_or_cache_features
from id_program import Population

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CSV helper functions for reading session_id and label columns
def load_unlabeled_batches(csv_path: str) -> Dict[str, List[str]]:
    """
    Load unlabeled images from CSV and group by session_id.

    Expected columns: rel_path, session_id, [date, Inferred, ...]
    Returns: {session_id: [rel_path1, rel_path2, ...], ...}
    """
    batches = defaultdict(list)

    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            if 'session_id' in row:
                session_id = row['session_id']  # e.g., '2021-5/1'
                rel_path = row['rel_path']
                batches[session_id].append(rel_path)

    return dict(batches)


def build_labeled_registry(csv_path: str, embeddings_dict: Dict[str, object] = None) -> Dict[int, List[str]]:
    """
    Build registry: identity → list of image paths from labeled data.

    Expected columns: rel_path, label, [date, ...]
    embeddings_dict: {rel_path: embedding_vector, ...} (optional, from previous embedding step)
    Returns: {identity: [rel_path1, rel_path2, ...], ...}
    """
    registry = defaultdict(list)

    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            if 'label' in row:
                identity = int(row['label'])
                rel_path = row['rel_path']
                # Only include if we have embedding (or if no embeddings provided)
                if embeddings_dict is None or rel_path in embeddings_dict:
                    registry[identity].append(rel_path)

    return dict(registry)


def parse_date_from_path(path: str) -> str:
    """Extract date from path like 'data/mask1/2013-12/14/IMGP0167.png'
    or 'data_uri/mask1/2016-12/1/DSC_3466.png'
    or 'data_uri/photos_bbox_sam/2015-2a/37/IMGP2567.png'

    Returns: '2013-12' or '2016-12' or '2015-2a' (with optional letter suffix)
    """
    parts = Path(path).parts
    # Look for a part that matches YYYY-M or YYYY-MM format (with optional letter suffix)
    for part in parts:
        if "-" in part:
            try:
                year_month = part.split("-")
                if len(year_month) == 2:
                    year = int(year_month[0])
                    # Extract numeric part of month (handle formats like '2a', '5b')
                    month_str = year_month[1]
                    # Remove any trailing letters
                    month_numeric = ''.join(c for c in month_str if c.isdigit())
                    if month_numeric:
                        month = int(month_numeric)
                        # Valid year and month
                        if 1900 <= year <= 2100 and 1 <= month <= 12:
                            return part
            except (ValueError, IndexError):
                continue
    raise ValueError(f"Cannot parse date from path: {path}")


def parse_temp_id_from_path(path: str) -> str:
    """Extract temp_id from path like 'data/mask1/2013-12/14/IMGP0167.png'
    or 'data_uri/mask1/2016-12/1/DSC_3466.png'

    Returns: '14' or '1'
    """
    parts = Path(path).parts
    # Find the date part first
    date_str = parse_date_from_path(path)
    date_idx = parts.index(date_str)

    # temp_id is the next folder after date
    if date_idx + 1 < len(parts):
        return parts[date_idx + 1]

    raise ValueError(f"Cannot parse temp_id from path: {path}")


def load_csv_entries(csv_path: str) -> Dict[str, List[str]]:
    """Load CSV and group images by unique_key (date/temp_id).

    This function expects all images to be already masked.
    Use run_masking_mode() first if images need masking.

    Args:
        csv_path: Path to CSV file with 'rel_path' column

    Returns:
        Dict mapping unique_key -> list of masked image paths
        e.g., {'2025-11/14': ['data_uri/masked/2025-11/14/img1.png', ...], ...}
    """
    entries = defaultdict(list)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["rel_path"]
            date = parse_date_from_path(path)
            temp_id = parse_temp_id_from_path(path)
            unique_key = f"{date}/{temp_id}"
            entries[unique_key].append(path)

    print(f"✓ Loaded {len(entries)} unique date/temp_id combinations from CSV")
    return dict(entries)


def parse_date_to_comparable(date_str: str) -> Tuple[int, int, int]:
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
    month_numeric = ''.join(c for c in month_str if c.isdigit())
    letter_suffix = ''.join(c for c in month_str if c.isalpha())

    # Convert letter to number: 'a' -> 1, 'b' -> 2, etc.
    # If no letter, use 0
    sub_visit = 0
    if letter_suffix:
        sub_visit = ord(letter_suffix.lower()) - ord('a') + 1

    return (int(year), int(month_numeric), sub_visit)


def build_bina_registry_from_csv(csv_path: str) -> Dict:
    """Build Bina registry from CSV.

    For Bina, we group by the ID folder (no date parsing needed).
    Path format: data/mask1/<date>/<id>/image.png -> use just <id>

    Args:
        csv_path: Path to CSV file with 'rel_path' column

    Returns:
        Dict mapping ID -> {images: [...], dataset: "Bina"}
    """
    from collections import defaultdict

    registry = defaultdict(lambda: {"images": [], "dataset": "Bina"})

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["rel_path"]

            # Extract ID from folder path
            # Bina: data/mask1/<date>/<id>/image.png -> extract <id>
            parts = Path(path).parts
            if len(parts) >= 4:
                # parts[3] is the ID folder
                unique_key = parts[3]
                registry[unique_key]["images"].append(path)

    # Convert defaultdict to regular dict
    registry = {key: dict(data) for key, data in registry.items()}

    print(f"✓ Built Bina registry with {len(registry)} entries from {csv_path}")
    return registry


def build_uri_registry_from_csv(csv_path: str) -> Dict:
    """Build URI registry from CSV.

    For URI, we group by date/temp_id.
    Path format: data_uri/mask1/<date>/<temp_id>/image.png -> use <date>/<temp_id>

    Args:
        csv_path: Path to CSV file with 'rel_path' column

    Returns:
        Dict mapping "date/temp_id" -> {images: [...], dataset: "Uri"}
    """
    from collections import defaultdict

    registry = defaultdict(lambda: {"images": [], "dataset": "Uri"})

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["rel_path"]

            # Extract date and temp_id from folder path
            # URI: data_uri/mask1/<date>/<temp_id>/image.png -> extract <date>/<temp_id>
            date = parse_date_from_path(path)
            temp_id = parse_temp_id_from_path(path)
            unique_key = f"{date}/{temp_id}"

            registry[unique_key]["images"].append(path)

    # Convert defaultdict to regular dict
    registry = {key: dict(data) for key, data in registry.items()}

    print(f"✓ Built Uri registry with {len(registry)} entries from {csv_path}")
    return registry


def load_combined_registries() -> Tuple[Dict, Dict]:
    """Load both Bina and URI registries.

    Returns:
        Tuple of (bina_registry, uri_registry)
    """
    bina_csv = "bina_photos_mask.csv"
    uri_csv = "data_uri/uri_photos_mask.csv"

    bina_registry = build_bina_registry_from_csv(bina_csv)
    uri_registry = build_uri_registry_from_csv(uri_csv)

    return bina_registry, uri_registry


# Feature extraction and caching now handled by feature_cache module


def match_single_query_vs_registry(
    query_features: Dict,
    registry: Dict,
    extractor,
    matcher,
    query_date: str = None,
    query_unique_key: str = None,
    apply_temporal_filter: bool = False,
) -> List[Tuple[str, int, str, List[str], str]]:
    """Match a single query image against all registry IDs.

    Args:
        query_features: Features extracted from query image
        registry: Registry dict (unique_key -> {images: [...], dataset: "Bina"/"Uri"})
        extractor: Feature extractor model
        matcher: Matcher model
        query_date: Date of query image (for temporal filtering)
        query_unique_key: Unique key of query to exclude from matching
        apply_temporal_filter: If True, only match against earlier dates (for URI registry)

    Returns:
        List of (registry_id, match_count, best_match_path, gallery_paths, dataset)
        sorted by match_count descending
    """
    # Move query features to device for matching
    query_features = move_to_device(query_features, device)

    # Parse query date for comparison if temporal filtering is enabled
    query_date_tuple = None
    if apply_temporal_filter and query_date:
        query_date_tuple = parse_date_to_comparable(query_date)

    # Manual matching: query vs all registry images
    candidate_scores = {}  # unique_key -> (match_count, img_path, dataset)

    for unique_key, registry_data in registry.items():
        # Skip matching against the exact same date/temp_id combination
        if query_unique_key and unique_key == query_unique_key:
            continue

        # Apply temporal filter if enabled
        if apply_temporal_filter and query_date_tuple:
            registry_date = unique_key.split("/")[0]
            try:
                registry_date_tuple = parse_date_to_comparable(registry_date)
                # ONLY match against EARLIER dates
                if registry_date_tuple >= query_date_tuple:
                    continue
            except Exception:
                continue

        dataset = registry_data.get("dataset", "Unknown")

        for img_path in registry_data.get("images", []):
            # Get or create feature cache for gallery images using unified cache
            gallery_features = get_or_cache_features(img_path, extractor, device)
            gallery_features = move_to_device(gallery_features, device)

            # Match query vs gallery
            with torch.inference_mode():
                matches = matcher(
                    {"image0": query_features, "image1": gallery_features}
                )

                matches = rbd(matches)

            # Extract match count
            match_idx = matches["matches"]
            if isinstance(match_idx, torch.Tensor) and match_idx.ndim == 3:
                match_idx = match_idx[0]
            match_count = int(match_idx.shape[0])

            # Keep best match per unique_key
            current = candidate_scores.get(unique_key)
            if current is None or match_count > current[0]:
                candidate_scores[unique_key] = (match_count, img_path, dataset)

    # Sort candidates by match count (descending)
    candidates = sorted(
        (
            (unique_key, score, path, dataset)
            for unique_key, (score, path, dataset) in candidate_scores.items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    # Build results with gallery paths and dataset
    results = []
    for unique_key, match_count, best_match_path, dataset in candidates:
        gallery_images = registry.get(unique_key, {}).get("images", [])
        results.append(
            (unique_key, match_count, best_match_path, gallery_images, dataset)
        )

    return results


def get_top3_predictions_for_best_image(
    masked_images: List[str],
    bina_registry: Dict,
    uri_registry: Dict,
    extractor,
    matcher,
    query_date: str,
    query_unique_key: str,
) -> Tuple[List[Dict], str, int]:
    """Get top-3 predictions using the image with highest match count.

    Tests all images in the group and selects the one with the highest top-1 match count.
    Matches against both Bina registry (all dates) and URI registry (earlier dates only).

    Returns:
        Tuple of (top_3_predictions, best_image_path, best_image_index)
    """
    best_image = None
    best_idx = 0
    best_predictions = []
    best_match_count = -1

    # Test each image and find the one with highest top-1 match count
    for idx, query_path in enumerate(masked_images):
        # Extract features for query image using unified cache
        query_features = get_or_cache_features(query_path, extractor, device)

        # Match against Bina registry (no temporal filter)
        bina_predictions = match_single_query_vs_registry(
            query_features,
            bina_registry,
            extractor,
            matcher,
            query_date=query_date,
            query_unique_key=query_unique_key,
            apply_temporal_filter=False,
        )

        # Match against URI registry (with temporal filter - earlier dates only)
        uri_predictions = match_single_query_vs_registry(
            query_features,
            uri_registry,
            extractor,
            matcher,
            query_date=query_date,
            query_unique_key=query_unique_key,
            apply_temporal_filter=True,
        )

        # Combine and sort all predictions by match count
        combined_predictions = bina_predictions + uri_predictions
        combined_predictions.sort(key=lambda x: x[1], reverse=True)

        # Check if this image has better top-1 match count
        if combined_predictions and len(combined_predictions) > 0:
            top1_match_count = combined_predictions[0][
                1
            ]  # match_count is second element
            if top1_match_count > best_match_count:
                best_match_count = top1_match_count
                best_predictions = combined_predictions
                best_image = query_path
                best_idx = idx

    # Format top-3 predictions from best image
    top_3 = []
    for rank, (reg_id, match_count, best_match_path, gallery, dataset) in enumerate(
        best_predictions[:3], 1
    ):
        # Format registry_id based on dataset
        # For URI: use format "<date>/<temp_id>" (e.g., "2016-12/1")
        # For Bina: use just ID (e.g., "14", "15") - extracted from folder structure
        # reg_id is already in the correct format for both
        formatted_id = reg_id

        top_3.append(
            {
                "rank": rank,
                "registry_id": formatted_id,
                "match_count": int(match_count),
                "best_match_image": best_match_path,
                "gallery": gallery,
                "dataset": dataset,
            }
        )

    return top_3, best_image, best_idx


def process_batch(
    entries: Dict[str, List[str]],
    bina_registry: Dict,
    uri_registry: Dict,
    extractor,
    matcher,
) -> Dict:
    """Process all entries and generate predictions.

    Returns:
        Dict mapping unique_key -> prediction entry
    """
    results = {}

    for unique_key, masked_images in tqdm(entries.items(), desc="Processing entries"):
        date, temp_id = unique_key.split("/")

        # Get top-3 predictions using the image with highest match count
        top_3_preds, best_image, best_idx = get_top3_predictions_for_best_image(
            masked_images,
            bina_registry,
            uri_registry,
            extractor,
            matcher,
            date,
            unique_key,
        )

        # Build entry
        entry = {
            "date": date,
            "temp_id": temp_id,
            "unique_key": unique_key,
            "query_folder": str(Path(masked_images[0]).parent),
            "query_images": masked_images,  # All masked images for this date/temp_id
            "best_query_image": best_image,
            "best_query_index": best_idx,
            "predictions": top_3_preds,
            "status": "pending",
            "decision": None,
            "confirmed_registry_id": None,
            "timestamp_processed": datetime.now().isoformat(),
        }

        results[unique_key] = entry

    return results


def run_prediction_mode(csv_path: str, output_path: str):
    """Run prediction mode: match query images against both Bina and URI registries.

    Args:
        csv_path: Path to CSV with masked image paths
        output_path: Path to save predictions JSON
    """
    print(f"🔍 Prediction Mode")
    print(f"Input CSV: {csv_path}")
    print(f"Output: {output_path}")
    print()

    # Load models
    print("Loading LightGlue models...")
    extractor = (
        ALIKED(max_num_keypoints=1432, detection_threshold=0.01).to(device).eval()
    )
    matcher = LightGlue(features="aliked").to(device).eval()
    print("✓ Models loaded")
    print()

    # Load data (expects pre-masked images)
    entries = load_csv_entries(csv_path)
    print()

    # Load both registries
    print("Loading Bina and URI registries...")
    bina_registry, uri_registry = load_combined_registries()
    print()

    # Process all entries
    print("Processing batch predictions...")
    print("  - Matching against Bina registry (all dates)")
    print("  - Matching against URI registry (earlier dates only)")
    results = process_batch(entries, bina_registry, uri_registry, extractor, matcher)
    print(f"✓ Processed {len(results)} entries")
    print()

    # Save results
    print(f"Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved batch predictions")
    print()

    # Summary
    total = len(results)
    print("📊 Summary:")
    print(f"  Total entries: {total}")
    print(f"  Bina registry size: {len(bina_registry)}")
    print(f"  URI registry size: {len(uri_registry)}")
    print(f"  Status breakdown:")
    print(f"    Pending: {total}")
    print(f"    Confirmed: 0")
    print(f"    New ID: 0")
    print(f"    Skipped: 0")
    print()
    print("✅ Done! Ready for review in Gradio app.")


def main():
    parser = argparse.ArgumentParser(
        description="Batch prediction for Latonia ReID - Match query images against Bina and URI registries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python batch_predict.py --csv uri_photos_mask.csv --output batch_predictions.json

Notes:
  - Matches against bina_photos_mask.csv (all dates)
  - Matches against uri_photos_mask.csv (earlier dates only)
  - Returns top-3 predictions with "dataset" key ("Bina" or "Uri")
        """,
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV with masked image paths (query images)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file for predictions",
    )
    parser.add_argument(
        "--cache", default="cache", help="Directory for feature cache (default: cache)"
    )

    args = parser.parse_args()

    print(f"🐸 Latonia ReID - Batch Prediction Script")
    print(f"Device: {device}")
    print(f"Matching against: bina_photos_mask.csv + uri_photos_mask.csv")
    print()

    run_prediction_mode(args.csv, args.output)


if __name__ == "__main__":
    main()


# Example usage:
# python batch_predict.py --csv uri_photos_mask.csv --output batch_predictions.json
#
# Key changes:
# - Uses BOTH bina_photos_mask.csv and uri_photos_mask.csv as reference registries
# - For URI registry: applies temporal filtering (only matches with earlier dates)
# - Returns top-3 predictions (changed from top-2)
# - Each prediction includes "dataset" key: "Bina" or "Uri"
