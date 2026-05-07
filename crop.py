import csv
import os
from pathlib import Path
from typing import Dict, List, Any

import click
import tqdm
from PIL import Image, ImageOps
from megadetector.detection import run_detector
from megadetector.visualization import visualization_utils as vis_utils

megadetector_model_name = 'MDV5A'


def bbox_image(image_path: str) -> List[float]:
    """Get bounding box for a single image using MegaDetector."""
    image = vis_utils.load_image(image_path)
    model = run_detector.load_detector(megadetector_model_name)
    result = model.generate_detections_one_image(image)
    detections_above_threshold = [d for d in result['detections'] if d['conf'] > 0.2]
    if not detections_above_threshold:
        raise ValueError(f"No detections found in {image_path}")
    bbox = detections_above_threshold[0]['bbox']
    return bbox


def crop_bbox_and_pad_square(img: Image.Image, bbox: List[float], pad: bool = True, pad_color=(0,0,0)) -> Image.Image:
    """Crop image to bounding box and optionally pad to square."""
    w, h = img.size
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int((bbox[0] + bbox[2]) * w)
    y2 = int((bbox[1] + bbox[3]) * h)
    crop = img.crop((x1, y1, x2, y2))
    if pad:
        crop = ImageOps.pad(crop, (max(crop.size), max(crop.size)), color=pad_color)
    return crop


def load_csv_entries(csv_path: str) -> List[Dict[str, Any]]:
    """Load all rows from CSV, preserving all columns."""
    entries = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    return entries


def save_csv_output(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    """Save rows to CSV, preserving all columns from input."""
    if not rows:
        click.echo("No rows to write", err=True)
        return

    fieldnames = rows[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@click.command()
@click.argument('input_csv', type=click.Path(exists=True, path_type=Path))
@click.option('--output_root', default='data/labeled_bbox',
              help='Output folder for cropped images')
@click.option('--pad/--no-pad', default=True,
              help='Pad images to square (default: pad)')
@click.option('--device', default='cuda',
              help='Device for MegaDetector (cuda or cpu)')
def main(input_csv: Path, output_root: str, pad: bool, device: str) -> None:
    """
    Crop images from CSV to bounding boxes.

    INPUT CSV: rel_path, label, [date, session_id, Inferred, ...]
    OUTPUT CSV: rel_path (updated), label, [date, session_id, ...] (preserved)
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load all rows from input CSV
    entries = load_csv_entries(str(input_csv))
    click.echo(f"Processing {len(entries)} images from {input_csv}")

    # Process each row
    updated_rows = []
    for row in tqdm.tqdm(entries, desc="Cropping images"):
        rel_path = row['rel_path']

        try:
            # Load image
            image = Image.open(rel_path)

            # Get bounding box
            bbox = bbox_image(rel_path)

            # Crop and pad
            cropped = crop_bbox_and_pad_square(image, bbox, pad=pad)

            # Save cropped image with same folder structure
            output_path = output_root / Path(rel_path).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(str(output_path))

            # Update row with new path (relative to repo root)
            row['rel_path'] = str(output_path)
            updated_rows.append(row)

        except Exception as e:
            click.echo(f"Error processing {rel_path}: {e}", err=True)
            continue

    # Write output CSV
    output_csv = output_root.parent / f"{output_root.name}.csv"
    save_csv_output(str(output_csv), updated_rows)
    click.echo(f"✓ Cropped images saved to {output_root}")
    click.echo(f"✓ Output CSV saved to {output_csv}")


if __name__ == '__main__':
    main()
