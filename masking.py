import csv
from pathlib import Path
from typing import Dict, List, Any

import click
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png")


def initialize_sam(sam_checkpoint: Path, sam_type: str, device: torch.device) -> SamPredictor:
    """Load SAM predictor on the requested device."""
    if not sam_checkpoint.exists():
        raise click.ClickException(f"Checkpoint not found: {sam_checkpoint}")

    click.echo(f"Loading {sam_type} model on {device}...")
    sam = sam_model_registry[sam_type](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    return SamPredictor(sam)


def apply_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to image (black background where mask is 0)."""
    mask_bool = mask.astype(bool)
    masked = np.zeros_like(image_rgb)
    masked[mask_bool] = image_rgb[mask_bool]
    return masked


def save_masked_image(save_path: Path, masked_image: np.ndarray) -> None:
    """Save masked image as PNG."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(masked_image).save(str(save_path), format="PNG")


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
@click.option('--output_root', default='data/labeled_mask',
              help='Output folder for masked images')
@click.option('--sam_checkpoint', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=Path("checkpoints/sam_vit_b_01ec64.pth"),
              help='Path to SAM checkpoint')
@click.option('--sam_type', type=str, default='vit_b',
              help='SAM model type (vit_b, vit_l, vit_h)')
@click.option('--device', type=str, default='cuda',
              help='Device for inference (cuda or cpu)')
@click.option('--overwrite/--skip-existing', default=False,
              help='Overwrite existing masked images')
def main(input_csv: Path, output_root: str, sam_checkpoint: Path, sam_type: str,
         device: str, overwrite: bool) -> None:
    """
    Apply SAM masking to images from CSV.

    INPUT CSV: rel_path, [label, date, session_id, ...]
    OUTPUT CSV: rel_path (updated), [label, date, session_id, ...] (preserved)
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load all rows from input CSV
    entries = load_csv_entries(str(input_csv))
    click.echo(f"Processing {len(entries)} images from {input_csv}")

    # Initialize SAM
    torch_device = torch.device(device)
    predictor = initialize_sam(sam_checkpoint, sam_type, torch_device)

    # Process each row
    updated_rows = []
    with torch.inference_mode():
        for row in tqdm(entries, desc="Masking images"):
            rel_path = row['rel_path']

            try:
                # Load image
                with Image.open(rel_path) as img_pil:
                    image_rgb = np.array(img_pil.convert('RGB'))

                # Set image for SAM
                predictor.set_image(image_rgb)

                # Predict mask (center point, positive label)
                h, w = image_rgb.shape[:2]
                input_point = np.array([[w // 2, h // 2]])
                input_label = np.array([1])
                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                selected_mask = masks[0]

                # Apply mask
                masked_image = apply_mask(image_rgb, selected_mask)

                # Save masked image
                output_path = output_root / Path(rel_path).name
                output_path = output_path.with_suffix('.png')
                save_masked_image(output_path, masked_image)

                # Update row with new path
                row['rel_path'] = str(output_path)
                updated_rows.append(row)

            except Exception as e:
                click.echo(f"Error processing {rel_path}: {e}", err=True)
                continue

    # Write output CSV with correct name
    output_csv = output_root.parent / "labeled_mask.csv"
    save_csv_output(str(output_csv), updated_rows)
    click.echo(f"Masked images saved to {output_root}")
    click.echo(f"Output CSV saved to {output_csv}")


if __name__ == "__main__":
    main()
