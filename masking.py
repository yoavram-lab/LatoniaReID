from __future__ import annotations

from pathlib import Path
from typing import Iterable

import click
import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

DEFAULT_EXTENSIONS = (".jpg", ".jpeg")


def initialize_sam(sam_checkpoint: Path, sam_type: str, device: torch.device) -> SamPredictor:
    """Load SAM predictor on the requested device."""
    if not sam_checkpoint.exists():
        raise click.ClickException(f"Checkpoint not found: {sam_checkpoint}")

    click.echo(f"Loading {sam_type} model on {device}...")
    sam = sam_model_registry[sam_type](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    return SamPredictor(sam)


def collect_images(root: Path, extensions: Iterable[str]) -> list[Path]:
    ext_set = {ext.lower() for ext in extensions}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext_set])


def save_mask(save_path: Path, mask: np.ndarray) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask * 255).astype(np.uint8)
    success, buffer = cv2.imencode(".png", mask_uint8)
    if success:
        save_path.write_bytes(buffer)
    else:
        click.echo(f"Warning: failed to encode mask for {save_path}", err=True)


def process_folder(
    predictor: SamPredictor,
    data_root: Path,
    mask_root: Path,
    extensions: Iterable[str],
    overwrite: bool,
) -> None:
    all_files = collect_images(data_root, extensions)
    click.echo(f"Found {len(all_files)} images under {data_root}")
    if not all_files:
        return

    with torch.inference_mode():
        with tqdm(total=len(all_files), desc="Masking", unit="image") as progress:
            for img_path in all_files:
                relative_path = img_path.relative_to(data_root)
                save_path = mask_root / relative_path.with_suffix(".png")

                if save_path.exists() and not overwrite:
                    progress.update(1)
                    continue

                image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    click.echo(f"Warning: could not read {img_path}", err=True)
                    progress.update(1)
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                predictor.set_image(image_rgb)

                h, w = image.shape[:2]
                input_point = np.array([[w // 2, h // 2]])
                input_label = np.array([1])

                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                selected_mask = masks[0]
                save_mask(save_path, selected_mask)
                progress.update(1)

    click.echo(f"Done. Masks saved to {mask_root}")


@click.command()
@click.option(
    "--data_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("rotated"),
    show_default=True,
    help="Root directory of images to mask.",
)
@click.option(
    "--mask_root",
    type=click.Path(path_type=Path),
    default=None,
    show_default=True,
    help="Output directory for masks; defaults to <data_root>_sam next to the images.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    show_default=True,
    help="Device to run inference on (e.g., cuda or cpu).",
)
@click.option(
    "--sam_checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("sam_vit_b_01ec64.pth"),
    show_default=True,
    help="Path to the SAM checkpoint.",
)
@click.option(
    "--sam_type",
    type=click.Choice(sorted(sam_model_registry.keys())),
    default="vit_b",
    show_default=True,
    help="SAM backbone type.",
)
@click.option(
    "--extensions",
    type=str,
    default=",".join(ext.lstrip(".") for ext in DEFAULT_EXTENSIONS),
    show_default=True,
    help="Comma-separated list of file extensions to process.",
)
@click.option(
    "--overwrite/--skip-existing",
    default=False,
    show_default=True,
    help="Overwrite existing masks or skip them.",
)
def main(
    data_root: Path,
    mask_root: Path | None,
    device: str,
    sam_checkpoint: Path,
    sam_type: str,
    extensions: str,
    overwrite: bool,
) -> None:
    mask_root = mask_root or data_root.parent / f"{data_root.name}_sam"
    ext_list = [f".{ext.strip().lstrip('.').lower()}" for ext in extensions.split(",") if ext.strip()]
    if not ext_list:
        raise click.BadParameter("Provide at least one valid extension", param_hint="--extensions")

    torch_device = torch.device(device)
    predictor = initialize_sam(sam_checkpoint, sam_type, torch_device)
    process_folder(predictor, data_root, mask_root, ext_list, overwrite)


if __name__ == "__main__":
    main()
