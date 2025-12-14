from pathlib import Path
import re

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from models import get_model
from similarities import get_similarity_function


def prepare_image(path: Path, preprocess, device: torch.device):
    """Load, preprocess, and resize the RGB image to match the extractor input."""
    image = Image.open(path).convert("RGB")
    tensor = preprocess(image)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    width, height = tensor.shape[-1], tensor.shape[-2]
    resized = image.resize((width, height))
    return tensor, np.array(resized)


def plot_matches(img1, img2, kpts0, kpts1):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    kpts1_shifted = kpts1.copy()
    kpts1_shifted[:, 0] += w1

    plt.imshow(canvas)
    for (x0, y0), (x1, y1) in zip(kpts0, kpts1_shifted):
        plt.plot([x0, x1], [y0, y1], c="lime", lw=0.5, alpha=0.7)
        plt.scatter(x0, y0, c="lime", s=3)
        plt.scatter(x1, y1, c="lime", s=3)
    plt.axis("off")


def build_output_path(img_a: Path, img_b: Path) -> Path:
    output_dir = Path("results") / "lightglue"
    output_dir.mkdir(parents=True, exist_ok=True)
    strip_ext = lambda p: Path(p).with_suffix("")  # remove final extension only
    sanitize = lambda p: re.sub(r"[^A-Za-z0-9._-]+", "_", strip_ext(p).as_posix())
    filename = f"{sanitize(img_a)}-{sanitize(img_b)}.png"
    return output_dir / filename


def parse_date_id(path: Path):
    """Extract date/id assuming .../<date>/<id>/<filename> structure."""
    parts = path.parts
    if len(parts) < 3:
        return "unknown", "unknown"
    return parts[-3], parts[-2]


@click.command()
@click.argument("image_a", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("image_b", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--extractor", "-e", default="ALIKED", show_default=True, help="Extractor name as defined in models.py")
@click.option("--device", "-d", default="cpu", show_default=True, help="Device for inference (e.g., cpu or cuda)")
def main(image_a: Path, image_b: Path, extractor: str, device: str):
    device = torch.device(device)
    extractor_model, preprocess, model_name = get_model(extractor)
    if model_name.lower() != "aliked":
        raise click.ClickException(f"LightGlue visualization supports the ALIKED extractor; got {model_name}.")
    extractor_model = extractor_model.eval().to(device)

    matcher_wrapper = get_similarity_function("lightglue").to(device)
    matcher = matcher_wrapper.model
    rbd = matcher_wrapper.rbd

    tensor_a, vis_a = prepare_image(image_a, preprocess, device)
    tensor_b, vis_b = prepare_image(image_b, preprocess, device)

    with torch.inference_mode():
        feats_a = extractor_model(tensor_a)
        feats_b = extractor_model(tensor_b)
        matches = matcher({"image0": feats_a, "image1": feats_b})
        feats_a, feats_b, matches = [rbd(x) for x in (feats_a, feats_b, matches)]

    kpts0, kpts1 = feats_a["keypoints"], feats_b["keypoints"]
    match_idx = matches["matches"]
    if isinstance(match_idx, torch.Tensor) and match_idx.ndim == 3:
        match_idx = match_idx[0]
    matched0, matched1 = kpts0[match_idx[..., 0]], kpts1[match_idx[..., 1]]

    click.echo(f"Found {len(matched0)} matches between {image_a} and {image_b}.")

    plt.figure(figsize=(12, 8))
    plot_matches(vis_a, vis_b, matched0.cpu().numpy(), matched1.cpu().numpy())
    date_a, id_a = parse_date_id(image_a)
    date_b, id_b = parse_date_id(image_b)
    title = (
        f"{image_a.name} ({date_a}/{id_a}) ↔ "
        f"{image_b.name} ({date_b}/{id_b}) | "
        f"{model_name} + LightGlue | {len(matched0)} matches"
    )
    plt.title(title)

    output_path = build_output_path(image_a, image_b)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    click.echo(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
