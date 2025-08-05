import os
import json
from megadetector.detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file
from megadetector.utils import path_utils
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.detection import run_detector
from PIL import ImageOps, ImageDraw, Image
from pathlib import Path
import click
import tqdm

# to setup megadetector we neet yolov5
# git clone https://github.com/ultralytics/yolov5.git
# cd yolov5
# pip install -r requirements.txt
# git checkout v6.0
import sys
sys.path.insert(0, 'yolov5')
# pip install megadetector

megadetector_model_name = 'MDV5A'

def bbox_image(image_path):
    image = vis_utils.load_image(image_path)
    model = run_detector.load_detector(megadetector_model_name)
    result = model.generate_detections_one_image(image)
    detections_above_threshold = [d for d in result['detections'] if d['conf'] > 0.2]
    bbox = detections_above_threshold[0]['bbox']
    return bbox


def crop_bbox_and_pad_square(img : Image.Image, bbox : list, pad : bool, pad_color=(0,0,0)):
    w, h = img.size
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int((bbox[0] + bbox[2]) * w)
    y2 = int((bbox[1] + bbox[3]) * h)
    crop = img.crop((x1, y1, x2, y2))
    if pad:
        crop = ImageOps.pad(crop, (max(crop.size), max(crop.size)), color=pad_color)
    return crop


def draw_image_with_bbox(image_path : Path, bbox : list):
    image = vis_utils.load_image(image_path)
    draw = ImageDraw.Draw(image)
    w, h = image.size
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int((bbox[0] + bbox[2]) * w)
    y2 = int((bbox[1] + bbox[3]) * h)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return image


def bbox_image_folder(image_folder, output=None):
    image_file_names = path_utils.find_images(image_folder, recursive=True)
    results = load_and_run_detector_batch(megadetector_model_name, image_file_names, quiet=True)
    if not output is None: 
        # Write results to a format that Timelapse and other downstream tools like.
        write_results_to_file(results, output)
    return results


def crop_and_pad_folder(image_folder, bbox_file, output_folder, pad):
    bbox_dict = json_to_dict(bbox_file)
    for path, bbox in tqdm.tqdm(bbox_dict.items(), desc=f"Cropping images from {bbox_file} to {output_folder}"):
        img = Image.open(path)
        crop = crop_bbox_and_pad_square(img, bbox, pad=pad)
        output_path = os.path.join(output_folder, *Path(path).parts[1:])
        os.makedirs(Path(output_path).parent, exist_ok=True)
        crop.save(output_path)


def json_to_dict(json_path):
    with open(json_path, 'r') as f:
        results = json.load(f)    
    return {r['file']: r['detections'][0]['bbox'] for r in results['images']}


@click.command()
@click.argument('image_folder', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output', type=click.Path(path_type=Path))
@click.option('--overwrite/--skip-existing', default=False, help="Overwrite existing output files or skip them (default: skip)")
@click.option('--crop/--no-crop', default=False, help="Crop images to bounding box (default: no-crop)")
@click.option('--pad/--no-pad', default=False, help="Pad images around bounding box (default: no-pad)")
def main(image_folder, output, overwrite, crop, pad):
    if not output.suffix == '.json':
        raise click.BadParameter('Output file must be .json')
    if os.path.exists(output) and not overwrite:
        click.echo("Output file already exists. Use --overwrite to overwrite it.")
    else:
        click.echo(f"Detecting bounding boxes in {image_folder} using {megadetector_model_name}...")
        bbox_image_folder(image_folder, output)
        click.echo(f"Wrote results to {output}")
    if crop:
        output_folder = image_folder.parent / (image_folder.name + '_bbox')
        click.echo(f"Cropping images to bounding box from {image_folder} to {output_folder}...")
        crop_and_pad_folder(image_folder, output, output_folder, pad=pad)


if __name__ == '__main__':
    main()
