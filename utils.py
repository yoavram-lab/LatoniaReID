from pathlib import Path
import numpy as np

def find_image_subfolders(folder):
    """
    Find all subfolders containing images.
    
    Args:
        folder: Folder to search for subfolders
        
    Returns:
        list: List of subfolders containing images
    """
    image_paths = find_image_paths(folder)
    subfolders = set()
    for path in image_paths:
        subfolders.add(str(Path(path).parent))
    subfolders = list(subfolders)
    return subfolders


def find_image_paths(folder, extensions=('.jpg', '.jpeg', '.png')):
    """Get all image paths in a folder."""
    folder = Path(folder)
    return [
        str(f) 
        for f in folder.rglob('*') 
        if f.suffix.lower() in extensions
    ]


def count_images(folder):
    paths = find_image_paths(folder)
    # count images per subfolder
    subfolders = find_image_subfolders(folder)
    counts = {subfolder.replace(str(folder)+'/', ""): 0 for subfolder in subfolders}
    for path in paths:
        counts[str(Path(path).parent).replace(str(folder)+'/', "")] += 1
    return counts


def extract_ids(paths):
    """
    Extract IDs from paths.
    
    Args:
        paths: Array of paths of shape (n_samples,)
        
    Returns:
        Array of IDs of shape (n_samples,)
    """
    return np.array(['/'.join(path.split('/')[-3:-1]) for path in paths])
