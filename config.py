from pathlib import Path
import os


DATA_FOLDER   = Path("data/")
CHECKPOINTS_FOLDER = Path("checkpoints/")
os.makedirs(CHECKPOINTS_FOLDER, exist_ok=True)
RESULTS_FOLDER =  Path("results/")
os.makedirs(RESULTS_FOLDER, exist_ok=True)
SEGMENTED_FOLDER = Path("segmented")
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)
