import os
import zipfile
from pathlib import Path

import requests

DEST_DATA_DIR = "data/"
IMAGE_DATA_DIR = "pizza_steak_sushi"
PIZZA_STEAK_SUSHI_ZIP_NAME = "pizza_steak_sushi.zip"
PIZZA_STEAK_SUSHI_SOURCE_URL = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
# Set up path to data folder
data_path = Path(DEST_DATA_DIR)
image_path = data_path / IMAGE_DATA_DIR

# If image folder doesn't exist, download it and prepare it
if image_path.is_dir():
    print(f"Image data directory {image_path} already exists, skipping download...")
else:
    print(f"{image_path} directory not found, creating it...")
    image_path.mkdir(parents=True, exist_ok=True)

# Download the image data
with open(data_path / PIZZA_STEAK_SUSHI_ZIP_NAME, "wb") as f:
    request = requests.get(PIZZA_STEAK_SUSHI_SOURCE_URL)
    print("Downloading pizza steak sushi data...")
    f.write(request.content)

# Unzip to data file to image directory
with zipfile.ZipFile(data_path / PIZZA_STEAK_SUSHI_ZIP_NAME, "r") as zip_ref:
    print(f"Unzipping {PIZZA_STEAK_SUSHI_ZIP_NAME} to {image_path}")
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / PIZZA_STEAK_SUSHI_ZIP_NAME)
