import os
import random
import shutil
from pathlib import Path

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Setup data directory
DATA_DIR = "data"
SOURCE_IMG_DIR = "food-101/images"
SOURCE_META_DIR = "food-101/meta"
# Setup data paths
# data_path = data_dir / "food-101" / "images"

# Change amount of data to get (e.g. 0.1 = random 10%, 0.2 = random 20%)
# amount_to_get = 0.2


def download_food_101(dest_dir: str):
    datasets.Food101(
        root=dest_dir,
        split="train",
        # transform=transforms.ToTensor(),
        download=True,
    )

    # Get testing data
    datasets.Food101(
        root=dest_dir,
        split="test",
        # transform=transforms.ToTensor(),
        download=True,
    )


# Create function to separate a random amount of data
def get_subset(
    image_path: Path,
    meta_path: Path,
    data_splits=["train", "test"],
    target_classes=["pizza", "steak", "sushi"],
    amount=0.1,
    seed=42,
):
    random.seed(seed)
    label_splits = {}

    # Get labels
    for data_split in data_splits:
        print(f"[INFO] Creating image split for: {data_split}...")
        label_path = meta_path / f"{data_split}.txt"
        with open(label_path, "r") as f:
            labels = [
                line.strip("\n")
                for line in f.readlines()
                if line.split("/")[0] in target_classes
            ]
        print(f"labels for {data_split}: {labels}")

        # Get random subset of target classes image ID's
        number_to_sample = round(amount * len(labels))
        print(
            f"[INFO] Getting random subset of {number_to_sample} images for {data_split}..."
        )
        sampled_images = random.sample(labels, k=number_to_sample)
        print(f"sampled_images for {data_split}: {sampled_images}")

        # Apply full paths
        image_paths = [
            Path(str(image_path / sample_image) + ".jpg")
            for sample_image in sampled_images
        ]
        label_splits[data_split] = image_paths
        print(f"image_paths for {data_split}: {image_paths}")
    return label_splits


def download_food_101_custom(
    data_dir: str,
    source_img_dir: str,
    source_meta_dir: str,
    target_classes: list[str] = ["pizza", "steak", "sushi"],
    amount=0.01,
):
    data_path = Path(data_dir)
    source_image_path = data_path / source_img_dir
    meta_path = data_path / source_meta_dir
    label_splits = get_subset(
        source_image_path,
        meta_path,
        target_classes=target_classes,
        amount=amount,
    )

    # Create target directory path
    sub_dir = "_".join(target_classes)
    target_dir_name = f"{data_dir}/{sub_dir}_{str(int(amount * 100))}_percent"

    # Setup the directories
    target_dir = Path(target_dir_name)
    if target_dir.exists():
        print(f"{target_dir} already exists, exiting without copying files.")
        return

    print(f"Creating directory: '{target_dir_name}'")
    # Make the directories
    target_dir.mkdir(parents=True, exist_ok=True)

    for image_split in label_splits.keys():
        for image_path in label_splits[str(image_split)]:
            dest_dir = (
                target_dir / image_split / image_path.parent.stem / image_path.name
            )
            if not dest_dir.parent.is_dir():
                dest_dir.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Copying {image_path} to {dest_dir}...")
            shutil.copy2(image_path, dest_dir)


# Check lengths of directories
def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """

    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


# download_food_101(DATA_DIR)

# download_food_101_custom(
#    DATA_DIR, SOURCE_IMG_DIR, SOURCE_META_DIR, ["pizza", "steak", "sushi"], 1
# )


walk_through_dir("data/pizza_steak_sushi_50_percent")
walk_through_dir("data/pizza_steak_sushi_75_percent")
walk_through_dir("data/pizza_steak_sushi_100_percent")
