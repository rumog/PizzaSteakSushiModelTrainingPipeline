from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RamImageDataset(Dataset):
    """
    Dataset class that supports loading all dataset images into RAM so they don't have to be decoded during the training loop on-the-fly
    (for exapmle, like when using the ImageFolder dataset)

    Images are intentionally NOT resized here, so that images of different sizes and aspect ratios can co-exist in ram.  When using this Dataset
    it is expected that a separate transform process (e.g. GPU autmentation pipeline) will produce the final image tensor shape expected by the model
    """

    MAX_IMAGE_SIZE = 350

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        self.classes = sorted(
            directory.name
            for directory in self.root_dir.iterdir()
            if directory.is_dir()
        )

        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(self.classes)
        }

        self.samples = []
        self.images = []

        # Decode every image. This is a one time up-front cost when creating the dataset as opposed to using
        # A dataset like ImageFolder which will decode the image (e.g. jpg file) for every batch load every epoch.
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_index = self.class_to_idx[class_name]

            for image_path in sorted(class_dir.iterdir()):
                # improve this list, although it shouldn't be necessary for our current case as we always
                # preprocess all images to jpg
                if image_path.suffix.lower() not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                }:
                    continue
                with Image.open(image_path) as image:
                    image = image.convert("RGB")

                    image.thumbnail(
                        (self.MAX_IMAGE_SIZE, self.MAX_IMAGE_SIZE),
                        Image.Resampling.BILINEAR,
                    )

                    if len(self.images) == 0:
                        print(f"DEBUG: first image after thumbnail = {image.size}")

                    # H x W x C -> C x H x W
                    tensor = (
                        torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()
                    )
                    self.images.append(tensor)
                    self.samples.append((image_path, class_index))
        self.targets = [class_index for _, class_index in self.samples]
        print(f"Loaded {len(self.images)} into RAM from: {self.root_dir}")

        total_bytes = sum(image.numel() * image.element_size() for image in self.images)

        print(f"Loaded {len(self.images)} images into RAM from: {self.root_dir}")
        print(f"Decoded image RAM: {total_bytes / (1024**3):.2f} GiB")

    def __getitem__(self, index):
        return self.images[index], self.samples[index][1]

    def __len__(self):
        return len(self.images)

    def get_class_list(self):
        return self.classes


def ram_collate_fn(batch):
    """
    DO NOT stack images here, as images are different sizes/aspect ratios- i.e. different shape tensors
    """
    # batch is a list of (image, label) tuples, this unpacks them to separate lists of image and label
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)
