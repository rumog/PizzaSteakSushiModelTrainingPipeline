from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torchvision
import torchvision.transforms.v2 as transforms_v2
from torchvision import transforms


@dataclass(frozen=True)
class AugmentationConfig:
    cpu: Callable
    gpu: Callable


B0_DEFAULT_WEIGHTS = torchvision.models.EfficientNet_B0_Weights.DEFAULT
B0_AUGMENTATION_EXPERIMENTS = {
    # --------
    # experiment1:
    # Initial reize 256 on short edge- retains aspect ratio
    # Random Crop and Horizontal Flip
    # --------
    "experiment_1": AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.Resize(
                    256,
                    antialias=True,
                ),
                transforms_v2.RandomCrop(
                    224,
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
    ),
    # --------
    # experiment2:
    # Random Resize Crop and Horizontal Flip + Color Jitter
    # --------
    "experiment_2": AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.8, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.8, 1.0),
                    antialias=True,
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
    ),
    # --------
    # experiment3:
    # Initial reize 256x256
    # Random Crop and Horizontal Flip + mild Color Jitter + Ratation + Affine
    # --------
    "experiment_3": AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=10,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                ),
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.Resize(
                    size=(256, 256),
                    antialias=True,
                ),
                transforms_v2.RandomCrop(
                    size=(224, 224),
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomRotation(
                    degrees=10,
                ),
                transforms_v2.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                ),
                transforms_v2.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
    ),
    # --------
    # experiment4:
    # Random Resize Crop 224x224 and Horizontal Flip
    # --------
    "experiment_4": AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.8, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.8, 1.0),
                    antialias=True,
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
    ),
    # --------
    # experiment5:
    # Initial reize 256 on short edge- retains aspect ratio
    # Random Crop 224x224 only
    # --------
    "experiment_5": AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.Resize(
                    256,
                    antialias=True,
                ),
                transforms_v2.RandomCrop(
                    224,
                ),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
    ),
    # --------
    # experiment6:
    # Initial reize 256 on short edge- retains aspect ratio
    # Center Crop 224x224 + Horizontal Flip only
    # --------
    "experiment_6": AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.Resize(
                    256,
                    antialias=True,
                ),
                transforms_v2.CenterCrop(224),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
    ),
    # --------
    # experiment1_300:
    # Initial reize 343 on short edge- retains aspect ratio
    # Random Crop and Horizontal Flip
    # --------
    "experiment_1_300": AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.Resize(343),
                transforms.RandomCrop(300),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.Resize(
                    343,
                    antialias=True,
                ),
                transforms_v2.RandomCrop(
                    300,
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
    ),
    # --------
    # experiment_2_300:
    # Random Resize Crop and Horizontal Flip + Color Jitter
    # --------
    "experiment_2_300": AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    300,
                    scale=(0.8, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    size=(300, 300),
                    scale=(0.8, 1.0),
                    antialias=True,
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
    ),
    # --------
    # experiment_3_300:
    # Initial reize 343x343
    # Random Crop and Horizontal Flip + mild Color Jitter + Ratation + Affine
    # --------
    "experiment_3_300": AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.Resize((343, 343)),
                transforms.RandomCrop((300, 300)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=10,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                ),
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.Resize(
                    size=(343, 343),
                    antialias=True,
                ),
                transforms_v2.RandomCrop(
                    size=(300, 300),
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomRotation(
                    degrees=10,
                ),
                transforms_v2.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                ),
                transforms_v2.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
    ),
}

B0_VALIDATION_TRANSFORMS = {
    # --------
    # experiment1:
    # Initial reize 343 on short edge- retains aspect ratio
    # Random Crop and Horizontal Flip
    # --------
    "experiment_1_300": AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.Resize(
                    343,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.Resize(
                    343,
                    interpolation=transforms_v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms_v2.CenterCrop(300),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=B0_DEFAULT_WEIGHTS.transforms().mean,
                    std=B0_DEFAULT_WEIGHTS.transforms().std,
                ),
            ]
        ),
    )
}


# --------
# experiment 1:
# Initial reize on short edge- retains aspect ratio
# Random Crop and Horizontal Flip
# --------
def experiment_1_factory(default_weights, image_size: int) -> AugmentationConfig:
    validate_factory_args(default_weights, image_size)
    resize_size = round(image_size * 8 / 7)
    return AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=default_weights.transforms().mean,
                    std=default_weights.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.Resize(
                    resize_size,
                    antialias=True,
                ),
                transforms_v2.RandomCrop(
                    image_size,
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=default_weights.transforms().mean,
                    std=default_weights.transforms().std,
                ),
            ]
        ),
    )


# --------
# experiment 2:
# Random Resize Crop and Horizontal Flip + Color Jitter
# --------
def experiment_2_factory(default_weights, image_size: int) -> AugmentationConfig:
    validate_factory_args(default_weights, image_size)
    return AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=default_weights.transforms().mean,
                    std=default_weights.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(0.8, 1.0),
                    antialias=True,
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=default_weights.transforms().mean,
                    std=default_weights.transforms().std,
                ),
            ]
        ),
    )


# --------
# experiment 3:
# Initial reize (alters aspect ratio)
# Random Crop and Horizontal Flip + mild Color Jitter + Ratation + Affine
# --------
def experiment_3_factory(default_weights, image_size: int) -> AugmentationConfig:
    validate_factory_args(default_weights, image_size)
    resize_size = round(image_size * 8 / 7)
    return AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=10,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                ),
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=default_weights.transforms().mean,
                    std=default_weights.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.Resize(
                    size=(resize_size, resize_size),
                    antialias=True,
                ),
                transforms_v2.RandomCrop(
                    size=(image_size, image_size),
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomRotation(
                    degrees=10,
                ),
                transforms_v2.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                ),
                transforms_v2.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=default_weights.transforms().mean,
                    std=default_weights.transforms().std,
                ),
            ]
        ),
    )


# --------
# experiment 4:
# Random Resize Crop + Horizontal Flip
# --------
def experiment_4_factory(default_weights, image_size: int) -> AugmentationConfig:
    validate_factory_args(default_weights, image_size)
    return AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=default_weights.transforms().mean,
                    std=default_weights.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(0.8, 1.0),
                    antialias=True,
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=default_weights.transforms().mean,
                    std=default_weights.transforms().std,
                ),
            ]
        ),
    )


# --------
# experiment 5:
# Initial reize on short edge- retains aspect ratio
# Random Crop
# --------
def experiment_5_factory(default_weights, image_size: int) -> AugmentationConfig:
    validate_factory_args(default_weights, image_size)
    resize_size = round(image_size * 8 / 7)
    return AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=default_weights.transforms().mean,
                    std=default_weights.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.Resize(
                    resize_size,
                    antialias=True,
                ),
                transforms_v2.RandomCrop(
                    image_size,
                ),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=default_weights.transforms().mean,
                    std=default_weights.transforms().std,
                ),
            ]
        ),
    )


# --------
# experiment 6:
# Initial reize on short edge- retains aspect ratio
# Center Crop + Horizontal Flip
# --------
def experiment_6_factory(default_weights, image_size: int) -> AugmentationConfig:
    validate_factory_args(default_weights, image_size)
    resize_size = round(image_size * 8 / 7)
    return AugmentationConfig(
        cpu=transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=default_weights.transforms().mean,
                    std=default_weights.transforms().std,
                ),
            ]
        ),
        gpu=transforms_v2.Compose(
            [
                transforms_v2.Resize(
                    resize_size,
                    antialias=True,
                ),
                transforms_v2.CenterCrop(image_size),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
                transforms_v2.Normalize(
                    mean=default_weights.transforms().mean,
                    std=default_weights.transforms().std,
                ),
            ]
        ),
    )


def validate_factory_args(default_weights, image_size):
    if not (default_weights and image_size):
        raise ValueError(
            f"Augmentation pipeline generation requires valid values for both default_weights: "
            f"{default_weights} and image_size: {image_size} to be set"
        )


AUGMENTATION_EXPERIMENTS = {
    "experiment_1": experiment_1_factory,
    "experiment_2": experiment_2_factory,
    "experiment_3": experiment_3_factory,
    "experiment_4": experiment_4_factory,
    "experiment_5": experiment_5_factory,
    "experiment_6": experiment_6_factory,
}
