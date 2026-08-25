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
}
