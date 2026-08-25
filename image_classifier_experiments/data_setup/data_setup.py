"""
Contains functionality for creating PyTorch Dataloaders for
Image Classification data.
"""

import os
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from image_classifier_experiments.data_setup.cached_features_dataset import (
    CatchedFeaturesDataset,
)
from image_classifier_experiments.data_setup.ram_image_dataset import (
    RamImageDataset,
    ram_collate_fn,
)


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: Callable,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
):
    """
    Create training and testing Dataloaders from the root directories provided.
    Args:
    [fill in later]
    """
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle_train,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    class_names = train_dataset.classes

    return train_dataloader, test_dataloader, class_names


def create_image_folder_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transform: Callable,
    test_transform: Callable,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
):
    """
    Create training and testing Dataloaders from the root directories provided.
    Args:
    [fill in later]
    """
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    if num_workers >= os.cpu_count():
        print(
            f"num_workers {num_workers} is at or above the os cpu count: {os.cpu_count()}. Ignoring provided"
            f"count and setting num_workers to {os.cpu_count() - 1}"
        )
        num_workers = os.cpu_count() - 1

    if num_workers > 0:
        pin_memory_arg = torch.cuda.is_available()
        persistent_worker_arg = True
        prefetch_factor_arg = 4
    else:
        pin_memory_arg = False
        persistent_worker_arg = False
        prefetch_factor_arg = None

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle_train,
        pin_memory=pin_memory_arg,
        persistent_workers=persistent_worker_arg,
        prefetch_factor=prefetch_factor_arg,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory_arg,
        persistent_workers=persistent_worker_arg,
        prefetch_factor=prefetch_factor_arg,
    )
    class_names = train_dataset.classes

    return train_dataloader, test_dataloader, class_names


def create_image_ram_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
):
    """
    Create training and testing Dataloaders from the root directories provided.
    Args:
    [fill in later]
    """
    train_dataset = RamImageDataset(root_dir=train_dir)
    test_dataset = RamImageDataset(root_dir=test_dir)

    if num_workers >= os.cpu_count():
        print(
            f"num_workers {num_workers} is at or above the os cpu count: {os.cpu_count()}. Ignoring provided"
            f"count and setting num_workers to {os.cpu_count() - 1}"
        )
        num_workers = os.cpu_count() - 1

    if num_workers > 0:
        pin_memory_arg = torch.cuda.is_available()
        persistent_worker_arg = True
        prefetch_factor_arg = 2
    else:
        pin_memory_arg = False
        persistent_worker_arg = False
        prefetch_factor_arg = None

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle_train,
        pin_memory=pin_memory_arg,
        persistent_workers=persistent_worker_arg,
        prefetch_factor=prefetch_factor_arg,
        collate_fn=ram_collate_fn,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory_arg,
        persistent_workers=persistent_worker_arg,
        prefetch_factor=prefetch_factor_arg,
        collate_fn=ram_collate_fn,
    )
    class_names = train_dataset.classes

    return train_dataloader, test_dataloader, class_names


# TODO: Remove this after testing the version below and cleaning up old
# training code that uses it
def create_feature_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
):
    """
    Create training and testing Dataloaders from the root directories provided.
    Args:
    [fill in later]
    """

    if num_workers >= os.cpu_count():
        print(
            f"num_workers {num_workers} is at or above the os cpu count: {os.cpu_count()}. Ignoring provided"
            f"count and setting num_workers to {os.cpu_count() - 1}"
        )
        num_workers = os.cpu_count() - 1

    if num_workers > 0:
        pin_memory_arg = torch.cuda.is_available()
        persistent_worker_arg = True
        prefetch_factor_arg = 2
    else:
        pin_memory_arg = False
        persistent_worker_arg = False
        prefetch_factor_arg = None

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle_train,
        pin_memory=pin_memory_arg,
        persistent_workers=persistent_worker_arg,
        prefetch_factor=prefetch_factor_arg,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory_arg,
        persistent_workers=persistent_worker_arg,
        prefetch_factor=prefetch_factor_arg,
    )

    return train_dataloader, test_dataloader


def create_cached_feature_dataloaders(
    train_cached_features_path: str,
    test_cached_features_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
):
    """
    Create training and testing Dataloaders from the root directories provided.
    Args:
    [fill in later]
    """

    train_dataset = CatchedFeaturesDataset(train_cached_features_path)
    test_dataset = CatchedFeaturesDataset(test_cached_features_path)

    if num_workers >= os.cpu_count():
        print(
            f"num_workers {num_workers} is at or above the os cpu count: {os.cpu_count()}. Ignoring provided"
            f"count and setting num_workers to {os.cpu_count() - 1}"
        )
        num_workers = os.cpu_count() - 1

    if num_workers > 0:
        pin_memory_arg = torch.cuda.is_available()
        persistent_worker_arg = True
        prefetch_factor_arg = 2
    else:
        pin_memory_arg = False
        persistent_worker_arg = False
        prefetch_factor_arg = None

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle_train,
        pin_memory=pin_memory_arg,
        persistent_workers=persistent_worker_arg,
        prefetch_factor=prefetch_factor_arg,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory_arg,
        persistent_workers=persistent_worker_arg,
        prefetch_factor=prefetch_factor_arg,
    )

    return train_dataloader, test_dataloader
