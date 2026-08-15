"""
Contains functionality for creating PyTorch Dataloaders for
Image Classification data.
"""

from collections.abc import Callable

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


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

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle_train,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    class_names = train_dataset.classes

    return train_dataloader, test_dataloader, class_names


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

    return train_dataloader, test_dataloader
