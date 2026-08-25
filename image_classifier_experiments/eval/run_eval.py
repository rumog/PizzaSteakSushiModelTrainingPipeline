import multiprocessing as mp
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix

from image_classifier_experiments.data_setup import data_setup
from image_classifier_experiments.eval.eval import make_predictions
from image_classifier_experiments.model_build.efficientnet_b0_transfer import (
    EfficientNetB0TransferLearningModel,
)

DATA_PATH_PARENT_DIR = "data"
IMAGE_PATH_PARENT_DIR = "seattlement_birds_50_100_percent"
MODEL_SAVE_DIR = "model"


def log_and_display_confusion_matrix(model_artifact=str, device: str = "cpu"):
    data_path = Path(DATA_PATH_PARENT_DIR)
    image_path = data_path / IMAGE_PATH_PARENT_DIR
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Get default EfficientNet_B0 pre_processing transform
    default_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    transform = default_weights.transforms()

    _, test_dataloader, class_list = data_setup.create_image_folder_dataloaders(
        train_dir,
        test_dir,
        transform,
        transform,
        batch_size=64,
        num_workers=os.cpu_count() - 2,
        shuffle_train=True,
    )
    # Create model instance
    model_0 = EfficientNetB0TransferLearningModel(
        num_classes=len(class_list),
        from_artifact=model_artifact,
    ).to(device)

    predictions = make_predictions(
        model=model_0, data_loader=test_dataloader, device=device
    )

    confmat = ConfusionMatrix(num_classes=len(class_list), task="multiclass")
    # note that there is a torch inconsistency in datasets to pay attention to.  For the FashionMNIST
    # dataset we just used dataset.targets, because it returns that as a single tensor for you.  For
    # ImageFolder dataset, 'targets' doen't do that, so we have to do it manually.
    confmat_tensor = confmat(
        preds=predictions, target=torch.tensor(test_dataloader.dataset.targets)
    )

    num_classes = len(class_list)
    fig_size_dim = max(12, int(num_classes * 0.4))
    # plot the confusion matrix tensor
    # matplotlib likes numpy so converting
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_list,
        figsize=(fig_size_dim, fig_size_dim),
        show_absolute=True,
        show_normed=False,  # Turn off percentages to prevent text clipping/clutter
        colorbar=True,  # Adds a clean color scale bar on the right
    )
    plt.show()

    # 2. Fix the axis labels formatting
    # Rotate labels 90 degrees so they read vertically instead of smashing horizontally
    ax.set_xticklabels(class_list, rotation=90, ha="right", fontsize=8)
    ax.set_yticklabels(class_list, fontsize=8)

    # 3. Prevent text overlapping inside cells
    # If text labels are completely washing out the grid colors,
    # hide cell text for very small matrix sizes or high dimensions
    if num_classes > 30:
        for text in ax.texts:
            text.set_fontsize(6)  # Shrink cell numbers down so grid colors show through

    # 4. Enforce tight layout bounds so labels don't get cut off on export
    plt.tight_layout()

    # Save a crisp high-res image to view cleanly in VS Code
    fig.savefig("eval_logging/confusion_matrix_fixed.png", dpi=300, bbox_inches="tight")
    print("Saved clean, high-resolution confusion matrix to confusion_matrix_fixed.png")

    # save to csv
    # Convert matrix to a DataFrame with headers
    df_cm = pd.DataFrame(confmat_tensor.numpy(), index=class_list, columns=class_list)

    # Save to your data subdirectory
    csv_path = Path("eval_logging/confusion_matrix.csv")
    df_cm.to_csv(csv_path)
    print(f"Raw matrix text saved to {csv_path}")


if __name__ == "__main__":
    # mp.set_start_method("fork", force=True)
    log_and_display_confusion_matrix(
        model_artifact="EfficientNetB0TransferLearningModel_ep90_20260821_195405_v0.0.pth",
        device="mps",
    )
