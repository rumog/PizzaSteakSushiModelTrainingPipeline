from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torchvision
from torch import nn

from image_classifier_experiments.data_setup import data_setup
from image_classifier_experiments.model_build.efficient_net_b0 import EfficientNetB0Pss
from image_classifier_experiments.training import engine
from image_classifier_experiments.utils.helper_functions import (
    accuracy_fn,
    plot_loss_curves,
    save_model_checkpoint,
    save_model_checkpoint_s3,
)

MODEL_NAME = "EfficientNetB0Pss_ep2_20260802_020056_v0.0.pth"
S3_MODEL_BUCKET = "pss-classifier-models-613693331461-us-west-2-an"
MODEL_KEY_PREFIX = "pss-classifier/0.1.0"

# training and model hyperparams
EPOCHS = 2
BATCH_SIZE = 32
NUM_WORKERS = 0  # os.cpu_count()
DATA_PATH_PARENT_DIR = "data/"
IMAGE_PATH_PARENT_DIR = "pizza_steak_sushi_100_percent"
MODEL_SAVE_DIR = "model"
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)
SAVE_MODEL = "file"  # None, s3, file


def redact_dict(d):
    # Base case: if it's not a dictionary, return it as is
    if not isinstance(d, dict):
        return d

    new_dict = {}
    for key, value in d.items():
        if key == "state_dict":
            new_dict[key] = "<redacted>"
        elif isinstance(value, dict):
            new_dict[key] = redact_dict(value)  # Recurse into nested dicts
        elif isinstance(value, list):
            # Recurse into lists in case dicts are hidden inside them
            new_dict[key] = [redact_dict(item) for item in value]
        else:
            new_dict[key] = value

    return new_dict


torch.manual_seed(42)
torch.mps.manual_seed(42)

data_path = Path(DATA_PATH_PARENT_DIR)
image_path = data_path / IMAGE_PATH_PARENT_DIR
train_dir = image_path / "train"
test_dir = image_path / "test"

# Hard code device for compatibility with M1 mac for now
# change this if expanding.
device = "mps" if torch.mps.is_available() else "cpu"

# Create a simple transform with minimal augmentaiton.
# Can expand on this with a collection of common transforms used
# for experimentation.
transform = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()

simple_train_dataloader, simple_test_dataloader, class_list = (
    data_setup.create_dataloaders(
        train_dir,
        test_dir,
        transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle_train=True,
    )
)

model_0 = EfficientNetB0Pss(num_classes=len(class_list)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=LEARNING_RATE)

results = engine.train_model(
    model=model_0,
    train_dataloader=simple_train_dataloader,
    test_dataloader=simple_test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    epochs=EPOCHS,
    device=device,
)


print(f"train_model output: {redact_dict(results)}")

# Save model if requested
if SAVE_MODEL:
    model_architecture_name = model_0.__class__.__name__

    # MODEL_NAME = "intro_pytorch_computer_vision_model_2.pth"
    model_architecture = {"name": model_architecture_name, "weights": "DEFAULT"}

    model_preprocessing = {"image_size": IMAGE_SIZE}

    # NOTE:
    # Currently using the course train/test split.
    # The test split is acting as validation during model selection.
    # When a true validation split is added, this contract should remain unchanged
    # but the internal training pipeline names like "test_loss" should be updated to "validation_loss" etc.
    model_training = {
        "epoch": results["best_checkpoint"]["epoch"],
        "validation_accuracy": results["best_checkpoint"]["test_acc"],
        "validation_loss": results["best_checkpoint"]["test_loss"],
    }

    model_metadata = {
        "class_list": class_list,
        "architecture": model_architecture,
        "preprocessing": model_preprocessing,
        "training": model_training,
    }

    version = 0.0
    timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
    filename = f"{model_architecture_name}_ep{results['best_checkpoint']['epoch']}_{timestamp}_v{version}.pth"

    if SAVE_MODEL == "s3":
        save_model_checkpoint_s3(
            state_dict=results["best_checkpoint"]["state_dict"],
            model_metadata=model_metadata,
            bucket_name=S3_MODEL_BUCKET,
            object_key=f"{MODEL_KEY_PREFIX}/{filename}",
        )
    elif SAVE_MODEL == "file":
        save_model_checkpoint(
            state_dict=results["best_checkpoint"]["state_dict"],
            model_metadata=model_metadata,
            target_dir=MODEL_SAVE_DIR,
            file_name=filename,
        )

plot_loss_curves(results["history"])
