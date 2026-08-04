from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torchvision
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from image_classifier_experiments.data_setup import data_setup
from image_classifier_experiments.model_build.efficient_net_b0 import EfficientNetB0Pss
from image_classifier_experiments.training import engine
from image_classifier_experiments.training.arg_parser import parse_train_args
from image_classifier_experiments.utils.helper_functions import (
    accuracy_fn,
    plot_loss_curves,
    save_model_checkpoint,
    save_model_checkpoint_s3,
)

S3_MODEL_BUCKET = "pss-classifier-models-613693331461-us-west-2-an"
MODEL_KEY_PREFIX = "pss-classifier/0.1.0"

# training and model hyperparams
NUM_WORKERS = 0  # os.cpu_count()
DATA_PATH_PARENT_DIR = "data/"
IMAGE_PATH_PARENT_DIR = "pizza_steak_sushi_100_percent"
MODEL_SAVE_DIR = "model"
IMAGE_SIZE = (224, 224)

torch.manual_seed(42)
torch.mps.manual_seed(42)

data_path = Path(DATA_PATH_PARENT_DIR)
image_path = data_path / IMAGE_PATH_PARENT_DIR
train_dir = image_path / "train"
test_dir = image_path / "test"


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


args = parse_train_args()
print(f"Args passed: {args}")

# Tensorboard integration
timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")

run_name = f"efnet_b0_ep{args.epochs}_lr{args.lr}_esp{args.early_stop_patience}_lrp{args.lr_schedule_patience}_wd{args.weight_decay}_ts_{timestamp}"
writer = SummaryWriter(log_dir=f"runs/efficientnet_b0/{run_name}")

# Hard code device for compatibility with M1 mac for now
# change this if expanding.
device = "mps" if torch.mps.is_available() else "cpu"

# Handle commandline arg parsing


# Create a simple transform with minimal augmentaiton.
# Can expand on this with a collection of common transforms used
# for experimentation.
transform = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()

simple_train_dataloader, simple_test_dataloader, class_list = (
    data_setup.create_dataloaders(
        train_dir,
        test_dir,
        transform,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        shuffle_train=True,
    )
)

# Create model instance
model_0 = EfficientNetB0Pss(num_classes=len(class_list)).to(device)

# Create loss function, lr scheduler, and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    params=model_0.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)
# min mode, as we will monitor and react to loss metric

scheduler = None
if args.lr_schedule_patience is not None:
    print(
        f"schedule_patience set to {args.lr_schedule_patience}, creating lr scheduler"
    )
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=args.lr_schedule_patience,
    )

try:
    results = engine.train_model(
        model=model_0,
        train_dataloader=simple_train_dataloader,
        test_dataloader=simple_test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        epochs=args.epochs,
        device=device,
        patience=args.early_stop_patience,
        writer=writer,
    )
    # collect hyperparams,
    writer.add_hparams(
        {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "scheduled_epochs": args.epochs,
            "early_stop_patience": args.early_stop_patience,
            "lr_schedule_patience": args.lr_schedule_patience,
        },
        {
            "best_accuracy": results["best_checkpoint"]["test_acc"],
            "best_loss": results["best_checkpoint"]["test_loss"],
            "epochs_completed": results["train_metadata"]["epochs_completed"],
            "early_stopped": results["train_metadata"]["stopped_early"],
        },
    )
finally:
    # collect hyperparams, close TensorBoard writer
    writer.close()

print(f"train_model output: {redact_dict(results)}")

# Save model if requested
if args.save is not None:
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

    if args.save == "s3":
        save_model_checkpoint_s3(
            state_dict=results["best_checkpoint"]["state_dict"],
            model_metadata=model_metadata,
            bucket_name=S3_MODEL_BUCKET,
            object_key=f"{MODEL_KEY_PREFIX}/{filename}",
        )
    elif args.save == "file":
        save_model_checkpoint(
            state_dict=results["best_checkpoint"]["state_dict"],
            model_metadata=model_metadata,
            target_dir=MODEL_SAVE_DIR,
            file_name=filename,
        )
else:
    print(f"args.save = {args.save}, skipping checkpoint artifact save")

# plot_loss_curves(results["history"])
