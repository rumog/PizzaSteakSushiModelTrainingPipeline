import os
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from zoneinfo import ZoneInfo

import torch
import torchvision
import torchvision.transforms.v2 as transforms_v2
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from image_classifier_experiments.data_setup import data_setup
from image_classifier_experiments.data_setup.cached_features_dataset import (
    CatchedFeaturesDataset,
)
from image_classifier_experiments.model_build.efficient_net_b0 import EfficientNetB0Pss
from image_classifier_experiments.training import engine
from image_classifier_experiments.training.arg_parser import parse_train_args
from image_classifier_experiments.utils.helper_functions import (
    accuracy_fn,
    plot_loss_curves,
    print_train_time,
    save_model_checkpoint,
    save_model_checkpoint_s3,
)

# S3_MODEL_BUCKET = "pss-classifier-models-613693331461-us-west-2-an"
# MODEL_KEY_PREFIX = "pss-classifier/0.1.0"

DEFAULT_WEIGHTS = torchvision.models.EfficientNet_B0_Weights.DEFAULT
RAW_TRANSFORM = transforms.PILToTensor()
# training and model hyperparams
NUM_WORKERS = 3
DATA_PATH_PARENT_DIR = "data/"
IMAGE_PATH_PARENT_DIR = "seattlement_birds_50_100_percent"
MODEL_SAVE_DIR = "model"
IMAGE_SIZE = (224, 224)
FEATURE_CACHE_ENABLED = False
ENABLE_CUSTOM_AUGMENTATION = True
CACHED_FEATURES_TRAIN_PATH = "model/cached_features/train.pt"
CACHED_FEATURES_TEST_PATH = "model/cached_features/test.pt"
ENABLE_GPU_AUGMENTATION = True
ENABLE_LOAD_IMAGES_TO_RAM = False

# Hard-coded GPU augmentation pipeline for testing performance
# On setups like the G5.xl where GPU is much more powerful than mac M1
# But CPU is less powerful and potential bottlneck
CPU_AUTMENTATION_PIPELINE = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=DEFAULT_WEIGHTS.transforms().mean,
            std=DEFAULT_WEIGHTS.transforms().std,
        ),
    ]
)

GPU_AUGMENTATION_PIPELINE = transforms_v2.Compose(
    [
        transforms_v2.RandomResizedCrop(
            size=(224, 224),
            scale=(0.8, 1.0),
            antialias=True,
        ),
        transforms_v2.RandomHorizontalFlip(),
        transforms_v2.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms_v2.Normalize(
            mean=DEFAULT_WEIGHTS.transforms().mean,
            std=DEFAULT_WEIGHTS.transforms().std,
        ),
    ]
)

torch.manual_seed(42)
torch.mps.manual_seed(42)


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


def run_training():
    data_path = Path(DATA_PATH_PARENT_DIR)
    image_path = data_path / IMAGE_PATH_PARENT_DIR
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    args = parse_train_args()
    print(f"Args passed: {args}")

    # Tensorboard integration
    timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")

    run_name = f"efnet_b0_ep{args.epochs}_lr{args.lr}_esp{args.early_stop_patience}_lrp{args.lr_schedule_patience}_wd{args.weight_decay}_ts_{timestamp}"
    writer = SummaryWriter(log_dir=f"runs/efficientnet_b0/{run_name}")

    # Hard code device for compatibility with M1 mac for now
    # change this if expanding.
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Handle commandline arg parsing

    # Get default EfficientNet_B0 pre_processing transform
    default_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    test_transform = default_weights.transforms()

    if ENABLE_CUSTOM_AUGMENTATION and not ENABLE_GPU_AUGMENTATION:
        ## Custom, currently hard coded autmentation pipeline for testing
        train_transform = CPU_AUTMENTATION_PIPELINE
        print(
            f"ENABLE_CUSTOM_AUGMENTATION: {ENABLE_CUSTOM_AUGMENTATION} "
            f"ENABLE_GPU_AUGMENTATION: {ENABLE_GPU_AUGMENTATION} "
            f"- Using custom CPU augmentation pipeline: {train_transform}"
        )
    elif ENABLE_CUSTOM_AUGMENTATION and ENABLE_GPU_AUGMENTATION:
        train_transform = RAW_TRANSFORM
        print(
            f"ENABLE_CUSTOM_AUGMENTATION: {ENABLE_CUSTOM_AUGMENTATION} "
            f"ENABLE_GPU_AUGMENTATION: {ENABLE_GPU_AUGMENTATION} "
            f"- Using RAW_TRANFORM: {train_transform}"
        )
    else:
        # Use the default EfficientNet_B0 transform pre-processing
        # For training as well as testing
        train_transform = test_transform
        print(
            f"ENABLE_CUSTOM_AUGMENTATION: {ENABLE_CUSTOM_AUGMENTATION} "
            f"ENABLE_GPU_AUGMENTATION: {ENABLE_GPU_AUGMENTATION} "
            f"- Using default EfficientNetB0 Transform: {train_transform}"
        )

    simple_train_dataloader, simple_test_dataloader, class_list = (
        data_setup.create_image_folder_dataloaders(
            train_dir,
            test_dir,
            train_transform,
            test_transform,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,
            shuffle_train=True,
        )
    )

    # Create model instance
    model_0 = EfficientNetB0Pss(num_classes=len(class_list)).to(device)

    # If feature caching enabled, we need to run a single forward pass and cache backbone features
    # Keep in mind that this will not be effective if randomized autmentation is enabled, so
    # Force disable using both here and default to not caching backbone
    start_time = timer()
    if FEATURE_CACHE_ENABLED and not ENABLE_CUSTOM_AUGMENTATION:
        print(
            f"FEATURE_CACHE_ENABLED: {FEATURE_CACHE_ENABLED}, ENABLE_CUSTOM_AUGMENTATION: {ENABLE_CUSTOM_AUGMENTATION} - extracting and saving backbonefeatures "
            f"and creating feature-based dataloaders for train/test"
        )
        if not Path(CACHED_FEATURES_TRAIN_PATH).is_file():
            engine.extract_backbone_features(
                model_0.backbone,
                simple_train_dataloader,
                device,
                CACHED_FEATURES_TRAIN_PATH,
            )
        if not Path(CACHED_FEATURES_TEST_PATH).is_file():
            engine.extract_backbone_features(
                model_0.backbone,
                simple_test_dataloader,
                device,
                CACHED_FEATURES_TEST_PATH,
            )

        train_dataset = CatchedFeaturesDataset(CACHED_FEATURES_TRAIN_PATH)
        test_dataset = CatchedFeaturesDataset(CACHED_FEATURES_TEST_PATH)

        train_dataloader, test_dataloader = data_setup.create_feature_dataloaders(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,
            shuffle_train=False,
        )
    else:
        print(
            f"FEATURE_CACHE_ENABLED: {FEATURE_CACHE_ENABLED}, ENABLE_CUSTOM_AUGMENTATION: {ENABLE_CUSTOM_AUGMENTATION} - SKIPPING bakcbone caching "
            f"and creating image-based dataloaders for train/test"
        )
        train_dataloader = simple_train_dataloader
        test_dataloader = simple_test_dataloader

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

    if (
        ENABLE_CUSTOM_AUGMENTATION
        and ENABLE_GPU_AUGMENTATION
        and not FEATURE_CACHE_ENABLED
    ):
        print(
            f"ENABLE_CUSTOM_AUGMENTATION: {ENABLE_CUSTOM_AUGMENTATION} "
            f"ENABLE_GPU_AUGMENTATION: {ENABLE_GPU_AUGMENTATION} "
            f"FEATURE_CACHE_ENABLED: {FEATURE_CACHE_ENABLED} "
            f"- Using default EfficientNetB0 Transform: {train_transform}"
        )
        gpu_transform = GPU_AUGMENTATION_PIPELINE
    else:
        gpu_transform = None

    try:
        results = engine.train_model(
            model=model_0,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            epochs=args.epochs,
            device=device,
            patience=args.early_stop_patience,
            writer=writer,
            caching_enabled=FEATURE_CACHE_ENABLED,
            gpu_transform=gpu_transform,
        )
        end_time = timer()
        train_time = print_train_time(start_time, end_time, device)
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
        timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
            "%Y%m%d_%H%M%S"
        )
        filename = f"{model_architecture_name}_ep{results['best_checkpoint']['epoch']}_{timestamp}_v{version}.pth"

        if args.save == "s3":
            save_model_checkpoint_s3(
                state_dict=results["best_checkpoint"]["state_dict"],
                model_metadata=model_metadata,
                bucket_name=args.s3_bucket,
                object_key=f"{args.s3_key_prefix}/{filename}",
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


if __name__ == "__main__":
    run_training()
