import multiprocessing as mp
from collections.abc import Callable
from datetime import datetime
from itertools import accumulate
from pathlib import Path
from timeit import default_timer as timer
from typing import Any
from zoneinfo import ZoneInfo

import torch
import torchvision
import torchvision.transforms.v2 as transforms_v2
from torch import nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from image_classifier_experiments.augmentation.augmentation_experiments import (
    B0_AUGMENTATION_EXPERIMENTS,
)
from image_classifier_experiments.data_setup import data_setup
from image_classifier_experiments.model_build.efficientnet_b0_transfer import (
    EfficientNetB0TransferLearningModel,
)
from image_classifier_experiments.training import engine
from image_classifier_experiments.training.arg_parser import TrainArgs, parse_train_args
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

# This is needed currently for the 'load images into ram first' scenario
# because the the ram loaded images retain their original size as opposed
# to being resized to matching tensors so they can be stacked by datasets like
# torch ImageFolder
RAW_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((500, 500)),
        transforms.PILToTensor(),
    ]
)
# training and model hyperparams
DATA_PATH_PARENT_DIR = "data/"
IMAGE_PATH_PARENT_DIR = "seattlement_birds_50_100_percent"
MODEL_SAVE_DIR = "model"
IMAGE_SIZE = (224, 224)
CACHED_FEATURES_TRAIN_PATH = "model/cached_features/train.pt"
CACHED_FEATURES_TEST_PATH = "model/cached_features/test.pt"
ENABLE_UNFROZEN_BACKBONE_TRAINING = True

torch.manual_seed(42)
torch.mps.manual_seed(42)
torch.cuda.manual_seed(42)


def run_training():

    # Handle commandline arg parsing
    args = parse_train_args()
    print(f"Args passed: {args}")

    # Tensorboard integration
    writer = get_tensorboard_writer(args) if args.enable_tensorboard else None

    device = get_device()

    # Create train and test transforms used in image dataloader creation
    train_transform, test_transform = get_dataloader_transforms(args)

    # Create image dataloaders
    train_image_dataloader, test_image_dataloader, class_list = get_image_dataloaders(
        args=args,
        train_transform=train_transform,
        test_transform=test_transform,
    )

    # Create the model
    model_0 = EfficientNetB0TransferLearningModel(
        num_classes=len(class_list),
        from_artifact=args.load_model_from_artifact,
        unfrozen_backbone_blocks=get_unfrozen_backbone_blocks(args),
        dropout_override=args.classifier_dropout,
    ).to(device)

    if args.enable_backbone_caching and not args.augmentation_config:
        # If feature caching enabled, cache backbone using the image dataloaders
        # and create feature-based dataloaders for use in train/test
        train_dataloader, test_dataloader = (
            cache_backbone_and_create_feature_dataloaders(
                model_0.backbone,
                args,
                train_image_dataloader,
                test_image_dataloader,
                device,
            )
        )
    else:
        # else, use the image dataloaders directly
        train_dataloader = train_image_dataloader
        test_dataloader = test_image_dataloader

    # Create loss function, lr scheduler, and optimizer
    loss_fn = get_loss_fn(args.label_smoothing)

    optimizer = get_optimizer(model=model_0, args=args)

    scheduler = create_scheduler_and_attach_to_optimizer(
        optimizer=optimizer,
        args=args,
    )

    # Get GPU-based transform if required, otherwise will be None
    gpu_transform = get_gpu_transform(args)

    try:
        start_time = timer()
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
            caching_enabled=args.enable_backbone_caching,
            gpu_transform=gpu_transform,
            test_transform=test_transform,
            ram_caching_enabled=args.enable_ram_loaded_images,
        )
        end_time = timer()
        train_time = print_train_time(start_time, end_time, device)
        if writer:
            write_tensorboard_hyperparams(writer, results, args)
    finally:
        if writer:
            writer.close()

    print(f"train_model output: {redact_dict(results)}")

    # Save training artifaacts if requested
    save_training_artifacts(
        model=model_0, results=results, class_list=class_list, args=args
    )

    # plot_loss_curves(results["history"])


def get_image_dataloaders(
    args: TrainArgs,
    train_transform: Callable,
    test_transform: Callable,
):
    data_path = Path(DATA_PATH_PARENT_DIR)
    image_path = data_path / IMAGE_PATH_PARENT_DIR
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # In the standard case where RAM loading of images is not enabled
    # generate dataloaders using the ImageFolder based creator
    if not args.enable_ram_loaded_images:
        train_dataloader, test_dataloader, class_list = (
            data_setup.create_image_folder_dataloaders(
                train_dir,
                test_dir,
                train_transform,
                test_transform,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle_train=True,
            )
        )
    # If RAM laoding of images is enabled, use custom RAM loading
    # based dataloaders. RAM loading is used in conjunction with
    # GPU-based augmentation, so no transform is suppoied here.
    # Transform will be applied on the fly during training loop
    else:
        train_dataloader, test_dataloader, class_list = (
            data_setup.create_image_ram_dataloaders(
                train_dir,
                test_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle_train=True,
            )
        )

    return train_dataloader, test_dataloader, class_list


def cache_backbone_and_create_feature_dataloaders(
    backbone: nn.Module,
    args: TrainArgs,
    train_image_dataloader,
    test_image_dataloader,
    device,
):
    # Run a single forward pass using train/test image dataloaders, and cache
    # features to train/test files
    cache_bb_features_start = timer()
    print("Caching backbone features for training image set")
    engine.extract_backbone_features(
        backbone,
        train_image_dataloader,
        device,
        CACHED_FEATURES_TRAIN_PATH,
    )
    print("Caching backbone features for test image set")
    engine.extract_backbone_features(
        backbone,
        test_image_dataloader,
        device,
        CACHED_FEATURES_TEST_PATH,
    )
    cache_bb_features_end = timer()
    print(
        f"Caching backbone features on train/test dataset took {cache_bb_features_end - cache_bb_features_start:.3f} seconds"
    )
    # Use cached features to create train/test feature dataloaders
    train_dataloader, test_dataloader = data_setup.create_cached_feature_dataloaders(
        train_cached_features_path=CACHED_FEATURES_TRAIN_PATH,
        test_cached_features_path=CACHED_FEATURES_TEST_PATH,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=False,
    )
    return train_dataloader, test_dataloader


def extract_backbone_param_groups(
    model: nn.Module,
    backbone_lr_stages: list[tuple[int, float]],
    backbone_wd_stages: list[float],
):
    param_groups = []
    # converting to list and using this is slightly more performant
    # than slicing and using the nn.sequential in-place.
    backbone_blocks = list(model.backbone.features)

    num_blocks_by_stage = [block for block, _ in backbone_lr_stages]
    # This verification is not really the job of this function, move this out later
    # to verify before calling this function
    if sum(num_blocks_by_stage) > len(backbone_blocks):
        raise ValueError(
            f"Total number of unfrozen backbone blocks {sum(num_blocks_by_stage)} "
            f"cannot be more than total existing backbone blocks in model: {len(backbone_blocks)}"
        )

    # create a list of indices that can be used to index into the desired
    # backbone block stages.  For example if you are freezing the backbone in
    # stages [2, 3, 3], the indices would be [2, 5, 8], and each stage of blocks could
    # be accessed using split notation [-2:], [-5:-2], and [-8:-5]
    frozen_block_stage_indices = list(accumulate(num_blocks_by_stage))

    for i, (_, blocks_lr) in enumerate(backbone_lr_stages):
        if i == 0:
            selected_blocks = backbone_blocks[-frozen_block_stage_indices[i] :]
        else:
            selected_blocks = backbone_blocks[
                -frozen_block_stage_indices[i] : -frozen_block_stage_indices[i - 1]
            ]

        params = []
        for block in selected_blocks:
            params.extend([p for p in block.parameters() if p.requires_grad])
        if params:
            param_groups.append(
                {
                    "params": params,
                    "lr": blocks_lr,
                    "weight_decay": backbone_wd_stages[i],
                }
            )

    return param_groups


# This logic is messy and hard codes assumptions for now
# [TODO] update scheduler creation to be cleaner/more configurable
"""
def create_scheduler_and_attach_to_optimizer(
    optimizer: torch.optim.Optimizer,
    unfrozen_backbone_blocks: int,
    training_args: TrainArgs,
):

    scheduler = None
    if training_args.epochs <= 1:
        print(
            f"Using scheduler: None. Maximum training epochs: {training_args.epochs}, is less than or equal to 1."
        )
        return None

    # lr_schedule_patience presence indicates to use ReduceLROnPlateau for now.  However currently
    # the choice is hard coded to use composite warmup + cosineAnnealing if any backbone blocks
    # are unfrozen, even if lr_schedule_patience is set.
    if (
        training_args.lr_schedule_patience is not None
        and not unfrozen_backbone_blocks > 0
    ):
        print(
            f"schedule_patience set to {training_args.lr_schedule_patience}, creating lr scheduler"
        )
        # min mode, as we will monitor and react to loss metric for now- can make this configurable later
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.1,
            patience=training_args.lr_schedule_patience,
            min_lr=1e-6,
        )

    # If unfrozen backbone layers exist, use cosine annealing scheduler instead
    # This is mutually exculsive with arts.lr_schedule-patience, that vlue should NOT be
    # Set when not using a scheduler that uses it.

    # Experimenting with warmup- note that this code won't be valid for epochs = 1
    # will make this more explicit later.
    elif unfrozen_backbone_blocks > 0:
        warmup_epochs = min(5, max(1, training_args.epochs // 10))
        cosine_epochs = training_args.epochs - warmup_epochs

        # Warm up every parameter group from 10% of it's configured LR
        warmup_scheduler = LinearLR(
            optimizer=optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cosine_epochs,
            eta_min=1e-6,
        )

        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[
                warmup_scheduler,
                cosine_scheduler,
            ],
            milestones=[warmup_epochs],
        )
        print(
            f"Using warmup + cosine scheduler: "
            f"warmup_epochs={warmup_epochs}, "
            f"cosine_epochs={cosine_epochs}"
        )

    if getattr(scheduler, "_schedulers", None):
        print(
            f"Using composite scheduler {scheduler.__class__.__name__}: with schedulers: {[s.__class__.__name__ for s in scheduler._schedulers]}"
        )
    elif scheduler is not None:
        print(f"Using scheduler {scheduler.__class__.__name__}")
    else:
        print("Using scheduler: None")
    return scheduler
    """


def create_scheduler_and_attach_to_optimizer(
    optimizer: torch.optim.Optimizer,
    args: TrainArgs,
):
    warmup_epochs = None
    warmup_scheduler = None
    scheduler = None
    if args.epochs <= 1:
        # This is also enforced in training args parsing, but protecting here as well
        print(
            f"Using scheduler: None. Maximum training epochs: {args.epochs}, is less than or equal to 1."
        )
        return None

    min_lr = args.min_scheduler_lr if args.min_scheduler_lr is not None else 0

    if args.enable_lr_warmup:
        # Set warmup defaults
        warmup_epochs = min(5, max(1, args.epochs // 10))
        warmup_start_factor = 0.1
        warmup_end_factor = 1.0

        # Set warmup overrides from training args
        if args.warmup_epochs:
            warmup_epochs = args.warmup_epochs
        if args.warmup_factors is not None:
            warmup_start_factor = args.warmup_factors[0]
            warmup_end_factor = args.warmup_factors[1]

        warmup_scheduler = LinearLR(
            optimizer=optimizer,
            start_factor=warmup_start_factor,
            end_factor=warmup_end_factor,
            total_iters=warmup_epochs,
        )

    if args.scheduler_type == "ReduceLROnPlateau":
        # reducelr_patience is validated during arg parse, must be present with valid value
        reducelr_patience = args.reducelr_patience

        if args.reducelr_factor is not None:
            reducelr_factor = args.reducelr_factor
        else:
            reducelr_factor = 0.1
            print(
                f"ReduceLROnPlateau scheduler specified, but no reducelr_factor set, using default: {reducelr_factor}"
            )

        # min mode, as we will monitor and react to loss metric
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=reducelr_factor,
            patience=reducelr_patience,
            min_lr=min_lr,
        )
    elif args.scheduler_type == "CosineAnnealingLR":
        cosine_epochs = args.epochs - warmup_epochs if warmup_scheduler else args.epochs
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cosine_epochs,
            eta_min=min_lr,
        )

    # NOTE The current warmup design uses a SequentialLR to create a composite with the
    # warmup scheduler (LinearLR) and the chosen scheduler type. SequentialLR cannot currently
    # compose LinearLR with metric-driven schedulers such as ReduceLROnPlateau.
    # Whle the current command line argument parsing code explicitly prevents such a combination
    # note that this code currently does NOT explicitly prevent any such combination. This may
    # be added in the future if more schedulers become supported, but for now just be aware of this
    if warmup_scheduler:
        composite_scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[
                warmup_scheduler,
                scheduler,
            ],
            milestones=[warmup_epochs],
        )
        print(
            f"Using composite scheduler {composite_scheduler.__class__.__name__}: with schedulers: {[s.__class__.__name__ for s in composite_scheduler._schedulers]}"
        )
        return composite_scheduler
    else:
        if scheduler:
            print(f"Using scheduler {scheduler.__class__.__name__}")
        else:
            print("No scheduler specified, using scheduler: None")
        return scheduler


def get_unfrozen_backbone_blocks(args: TrainArgs):
    if args.unfreeze_bb_blocks_with_lr:
        return sum([block for block, _ in args.unfreeze_bb_blocks_with_lr])
    else:
        return 0


def get_dataloader_transforms(args: TrainArgs):

    # Get default EfficientNet_B0 pre_processing transform
    default_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    test_transform = default_weights.transforms()

    # If custom augmentation is enabled, but gpu augmentation is not
    # Use the custom cpu augmentation pipeline
    if args.augmentation_config and not args.enable_gpu_augmentation:
        # Get the specified augmentation pipeline configuration
        try:
            augmentation_pipeline = B0_AUGMENTATION_EXPERIMENTS[
                args.augmentation_config
            ].cpu
        except KeyError as e:
            raise ValueError(
                f"Augmentation experiment: {args.augmentation_config} does not exist. {str(e)}"
            )

        train_transform = augmentation_pipeline
        print(
            f"augmentation_config: {args.augmentation_config} "
            f"enable_gpu_augmentation: {args.enable_gpu_augmentation} "
            f"- Using custom CPU augmentation pipeline: {train_transform}"
        )
    # Else if custom augmentation is enabled, and GPU-based augmentaiton is enabled
    elif args.augmentation_config and args.enable_gpu_augmentation:
        if args.enable_ram_loaded_images:
            # When loading images from ram, any necessary transform will be done during that
            # process, no train transform needed
            train_transform = None
        else:
            # when not loading images from ram, we currently use a raw transform
            # just to convert the image to tensor, and potentially resize
            # so images are the same shape for tensor stacking purposes.
            train_transform = RAW_TRANSFORM
        print(
            f"augmentation_config: {args.augmentation_config} "
            f"enable_gpu_agumentation: {args.enable_gpu_augmentation} "
            f"enable_ram_loaded_images: {args.enable_ram_loaded_images} "
            f"- Using default RAW_TRANFORM: {train_transform}"
        )
    else:
        # No custom augmentation is enabled, use standard EfficientNet B0 transform
        # [TODO]: this is coupling to efficientnet B0- update this to be more flexible
        train_transform = test_transform
        print(
            f"augmentation_config: {args.augmentation_config} "
            f"enable_gpu_agumentation: {args.enable_gpu_augmentation} "
            f"- Using default EfficientNetB0 Transform: {train_transform}"
        )

    return train_transform, test_transform


def get_loss_fn(label_smoothing: float = 0.0):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    print(
        f"Using loss_fn {loss_fn.__class__.__name__} with label smoothing: {label_smoothing}"
    )
    return loss_fn


def get_optimizer(model: nn.Module, args: TrainArgs):
    # Create optimizer param groups with discriminated initial learning rates and weight
    # decay. Start wtih the classifier which should always be present
    param_groups = [
        {
            "params": [
                param for param in model.classifier.parameters() if param.requires_grad
            ],
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
    ]

    if args.unfreeze_bb_blocks_with_lr:
        # A series of unfrozen backbone blocks with corresponding learning rates has been set,
        # create the optimizer params accordingly.
        # Note that total number of blocks to unfreeze has alredy been validated, so we don't need
        # to verify that again before proceeding
        param_groups.extend(
            extract_backbone_param_groups(
                model=model,
                backbone_lr_stages=args.unfreeze_bb_blocks_with_lr,
                backbone_wd_stages=args.bb_block_wd,
            )
        )
    # Top level lr here is a fallback, but should be overriden by values in
    # param groups
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    print(f"Using optimizer {optimizer.__class__.__name__} with param groups:")
    for i, group in enumerate(optimizer.param_groups):
        num_tensors = len(group["params"])
        print(
            f"Group {i} | LR: {group['lr']:.5f} | WD: {group['weight_decay']} | Tensors: {num_tensors}"
        )

    return optimizer


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


def get_gpu_transform(args: TrainArgs):
    # args should already be validated to not have invalid combinations
    # but doing some protection here
    if (
        args.augmentation_config
        and args.enable_gpu_augmentation
        and not args.enable_backbone_caching
    ):
        try:
            augmentation_pipeline = B0_AUGMENTATION_EXPERIMENTS[
                args.augmentation_config
            ].gpu
        except KeyError as e:
            raise ValueError(
                f"Augmentation experiment: {args.augmentation_config} does not exist. {str(e)}"
            )

        print(
            f"augmentation_config: {args.augmentation_config} "
            f"enable_gpu_augmentation: {args.enable_gpu_augmentation} "
            f"enable_backbone_caching: {args.enable_backbone_caching} "
            f"- setting gpu_transform : {augmentation_pipeline}"
        )
        return augmentation_pipeline
    else:
        return None


def save_training_artifacts(
    model: nn.Module, results: dict[str, Any], class_list: list[str], args: TrainArgs
):
    # Save model if requested
    if args.save is not None:
        model_architecture_name = model.__class__.__name__

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


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_tensorboard_writer(args: TrainArgs):
    timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
    run_name = f"efnet_b0_ep{args.epochs}_lr{args.lr}_esp{args.early_stop_patience}_lrp{args.lr_schedule_patience}_wd{args.weight_decay}_ts_{timestamp}"
    return SummaryWriter(log_dir=f"runs/efficientnet_b0/{run_name}")


def write_tensorboard_hyperparams(
    writer: SummaryWriter, results: dict[str, Any], args: TrainArgs
):
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


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    run_training()
