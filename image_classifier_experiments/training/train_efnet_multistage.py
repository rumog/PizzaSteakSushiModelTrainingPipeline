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
from torchvision import transforms

import wandb
from image_classifier_experiments.augmentation.augmentation_experiments import (
    AUGMENTATION_EXPERIMENTS,
    B0_AUGMENTATION_EXPERIMENTS,
)
from image_classifier_experiments.data_setup import data_setup
from image_classifier_experiments.model_build.artifact_loader.file_model_reader import (
    ModelArtifactFileReader,
)
from image_classifier_experiments.model_build.artifact_loader.s3_model_reader import (
    ModelArtifactS3Reader,
)
from image_classifier_experiments.model_build.efficientnet_b0_transfer import (
    EfficientNetB0TransferLearningModel,
)
from image_classifier_experiments.model_build.efficientnet_transfer import (
    EfficientNetTransferLearningModel,
)
from image_classifier_experiments.model_build.model_factory import BackboneFactory
from image_classifier_experiments.model_build.types.model_artifact import (
    ModelArtifactData,
)
from image_classifier_experiments.model_build.types.model_artifact_schema import (
    ModelMetadataSchema,
)
from image_classifier_experiments.training import engine
from image_classifier_experiments.training.checkpoint.reader.file_train_checkpoint_reader import (
    TrainingCheckpointFileReader,
)
from image_classifier_experiments.training.checkpoint.types.training_checkpoint import (
    TrainCheckpoint,
)
from image_classifier_experiments.training.cli.arg_parser import parse_train_args
from image_classifier_experiments.training.config.train_config import (
    TrainConfig,
)
from image_classifier_experiments.utils.helper_functions import (
    accuracy_fn,
    matches_state_dict,
    plot_loss_curves,
    print_checkpoint_values,
    print_train_time,
    save_model_checkpoint,
    save_model_checkpoint_s3,
)

# S3_MODEL_BUCKET = "pss-classifier-models-613693331461-us-west-2-an"
# MODEL_KEY_PREFIX = "pss-classifier/0.1.0"


# This is needed currently for the GPU-based augmentation scenario
# when loading images to ram is disabled. Currently in this scneario
# Augmentation is applied directly during the training loop and not
# injected into the dataloader. In this case this minimal cpu-based
# transform is injected into the dataloader to resize the images
# to the same time so they can be stacked/opearated on at batch level
# and converted to tensor.
#
# The ram-loaded image case applies augmentation at the image-level
# so the image tensors don't have to have the same shape.
#
# torch ImageFolder
RAW_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.PILToTensor(),
    ]
)
# training and model hyperparams
DATA_PATH_PARENT_DIR = "data/"
TRAINING_RUNS_DIR = "runs/"
IMAGE_PATH_PARENT_DIR = "seattlement_birds_50_100_percent"
MODEL_SAVE_DIR = "model"
IMAGE_SIZE = 224
CACHED_FEATURES_TRAIN_PATH = "model/cached_features/train.pt"
CACHED_FEATURES_TEST_PATH = "model/cached_features/test.pt"
DEFAULT_MODEL_ARCHITECTURE = "efficientnet_b0"
DEFAULT_WEIGHTS = torchvision.models.EfficientNet_B0_Weights.DEFAULT

torch.manual_seed(42)
torch.mps.manual_seed(42)
torch.cuda.manual_seed(42)


def run_training():

    # If user provides --train_config argument, they must not specify any
    # other test config in cli arguments, except --resume_training_checkpoint
    # fail_on_duplicate_training_config()

    # Handle commandline arg parsing
    args = parse_train_args()

    train_config = create_train_config(args)

    # If no run_name specified, generate one and inject to config
    if not train_config.run_name:
        train_config.run_name = generate_run_name()

    if train_config.enable_wandb:
        if not train_config.resume_training_checkpoint:
            # This is a new run, init wandb and log run_id to train_config
            run_id = init_wandb(train_config)
            train_config.wandb_run_id = run_id
        else:
            # Resuming existing run. Config validation ensures a
            # project and run id are present in this case
            resume_wandb_run(train_config.wandb_project, train_config.wandb_run_id)

    # [TODO] "Validate" checkpoint against training config
    # For example, if the model has 2 stages of unfrozen backbone layers, but
    # the optimizer has only 2 param groups, that means the config doesn't match
    # (it should have 3, 1 for classifier and 1 for each unfrozen 'stage' of backbone layers)

    # Create run directory. Used to save training config
    # and training checkpoints for the run.
    run_name = train_config.run_name or generate_run_name()
    run_dir = create_run_directory(run_name)
    save_training_config(train_config, run_dir)

    print(
        f"Starting training run: {train_config.run_name} with Training Config:\n{train_config}\n"
    )

    resume_checkpoint = None
    # Load training checkpoint for resume if specified
    if train_config.resume_training_checkpoint:
        resume_checkpoint = TrainingCheckpointFileReader.load_train_checkpoint(
            train_config.resume_training_checkpoint
        )
        # print_checkpoint_values(resume_checkpoint)
    else:
        print("No resume checkpoint specified, starting new training session")

    device = get_device()

    # construct backbone and use weights to construct dataloader transforms
    backbone_architecture = (
        train_config.model_architecture or DEFAULT_MODEL_ARCHITECTURE
    )
    backbone, origin_weights = BackboneFactory.build_backbone(backbone_architecture)

    # Create train and test transforms used in image dataloader creation
    train_transform, test_transform = get_dataloader_transforms(
        train_config, origin_weights
    )

    # Create image dataloaders
    train_image_dataloader, test_image_dataloader, class_list = get_image_dataloaders(
        args=train_config,
        train_transform=train_transform,
        test_transform=test_transform,
    )

    # If artifact loading is specified in training args, load artifact
    # else returns None
    model_artifact = None

    # Validation should prevent both checkpoint resume and load_artifact from
    # both being present, but in an unexpected case where both are present
    # resume_checpoint takes precedence.
    if resume_checkpoint is not None:
        print("Creating model artifact from resume checkpoint")
        model_artifact = ModelArtifactData(
            resume_checkpoint.model_state_dict, model_metadata={}
        )
    elif train_config.load_artifact_from is not None:
        model_artifact = load_model_artifact(train_config)

    # Create the model
    """
    model_0 = EfficientNetB0TransferLearningModel(
        num_classes=len(class_list),
        from_artifact=model_artifact,
        unfrozen_backbone_blocks=get_unfrozen_backbone_blocks(train_config),
        dropout_override=train_config.classifier_dropout,
    ).to(device)
    """

    model_0 = EfficientNetTransferLearningModel(
        backbone_arch=backbone_architecture,
        backbone=backbone,
        origin_weights=origin_weights,
        num_classes=len(class_list),
        from_artifact=model_artifact,
        unfrozen_backbone_blocks=get_unfrozen_backbone_blocks(train_config),
        dropout_override=train_config.classifier_dropout,
    ).to(device)

    if train_config.enable_backbone_caching and not train_config.augmentation_config:
        # If feature caching enabled, cache backbone using the image dataloaders
        # and create feature-based dataloaders for use in train/test
        train_dataloader, test_dataloader = (
            cache_backbone_and_create_feature_dataloaders(
                model_0.backbone,
                train_config,
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
    loss_fn = get_loss_fn(train_config.label_smoothing)

    optimizer_state_dict = None
    if resume_checkpoint is not None:
        optimizer_state_dict = resume_checkpoint.optimizer_state_dict

    optimizer = get_optimizer(
        model=model_0,
        args=train_config,
        state_dict=optimizer_state_dict,
    )

    scheduler_state_dict = None
    if resume_checkpoint is not None:
        scheduler_state_dict = resume_checkpoint.scheduler_state_dict

    scheduler = create_scheduler_and_attach_to_optimizer(
        optimizer=optimizer,
        args=train_config,
        state_dict=scheduler_state_dict,
    )

    # verification_checkpoint = TrainCheckpoint(
    #     model_state_dict=model_0.state_dict(),
    #    optimizer_state_dict=optimizer.state_dict(),
    #    scheduler_state_dict=scheduler.state_dict(),
    #    metadata=None,
    # )
    # print("=== DEBUG:VERIFY LOADED STATE DICTS ===")
    # print_checkpoint_values(verification_checkpoint)

    # Debug check to make sure the state_dicts in the resume_checkpoint
    # match those in the constructed model, otpimizier, and scheduler
    if resume_checkpoint is not None:
        verify_resume_checkpoint_state_dicts_loaded_successfully(
            model=model_0,
            optimizer=optimizer,
            scheduler=scheduler,
            resume_checkpoint=resume_checkpoint,
        )

    # Get GPU-based transform if required, otherwise will be None
    gpu_transform = get_gpu_transform(train_config, origin_weights)

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
            epochs=train_config.epochs,
            device=device,
            patience=train_config.early_stop_patience,
            caching_enabled=train_config.enable_backbone_caching,
            gpu_transform=gpu_transform,
            test_transform=test_transform,
            ram_caching_enabled=train_config.enable_ram_loaded_images,
            resume_checkpoint=resume_checkpoint,
            run_dir=run_dir,
        )
        end_time = timer()
        train_time = print_train_time(start_time, end_time, device)
        write_wandb_summary(train_time, results, train_config)

        print(f"train_model output: {redact_dict(results)}")

        # Save training artifaacts if requested
        save_training_artifacts(
            model=model_0, results=results, class_list=class_list, args=train_config
        )
    finally:
        if wandb.run is not None:
            wandb.finish()

    # plot_loss_curves(results["history"])


def get_image_dataloaders(
    args: TrainConfig,
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
    args: TrainConfig,
    train_image_dataloader,
    test_image_dataloader,
    device,
):
    train_path = Path(CACHED_FEATURES_TRAIN_PATH)
    test_path = Path(CACHED_FEATURES_TEST_PATH)
    if not (
        # Unless the user explicitly chose to bypass cache generation
        # and both cached feature files are present, re-generate the cache.
        args.bypass_cache_generation and train_path.is_file() and test_path.is_file()
    ):
        # Run a single forward pass using train/test image dataloaders, and cache
        # features to train/test files
        cache_bb_features_start = timer()
        print("Caching backbone features for training image set")
        train_path.parent.mkdir(parents=True, exist_ok=True)
        engine.extract_backbone_features(
            backbone,
            train_image_dataloader,
            device,
            train_path,
        )

        print("Caching backbone features for test image set")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        engine.extract_backbone_features(
            backbone,
            test_image_dataloader,
            device,
            test_path,
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


def create_scheduler_and_attach_to_optimizer(
    optimizer: torch.optim.Optimizer,
    args: TrainConfig,
    state_dict: dict[str, Any] | None = None,
):
    warmup_epochs = None
    warmup_scheduler = None
    main_scheduler = None
    output_scheduler = None
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
        main_scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=reducelr_factor,
            patience=reducelr_patience,
            min_lr=min_lr,
        )
    elif args.scheduler_type == "CosineAnnealingLR":
        cosine_epochs = args.epochs - warmup_epochs if warmup_scheduler else args.epochs
        main_scheduler = CosineAnnealingLR(
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
                main_scheduler,
            ],
            milestones=[warmup_epochs],
        )
        print(
            f"Using composite scheduler {composite_scheduler.__class__.__name__}: with schedulers: {[s.__class__.__name__ for s in composite_scheduler._schedulers]}"
        )
        output_scheduler = composite_scheduler
    else:
        if main_scheduler:
            print(f"Using scheduler {main_scheduler.__class__.__name__}")
            output_scheduler = main_scheduler
        else:
            print("No scheduler specified, using scheduler: None")
            return None

    if state_dict is not None:
        print("Scheduler state dict was provided, loading state_dict to scheduler")
        output_scheduler.load_state_dict(state_dict)
    return output_scheduler


def get_unfrozen_backbone_blocks(args: TrainConfig):
    if args.unfreeze_bb_blocks_with_lr:
        return sum([block for block, _ in args.unfreeze_bb_blocks_with_lr])
    else:
        return 0


def get_dataloader_transforms(train_config: TrainConfig, default_weights):

    # Get default pre_processing transform
    weights = default_weights or DEFAULT_WEIGHTS
    test_transform = weights.transforms()

    print(type(weights))
    crop_size = test_transform.crop_size
    image_size = crop_size[0] if isinstance(crop_size, (list, tuple)) else crop_size

    if train_config.image_size_override:
        print(
            f"Using custom base transform due to image_size_override: {train_config.image_size_override}"
        )
        image_size = train_config.image_size_override
        resize_size = round(image_size * 8 / 7)
        test_transform = transforms.Compose(
            [
                transforms.Resize(
                    resize_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=weights.transforms().mean,
                    std=weights.transforms().std,
                ),
            ]
        )
    print(f"Using test dataloader transform: {test_transform}")

    # If custom augmentation is enabled, but gpu augmentation is not
    # Use the custom cpu augmentation pipeline
    if train_config.augmentation_config and not train_config.enable_gpu_augmentation:
        # Get the specified augmentation pipeline configuration
        try:
            augmentation_pipeline = AUGMENTATION_EXPERIMENTS[
                train_config.augmentation_config
            ](weights, image_size).cpu
        except KeyError as e:
            raise ValueError(
                f"Augmentation experiment: {train_config.augmentation_config} does not exist. {str(e)}"
            )

        train_transform = augmentation_pipeline
        print(
            f"augmentation_config: {train_config.augmentation_config} "
            f"enable_gpu_augmentation: {train_config.enable_gpu_augmentation} "
            f"- Using custom CPU augmentation pipeline: {train_transform}"
        )
    # Else if custom augmentation is enabled, and GPU-based augmentaiton is enabled
    elif train_config.augmentation_config and train_config.enable_gpu_augmentation:
        if train_config.enable_ram_loaded_images:
            # When loading images from ram, any necessary transform will be done during that
            # process, no train transform needed
            train_transform = None
        else:
            # when not loading images from ram, we currently use a raw transform
            # just to convert the image to tensor, and potentially resize
            # so images are the same shape for tensor stacking purposes.
            train_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.PILToTensor(),
                ]
            )
        print(
            f"augmentation_config: {train_config.augmentation_config} "
            f"enable_gpu_agumentation: {train_config.enable_gpu_augmentation} "
            f"enable_ram_loaded_images: {train_config.enable_ram_loaded_images} "
            f"- Using raw transform: {train_transform}"
        )
    else:
        # No custom augmentation is enabled, use standard EfficientNet B0 transform
        # [TODO]: this is coupling to efficientnet B0- update this to be more flexible
        train_transform = test_transform
        print(
            f"augmentation_config: {train_config.augmentation_config} "
            f"enable_gpu_agumentation: {train_config.enable_gpu_augmentation} "
            f"- No custom augmentation enabled, train dataloader will use same as test: {train_transform}"
        )

    return train_transform, test_transform


def get_loss_fn(label_smoothing: float = 0.0):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    print(
        f"Using loss_fn {loss_fn.__class__.__name__} with label smoothing: {label_smoothing}"
    )
    return loss_fn


def get_optimizer(
    model: nn.Module,
    args: TrainConfig,
    state_dict: dict[str, Any] | None = None,
):
    # Create optimizer param groups with discriminated initial learning rates and weight
    # decay. Start wtih the classifier which should always be present
    param_groups = [
        {
            "params": [
                param for param in model.classifier.parameters() if param.requires_grad
            ],
            "lr": args.classifier_lr,
            "weight_decay": args.classifier_wd,
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
    optimizer = torch.optim.AdamW(param_groups, lr=args.classifier_lr)

    if state_dict is not None:
        print("Optimizer state dict was provided, loading state_dict to optimizer")
        optimizer.load_state_dict(state_dict)

    # Log initial param groups for debugging
    print(f"Using optimizer {optimizer.__class__.__name__} with param groups:")
    for i, group in enumerate(optimizer.param_groups):
        num_tensors = len(group["params"])
        print(
            f"Group {i} | LR: {group['lr']:.7g} | WD: {group['weight_decay']} | Tensors: {num_tensors}"
        )

    return optimizer


def save_training_config(config: TrainConfig, save_dir: str | Path):
    save_file = "train_config.yaml"
    config_save_path = Path(save_dir) / save_file

    print(f"Saving test config to project run folder at: {config_save_path}")
    config_string_yaml = config.to_yaml(config_save_path)
    # Add exception handling here
    config_save_path.write_text(config_string_yaml, encoding="utf-8")


def generate_run_name():
    timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}"


def create_run_directory(run_name: str) -> Path:
    run_dir_path = Path(TRAINING_RUNS_DIR) / run_name

    run_dir_path.mkdir(parents=True, exist_ok=True)
    return run_dir_path


def create_train_config(args) -> TrainConfig:
    if args.train_config is not None:
        # Load config from user-specified file
        print(f"Loading training config from: {args.train_config}")
        config_path = Path(args.train_config)

        if not config_path.is_file():
            raise ValueError(
                f"Training config file at {config_path} does not exist. "
                "Please check file exists and is specified correctly"
            )

        # Add exception handling
        config_string_yaml = config_path.read_text(encoding="utf-8")
        config = TrainConfig.from_yaml(config_string_yaml)
        # if resume_train_checkpoint override set, inject it into config
        if args.resume_training_checkpoint:
            print(
                f"Injecting resume_training_checkpoint override '{args.resume_training_checkpoint}' into test config."
            )
            config.resume_training_checkpoint = args.resume_training_checkpoint

        # Ensure if wandb_run_id is present, a training checkpoint to resume is also specified.
        # Because the wandb run id is saved to training config on the first run (e.g. when
        # no resume_training_checkpoint is provided or saved in the convig), and because
        # --resume_training_checkpoint can be overridden from command line, the validation
        # is deferred until after the TrainConfig is constructed.
        if config.enable_wandb and (
            bool(config.wandb_run_id) != bool(config.resume_training_checkpoint)
        ):
            raise ValueError(
                "When enabling wandb logging you must either 1. Be starting a clean run (no wandb_run_id or resume_training_checkpoint "
                "specified) or 2. Be resuming an existing checkpoint (resume_training_checkpoint) which requires supplying the wandb_run_id"
            )

        return config
    else:
        print("Creating training config from cli args.")
        # even though --train_config is None, it still exists as an argument.
        # remove this as it isn't part of the training_config contract
        args_dict = vars(args)
        args_dict.pop("train_config")
        return TrainConfig(**args_dict)


def redact_dict(d):
    # Base case: if it's not a dictionary, return it as is
    if not isinstance(d, dict):
        return d

    new_dict = {}
    for key, value in d.items():
        if "state_dict" in key:
            new_dict[key] = "<redacted>"
        elif isinstance(value, dict):
            new_dict[key] = redact_dict(value)  # Recurse into nested dicts
        elif isinstance(value, list):
            # Recurse into lists in case dicts are hidden inside them
            new_dict[key] = [redact_dict(item) for item in value]
        else:
            new_dict[key] = value

    return new_dict


def get_gpu_transform(args: TrainConfig, default_weights):
    weights = default_weights or DEFAULT_WEIGHTS
    crop_size = weights.transforms().crop_size
    image_size = crop_size[0] if isinstance(crop_size, (list, tuple)) else crop_size

    # apply override if present
    image_size = args.image_size_override or image_size

    # args should already be validated to not have invalid combinations
    # but doing some protection here
    if (
        args.augmentation_config
        and args.enable_gpu_augmentation
        and not args.enable_backbone_caching
    ):
        try:
            augmentation_pipeline = AUGMENTATION_EXPERIMENTS[args.augmentation_config](
                default_weights, image_size
            ).gpu
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
    model: nn.Module, results: dict[str, Any], class_list: list[str], args: TrainConfig
):
    # Save model if requested
    if args.save is not None:
        model_name = model.__class__.__name__
        model_architecture = {
            "backbone": model.backbone_arch,
            "name": model_name,
            "weights": "DEFAULT",
        }

        # default image size based on origin_weights if exists
        # or the hard-coded default (should match the default backbone arch)
        weights = model.origin_weights or DEFAULT_WEIGHTS
        crop_size = weights.transforms().crop_size
        image_dim = crop_size[0] if isinstance(crop_size, (list, tuple)) else crop_size

        # update if override is set
        image_dim = args.image_size_override or image_dim
        image_size = (image_dim, image_dim)

        model_preprocessing = {"image_size": image_size}

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
        filename = f"{model_name}_ep{results['best_checkpoint']['epoch']}_arch_{model.backbone_arch}_{timestamp}_v{version}.pth"

        # validate model metadata matches expected schema and
        # save dump from schema object to ensure consistent save
        print("Validating model artifact schema...")
        model_metadata_schema = ModelMetadataSchema.model_validate(model_metadata)
        validated_model_metadata = model_metadata_schema.model_dump()

        if args.save == "s3":
            save_model_checkpoint_s3(
                state_dict=results["best_checkpoint"]["state_dict"],
                model_metadata=validated_model_metadata,
                bucket_name=args.s3_bucket,
                object_key=f"{args.s3_key_prefix}/{filename}",
            )
        elif args.save == "file":
            save_model_checkpoint(
                state_dict=results["best_checkpoint"]["state_dict"],
                model_metadata=validated_model_metadata,
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


def write_wandb_summary(train_time, results: dict[str, Any], args: TrainConfig):
    if wandb.run is not None:
        wandb.summary["train_time_seconds"] = train_time
        wandb.summary["epochs_completed"] = results["train_metadata"][
            "epochs_completed"
        ]
        wandb.summary["early_stopped"] = results["train_metadata"]["stopped_early"]
        wandb.summary["best_epoch"] = results["best_checkpoint"]["epoch"]
        wandb.summary["best_test_accuracy"] = results["best_checkpoint"]["test_acc"]
        wandb.summary["best_test_loss"] = results["best_checkpoint"]["test_loss"]


def load_model_artifact(args: TrainConfig) -> ModelArtifactData:
    model_artifact = None
    if args.load_artifact_from == "s3":
        model_loader = ModelArtifactS3Reader()
        model_artifact = model_loader.load_model_artifact(
            args.artifact_s3_bucket,
            args.artifact_s3_key,
        )
    elif args.load_artifact_from == "file":
        model_loader = ModelArtifactFileReader()
        model_artifact = model_loader.load_model_artifact(args.artifact_file_path)

    return model_artifact


def verify_resume_checkpoint_state_dicts_loaded_successfully(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    resume_checkpoint: TrainCheckpoint,
):
    print(
        "Verifying resume_checkpoint state_dicts match loaded model, optimizer, and scheduler:"
    )

    print(
        f"Model state_dict matches checkpoint: "
        f"{matches_state_dict(model.state_dict(), resume_checkpoint.model_state_dict)}"
    )
    print(
        f"Optimizer state_dict matches checkpoint: "
        f"{matches_state_dict(optimizer.state_dict(), resume_checkpoint.optimizer_state_dict)}"
    )
    print(
        f"Scheduler state_dict matches checkpoint: "
        f"{matches_state_dict(scheduler.state_dict(), resume_checkpoint.scheduler_state_dict)}"
    )


def init_wandb(train_config: TrainConfig):
    run_name = train_config.run_name
    run = wandb.init(
        project=train_config.wandb_project,
        name=run_name,
        config=vars(train_config),
    )
    # according to documentation while the id is already implicity saved with run metadata
    # this makes runs more easily search and comparable by id
    wandb.config.update({"run_id": run.id})
    print(f"Successfully initialized new wandb run:{run_name} with id: {run.id}")
    return run.id


def resume_wandb_run(project: str, run_id: str):
    wandb.init(
        project=project,
        id=run_id,
        resume="must",
    )
    print(f"Successfully resumed wandb run for run_id: {run_id}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    run_training()
