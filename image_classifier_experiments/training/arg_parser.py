import argparse
from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainArgs:
    load_model_from_artifact: str | None = None
    augmentation_config: str | None = None
    enable_gpu_augmentation: bool = False
    enable_ram_loaded_images: bool = False
    enable_backbone_caching: bool = False
    bypass_cache_generation: bool = False
    epochs: int = 5
    early_stop_patience: int | None = None
    classifier_lr: float = 0.001
    classifier_wd: float = 0.0
    classifier_dropout: float | None = None

    # This argument expects a space delimited list of int:float pairs, e.g. 2:1e-3
    # or 2:0.001.
    # The fist number of the pair indicates how many blocks of the model
    # backbone to unfreeze.  The second number indicates the initial learning rate
    # set for those blocks in optimiziation.
    #
    # Blocks are unfrozen starting at the end of the backbone, starting with index 0.
    # So, for example the argument value "2:1e-3 3:1e-4" will result in the last 5 backbone blocks
    # being unfrozen. The final 2 blocks of the backbone will start with lr: 1e-3, and the next
    # 3 will start with lr 1e-4
    unfreeze_bb_blocks_with_lr: list[tuple[int, float]] | None = None
    bb_block_wd: list[float] | None = None
    batch_size: int = 32
    num_workers: int = 0
    # Experiment with adding more scheduler config
    scheduler_type: Literal["ReduceLROnPlateau", "CosineAnnealingLR"] | None = None
    # only compatible with ReduceLROnPlateau, ensure this is validated
    reducelr_patience: int | None = None
    reducelr_factor: float | None = None
    # can apply to either main scheduler
    min_scheduler_lr: float | None = None
    # scheduler warmup
    enable_lr_warmup: bool = False
    warmup_epochs: int | None = None
    warmup_factors: tuple[float, float] | None = None
    label_smoothing: float = 0.0
    save: Literal["file", "s3"] | None = None
    s3_bucket: str | None = None
    s3_key_prefix: str | None = None
    enable_wandb: bool = False
    wandb_run_name: str | None = None


# Currently the file and s3 saving locations are hard coded
# may update this in the future
def parse_train_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Training engine for image classifier")

    parser.add_argument(
        "--load_model_from_artifact",
        type=str,
    )

    parser.add_argument(
        "--augmentation_config",
        type=str,
    )

    parser.add_argument(
        "--enable_gpu_augmentation",
        action="store_true",
    )

    parser.add_argument(
        "--enable_ram_loaded_images",
        action="store_true",
    )

    parser.add_argument(
        "--enable_backbone_caching",
        action="store_true",
    )

    parser.add_argument(
        "--bypass_cache_generation",
        action="store_true",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--early_stop_patience",
        type=int,
    )

    parser.add_argument(
        "--classifier_lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--classifier_wd",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--classifier_dropout",
        type=float,
    )

    parser.add_argument(
        "--unfreeze_bb_blocks_with_lr",
        type=int_float_pair,
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--bb_block_wd",
        type=float,
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--scheduler_type",
        choices=["ReduceLROnPlateau", "CosineAnnealingLR"],
    )

    parser.add_argument(
        "--reducelr_patience",
        type=int,
    )

    parser.add_argument(
        "--reducelr_factor",
        type=float,
    )

    parser.add_argument(
        "--min_scheduler_lr",
        type=float,
    )

    parser.add_argument(
        "--enable_lr_warmup",
        action="store_true",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
    )

    parser.add_argument(
        "--warmup_factors",
        type=float_pair,
    )

    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--save",
        choices=["file", "s3"],
    )

    parser.add_argument(
        "--s3_bucket",
        type=str,
    )

    parser.add_argument(
        "--s3_key_prefix",
        type=str,
    )

    parser.add_argument(
        "--enable_wandb",
        action="store_true",
    )

    parser.add_argument(
        "--wandb_run_name",
        type=str,
    )

    try:
        args = parser.parse_args()
    except ValueError as e:
        parser.error(str(e))

    validate_args(args)
    return TrainArgs(
        load_model_from_artifact=args.load_model_from_artifact,
        augmentation_config=args.augmentation_config,
        enable_gpu_augmentation=args.enable_gpu_augmentation,
        enable_ram_loaded_images=args.enable_ram_loaded_images,
        enable_backbone_caching=args.enable_backbone_caching,
        bypass_cache_generation=args.bypass_cache_generation,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        classifier_lr=args.classifier_lr,
        classifier_wd=args.classifier_wd,
        classifier_dropout=args.classifier_dropout,
        unfreeze_bb_blocks_with_lr=args.unfreeze_bb_blocks_with_lr,
        bb_block_wd=args.bb_block_wd,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scheduler_type=args.scheduler_type,
        reducelr_patience=args.reducelr_patience,
        reducelr_factor=args.reducelr_factor,
        min_scheduler_lr=args.min_scheduler_lr,
        enable_lr_warmup=args.enable_lr_warmup,
        warmup_epochs=args.warmup_epochs,
        warmup_factors=args.warmup_factors,
        label_smoothing=args.label_smoothing,
        save=args.save,
        s3_bucket=args.s3_bucket,
        s3_key_prefix=args.s3_key_prefix,
        enable_wandb=args.enable_wandb,
        wandb_run_name=args.wandb_run_name,
    )


def int_float_pair(arg_value: str) -> tuple[int, float]:
    try:
        parsed_int, parsed_float = arg_value.split(":")
        return int(parsed_int), float(parsed_float)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid int:float entry pair: {arg_value}.  Must be of format 'int:float'. {str(e)}"
        )


def float_pair(arg_value: str) -> tuple[float, float]:
    try:
        parsed_float1, parsed_float2 = arg_value.split(":")
        return float(parsed_float1), float(parsed_float2)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid float:float entry pair: {arg_value}.  Must be of format 'float:float'. {str(e)}"
        )


def validate_args(args):

    # If  backbone caching is enabled, you cannot run custom data augmentation or set backbone as trainable
    if args.enable_backbone_caching and (
        args.augmentation_config
        or args.enable_gpu_augmentation
        or args.unfreeze_bb_blocks_with_lr
    ):
        raise ValueError(
            "Invalid Training Args: if enable_backbone_caching enabled, no augmentation or backbaone unfreezing args can be set"
        )
    if args.bypass_cache_generation and not args.enable_backbone_caching:
        raise ValueError(
            "Invalid Training Args: bypass_cache_generation cannot be specified without enable_backbone_caching."
        )

    # validate/normalize args related to staged backbone unfreeze and associated lr/wd per stage
    args.bb_block_wd = validate_and_normalize_bb_wd(
        args.bb_block_wd, args.unfreeze_bb_blocks_with_lr
    )

    # Validate scheduler args
    validate_scueduler_config(args)

    # Validate save for S3 path
    if args.save == "s3" and (not args.s3_bucket or not args.s3_key_prefix):
        raise ValueError(
            "Invalid Training Args: If 'save' is S3, both s3_bucket and s3_key_prefix Must be set"
        )

    if args.wandb_run_name and not args.enable_wandb:
        raise ValueError("wandb_run_name cannot be set without enable_wandb.")


def validate_and_normalize_bb_wd(
    bb_block_wd: list[float], unfreeze_bb_blocks_with_lr: list[tuple[int, float]]
):
    if unfreeze_bb_blocks_with_lr is None:
        if bb_block_wd is not None:
            raise ValueError(
                "Cannot specify backbone weight decay stages (--bb_block_wd) if no unfreeze block configuration is specified "
                "(--unfreeze_bb_blocks_with_lr)"
            )
        return None

    # Unfreeze block stages have been specified, fallback to default weight decay of 0.0
    # for any stages where backbone weight decay is unspecified
    num_unfrozen_bb_stages = len(unfreeze_bb_blocks_with_lr)
    if bb_block_wd is None:
        return [0.0] * num_unfrozen_bb_stages

    if len(bb_block_wd) < num_unfrozen_bb_stages:
        pad_len = num_unfrozen_bb_stages - len(bb_block_wd)
        bb_block_wd.extend([0.0] * pad_len)

    elif len(bb_block_wd) > num_unfrozen_bb_stages:
        raise ValueError(
            "Cannot specify more backbone weight decay values (--bb_block_wd) than unfrozen backbone stages "
            "(--unfreeze_bb_blocks_with_lr)"
        )

    return bb_block_wd


def validate_scueduler_config(args):
    if not args.scheduler_type and (
        args.reducelr_patience is not None
        or args.reducelr_factor is not None
        or args.min_scheduler_lr is not None
        or args.enable_lr_warmup
        or args.warmup_epochs is not None
        or args.warmup_factors is not None
    ):
        raise ValueError(
            "No scheduler_type set: cannot specify lr scheduler-related arguments if no scheduler type set."
        )

    if args.scheduler_type and args.epochs <= 1:
        raise ValueError(
            "Cannot enable scheduler with scheduler_type when training for less than 2 epochs, as scheduler would not be engaged"
        )

    if args.scheduler_type == "ReduceLROnPlateau" and args.reducelr_patience is None:
        raise ValueError(
            "Scheduler type ReduceLROnPlateau requires setting patience with reducelr_patience"
        )

    if args.scheduler_type == "ReduceLROnPlateau" and args.enable_lr_warmup:
        raise ValueError(
            "Scheduler type ReduceLROnPlateau is not compatible with current warmup mechanism (composite scheduler with LinearLR)"
        )

    if args.scheduler_type == "CosineAnnealingLR" and (
        args.reducelr_patience is not None or args.reducelr_factor is not None
    ):
        raise ValueError(
            "Scheduler type CosineAnnealingLR not compatible with reducelr_patience or reducelr_factor and must not specify these arguments."
        )

    if args.reducelr_factor is not None and not (0.0 < args.reducelr_factor < 1.0):
        raise ValueError("reducelr_factor must be greater than 0 and less than 1")

    if not args.enable_lr_warmup and (
        args.warmup_epochs is not None or args.warmup_factors is not None
    ):
        raise ValueError(
            "enable_lr_warmup must be set to use warmup_epochs, or warmup_factors. Enable warmup, or reomve warmup related arguments."
        )
    if args.min_scheduler_lr is not None and args.min_scheduler_lr < 0:
        raise ValueError("min_scheduler_lr must be >= 0.")

    if args.warmup_epochs is not None and args.warmup_epochs < 1:
        raise ValueError("warmup_epochs must be >= 1.")

    if args.warmup_epochs is not None and args.warmup_epochs >= args.epochs:
        raise ValueError(
            f"warmup_epochs: {args.warmup_epochs} must be less than epochs: {args.epochs}"
        )
    if args.warmup_factors is not None:
        start_factor, end_factor = args.warmup_factors
        if start_factor <= 0 or end_factor <= 0:
            raise ValueError(
                f"warmup_factor values {start_factor}:{end_factor} must both be greater than 0"
            )
        if start_factor > end_factor:
            raise ValueError(
                f"warmup_factor start value: {start_factor} must be less than or equal to end value: {end_factor}"
            )
