import argparse
from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainArgs:
    num_workers: int = 0
    # enable_custom_augmentation: bool = False
    enable_gpu_augmentation: bool = False
    enable_ram_loaded_images: bool = False
    enable_backbone_caching: bool = False
    epochs: int = 5
    lr: float = 0.001
    load_model_from_artifact: str | None = None
    unfreeze_backbone_blocks: int = 0
    backbone_ft_lr: float | None = None
    batch_size: int = 32
    early_stop_patience: int | None = None
    lr_schedule_patience: int | None = None
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    save: Literal["file", "s3"] | None = None
    s3_bucket: str | None = None
    s3_key_prefix: str | None = None

    # Expermenting with switching to multi-value flags for multi-stage
    # training routines
    # unfrozen_backbone_blocks: list[int] | None = None
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
    enable_tensorboard: bool = False
    augmentation_config: str | None = None
    classifier_dropout: float | None = None


# Currently the file and s3 saving locations are hard coded
# may update this in the future
def parse_train_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Training engine for image classifier")

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    """
    parser.add_argument(
        "--enable_custom_augmentation",
        action="store_true",
    )
    """

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
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--load_model_from_artifact",
        type=str,
    )

    parser.add_argument(
        "--unfreeze_backbone_blocks",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--backbone_ft_lr",
        type=float,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--early_stop_patience",
        type=int,
    )

    parser.add_argument(
        "--lr_schedule_patience",
        type=int,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
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
        "--enable_tensorboard",
        action="store_true",
    )

    parser.add_argument(
        "--augmentation_config",
        type=str,
    )

    parser.add_argument(
        "--classifier_dropout",
        type=float,
    )

    try:
        args = parser.parse_args()
    except ValueError as e:
        parser.error(str(e))

    validate_args(args)
    return TrainArgs(
        num_workers=args.num_workers,
        # enable_custom_augmentation=args.enable_custom_augmentation,
        enable_gpu_augmentation=args.enable_gpu_augmentation,
        enable_ram_loaded_images=args.enable_ram_loaded_images,
        enable_backbone_caching=args.enable_backbone_caching,
        epochs=args.epochs,
        lr=args.lr,
        load_model_from_artifact=args.load_model_from_artifact,
        unfreeze_backbone_blocks=args.unfreeze_backbone_blocks,
        backbone_ft_lr=args.backbone_ft_lr,
        batch_size=args.batch_size,
        early_stop_patience=args.early_stop_patience,
        lr_schedule_patience=args.lr_schedule_patience,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        save=args.save,
        s3_bucket=args.s3_bucket,
        s3_key_prefix=args.s3_key_prefix,
        unfreeze_bb_blocks_with_lr=args.unfreeze_bb_blocks_with_lr,
        bb_block_wd=args.bb_block_wd,
        enable_tensorboard=args.enable_tensorboard,
        augmentation_config=args.augmentation_config,
        classifier_dropout=args.classifier_dropout,
    )


def int_float_pair(arg_value: str) -> tuple[int, float]:
    try:
        parsed_int, parsed_float = arg_value.split(":")
        return int(parsed_int), float(parsed_float)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid int:float entry pair: {arg_value}.  Must be of format 'int:float'. {str(e)}"
        )


def validate_args(args):

    # If  backbone caching is enabled, you cannot cache or set backbone as trainable
    # as it doesn't make sense
    if args.enable_backbone_caching and (
        args.augmentation_config
        or args.enable_gpu_augmentation
        or args.unfreeze_backbone_blocks
        or args.unfreeze_bb_blocks_with_lr
    ):
        raise ValueError(
            "Invalid Training Args: if enable_backbone_caching enabled, no augmentation or backbaone unfreezing args can be set"
        )

    # By default right now, training happens with backbone frozen and classifier trainable, so the base "lr" parameter is used
    # for the classifier.  If you choose to unfreeze backbone feature blocks, you must explicitly set a learning rate for backbone
    if args.unfreeze_backbone_blocks and (not args.backbone_ft_lr):
        raise ValueError(
            "Invalid Training Args: unfreeze_backbone_blocks set, you must set backbone_ft_lr to set the backbone learning rate"
        )

    # Validate save for S3 path
    if args.save == "s3" and (not args.s3_bucket or not args.s3_key_prefix):
        raise ValueError(
            "Invalid Training Args: If 'save' is S3, both s3_bucket and s3_key_prefix Must be set"
        )

    args.bb_block_wd = validate_and_normalize_bb_wd(
        args.bb_block_wd, args.unfreeze_bb_blocks_with_lr
    )


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
