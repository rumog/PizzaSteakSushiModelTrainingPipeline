import argparse
from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainArgs:
    num_workers: int = 0
    enable_custom_augmentation: bool = False
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
    save: Literal["file", "s3"] | None = None
    s3_bucket: str | None = None
    s3_key_prefix: str | None = None


# Currently the file and s3 saving locations are hard coded
# may update this in the future
def parse_train_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Training engine for image classifier")

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--enable_custom_augmentation",
        action="store_true",
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

    args = parser.parse_args()
    validate_args(args)
    return TrainArgs(
        num_workers=args.num_workers,
        enable_custom_augmentation=args.enable_custom_augmentation,
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
        save=args.save,
        s3_bucket=args.s3_bucket,
        s3_key_prefix=args.s3_key_prefix,
    )


def validate_args(args):

    # If  backbone caching is enabled, you cannot cache or set backbone as trainable
    # as it doesn't make sense
    if args.enable_backbone_caching and (
        args.enable_custom_augmentation
        or args.enable_gpu_augmentation
        or args.unfreeze_backbone_blocks
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
