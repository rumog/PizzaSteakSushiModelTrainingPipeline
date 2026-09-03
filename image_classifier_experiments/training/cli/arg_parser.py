import argparse
import sys

from image_classifier_experiments.training.config.train_config import (
    TrainConfig,
)


# Currently the file and s3 saving locations are hard coded
# may update this in the future
def parse_train_args():  # -> TrainConfig:
    parser = argparse.ArgumentParser(description="Training engine for image classifier")

    parser.add_argument(
        "--train_config",
        type=str,
    )

    parser.add_argument(
        "--run_name",
        type=str,
    )

    parser.add_argument(
        "--resume_training_checkpoint",
        type=str,
    )

    parser.add_argument(
        "--load_artifact_from",
        choices=["file", "s3"],
    )

    parser.add_argument(
        "--artifact_s3_bucket",
        type=str,
    )

    parser.add_argument(
        "--artifact_s3_key",
        type=str,
    )

    parser.add_argument(
        "--artifact_file_path",
        type=str,
    )

    parser.add_argument(
        "--augmentation_config",
        type=str,
    )

    parser.add_argument(
        "--image_size_override",
        type=int,
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
        "--wandb_project",
        type=str,
    )

    parser.add_argument(
        "--wandb_run_id",
        type=str,
    )

    try:
        # args = parser.parse_args()
        fail_on_duplicate_training_config(parser)
        return parser.parse_args()
    except ValueError as e:
        parser.error(str(e))


def fail_on_duplicate_training_config(parser):
    provided_flags = {arg.split("=")[0] for arg in sys.argv[1:] if arg.startswith("-")}
    if "--train_config" in provided_flags and not provided_flags.issubset(
        {"--train_config", "--resume_training_checkpoint"}
    ):
        parser.error(
            "Training configuration cannot be specified both by config file and commandline.  When using "
            "--train_config, no arguments other than --resume_training_checkpoint can be used."
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
