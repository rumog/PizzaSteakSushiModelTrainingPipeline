import argparse
from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainArgs:
    epochs: int = 5
    lr: float = 0.001
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
    # validate_args(args)
    return TrainArgs(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stop_patience=args.early_stop_patience,
        lr_schedule_patience=args.lr_schedule_patience,
        weight_decay=args.weight_decay,
        save=args.save,
        s3_bucket=args.s3_bucket,
        s3_key_prefix=args.s3_key_prefix,
    )
