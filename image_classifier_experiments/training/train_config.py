# Avoids needing quotes around "TrainArgs"
# In from_yaml as TrainArgs isn't fully defined yet
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class TrainConfig:
    run_name: str | None = None
    resume_training_checkpoint: str | None = None
    load_artifact_from: Literal["file", "s3"] | None = None
    artifact_s3_bucket: str | None = None
    artifact_file_path: str | None = None
    artifact_s3_key: str | None = None
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

    def __post_init__(self):
        self.validate_args()

    # Putting this here in the TrainArgs class for now, but consider moving it into the
    # test entry point.  This should probably just be a dumb datatype for hoding the values
    # and tests define what argument combinations/values are valid for their use case.  Validity
    # is defined in the context of a given training scenario not by the object itself.
    def validate_args(self):
        print("Validating training config...")

        if (
            self.resume_training_checkpoint is not None
            and self.load_artifact_from is not None
        ):
            raise ValueError(
                "You cannot use both load_artifact_from and resume_training_checkpoint. If you want to load model weights from "
                "a source, you must choose either a model artifact, or a training checkpoint as the source, not both."
            )

        # Validate correct argument config for loading model artifact from file or s3
        self.validate_load_artifact_config()

        # If  backbone caching is enabled, you cannot run custom data augmentation or set backbone as trainable
        if self.enable_backbone_caching and (
            self.augmentation_config
            or self.enable_gpu_augmentation
            or self.unfreeze_bb_blocks_with_lr
        ):
            raise ValueError(
                "Invalid Training Args: if enable_backbone_caching enabled, no augmentation or backbaone unfreezing args can be set"
            )
        if self.bypass_cache_generation and not self.enable_backbone_caching:
            raise ValueError(
                "Invalid Training Args: bypass_cache_generation cannot be specified without enable_backbone_caching."
            )

        # validate/normalize args related to staged backbone unfreeze and associated lr/wd per stage
        self.bb_block_wd = self.validate_and_normalize_bb_wd()

        # Validate scheduler args
        self.validate_scueduler_config()

        # Validate save for S3 path
        if self.save == "s3" and (not self.s3_bucket or not self.s3_key_prefix):
            raise ValueError(
                "Invalid Training Args: If 'save' is S3, both s3_bucket and s3_key_prefix Must be set"
            )

        if self.wandb_run_name and not self.enable_wandb:
            raise ValueError("wandb_run_name cannot be set without enable_wandb.")

    def validate_load_artifact_config(self):
        if self.load_artifact_from == "s3" and (
            (not self.artifact_s3_bucket or not self.artifact_s3_key)
            or self.artifact_file_path
        ):
            raise ValueError(
                f"If load_artifact_from is 's3', must include artifact_s3_bucket: {self.artifact_s3_bucket} and artifact_s3_key: "
                f"{self.artifact_s3_key}, and must not include artifact_file_path: {self.artifact_file_path}"
            )

        if self.load_artifact_from == "file" and (
            not self.artifact_file_path
            or self.artifact_s3_bucket
            or self.artifact_s3_key
        ):
            raise ValueError(
                f"If load_artifact_from == 'file', must include artifact_file_path: {self.artifact_file_path}, and must not specify "
                f"artifact_s3_bucket: {self.artifact_s3_bucket} or artifact_s3_key: {self.artifact_s3_key}"
            )

        if not self.load_artifact_from and (
            self.artifact_s3_bucket or self.artifact_s3_key or self.artifact_file_path
        ):
            raise ValueError(
                "Cannot specify include artifact loading source arguments without specifying load_artifact_from to."
            )

    def validate_and_normalize_bb_wd(self):
        if self.unfreeze_bb_blocks_with_lr is None:
            if self.bb_block_wd is not None:
                raise ValueError(
                    "Cannot specify backbone weight decay stages (--bb_block_wd) if no unfreeze block configuration is specified "
                    "(--unfreeze_bb_blocks_with_lr)"
                )
            return None

        # Unfreeze block stages have been specified, fallback to default weight decay of 0.0
        # for any stages where backbone weight decay is unspecified
        num_unfrozen_bb_stages = len(self.unfreeze_bb_blocks_with_lr)
        if self.bb_block_wd is None:
            return [0.0] * num_unfrozen_bb_stages

        if len(self.bb_block_wd) < num_unfrozen_bb_stages:
            pad_len = num_unfrozen_bb_stages - len(self.bb_block_wd)
            self.bb_block_wd.extend([0.0] * pad_len)

        elif len(self.bb_block_wd) > num_unfrozen_bb_stages:
            raise ValueError(
                "Cannot specify more backbone weight decay values (--bb_block_wd) than unfrozen backbone stages "
                "(--unfreeze_bb_blocks_with_lr)"
            )

        return self.bb_block_wd

    def validate_scueduler_config(self):
        if not self.scheduler_type and (
            self.reducelr_patience is not None
            or self.reducelr_factor is not None
            or self.min_scheduler_lr is not None
            or self.enable_lr_warmup
            or self.warmup_epochs is not None
            or self.warmup_factors is not None
        ):
            raise ValueError(
                "No scheduler_type set: cannot specify lr scheduler-related arguments if no scheduler type set."
            )

        if self.scheduler_type and self.epochs <= 1:
            raise ValueError(
                "Cannot enable scheduler with scheduler_type when training for less than 2 epochs, as scheduler would not be engaged"
            )

        if (
            self.scheduler_type == "ReduceLROnPlateau"
            and self.reducelr_patience is None
        ):
            raise ValueError(
                "Scheduler type ReduceLROnPlateau requires setting patience with reducelr_patience"
            )

        if self.scheduler_type == "ReduceLROnPlateau" and self.enable_lr_warmup:
            raise ValueError(
                "Scheduler type ReduceLROnPlateau is not compatible with current warmup mechanism (composite scheduler with LinearLR)"
            )

        if self.scheduler_type == "CosineAnnealingLR" and (
            self.reducelr_patience is not None or self.reducelr_factor is not None
        ):
            raise ValueError(
                "Scheduler type CosineAnnealingLR not compatible with reducelr_patience or reducelr_factor and must not specify these arguments."
            )

        if self.reducelr_factor is not None and not (0.0 < self.reducelr_factor < 1.0):
            raise ValueError("reducelr_factor must be greater than 0 and less than 1")

        if not self.enable_lr_warmup and (
            self.warmup_epochs is not None or self.warmup_factors is not None
        ):
            raise ValueError(
                "enable_lr_warmup must be set to use warmup_epochs, or warmup_factors. Enable warmup, or reomve warmup related arguments."
            )
        if self.min_scheduler_lr is not None and self.min_scheduler_lr < 0:
            raise ValueError("min_scheduler_lr must be >= 0.")

        if self.warmup_epochs is not None and self.warmup_epochs < 1:
            raise ValueError("warmup_epochs must be >= 1.")

        if self.warmup_epochs is not None and self.warmup_epochs >= self.epochs:
            raise ValueError(
                f"warmup_epochs: {self.warmup_epochs} must be less than epochs: {self.epochs}"
            )
        if self.warmup_factors is not None:
            start_factor, end_factor = self.warmup_factors
            if start_factor <= 0 or end_factor <= 0:
                raise ValueError(
                    f"warmup_factor values {start_factor}:{end_factor} must both be greater than 0"
                )
            if start_factor > end_factor:
                raise ValueError(
                    f"warmup_factor start value: {start_factor} must be less than or equal to end value: {end_factor}"
                )

    def to_dict(self, include_nulls: bool = False) -> dict[str, Any]:
        # return asdict(self)
        # use this way if you want to strip out values with None
        # instead of writing them to yaml with "null" vlues
        if include_nulls:
            return asdict(self)
        else:
            return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        data = dict(data)

        if data.get("unfreeze_bb_blocks_with_lr") is not None:
            data["unfreeze_bb_blocks_with_lr"] = [
                (int(num_blocks), float(lr))
                for num_blocks, lr in data["unfreeze_bb_blocks_with_lr"]
            ]

        if data.get("warmup_factors") is not None:
            data["warmup_factors"] = tuple(
                float(factor) for factor in data["warmup_factors"]
            )

        return cls(**data)

    def to_yaml(self, path: str | Path, include_nulls: bool = False) -> None:

        return yaml.safe_dump(
            self.to_dict(include_nulls),
            sort_keys=False,
            default_flow_style=None,
        )

    @classmethod
    def from_yaml(cls, yaml_string: str) -> TrainConfig:
        data = yaml.safe_load(yaml_string)
        return cls.from_dict(data)
