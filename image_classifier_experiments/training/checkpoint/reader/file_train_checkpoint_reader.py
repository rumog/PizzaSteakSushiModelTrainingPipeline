from pathlib import Path
from typing import Any

import torch

from image_classifier_experiments.training.checkpoint.types.training_checkpoint import (
    BestEpochCheckpoint,
    TrainCheckpoint,
    TrainCheckpointMetadata,
)
from image_classifier_experiments.training.checkpoint.types.training_checkpoint_schema import (
    TrainingCheckpointMetadataSchema,
)


class TrainingCheckpointFileReader:  # Model storage format keys
    MODEL_STATE_DICT_KEY = "model_state_dict"
    OPTIMIZER_STATE_DICT_KEY = "optimizer_state_dict"
    SCHEDULER_STATE_DICT_KEY = "scheduler_state_dict"
    METADATA_KEY = "metadata"

    @classmethod
    def load_train_checkpoint(cls, checkpoint_file: str):
        print(f"Loading training checkpoint from {checkpoint_file}...")
        checkpoint_file_path = Path(checkpoint_file)
        if checkpoint_file_path.is_file():
            training_checkpoint_artifact = torch.load(
                checkpoint_file_path, map_location=torch.device("cpu")
            )
            cls.validate_state_dicts(artifact=training_checkpoint_artifact)
        else:
            raise ValueError(
                f"Checkpoint file: {checkpoint_file} does not exist. Check that file exists and is entered correctly."
            )

        print(f"Successfully loaded training checkpoint artifact: {checkpoint_file}")

        train_metadata_schema = TrainingCheckpointMetadataSchema.model_validate(
            training_checkpoint_artifact.get(cls.METADATA_KEY)
        )

        best_epoch_checkpoint = BestEpochCheckpoint(
            epoch=train_metadata_schema.best_epoch_checkpoint.epoch,
            test_loss=train_metadata_schema.best_epoch_checkpoint.test_loss,
            test_acc=train_metadata_schema.best_epoch_checkpoint.test_acc,
            state_dict=train_metadata_schema.best_epoch_checkpoint.state_dict,
        )

        train_checkpoint_metadata = TrainCheckpointMetadata(
            history=train_metadata_schema.history,
            last_epoch=train_metadata_schema.last_epoch,
            scheduled_epochs=train_metadata_schema.scheduled_epochs,
            epochs_without_improvement=train_metadata_schema.epochs_without_improvement,
            best_epoch_checkpoint=best_epoch_checkpoint,
        )

        train_checkpoint = TrainCheckpoint(
            model_state_dict=training_checkpoint_artifact.get(cls.MODEL_STATE_DICT_KEY),
            optimizer_state_dict=training_checkpoint_artifact.get(
                cls.OPTIMIZER_STATE_DICT_KEY
            ),
            scheduler_state_dict=training_checkpoint_artifact.get(
                cls.SCHEDULER_STATE_DICT_KEY
            ),
            metadata=train_checkpoint_metadata,
        )
        print(
            "Successfully validated and created training checkpoint from loaded artifact"
        )
        return train_checkpoint

    @classmethod
    def validate_state_dicts(cls, artifact: dict[str, Any]):

        # model state_dict must be present
        if (
            not artifact.get(cls.MODEL_STATE_DICT_KEY)
            or not artifact.get(cls.OPTIMIZER_STATE_DICT_KEY)
            or not artifact.get(cls.SCHEDULER_STATE_DICT_KEY)
        ):
            raise ValueError(
                f"artifact: {artifact} does not contain one of state_dict key: {cls.MODEL_STATE_DICT_KEY}, "
                f"{cls.OPTIMIZER_STATE_DICT_KEY}, or {cls.SCHEDULER_STATE_DICT_KEY}."
            )
