from typing import Any

from pydantic import BaseModel


class BestEpochCheckpointSchema(BaseModel):
    epoch: int
    test_loss: float
    test_acc: float
    state_dict: dict[str, Any]


class TrainingCheckpointMetadataSchema(BaseModel):
    history: dict[str, Any]
    last_epoch: int
    scheduled_epochs: int
    epochs_without_improvement: int
    best_epoch_checkpoint: BestEpochCheckpointSchema
