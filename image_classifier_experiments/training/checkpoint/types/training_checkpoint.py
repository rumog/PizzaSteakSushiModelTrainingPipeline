from dataclasses import dataclass
from typing import Any


@dataclass
class TrainCheckpoint:
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any]
    metadata: TrainCheckpointMetadata


@dataclass
class BestEpochCheckpoint:
    epoch: int
    test_loss: float
    test_acc: float
    state_dict: dict[str, Any]


@dataclass
class TrainCheckpointMetadata:
    history: dict[str, Any]
    last_epoch: int
    scheduled_epochs: int
    epochs_without_improvement: int
    best_epoch_checkpoint: BestEpochCheckpoint
