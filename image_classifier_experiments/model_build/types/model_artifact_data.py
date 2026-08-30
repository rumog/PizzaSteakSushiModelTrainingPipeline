from dataclasses import dataclass
from typing import Any


@dataclass
class ModelArtifactData:
    model_state_dict: dict[str, Any]
    model_metadata: ModelMetadata


@dataclass
class ModelArchitecture:
    name: str
    weights: str | None


@dataclass
class ModelPreprocessing:
    image_size: tuple[int, int]


@dataclass
class ModelTrainingInfo:
    epoch: int | None = None
    validation_loss: float | None = None
    validation_accuracy: float | None = None


@dataclass
class ModelMetadata:
    class_list: list[str]
    architecture: ModelArchitecture
    preprocessing: ModelPreprocessing
    training: ModelTrainingInfo | None
