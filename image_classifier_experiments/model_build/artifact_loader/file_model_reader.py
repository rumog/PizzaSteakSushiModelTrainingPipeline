from pathlib import Path
from typing import Any

import torch

from image_classifier_experiments.model_build.types.model_artifact import (
    ModelArchitecture,
    ModelArtifactData,
    ModelMetadata,
    ModelPreprocessing,
    ModelTrainingInfo,
)
from image_classifier_experiments.model_build.types.model_artifact_schema import (
    ModelMetadataSchema,
)


class ModelArtifactFileReader:  # Model storage format keys
    STATE_DICT_KEY = "state_dict"
    METADATA_KEY = "metadata"

    def load_model_artifact(self, artifact_file: str):
        print(f"Loading model artifact from {artifact_file}")
        artifact_path = Path(artifact_file)
        if artifact_path.is_file():
            artifact = torch.load(artifact_path, map_location=torch.device("cpu"))
            self.validate_state_dict(artifact=artifact)
            print(f"Successfully loaded model weights from artifact: {artifact_file}")
        else:
            raise ValueError(
                f"artifact file: {artifact_file} does not exist. Check that file exists and is entered correctly."
            )
        print("successfully validated and loaded model artifact")

        model_metadata_schema = ModelMetadataSchema.model_validate(
            artifact.get(self.METADATA_KEY)
        )

        model_architecture = ModelArchitecture(
            backbone=model_metadata_schema.architecture.backbone,
            name=model_metadata_schema.architecture.name,
            weights=model_metadata_schema.architecture.weights,
        )

        model_preprocessing = ModelPreprocessing(
            image_size=model_metadata_schema.preprocessing.image_size
        )

        model_training_info = None
        if model_metadata_schema.training:
            model_training_info = ModelTrainingInfo(
                epoch=model_metadata_schema.training.epoch,
                validation_loss=model_metadata_schema.training.validation_loss,
                validation_accuracy=model_metadata_schema.training.validation_accuracy,
            )

        model_metadata = ModelMetadata(
            class_list=model_metadata_schema.class_list,
            architecture=model_architecture,
            preprocessing=model_preprocessing,
            training=model_training_info,
        )

        model_artifact_data = ModelArtifactData(
            model_state_dict=artifact.get(self.STATE_DICT_KEY),
            model_metadata=model_metadata,
        )

        return model_artifact_data

    def validate_state_dict(self, artifact: dict[str, Any]):

        # model state_dict must be present
        if not artifact.get(self.STATE_DICT_KEY):
            raise ValueError(
                f"artifact: {artifact} does not contain model state_dict key: {self.STATE_DICT_KEY}"
            )
