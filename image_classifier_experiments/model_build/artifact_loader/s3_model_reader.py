import io
from typing import Any

import boto3
import torch
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    ParamValidationError,
)

from image_classifier_experiments.model_build.types.model_artifact_data import (
    ModelArchitecture,
    ModelArtifactData,
    ModelMetadata,
    ModelPreprocessing,
    ModelTrainingInfo,
)
from image_classifier_experiments.model_build.types.model_checkpoint import (
    ModelMetadataSchema,
)


class ModelArtifactS3Reader:  # Model storage format keys
    STATE_DICT_KEY = "state_dict"
    METADATA_KEY = "metadata"

    # use dependency injection instead
    def __init__(self):
        self.s3 = boto3.client("s3")

    def load_model_artifact(self, model_bucket, model_key) -> ModelArtifactData:

        try:
            response = self.s3.get_object(Bucket=model_bucket, Key=model_key)
            content = response["Body"].read()

        except ParamValidationError as e:
            print(
                f"Local param validation failure, check artifact bucket: {model_bucket} and key: {model_key} values. Error: {e}"
            )
            raise
        except NoCredentialsError as e:
            print(
                f"AWS Credential Error: AWS credentials not found or configured. Error: {e}"
            )
            raise
        except EndpointConnectionError as e:
            print(f"Network Error: Could not contact AWS S3 endpoint. Error: {e}")
            raise
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")

            match error_code:
                case "NoSuchKey":
                    print(f"The artifact key {model_key} wasn't found.")
                case "NoSuchBucket":
                    print(f"The artifact bucket {model_bucket} wasn't found.")
                case "AccessDenied":
                    print(
                        f"Access denied to S3 object {model_bucket}/{model_key}.  Check IAM permissions"
                    )
                case _:
                    print(f"AWS S3 ClientError ({error_code}): {e}")
            raise

        print(
            f"Successfully retrieved and read artifact object from {model_bucket}/{model_key}"
        )

        artifact = torch.load(io.BytesIO(content), map_location="cpu")
        self.validate_state_dict(artifact=artifact)
        print("successfully validated and loaded model artifact")

        model_metadata_schema = ModelMetadataSchema.model_validate(
            artifact.get(self.METADATA_KEY)
        )

        model_architecture = ModelArchitecture(
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
