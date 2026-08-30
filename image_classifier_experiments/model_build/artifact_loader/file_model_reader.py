from pathlib import Path
from typing import Any

import torch


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

        return artifact

    def validate_state_dict(self, artifact: dict[str, Any]):

        # model state_dict must be present
        if not artifact.get(self.STATE_DICT_KEY):
            raise ValueError(
                f"artifact: {artifact} does not contain model state_dict key: {self.STATE_DICT_KEY}"
            )
