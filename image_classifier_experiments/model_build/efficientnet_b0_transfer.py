from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import nn

from image_classifier_experiments.model_build.types.model_artifact_data import (
    ModelArtifactData,
)

WEIGHTS = torchvision.models.EfficientNet_B0_Weights.DEFAULT
DEFAULT_CLASSIFIER_DROPOUT = 0.2
STATE_DICT_KEY = "state_dict"


class EfficientNetB0TransferLearningModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        from_artifact: ModelArtifactData | None = None,
        unfrozen_backbone_blocks: int = 0,
        dropout_override: float | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.from_artifact = from_artifact
        self.unfrozen_backbone_blocks = unfrozen_backbone_blocks
        self.classifier_dropout = (
            dropout_override
            if dropout_override is not None
            else DEFAULT_CLASSIFIER_DROPOUT
        )

        # Create efficient_net_b0 model and initialize with DEFAULT weights
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = torchvision.models.efficientnet_b0(weights=weights)

        # No-op the backbone classifier to effectively nullify it, and replace
        # with custom classifier
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.classifier_dropout, inplace=True),
            nn.Linear(
                in_features=1280,  # same as original
                out_features=num_classes,
                bias=True,
            ),
        )

        # If a load-from artifact was specified, load state dict from artifact
        if self.from_artifact is not None:
            print("Load from artifact specified, loading artifact state_dict")
            self.load_state_dict(self.from_artifact.model_state_dict)
            print("Successfully loaded model weights from artifact")
            print(
                f"Verify state_dict matches artifact: {self.matches_state_dict(self.from_artifact.model_state_dict)}"
            )

        # Set trainability params

        # First freeze entire backbone, and leave classifier trainable
        self.backbone.requires_grad_(False)
        self.classifier.requires_grad_(True)

        # If rquested, unfreeze specified number of trailing backbone blocks
        if self.unfrozen_backbone_blocks < 0 or self.unfrozen_backbone_blocks > len(
            self.backbone.features
        ):
            raise ValueError(
                f"unfrozen_backbone_blocks must be between 0 and "
                f"{len(self.backbone.features)}"
            )
        elif self.unfrozen_backbone_blocks > 0:
            print(
                f"Unfreezing {self.unfrozen_backbone_blocks} trailing backbone feature layers"
            )
            self.backbone.features[-self.unfrozen_backbone_blocks :].requires_grad_(
                True
            )

    @classmethod
    def inference_transform(cls):
        return cls.WEIGHTS.transforms()

    def forward(self, X):
        return self.classifier(self.backbone(X))

    def train(self, mode: bool = True):
        """
        override the nn.Module train method so calling model.train() always sets the correct train/eval
        state for our current transfer learning configuration
        """
        # reset EVERYTHING to train() first, this covers both backbone and classifier
        super().train(mode)

        # Only set custom train state when mode is True (e.g. .train() was called)
        # not when mode is False - (e.g. scenarios like calling model.eval()
        # which under the hood is simply calling model.train(False)).  We don't want to set
        # anything to .train() in that case.
        if mode:
            # Set entire backbone to eval
            self.backbone.eval()

            # If we have unfrozen backbone blocks, set to train
            if self.unfrozen_backbone_blocks > 0:
                self.backbone.features[-self.unfrozen_backbone_blocks :].train()

        return self

    def matches_state_dict(self, match_dict):
        source_dict = self.state_dict()
        # 1. Check if they have the same number of layers/parameters
        if len(source_dict) != len(match_dict):
            return False

        # 2. Check if all keys match exactly
        if set(source_dict.keys()) != set(match_dict.keys()):
            return False

        # 3. Check if all individual tensors are identical
        for key, tensor1 in source_dict.items():
            tensor2 = match_dict[key]
            if not torch.equal(tensor1, tensor2):
                return False

        return True

    def print_training_modes(self):
        print(f"model: {'train' if self.training else 'eval'}")
        print(f"backbone: {'train' if self.backbone.training else 'eval'}")
        print(f"classifier: {'train' if self.classifier.training else 'eval'}")

        for i, block in enumerate(self.backbone.features):
            mode = "train" if block.training else "eval"
            print(f"  features[{i}]: {mode}")
