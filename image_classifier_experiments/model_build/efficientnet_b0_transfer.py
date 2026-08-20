from pathlib import Path

import torch
import torchvision
from torch import nn

WEIGHTS = torchvision.models.EfficientNet_B0_Weights.DEFAULT
ROOT_DIR = root_dir = Path("model")


class EfficientNetB0TransferLearningModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        from_artifact: str | None = None,
        unfrozen_backbone_blocks: int = 0,
        # device defult to cpu for loading purposes, so you can load a model trained on
        # a different device.  After model creation you can move it to device associated
        # with current training platform
        device: str = "cpu",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.from_artifact = from_artifact
        self.unfrozen_backbone_blocks = unfrozen_backbone_blocks

        # Create efficient_net_b0 model and initialize with DEFAULT weights
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = torchvision.models.efficientnet_b0(weights=weights)

        # No-op the backbone classifier to effectively nullify it, and replace
        # with custom classifier
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(
                in_features=1280,  # same as original
                out_features=num_classes,
                bias=True,
            ),
        )

        artifact = None
        if self.from_artifact is not None:
            artifact_file = ROOT_DIR / from_artifact
            if artifact_file.is_file():
                artifact = torch.load(artifact_file, map_location=torch.device(device))
                self.load_state_dict(artifact["state_dict"])
            else:
                print(
                    f"from_artifact argument {self.from_artifact} not found, ignoring artifact loading and using default weights"
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
            self.backbone.features[-self.unfrozen_backbone_blocks :].requires_grad_(
                True
            )

        # print(f"----DEBUG-----: {self}")

        if artifact is not None:
            print(
                f"State dicts equal: {self.are_state_dicts_equal(self.state_dict(), artifact['state_dict'])}"
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

    def are_state_dicts_equal(self, dict1, dict2):
        # 1. Check if they have the same number of layers/parameters
        if len(dict1) != len(dict2):
            return False

        # 2. Check if all keys match exactly
        if set(dict1.keys()) != set(dict2.keys()):
            return False

        # 3. Check if all individual tensors are identical
        for key, tensor1 in dict1.items():
            tensor2 = dict2[key]
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
