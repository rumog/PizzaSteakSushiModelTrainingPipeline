import torchvision
from torch import nn

WEIGHTS = torchvision.models.EfficientNet_B0_Weights.DEFAULT


class EfficientNetB0Pss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # Create pretrained efficient_net_b0 model
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = torchvision.models.efficientnet_b0(weights=weights)

        # Freeze all pretrained backbone parameters so the custom head remains
        # independent of the backbone's internal architecture
        for param in self.backbone.parameters():
            param.requires_grad = False

        # make backbone classifier essentially a no-op
        self.backbone.classifier = nn.Identity()

        # Create custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(
                in_features=1280,  # same as original
                out_features=num_classes,
                bias=True,
            ),
        )

    @classmethod
    def inference_transform(cls):
        return cls.WEIGHTS.transforms()

    def forward(self, X):
        return self.classifier(self.backbone(X))
