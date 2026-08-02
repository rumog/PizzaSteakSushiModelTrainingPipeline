from torch import nn


class TinyVGGBaseline(nn.Module):
    def __init__(self, in_channels: int, hidden_units: int, out_channels: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # TODO: make this configurable along with other magic numbers in here
            nn.Linear(in_features=hidden_units * 16 * 16, out_features=out_channels),
        )

        self.layer_stack = nn.Sequential(
            self.conv_block_1, self.conv_block_2, self.classifier
        )

    def forward(self, X):
        # Look into operator fusion vs using nn.sequential.
        return self.layer_stack(X)
