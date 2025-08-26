import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, image_size=8, num_channels=1, num_filters=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(
                num_channels, 16, kernel_size=3, stride=1, padding=1
            ),  # (16x16x16)
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(2),  # (8x8x8)
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),  # (8x8x8)
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),  # (4x4x4)
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),  # (4x4x4)
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),  # (4x4x4)
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Global Average Pooling
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, num_filters)

    def forward(self, x):
        """
        x: (batch_size, 1, depth, height, width) - N binary 3D images
        """
        x = x.unsqueeze(1)  # # (batch_size, channel, depth, height, width)
        x = self.conv_layers(x)  # (batch_size, num_filters, 1, 1, 1)
        x = self.flatten(x)  # (batch_size, num_filters)
        x = self.fc(x)  # (batch_size, num_filters)
        return x


if __name__ == "__main__":
    # Create a model
    image_size = 16
    model = CNNEncoder(image_size=image_size).to("cuda:0")
    # patches = torch.randn(100, 16, 16, 16)  # (batch_size, depth, height, width)
    patches = torch.randn(21641, image_size, image_size, image_size).to(
        "cuda:0"
    )  # (batch_size, depth, height, width)
    print(f"Input shape: {patches.shape}")
    output = model(patches)
    print(output.shape)  # Expected output: (100, 128)
