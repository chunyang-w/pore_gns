"""
Source Code adapted from: https://github.com/zongyi-li/fourier_neural_operator
Simplification code based on: https://github.com/gegewen/ufno

Modified by: Chunyang Wang
GitHub username: chunyang-w

Main modification:
1. Permute the input and output to facilitate the FC layer

Example usage: (test model setup)
python pore_net/fno/model.py
"""

import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3],
            self.weights2,
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3],
            self.weights3,
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3],
            self.weights4,
        )

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1=8, modes2=8, modes3=8, width=36):
        super(FNO3d, self).__init__()
        """
        3D Fourier Neural Operator (FNO) for solving PDEs in spatiotemporal domains.

        This model consists of three Fourier layers followed by point-wise feed-forward
        layers to learn complex spatial-temporal dependencies.

        Args:
            modes1 (int, optional): Number of Fourier modes to keep in the first dimension. Default is 8.
            modes2 (int, optional): Number of Fourier modes to keep in the second dimension. Default is 8.
            modes3 (int, optional): Number of Fourier modes to keep in the third (temporal) dimension. Default is 8.
            width (int, optional): Number of feature channels in Fourier and convolution layers. Default is 36.

        Attributes:
            fc0 (nn.Linear): Initial linear layer to project input channels to `width`.
            conv0, conv1, conv2 (SpectralConv3d): Spectral convolution layers for learning frequency representations.
            w0, w1, w2 (nn.Conv1d): Pointwise convolution layers for additional processing.
            fc1, fc2 (nn.Linear): Fully connected layers for post-processing.

        Task:
        - Implement the `forward` method:
            - Apply `fc0` to project input to `width` channels.
            - Use `permute` to rearrange dimensions for spectral convolutions.
            - Apply `conv0`, `conv1`, `conv2` sequentially, with `relu` activations.
            - Apply `w0`, `w1`, `w2` to perform pointwise transformations.
            - Pass through `fc1` and `fc2` to obtain final outputs.
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        """
        3 channels for velocity_x, velocity_y, pressure
        """
        self.fc0 = nn.Linear(1, self.width)
        self.conv0 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv1 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv2 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """# noqa: E501
        The original model will take in (BS, H, W, D, num_c)
        and output (BS, H, W, D, 1)
        In this pore shape experiment, we would like the input to be (BS, C, H, W, D)
        So lets first permute the input to (BS, C, H, W, D), and before we return the output,
        we permute the output back to (BS, H, W, D, 1)
        """
        # we move the Channel dim to the last to facilitate the FC layer  # noqa: E501
        x = x.permute(0, 2, 3, 4, 1)
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        x = self.fc0(x)
        # This is the shape needed for the spectral conv
        # The spectral conv will take in (BS, C, H, W, D)
        # This is bringing the Channel dim from the last to the second position  # noqa: E501
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(
            batchsize, self.width, size_x, size_y, size_z
        )
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(
            batchsize, self.width, size_x, size_y, size_z
        )
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(
            batchsize, self.width, size_x, size_y, size_z
        )
        x = x1 + x2
        x = F.relu(x)

        # we move the Channel dim to the last to facilitate the FC layer  # noqa: E501
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # Apply sigmoid to the output
        # x = torch.sigmoid(x)

        # Now we bring the Channel dim back to the second position  # noqa: E501
        x = x.permute(0, 4, 1, 2, 3)
        return x


if __name__ == "__main__":

    model = FNO3d()
    x = torch.randn(1, 1, 128, 128, 128)
    # x = torch.randn(1, 128, 128, 128, 1)
    # x = torch.randn(1, 1, 128, 128, 128)
    print(model(x).shape)
