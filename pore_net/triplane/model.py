"""
Source Code from: https://github.com/JRyanShue/NFD
Code for paper 3D Neural Field Generation using Triplane Diffusion, https://arxiv.org/abs/2211.16677  # noqa: E501

Modified by: Chunyang Wang
GitHub username: chunyang-w

test MultiTriplane model code:

python pore_net/triplane/model.py
"""

import torch
import torch.nn as nn
import numpy as np


def first_layer_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            # if hasattr(m, 'weight'):
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(
                    -np.sqrt(6 / num_input) / freq,
                    np.sqrt(6 / num_input) / freq,
                )

    return init


class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(
            torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False
        )  # noqa

    def forward(self, x):
        B, N, C = x.shape
        x = (x.reshape(B * N, C) @ self._B).reshape(B, N, -1)
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class MultiTriplane(nn.Module):
    def __init__(
        self,
        num_objs,
        input_dim=3,
        output_dim=1,
        noise_val=None,
        resolution=128,
        num_channels=32,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.num_objs = num_objs
        self.resolution = resolution
        self.resolution = resolution
        self.num_channels = num_channels
        self.embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(1, num_channels, resolution, resolution) * 0.001
                )  # noqa
                for _ in range(3 * num_objs)
            ]
        )
        self.noise_val = noise_val
        # Use this if you want a PE
        self.net = nn.Sequential(
            # FourierFeatureTransform(32, 64, scale=1),
            FourierFeatureTransform(num_channels, 64, scale=1),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
        )

        self.__init_print__()

    def __init_print__(self):
        """Print model initialization details in a fancy format"""
        print("\n" + "=" * 50)
        print("Initializing MultiTriplane Model:")
        print("=" * 50)
        print(f"{'Number of Objects:':<25} {self.num_objs}")
        print(f"{'Resolution:':<25} {self.resolution}")
        print(f"{'Number of Channels:':<25} {self.num_channels}")
        print(f"{'Device:':<25} {self.device}")
        print(f"{'Noise Value:':<25} {self.noise_val}")

        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"{'Total Parameters:':<25} {total_params:,}")

        # Break down parameters by component
        triplane_params = sum(p.numel() for p in self.embeddings.parameters())
        decoder_params = sum(p.numel() for p in self.net.parameters())
        print("\nParameter Distribution:")
        print(f"{'Triplane Parameters:':<25} {triplane_params:,}")
        print(f"{'Decoder Parameters:':<25} {decoder_params:,}")
        print("=" * 50 + "\n")

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        # print("grid_shape", coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]).shape)  # [1, 1, 1000, 2]  # noqa: E501
        sampled_features = torch.nn.functional.grid_sample(
            plane,
            coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        # print("plane.shape", plane.shape)  # [1, 32, 128, 128]
        # print("sampled_features.shape", sampled_features.shape)  # [1, 32, 1, 1000]  # noqa: E501
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H * W).permute(0, 2, 1)
        return sampled_features

    def forward(self, obj_idx, coordinates, debug=False):
        # obj_idx: frame number, int
        # Coordinates: Could point (BS, M, 3)
        batch_size, n_coords, n_dims = coordinates.shape
        # print(f"coordinates.shape: {coordinates.shape}")
        # print(f"xy slice shape: {coordinates[..., 0:2].shape}")
        # print(f"yz slice shape: {coordinates[..., 1:3].shape}")
        # print(f"xz slice shape: {coordinates[..., :3:2].shape}") # ([1, 1000, 2])  # noqa: E501
        xy_embed = self.sample_plane(
            coordinates[..., 0:2], self.embeddings[3 * obj_idx + 0]
        )
        yz_embed = self.sample_plane(
            coordinates[..., 1:3], self.embeddings[3 * obj_idx + 1]
        )
        xz_embed = self.sample_plane(
            coordinates[..., :3:2], self.embeddings[3 * obj_idx + 2]
        )

        # print("xy_embed.shape", xy_embed.shape)  # ([1, 1000, 32])  # noqa: E501

        # if self.noise_val != None:
        #    xy_embed = xy_embed + self.noise_val*torch.empty(xy_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)  # noqa
        #    yz_embed = yz_embed + self.noise_val*torch.empty(yz_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)  # noqa
        #    xz_embed = xz_embed + self.noise_val*torch.empty(xz_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)  # noqa
        # print(f"xy_embed.shape: {xy_embed.shape}")
        # print(f"yz_embed.shape: {yz_embed.shape}")
        # print(f"xz_embed.shape: {xz_embed.shape}")

        features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        if self.noise_val is not None and self.training:
            features = features + self.noise_val * torch.empty(features.shape).normal_(
                mean=0, std=0.5
            ).to(self.device)
        # out: (BS, M, 1)
        return self.net(features)

    def tvreg(self):
        l = 0  # noqa
        for embed in self.embeddings:
            l += ((embed[:, :, 1:] - embed[:, :, :-1]) ** 2).sum() ** 0.5
            l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1]) ** 2).sum() ** 0.5
        return l / self.num_objs

    def l2reg(self):
        l = 0  # noqa
        for embed in self.embeddings:
            l += (embed**2).sum() ** 0.5
        return l / self.num_objs


class TriplaneAutoDecoder(nn.Module):
    def __init__(
        self,
        resolution,
        channels,
        how_many_scenes,
        input_dim=3,
        output_dim=1,
        aggregate_fn="sum",  # vs sum
        use_tanh=False,
        view_embedding=True,
        rendering_kwargs={},
        triplane_cpu_intermediate=False,
        device="cpu",
    ):
        super().__init__()

        self.aggregate_fn = aggregate_fn
        print(f"Using aggregate_fn {aggregate_fn}")

        self.resolution = resolution
        self.channels = channels
        self.embeddings = nn.ModuleList(
            [
                torch.nn.Embedding(
                    1, 3 * self.channels * self.resolution * self.resolution
                )
                for i in range(how_many_scenes)
            ]
        )  # , sparse=True)
        self.use_tanh = use_tanh
        self.view_embedding = view_embedding
        self.embedder_type = "positional"
        self.view_multires = 4
        self.triplane_cpu_intermediate = triplane_cpu_intermediate
        self.device = device

        if not self.triplane_cpu_intermediate:
            for embedding in self.embeddings:
                embedding = embedding.to(self.device)

        self.mode = "shape"

        self.view_embed_dim = 0

        # Lightweight decoder
        self.net = nn.Sequential(
            # https://arxiv.org/abs/2006.10739 - Fourier FN
            nn.Linear(self.channels + self.view_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        ).to(self.device)

        self.net.apply(frequency_init(30))
        self.net[0].apply(first_layer_sine_init)

        self.rendering_kwargs = rendering_kwargs

        if self.triplane_cpu_intermediate:
            # We need to store the currently used triplanes on GPU memory, but don't want to load them each time we make a forward pass.  # noqa: E501
            self.current_embeddings = None  # Embedding object within list of embeddings. Need this intermediate step for gradient to pass through  # noqa: E501
            self.current_triplanes = None
            self.current_obj_idx = None

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(
            plane,
            coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H * W).permute(0, 2, 1)
        return sampled_features

    def forward(self, obj_idx, coordinates):

        # print(f'coordinates.shape: {coordinates.shape}')  # e.g. [1, 16359, 256, 3]  # noqa: E501

        if len(coordinates.shape) == 3:
            batch_size, n_coords, n_dims = coordinates.shape
        elif len(coordinates.shape) == 4:
            batch_size, ray_batch_size, n_coords, n_dims = coordinates.shape
        assert batch_size == obj_idx.shape[0]

        # Get embedding at index and reshape to (N, 3, channels, H, W)
        # self.embeddings[obj_idx].to(self.device)
        # print(f'obj_idx: {obj_idx}')  # e.g. tensor([[0]], device='cuda:0')

        if self.triplane_cpu_intermediate:
            # Move triplane from CPU to GPU. Only happens once per scene.
            if obj_idx != self.current_obj_idx:
                print(f"Moving triplane at obj_idx {obj_idx} from CPU to GPU...")
                self.current_obj_idx = obj_idx
                self.current_embeddings = self.embeddings[obj_idx.to("cpu")].to(
                    self.device
                )

            self.current_triplanes = self.current_embeddings(
                torch.tensor(0, dtype=torch.int64).to(self.device)
            ).view(batch_size, 3, self.channels, self.resolution, self.resolution)
            triplanes = self.current_triplanes
        # Coord (bs, num_points, 3)
        else:
            triplanes = self.embeddings[obj_idx.to("cpu")](  # (128, 128, 3)
                torch.tensor(0, dtype=torch.int64).to(self.device)
            ).view(batch_size, 3, self.channels, self.resolution, self.resolution)

        # Use tanh to clamp triplanes
        if self.use_tanh:
            triplanes = torch.tanh(triplanes)

        # Triplane aggregating fn.

        # TODO: Make sure all these coordinates line up.
        xy_embed = self.sample_plane(
            coordinates[..., 0:2], triplanes[:, 0]
        )  # ex: [batch_size, 20000, 64]
        yz_embed = self.sample_plane(coordinates[..., 1:3], triplanes[:, 1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], triplanes[:, 2])
        # (M, C)

        # aggregate - product or sum?
        if self.aggregate_fn == "sum":
            features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        else:
            features = torch.prod(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        # (M, C)

        # decoder
        if self.mode == "shape":
            return self.net(features)


# For single-scene fitting
class CartesianPlaneNonSirenEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, aggregate_fn="prod"):  # vs sum
        super().__init__()

        self.aggregate_fn = aggregate_fn
        print(f"Using aggregate_fn {aggregate_fn}")

        self.embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(1, 64, 32, 32) * 0.1) for _ in range(3)]
        )

        self.net = nn.Sequential(
            # https://arxiv.org/abs/2006.10739
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        self.net.apply(frequency_init(30))
        self.net[0].apply(first_layer_sine_init)

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(
            plane,
            coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H * W).permute(0, 2, 1)
        return sampled_features

    def forward(self, coordinates, debug=False):
        batch_size, n_coords, n_dims = coordinates.shape

        xy_embed = self.sample_plane(
            coordinates[..., 0:2], self.embeddings[0]
        )  # ex: [1, 20000, 64]
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[2])
        # (M, C)

        # aggregate - product or sum?
        if self.aggregate_fn == "sum":
            features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        else:
            features = torch.prod(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        # (M, C)

        # decoder
        return self.net(features)


if __name__ == "__main__":
    model = MultiTriplane(
        num_objs=1,
        resolution=128,
        num_channels=32,
        device="cuda",
    )
    print("model initialized")
    dummy_input = torch.randn(1, 1000, 3)
    output = model(0, dummy_input)
    print(output.shape)
