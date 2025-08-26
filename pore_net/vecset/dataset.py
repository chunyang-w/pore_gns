"""
Inspired by: https://github.com/1zb/3DShape2VecSet?tab=readme-ov-file

Written by: Chunyang Wang
GitHub username: chunyang-w
"""

import os
import glob
import torch
import numpy as np

from natsort import natsorted
from torch.utils.data import Dataset


class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter

    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, point


class PoreShapeDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        num_entries=None,  # how many 3d shapes to use
        sampling=True,
        num_samples=1024,  # how many points to sample from the 3d shape
        surface_sampling=True,
        pc_size=2048,
        scale=True,
        transform=AxisScaling((0.75, 1.25), True),
    ):
        self.num_entries = num_entries
        self.transform = transform
        self.sampling = sampling
        self.num_samples = num_samples
        self.surface_sampling = surface_sampling
        self.pc_size = pc_size
        self.split = split
        self.scale = scale
        self.transform = transform

        self.data_names = natsorted(glob.glob(data_dir + "*.npz"))

    def __len__(self):
        if self.num_entries is not None:
            return self.num_entries
        else:
            return len(self.data_names)

    def __getitem__(self, index):
        data_name = self.data_names[index]
        data = np.load(data_name)

        points_vol = data["points_vol"]
        labels_vol = data["labels_vol"]
        points_near = data["points_near"]
        labels_near = data["labels_near"]
        surface_points = data["surface_points"]

        if self.sampling:
            idx = np.random.choice(
                len(points_vol),
                self.num_samples,
                replace=False,
            )
            points_vol = points_vol[idx]
            labels_vol = labels_vol[idx]

            idx = np.random.choice(
                len(points_near),
                self.num_samples,
                replace=False,
            )
            points_near = points_near[idx]
            labels_near = labels_near[idx]

        if self.surface_sampling:
            idx = np.random.choice(
                len(surface_points),
                self.pc_size,
                replace=False,
            )
            surface_points = surface_points[idx]

        if self.split == "train":
            points = np.concat([points_vol, points_near], axis=0)
            labels = np.concat([labels_vol, labels_near], axis=0)
        else:
            points = points_vol
            labels = labels_vol

        if self.scale:
            epsilon = 1e-8
            min_val = np.min(points, axis=0)
            max_val = np.max(points, axis=0)

            points = 2 * (points - min_val) / (max_val - min_val) - 1 + epsilon  # noqa
            surface_points = (
                2 * (surface_points - min_val) / (max_val - min_val)
                - 1
                + epsilon  # noqa
            )  # noqa

        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).float()
        surface_points = torch.from_numpy(surface_points).float()

        if self.transform:
            surface_points, points = self.transform(surface_points, points)

        return points, labels, surface_points


if __name__ == "__main__":
    data_dir = "./output/pad_size5_down10dist_thresh1/"  # noqa
    dataset = PoreShapeDataset(
        data_dir=data_dir,
        transform=None,
        sampling=True,
        num_samples=2048,
        surface_sampling=True,
        pc_size=2048,
    )
    print("dataset length: ", len(dataset))
    print("dataset[0]: ", dataset[0])

    points, labels, surface_points = dataset[0]
    print("min points: ", torch.min(points, dim=0))
    print("max points: ", torch.max(points, dim=0))
    print(
        "min surface_points: ",
        torch.min(surface_points, dim=0),
    )
    print(
        "max surface_points: ",
        torch.max(surface_points, dim=0),
    )
