"""
Code for loading cloud point dataset for triplane fitting.
Inspired by work from NFD: https://github.com/JRyanShue/NFD/blob/main/nfd/neural_field_diffusion/triplane_fitting/dataset.py  # noqa: E501

Written by: Chunyang Wang
GitHub username: chunyang-w

Example testing code:
python pore_net/triplane/dataset.py
"""

import os
import glob
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset


class CloudPointDataset(Dataset):
    def __init__(
        self,
        data_path,
        num_near_points=5e5,
        num_voxel_points=5e5,
        subset_size=None,
        replace=False,
    ):
        self.data_path = data_path
        self.num_near_points = int(num_near_points)
        self.num_voxel_points = int(num_voxel_points)
        self.subset_size = subset_size
        self.replace = replace
        self.data_names = natsorted(glob.glob(os.path.join(data_path, "*.npz")))

    def __len__(self):
        if self.subset_size is None:
            return len(self.data_names)
        else:
            return self.subset_size

    def __getitem__(self, idx):
        if self.subset_size is not None:
            assert (
                idx < self.subset_size
            ), f"Index {idx} is greater than the subset size {self.subset_size}"  # noqa: E501

        data_name = self.data_names[idx]
        cp_data = np.load(data_name)

        # Sample 10000 points randomly
        points_vol = cp_data["points_vol"]
        labels_vol = cp_data["labels_vol"]

        points_near = cp_data["points_near"]
        labels_near = cp_data["labels_near"]

        num_points_vol = len(points_vol)
        num_points_near = len(points_near)

        if not self.replace:
            assert num_points_vol > self.num_voxel_points, (
                f"Number of points in voxel volume is {num_points_vol}, "
                f"which is greater less the sample requirement {self.num_voxel_points}"  # noqa: E501
            )
            assert num_points_near > self.num_near_points, (
                f"Number of points in near points is {num_points_near}, "
                f"which is less than the sample requirement {self.num_near_points}"
            )

        idx_vol = np.random.choice(
            len(points_vol), self.num_voxel_points, replace=self.replace
        )  # noqa: E501
        idx_near = np.random.choice(
            len(points_near), self.num_near_points, replace=self.replace
        )  # noqa: E501

        points_vol = points_vol[idx_vol]
        labels_vol = labels_vol[idx_vol]

        points_near = points_near[idx_near]
        labels_near = labels_near[idx_near]

        points = np.concatenate([points_near, points_vol], axis=0)
        labels = np.concatenate([labels_near, labels_vol], axis=0)

        # Add batch dimension ahead of all other dims
        points = np.expand_dims(points, axis=0)  # (N, 3) -> (1, N, 3)
        labels = np.expand_dims(labels, axis=0)  # (N,) -> (1, N)

        return points, labels


if __name__ == "__main__":
    data_path = "./data/PC_073_segmented_tifs_pad_0_ds6near_thresh3.0"
    num_points = 5e5
    num_near_points = num_points
    num_voxel_points = num_points
    # Test the dataset
    print("\n")
    print("#" * 80)
    print("=" * 80)
    print("Testing CloudPointDataset with parameters:")
    print(f"  Data path:          {data_path}")  # noqa: E501
    print(f"  Num near points:    {num_near_points}")
    print(f"  Num voxel points:   {num_voxel_points}")
    print("=" * 80)
    print("#" * 80)
    print("\n")
    dataset = CloudPointDataset(
        data_path=data_path,
        num_near_points=num_near_points,
        num_voxel_points=num_voxel_points,
    )
    print(len(dataset))
    print("points shape:", dataset[0][0].shape)
    print("labels shape:", dataset[0][1].shape)

    for i in range(10):
        print(dataset[i][0].shape)
        print(dataset[i][1].shape)
        print("-" * 80)
