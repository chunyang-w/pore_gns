"""
Dataset implementation for the UNET model - this dataset is used to test the
reconstruction performance of the UNET model. Only shape info is encoded in the
dataset, no velocity info is used. The dataset returns the voxelized
representation of the same frame for a tif file when accessed by the index.

Example usage (test dataset):

python pore_net/unet_recon/dataset.py
"""

import glob
import os
import numpy as np

from torch.utils.data import Dataset
from pore_net.utils import (
    down_sample,
)  # ensure this function is available
from natsort import natsorted


class PoreShapeDatasetUnet(Dataset):
    def __init__(
        self,
        data_dir,
        frame_idxs,
        transform=None,
    ):
        # fame_idxs is an list containing the index of the items to be selected
        self.frame_idxs = frame_idxs
        self.transform = transform

        # Load velocity (npy) and tif file paths
        self.tif_names = natsorted(glob.glob(os.path.join(data_dir, "*.npy")))
        self.tif_names = [self.tif_names[i] for i in frame_idxs]
        assert len(self.tif_names) == len(
            frame_idxs
        ), "Mismatch between number of frames and TIF files"  # noqa: E501
        print(f"Loaded {len(self.tif_names)} TIF files")
        print("fist 3 tif names: ", self.tif_names[:3])
        print("last 3 tif names: ", self.tif_names[-3:])

        print("dataset init done.")

    def down_sample(self, cube, factor):
        return down_sample(cube, factor)

    def __len__(self):
        # Assuming two frames per data sample
        return len(self.tif_names)

    def __getitem__(self, idx):
        # Get the input and output frame indices
        voxel_data = np.load(self.tif_names[idx])
        voxel_data = voxel_data.astype(np.float32)
        voxel_data = voxel_data[None, :, :, :]  # Add channel axis

        return voxel_data


if __name__ == "__main__":
    dataset = PoreShapeDatasetUnet(
        data_dir="./data/tif_073_segmented_tifs_down8/",
        frame_idxs=[i for i in range(150)],
    )
    print(len(dataset))
    for i in range(10):
        print(dataset[i].shape)
