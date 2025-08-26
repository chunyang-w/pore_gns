"""
Code for building cloud point dataset raw format for triplane fitting.

Written by: Chunyang Wang, in collaboration with ChatGPT-o3 mini-high.
GitHub username: chunyang-w

Example usage:
1. Build Cloud Point Dataset for 073 dataset
python pore_net/vecset/build_ae_dataset.py \
    --data_path=../data/Segmentations/073_segmented_tifs/ \
    --out_folder=./data \
    --near_dist_thresh=5 \
    --down_factor=10 \

2. Build Cloud Point Dataset for 073 dataset - ds = 8
python pore_net/vecset/build_ae_dataset.py \
    --data_path=../data/Segmentations/073_segmented_tifs/ \
    --out_folder=./data \
    --near_dist_thresh=5 \
    --down_factor=8 \
"""
import os
import glob
import shutil
import argparse
import numpy as np

from skimage import io
from natsort import natsorted
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes

parser = argparse.ArgumentParser(
    description="Process dataset and file parameters."
)  # noqa

# Dataset parameters
parser.add_argument(
    "--pad_size",
    type=int,
    default=20,
    help="Padding size",
)  # noqa
parser.add_argument(
    "--pad_val",
    type=int,
    default=0,
    help="Padding value",
)  # noqa
parser.add_argument(
    "--down_factor",
    type=int,
    default=10,
    help="Downsampling factor",
)  # noqa
parser.add_argument(
    "--near_dist_thresh",
    type=float,
    default=5,
    help="Near distance threshold",
)  # noqa

# File parameters
parser.add_argument(
    "--data_path",
    type=str,
    default="../data/CombinedResults/Segmentations/073_segmented_tifs/",
    help="Path to data",
)  # noqa
parser.add_argument(
    "--out_folder",
    type=str,
    default="./data/",
    help="Output folder",
)  # noqa

args = parser.parse_args()

# Assign parameters to local variables if needed
pad_size = args.pad_size
pad_val = args.pad_val
down_factor = args.down_factor
near_dist_thresh = args.near_dist_thresh
data_path = args.data_path
out_folder = args.out_folder


dataset_name = (
    "PC_AE_" +
    os.path.basename(os.path.normpath(data_path)) +
    f"_pad_size{pad_size}_down{down_factor}_dist_thresh{near_dist_thresh}"
)

# Helper functions & variables
pad_arr = (
    (pad_size, pad_size),
    (pad_size, pad_size),
    (pad_size, pad_size),
)


def get_vol_points(vol):
    x, y, z = np.indices(vol.shape)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))  # (N, 3)
    labels = vol.ravel().astype(np.uint8)  # (N,)
    return points, labels


def get_near_surf_points(vol, dist_thresh=5):
    fluid_distance = distance_transform_edt(vol)
    # Compute distances
    fluid_distance = distance_transform_edt(vol)
    rock_distance = distance_transform_edt(1 - vol)

    distance_to_interface = np.where(tif, fluid_distance, rock_distance)

    near_interface_mask = distance_to_interface <= dist_thresh

    x, y, z = np.where(near_interface_mask)
    near_surf_points = np.column_stack((x, y, z))  # (N, 3)
    near_surf_labels = tif[x, y, z].astype(np.uint8)  # (N,)
    return near_surf_points, near_surf_labels


def get_surface(vol):
    surface_points, _, _, _ = marching_cubes(vol, level=0.5)
    return surface_points


if __name__ == "__main__":
    # Get all tif file's name in the data path
    data_pattern = data_path + "/*.tif"
    data_names = glob.glob(data_pattern)
    data_names = natsorted(data_names)

    # Init output path
    out_path = os.path.join(os.path.abspath(out_folder), dataset_name)

    Robustness check
    if os.path.exists(out_path):
        print(
            f"\n[Warning] Output path {out_path} already exists. Press [y] to overwrite.\n"  # noqa
        )  # noqa
        if input() != "y":
            exit(0)
        else:
            shutil.rmtree(out_path)
            print("\n[Info] Originl folder cleared. \n")

    os.makedirs(out_path, exist_ok=True)

    # Process each tif file
    for data_name in tqdm(data_names):
        tif = io.imread(data_name)  # Read tif file
        tif = np.pad(  # Pad the tif file
            tif,
            pad_width=pad_arr,
            mode="constant",
            constant_values=pad_val,
        )
        tif = tif[
            ::down_factor,
            ::down_factor,
            ::down_factor,
        ]
        # print("tif.shape", tif.shape)
        # exit(0)
        # Get volume points & labels
        points, labels = get_vol_points(tif)

        # Get near surface points and labels
        points_near, labels_near = get_near_surf_points(tif, near_dist_thresh)

        # Get surface points
        surface_points = get_surface(tif)

        # Save the points and labels
        name = os.path.join(
            out_path,
            os.path.basename(data_name).replace(".tif", ".npz"),
        )
        np.savez(
            name,
            points_vol=points,
            labels_vol=labels,
            points_near=points_near,
            labels_near=labels_near,
            surface_points=surface_points,
        )
