"""
Code for building cloud point dataset raw format for triplane fitting.

Written by: Chunyang Wang, in collaboration with ChatGPT-o3 mini-high.
GitHub username: chunyang-w

Example usage:
1. Build Cloud Point Dataset for 073 dataset
python pore_net/triplane/build_cloud_point.py \
    --data_path ../data/Segmentations/073_segmented_tifs/ \
    --out_folder ./data/ \
    --pad_size 0 \
    --down_factor 6 \
    --near_dist_thresh 3 \
    --box_size 204,261,261 \

2. Build Cloud Point Dataset for 073 dataset - less downsampling -
more accurate surface points

python pore_net/triplane/build_cloud_point.py \
    --data_path ../data/Segmentations/073_segmented_tifs/ \
    --out_folder ./data/ \
    --pad_size 0 \
    --down_factor 4 \
    --near_dist_thresh 3 \
    --box_size 307,391,391 \

3. Build Cloud Point Dataset for 073 dataset - standard downsampling - 8
for comparison with UNet

python pore_net/triplane/build_cloud_point.py \
    --data_path ../data/Segmentations/073_segmented_tifs/ \
    --out_folder ./data/ \
    --pad_size 0 \
    --down_factor 8 \
    --near_dist_thresh 3 \
    --box_size 153,195,195 \

4. Build Cloud Point Dataset for 072 dataset - standard downsampling - 8

python pore_net/triplane/build_cloud_point.py \
    --data_path ../data/Segmentations/072/ \
    --out_folder ./data/ \
    --pad_size 0 \
    --down_factor 8 \
    --near_dist_thresh 3 \
    --box_size 89,195,195 \
    --exp_id 72 \
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

from pore_net.utils import down_sample

parser = argparse.ArgumentParser(
    description="Process dataset and file parameters."
)  # noqa

# Dataset parameters
parser.add_argument(
    "--pad_size",
    type=int,
    default=0,
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
    default=8,
    help="Downsampling factor",
)  # noqa
parser.add_argument(
    "--near_dist_thresh",
    type=float,
    default=2,
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
parser.add_argument(
    "--box_size",
    type=str,
    default="153,195,195",
    help="Box size in format 'z, y, x'",
)
parser.add_argument(
    "--exp_id",
    type=int,
    default=73,
    help="Experiment ID",
)  # noqa

args = parser.parse_args()

# Assign parameters to local variables if needed
pad_size = args.pad_size
pad_val = args.pad_val
down_factor = args.down_factor
near_dist_thresh = args.near_dist_thresh
data_path = args.data_path
out_folder = args.out_folder
box_size = tuple(map(int, args.box_size.split(",")))

dataset_name = (
    "PC_"
    + os.path.basename(os.path.normpath(data_path))
    + "_"
    + f"pad_{pad_size}_ds{down_factor}"
    f"near_thresh{near_dist_thresh}"
)

# Helper functions & variables
pad_arr = (
    (pad_size, pad_size),
    (pad_size, pad_size),
    (pad_size, pad_size),
)


def get_vol_points(vol, exp_id=73):
    x, y, z = np.indices(vol.shape)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))  # (N, 3)
    labels = vol.ravel().astype(np.uint8)  # (N,)
    if exp_id == 72:
        labels = labels // 255
    return points, labels


def get_near_surf_points(vol, dist_thresh=2, exp_id=73):
    fluid_distance = distance_transform_edt(vol)
    # Compute distances
    fluid_distance = distance_transform_edt(vol)
    rock_distance = distance_transform_edt(1 - vol)

    distance_to_interface = np.where(vol, fluid_distance, rock_distance)

    near_interface_mask = distance_to_interface <= dist_thresh

    x, y, z = np.where(near_interface_mask)
    near_surf_points = np.column_stack((x, y, z))  # (N, 3)
    near_surf_labels = vol[x, y, z].astype(np.uint8)  # (N,)
    if exp_id == 72:
        near_surf_labels = near_surf_labels // 255
    return near_surf_points, near_surf_labels


def get_surface(vol, exp_id=73):
    if exp_id == 72:
        vol = vol // 255
    surface_points, _, _, _ = marching_cubes(vol, level=0.5)
    return surface_points


def normalize_points(points, box_size):
    """
    Normalize points to range [-1, 1] based on box dimensions.

    Args:
        points: numpy array of shape (N, 3) containing point coordinates
        box_size: tuple of (z, y, x) dimensions of the box

    Returns:
        Normalized points in range [-1, 1]
    """
    normalized_points = points.copy().astype(np.float32)
    for i in range(3):
        normalized_points[:, i] = 2 * (points[:, i] / (box_size[i] - 1)) - 1
    return normalized_points


if __name__ == "__main__":
    print("\n")
    print("#" * 80)
    print("=" * 80)
    print("Building point cloud dataset with parameters:")
    print(f"  Data path:          {data_path}")
    print(f"  Output folder:      {out_folder}")
    print(f"  Padding size:       {pad_size}")
    print(f"  Padding value:      {pad_val}")
    print(f"  Downsampling:       {down_factor}")
    print(f"  Near dist thresh:   {near_dist_thresh}")
    print(f"  Box size:           {box_size}")
    print("=" * 80)
    print("#" * 80)
    print("\n")
    # Get all tif file's name in the data path
    exp_id = args.exp_id
    data_pattern = data_path + "*.tif"
    data_names = glob.glob(data_pattern)
    data_names = natsorted(data_names)

    print(f"Found {len(data_names)} tif files in {data_path}...")

    # Init output path
    out_path = os.path.join(os.path.abspath(out_folder), dataset_name)

    # Robustness check
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
    for data_name in tqdm(data_names, desc="Processing files"):
        tif = io.imread(data_name)  # Read tif file
        tif = np.pad(  # Pad the tif file
            tif,
            pad_width=pad_arr,
            mode="constant",
            constant_values=pad_val,
        )
        tif = down_sample(tif, down_factor)

        # Get volume points & labels
        points, labels = get_vol_points(tif, exp_id)
        points = normalize_points(points, box_size)

        # Get near surface points and labels
        points_near, labels_near = get_near_surf_points(
            tif, near_dist_thresh, exp_id
        )  # noqa: E501
        points_near = normalize_points(points_near, box_size)

        # Get surface points
        surface_points = get_surface(tif, exp_id)
        surface_points = normalize_points(surface_points, box_size)

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
