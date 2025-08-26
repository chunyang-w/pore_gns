"""
Author: Chunyang Wang  # noqa: E501
Github: https://github.com/chunyang-w

A script to pre-compute the velocity dataset for the GNS model.

Example Usage:
===================================
## Downsample the pore wall by 2
===================================
1. Pre-compute 073 dataset
python3 script/pre_compute_vel.py \
    --exp_id 73 \
    --vel_path ../data/Velocity_smooth/073_final.csv \
    --tif_dir ../data/Segmentations/073_segmented_tifs \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif
    --tag 'super-smoothed' \

2. Pre-compute 072 dataset
python3 script/pre_compute_vel.py \
    --exp_id 72 \
    --vel_path ../data/Velocity_smooth/072_final.csv \
    --tif_dir ../data/Segmentations/072 \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --tag 'super-smoothed' \

===================================
## Downsample the pore wall by 8
===================================
1. Pre-compute 073 dataset
python3 script/pre_compute_vel.py \
    --exp_id 73 \
    --vel_path ../data/Velocity_smooth/073_final.csv \
    --tif_dir ../data/Segmentations/073_segmented_tifs \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --ds_patch 8 \
    --tag 'super-smoothed' \

2. Pre-compute 072 dataset
python3 script/pre_compute_vel.py \
    --exp_id 72 \
    --vel_path ../data/Velocity_smooth/072_final.csv \
    --tif_dir ../data/Segmentations/072 \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --ds_patch 8 \
    --tag 'super-smoothed' \


3. Pre-compute 073 dataset - none-smoothed
python3 script/pre_compute_vel.py \
    --exp_id 73 \
    --vel_path ../data/Velocity/073_RobuGlass3_drainage_174nl_min_run5_velocityPoints_surface_masked.csv \
    --tif_dir ../data/Segmentations/073_segmented_tifs \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --ds_patch 8 \
    --tag 'none-smoothed' \

4. Pre-compute 073 dataset - get acceleration
python3 script/pre_compute_vel.py \
    --exp_id 73 \
    --vel_path ../data/Velocity_smooth/073_final.csv \
    --tif_dir ../data/Segmentations/073_segmented_tifs \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --ds_patch 8 \
    --get_acc \
    --tag 'get_acc' \

4. Pre-compute 072 dataset - get acceleration
python3 script/pre_compute_vel.py \
    --exp_id 72 \
    --vel_path ../data/Velocity_smooth/072_final.csv \
    --tif_dir ../data/Segmentations/072 \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --ds_patch 8 \
    --get_acc \
    --tag 'get_acc' \
"""

import os
import torch
import pandas as pd
import argparse
import tifffile


from tqdm import tqdm
from pore_net.dataset import (
    CustomDatasetPredictVelocity_no_synthetic_autoregressive,
)  # noqa: E501
from pore_net.utils import get_stats
from pore_net.utils import extract_patches, down_sample, load_tif

parser = argparse.ArgumentParser()
parser.add_argument("--vel_path", type=str, help="path to the velocity data")
parser.add_argument("--tif_dir", type=str, help="path to the tif directory")
parser.add_argument("--pore_path", type=str, help="path to the pore wall tif file")  # noqa: E501
parser.add_argument(
    "--ds_patch",
    type=int,
    default=2,
    help="downsample factor for pore wall patch",
)
parser.add_argument("--exp_id", type=int, help="experiment id")
parser.add_argument("--tag", default="", type=str, help="tag for the dataset")
parser.add_argument("--get_acc", action="store_true", help="get acceleration")

args = parser.parse_args()

C = 5
radius = 32
autoregressive_step = 5
vel_path = args.vel_path
exp_id = args.exp_id
pore_path = args.pore_path
tif_dir = args.tif_dir
ds_patch = args.ds_patch


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("## Pre-computing Velocity Dataset")
    print("## " + "=" * 76)
    print(f"## Experiment ID:        {exp_id}")
    print(f"## Velocity Data Path:   {vel_path}")
    print(f"## Pore Wall Path:       {pore_path}")
    print(f"## TIF Directory:        {tif_dir}")
    print(f"## Context Size (C):     {C}")
    print(f"## Radius:               {radius}")
    print(f"## Downsample Factor:    {ds_patch}")
    print(f"## Autoregressive Steps: {autoregressive_step}")
    print("## " + "=" * 76)
    print("=" * 80 + "\n")

    data_list = []
    df = pd.read_csv(vel_path)

    # Load the pore mask, downsample the data by 2
    pore_data = tifffile.imread(pore_path)
    pore_data = pore_data[:, 50:-50, 50:-50] // 255
    # pore_data = pore_data[::2, ::2, ::2]
    pore_data = down_sample(pore_data, ds_patch)
    pore_data = torch.tensor(pore_data, dtype=torch.short)
    pore_mask = pore_data == 1

    # Pre-compute the velocity dataset
    dataset = CustomDatasetPredictVelocity_no_synthetic_autoregressive(
        df=df,
        radius=radius,
        C=C,
        autoregressive_step=autoregressive_step,
        get_acc=args.get_acc,
    )
    for t in tqdm(range(10, 188), desc="Pre-computing velocity dataset"):
        data = dataset[t]
        tif_data = load_tif(
            t_idx=t,
            tif_dir=tif_dir,
            pore_mask=pore_mask,
            exp_id=exp_id,
            ds_patch=ds_patch,
        )
        data.image_3D = extract_patches(
            positions=data.x[:, :3],
            tif_data=tif_data,
            ds_patch=ds_patch,
        )
        data_list.append(data)

    mean_std_list = get_stats(data_list)

    # Create output directory based on generation parameters
    output_folder = (
        f"{exp_id:03d}_C{C}_r{radius}_ar{autoregressive_step}_ds{ds_patch}"  # noqa E501
    )
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        output_folder,
    )
    os.makedirs(output_dir, exist_ok=True)

    # Define output file path
    output_file_path = os.path.join(
        output_dir,
        f"{exp_id:03d}_"
        f"{args.tag}_"
        f"ds{ds_patch}_"
        f"{autoregressive_step}_"
        f"autoregressive.pt",
    )
    output_file_path_stat = os.path.join(
        output_dir,
        f"{exp_id:03d}_"
        f"{args.tag}_"
        f"ds{ds_patch}_"
        f"{autoregressive_step}_"
        f"stats.pt",
    )
    # Save the dataset
    torch.save(data_list, output_file_path)
    torch.save(mean_std_list, output_file_path_stat)
    print(f"Dataset saved to: {output_file_path}")
