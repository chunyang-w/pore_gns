#!/usr/bin/env python3
"""
This script precomputes physics information from a velocity CSV,
applies max pooling to the resulting grid, and stores the pooled grids
to a specified directory.

It implements a max pooling function in NumPy that groups the grid into
blocks and takes the maximum over each block. It also follows the logic
from your dataset class to assign vx, vy, and vz values to a grid.

----------------------------------------------
Usage:
python script/precompute_phy_info.py \
    --vel_path /path/to/velocity.csv \
    [--out_dir /path/to/output_directory] \
    --grid_size 1228 1566 1566 \
    --pool_factor 8 \
    [--vx_key vx] [--vy_key vy] [--vz_key vz] \
    [--num_workers 4]

----------------------------------------------
particle

Create for 073:
python script/precompute_phy_info.py \
    --vel_path ../data/Velocity_smooth/073_final.csv \
    --grid_size 1228 1566 1566 \
    --pool_factor=8 \
    --off_set 50 50 0 \

Create for 072:
python script/precompute_phy_info.py \
    --vel_path ../data/Velocity_smooth/072_final.csv \
    --grid_size 718 1566 1566 \
    --pool_factor 8 \
    --off_set 50 50 -460 \

prediction
Create for 073
python script/precompute_phy_info.py \
    --vel_path ../data/Velocity_pred/073_autoregressive_5noise_predictions.csv \  # noqa E501
    --grid_size 1228 1566 1566 \
    --pool_factor 8 \
    --off_set 50 50 0 \

Create for 072:
python script/precompute_phy_info.py \
    --vel_path ../data/Velocity_pred/072_autoregressive_noise5_predictions.csv \  # noqa E501
    --grid_size 718 1566 1566 \
    --pool_factor 8 \
    --off_set 50 50 -460 \
    --num_workers 6
"""

import os
import argparse
import numpy as np
import pandas as pd
import tqdm
from concurrent.futures import ProcessPoolExecutor
from pore_net.utils import process_frame, max_pool3d_numpy


def process_and_save_frame(frame, df, grid_size, args):
    """
    Process a single frame: compute its physics grid, apply max pooling,
    and save to disk. This function is intended to be called in parallel.
    """
    grid = process_frame(
        df,
        frame,
        grid_size,
        args.x_key,
        args.y_key,
        args.z_key,
        args.frame_key,
        args.vx_key,
        args.vy_key,
        args.vz_key,
        args.off_set,
    )
    pooled_grid = max_pool3d_numpy(grid, args.pool_factor)
    filename = os.path.join(args.out_dir, f"phy_info_frame_{frame}.npy")
    np.save(filename, pooled_grid)
    return frame  # Return frame for progress reporting


def main(args):
    # Load velocity data from CSV.
    df = pd.read_csv(args.vel_path)

    # If no output directory is specified, create one based on the dataset name, downsampling factor, and pooling method.  # noqa E501
    if not args.out_dir:
        dataset_name = os.path.splitext(os.path.basename(args.vel_path))[0]
        args.out_dir = f"data/{dataset_name}_down{args.pool_factor}_maxpool"
        print(f"No output directory specified. Using: {args.out_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    grid_size = tuple(args.grid_size)
    frames = sorted(df[args.frame_key].unique())
    print(f"Found {len(frames)} unique frames in the CSV.")

    # Use ProcessPoolExecutor to process frames in parallel.
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for frame in frames:
            futures.append(
                executor.submit(
                    process_and_save_frame, frame, df, grid_size, args
                )  # noqa E501
            )
        # Optionally, display a progress bar.
        for future in tqdm.tqdm(
            futures, desc="Processing frames", total=len(futures)
        ):  # noqa E501
            # Retrieve result to catch exceptions.
            future.result()

    print(
        f"Precomputation complete. Pooled physics info saved in {args.out_dir}"
    )  # noqa E501


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute physics info from velocity CSV, apply max pooling, and save to disk."  # noqa E501
    )
    parser.add_argument(
        "--vel_path",
        type=str,
        required=True,
        help="Path to the velocity CSV file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Directory to save the pooled physics info files. If not provided, a folder is created automatically.",  # noqa E501
    )
    parser.add_argument(
        "--frame_key",
        type=str,
        default="frame",
        help="Column name for frame in CSV.",
    )
    parser.add_argument(
        "--x_key",
        type=str,
        default="x",
        help="Column name for x coordinate in CSV.",
    )
    parser.add_argument(
        "--y_key",
        type=str,
        default="y",
        help="Column name for y coordinate in CSV.",
    )
    parser.add_argument(
        "--z_key",
        type=str,
        default="z",
        help="Column name for z coordinate in CSV.",
    )
    parser.add_argument(
        "--vx_key",
        type=str,
        default="vx",
        help="Column name for x-velocity in CSV.",
    )
    parser.add_argument(
        "--vy_key",
        type=str,
        default="vy",
        help="Column name for y-velocity in CSV.",
    )
    parser.add_argument(
        "--vz_key",
        type=str,
        default="vz",
        help="Column name for z-velocity in CSV.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs=3,
        default=[1228, 1566, 1566],
        help="Grid size as three integers, e.g. 1228 1566 1566.",
    )
    parser.add_argument(
        "--off_set",
        type=int,
        nargs=3,
        default=[50, 50, 0],
        help="particle position offset arr",
    )
    parser.add_argument(
        "--pool_factor",
        type=int,
        default=8,
        help="Downsampling factor to use for max pooling.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel worker processes to use.",
    )
    args = parser.parse_args()

    # Print the parsed argument table before executing

    main(args)
