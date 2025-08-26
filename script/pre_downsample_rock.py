#!/usr/bin/env python3
"""
This script downsamples rock TIF files and stores them as NPY files.
Rock TIF files represent the static structure of porous media.

----------------------------------------------
Usage:
python script/pre_downsample_rock.py \
    --rock_tif_path /path/to/rock.tif \
    [--out_dir /path/to/output_directory] \
    --down_factor 8

----------------------------------------------
Example:
python script/pre_downsample_rock.py \
    --rock_tif_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --down_factor 8
"""

import os
import argparse
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

from pore_net.utils import down_sample


def downsample_rock_tif(rock_tif_path, down_factor, out_dir, visualize=False):
    """
    Downsample a rock TIF file and save as NPY.

    Args:
        rock_tif_path: Path to the rock TIF file
        down_factor: Downsampling factor
        out_dir: Output directory
        visualize: Whether to visualize slices before and after downsampling
    """
    print(f"Loading rock TIF file: {rock_tif_path}")

    # Load rock TIF file
    rock_tif = io.imread(rock_tif_path)

    # Binarize if needed (assuming rock is represented as binary values)
    rock_binary = (rock_tif // 255) == 1

    print(f"Original rock TIF shape: {rock_tif.shape}")

    # Downsample the data
    rock_downsampled = down_sample(rock_binary, down_factor)

    print(f"Downsampled rock shape: {rock_downsampled.shape}")

    # Get name for the output file
    rock_name = os.path.splitext(os.path.basename(rock_tif_path))[0]
    out_filename = os.path.join(out_dir, f"rock_{rock_name}_down{down_factor}.npy")

    # Save as NPY file
    np.save(out_filename, rock_downsampled)
    print(f"Saved downsampled rock to: {out_filename}")

    if visualize:
        visualize_slices(rock_binary, rock_downsampled)

    return out_filename


def visualize_slices(original, downsampled, num_slices=3):
    """
    Visualize slices from the original and downsampled rock volumes.

    Args:
        original: Original rock volume
        downsampled: Downsampled rock volume
        num_slices: Number of slices to visualize
    """
    # Get slice indices evenly distributed through the volume
    indices = [
        i * original.shape[0] // (num_slices + 1) for i in range(1, num_slices + 1)
    ]

    # Create a figure with subplots
    fig, axes = plt.subplots(num_slices, 2, figsize=(10, 4 * num_slices))

    for i, idx in enumerate(indices):
        # Calculate corresponding index in downsampled volume
        ds_idx = idx // downsampled.shape[0]

        # Original slice
        axes[i, 0].imshow(original[idx], cmap="gray")
        axes[i, 0].set_title(f"Original Slice {idx}")
        axes[i, 0].axis("off")

        # Downsampled slice
        axes[i, 1].imshow(downsampled[ds_idx], cmap="gray")
        axes[i, 1].set_title(f"Downsampled Slice {ds_idx}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


def main(args):
    # Create output directory if it doesn't exist
    if not args.out_dir:
        # Create directory based on the rock file name
        rock_name = os.path.splitext(os.path.basename(args.rock_tif_path))[0]
        args.out_dir = f"data/rock_{rock_name}_down{args.down_factor}"
        print(f"No output directory specified. Using: {args.out_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Process the rock TIF file
    downsample_rock_tif(
        args.rock_tif_path,
        args.down_factor,
        args.out_dir,
        args.visualize,
    )

    print("Rock downsampling complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample rock TIF files and save as NPY files."
    )
    parser.add_argument(
        "--rock_tif_path",
        type=str,
        required=True,
        help="Path to the rock TIF file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Directory to save the downsampled NPY file. "
        "If not provided, a folder is created automatically.",
    )
    parser.add_argument(
        "--down_factor",
        type=int,
        default=8,
        help="Downsampling factor to use.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize slices before and after downsampling.",
    )
    args = parser.parse_args()

    # Print the parsed argument table before executing
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    main(args)
