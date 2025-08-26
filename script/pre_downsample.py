#!/usr/bin/env python3
"""
This script downsamples pore TIF files and stores them as NPY files.
It follows a similar naming convention as the precompute_phy_info.py script.

There is also a switch to determine whether or not to normalise the pixel value
to 0 & 1

----------------------------------------------
Usage:
python script/pre_downsample.py \
    --tif_dir /path/to/tif_directory \
    [--out_dir /path/to/output_directory] \
    --down_factor 8 \
    [--num_workers 4]

----------------------------------------------
Example - 073:
python script/pre_downsample.py \
    --tif_dir ../data/Segmentations/073_segmented_tifs/ \
    --down_factor 8 \
    --num_workers 6

Example - 072:
python script/pre_downsample.py \
    --tif_dir ../data/Segmentations/072/ \
    --down_factor 8 \
    --num_workers 6
"""

import os
import argparse
import numpy as np
from skimage import io
import glob
from natsort import natsorted
import tqdm
import re
from concurrent.futures import ProcessPoolExecutor

from pore_net.utils import down_sample


def process_tif_file(tif_path, down_factor, out_dir, file_index, normalise):
    """
    Process a single TIF file: load it, downsample it, and save as NPY.

    Args:
        tif_path: Path to the TIF file
        down_factor: Downsampling factor
        out_dir: Output directory
        file_index: Index of file in the sorted list (fallback for frame number)
    """
    # Load TIF file
    tif_data = io.imread(tif_path)

    # Downsample the data
    downsampled = down_sample(tif_data, down_factor)
    # Normalize if requested
    if normalise:
        downsampled = downsampled / np.max(downsampled)

    # Get frame number from filename using regex to extract digits
    basename = os.path.basename(tif_path)
    # Find the last sequence of digits before the file extension
    match = re.search(r"(\d+)(?=[^0-9]*\.tif$)", basename)
    if match:
        frame_number = int(match.group(1))
    else:
        # Fallback: use position in sorted list
        frame_number = file_index
        print(
            f"Warning: Could not extract frame number from {basename}, "
            f"using {frame_number}"
        )

    out_filename = os.path.join(out_dir, f"tif_frame_{frame_number}.npy")

    # Save as NPY file
    np.save(out_filename, downsampled)

    return tif_path


def main(args):
    # Get list of TIF files from the directory
    tif_files = natsorted(glob.glob(os.path.join(args.tif_dir, "*.tif")))

    if len(tif_files) == 0:
        print(f"No TIF files found in {args.tif_dir}")
        return

    print(f"Found {len(tif_files)} TIF files in {args.tif_dir}")

    # If no output directory is specified, create one based on the dataset name
    # and downsampling factor
    if not args.out_dir:
        dataset_name = os.path.basename(os.path.normpath(args.tif_dir))
        args.out_dir = f"data/tif_{dataset_name}_down{args.down_factor}"
        print(f"No output directory specified. Using: {args.out_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for i, tif_path in enumerate(tif_files):
            futures.append(
                executor.submit(
                    process_tif_file,
                    tif_path,
                    args.down_factor,
                    args.out_dir,
                    i,
                    args.normalise,
                )
            )

        # Display progress
        for future in tqdm.tqdm(
            futures, desc="Processing TIF files", total=len(futures)
        ):
            # Retrieve result to catch exceptions
            future.result()

    print(f"Downsampling complete. {len(tif_files)} files saved to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample TIF files and save as NPY files."
    )
    parser.add_argument(
        "--tif_dir",
        type=str,
        required=True,
        help="Directory containing TIF files to downsample.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Directory to save the downsampled NPY files. "
        "If not provided, a folder is created automatically.",
    )
    parser.add_argument(
        "--normalise",
        type=bool,
        default=True,
        help="Normalise the pixel values to 0 & 1",
    )
    parser.add_argument(
        "--down_factor",
        type=int,
        default=8,
        help="Downsampling factor to use.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel worker processes to use.",
    )
    args = parser.parse_args()

    # Print the parsed argument table before executing
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    main(args)
