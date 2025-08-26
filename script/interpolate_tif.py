"""
python script/interpolate_tif.py \
  --input_dir /gpfs/home/cw1722/particle/pore_unet/data/tif_072_down8 \
  --out_dir /gpfs/home/cw1722/particle/pore_unet/data/tif_072_down8_interpolated \
  --num_interpolations 1

"""

import numpy as np
from scipy.ndimage import distance_transform_cdt
import os
import argparse
import glob
from natsort import natsorted, ns
import tqdm
import re


from scipy.ndimage import distance_transform_edt


def interpolate(V0, V1, t=0.5):
    sdf0 = distance_transform_edt(V0) - distance_transform_edt(~V0)
    sdf1 = distance_transform_edt(V1) - distance_transform_edt(~V1)
    sdf_t = (1 - t) * sdf0 + t * sdf1
    # V_t = (sdf_t >= 0).astype(np.uint8)
    V_t = (sdf_t >= -0.25).astype(np.uint8)
    return V_t


# def interpolate(V0, V1, t=0.5):
#     sdf0 = distance_transform_cdt(V0, metric="chessboard") - distance_transform_cdt(  # noqa: E501
#         ~V0, metric="chessboard"
#     )
#     sdf1 = distance_transform_cdt(V1, metric="chessboard") - distance_transform_cdt(  # noqa: E501
#         ~V1, metric="chessboard"
#     )
#     sdf_t = (1 - t) * sdf0 + t * sdf1
#     V_t = (sdf_t >= 0).astype(np.uint8)
#     return V_t


def interpolate_frame_pair(
    frame1_path, frame2_path, t, out_dir, frame1_idx, frame2_idx
):
    """
    Interpolate between two frames and save the result.

    Args:
        frame1_path: Path to the first frame
        frame2_path: Path to the second frame
        t: Interpolation parameter (0-1)
        out_dir: Output directory
        frame1_idx: Index of the first frame
        frame2_idx: Index of the second frame
    """
    # Load frames
    frame1 = np.load(frame1_path)  # Frames are stored with shape (D, H, W)
    frame2 = np.load(frame2_path)  # Extract the first dimension

    # Convert to binary if needed
    frame1_binary = frame1.astype(bool)
    frame2_binary = frame2.astype(bool)

    # Interpolate
    interpolated = interpolate(frame1_binary, frame2_binary)
    print("frame1_idx", frame1_idx)
    print("frame2_idx", frame2_idx)
    # New frame index - calculate based on original frames with decimal point
    interpolated_idx = frame1_idx + (frame2_idx - frame1_idx) * t

    # Format name with high precision to ensure correct ordering
    out_filename = os.path.join(out_dir, f"tif_frame_{interpolated_idx:.1f}.npy")

    # Save as NPY with shape (D, H, W) to match original format
    interpolated = interpolated.astype(np.float32)
    np.save(out_filename, interpolated)

    return out_filename


def rename_and_reorder_files(directory):
    """
    Rename all files in the directory to sequential integers.

    Args:
        directory: Directory containing the files to rename
    """
    # Get all NPY files
    files = natsorted(glob.glob(os.path.join(directory, "*.npy")), alg=ns.FLOAT)

    # Create a temporary directory for staging
    temp_dir = os.path.join(directory, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    print("natsorted:", files)

    # Move files to temp with new sequential names
    for i, file_path in enumerate(files):
        new_name = f"tif_frame_{i}.npy"
        new_path = os.path.join(temp_dir, new_name)
        os.rename(file_path, new_path)

    # Move files back to original directory
    temp_files = glob.glob(os.path.join(temp_dir, "*.npy"))
    for file_path in temp_files:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(directory, file_name)
        os.rename(file_path, dest_path)

    # Remove temporary directory
    os.rmdir(temp_dir)

    print(f"Renamed {len(files)} files to sequential indices")


def extract_frame_number(filename):
    """
    Extract the frame number from a filename.

    Args:
        filename: Filename to extract frame number from

    Returns:
        float: Frame number
    """
    # Extract the number after "tif_frame_" and before ".npy"
    match = re.search(r"tif_frame_(\d+(?:\.\d+)?)", filename)
    if match:
        return float(match.group(1))
    return 0  # Default if no match


def main(args):
    # Get list of NPY files from the directory
    npy_files = natsorted(glob.glob(os.path.join(args.input_dir, "*.npy")))

    if len(npy_files) == 0:
        print(f"No NPY files found in {args.input_dir}")
        return

    print(f"Found {len(npy_files)} NPY files in {args.input_dir}")

    # If no output directory is specified, create one based on the input directory  # noqa: E501
    if not args.out_dir:
        # input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
        args.out_dir = f"{args.input_dir}_interpolated"
        print(f"No output directory specified. Using: {args.out_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Extract frame numbers for each file
    frame_indices = []
    for file_path in npy_files:
        basename = os.path.basename(file_path)
        frame_indices.append(extract_frame_number(basename))

    # Create pairs of frames for interpolation
    frame_pairs = []
    interpolated_count = 0

    for i in range(len(npy_files) - 1):
        # Copy original frames to output directory
        frame1_path = npy_files[i]
        frame2_path = npy_files[i + 1]
        frame1_idx = frame_indices[i]
        frame2_idx = frame_indices[i + 1]

        frame1_dest = os.path.join(args.out_dir, os.path.basename(frame1_path))
        if not os.path.exists(frame1_dest):
            import shutil

            shutil.copy(frame1_path, frame1_dest)

        # Last frame will be copied after the loop

        # Create interpolation tasks
        for t_idx in range(1, args.num_interpolations + 1):
            t = t_idx / (args.num_interpolations + 1)
            frame_pairs.append(
                (
                    frame1_path,
                    frame2_path,
                    t,
                    args.out_dir,
                    frame1_idx,
                    frame2_idx,
                )  # noqa: E501
            )
            interpolated_count += 1

    # Copy the last frame
    last_frame_path = npy_files[-1]
    last_frame_dest = os.path.join(
        args.out_dir, os.path.basename(last_frame_path)
    )  # noqa: E501
    if not os.path.exists(last_frame_dest):
        import shutil

        shutil.copy(last_frame_path, last_frame_dest)

    # Process frame pairs sequentially with progress bar
    for i, (
        frame1_path,
        frame2_path,
        t,
        out_dir,
        frame1_idx,
        frame2_idx,
    ) in enumerate(tqdm.tqdm(frame_pairs, desc="Interpolating frames")):
        # print(frame1_path, frame2_path, t, out_dir, frame1_idx, frame2_idx)
        interpolate_frame_pair(
            frame1_path,
            frame2_path,
            t,
            out_dir,
            frame1_idx,
            frame2_idx,
        )

    print(f"Interpolation complete. {interpolated_count} new frames created.")

    if args.sequential_names:
        print("Renaming files to sequential indices...")
        rename_and_reorder_files(args.out_dir)

    print(f"All frames saved to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpolate between frames in a sequence."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/gpfs/home/cw1722/particle/pore_unet/data/tif_072_down8",
        help="Directory containing NPY files to interpolate.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Directory to save the interpolated frames. "
        "If not provided, a folder is created automatically.",
    )
    parser.add_argument(
        "--num_interpolations",
        type=int,
        default=1,
        help="Number of frames to interpolate between each original pair.",
    )
    parser.add_argument(
        "--sequential_names",
        action="store_true",
        default=True,
        help="Rename all files to sequential integers after interpolation.",
    )
    args = parser.parse_args()

    # Print the parsed argument table before executing
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    main(args)
