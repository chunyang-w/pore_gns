#!/usr/bin/env python3
"""
This script benchmarks a 3D U-Net model using the PoreShapeDataset and calculates dice scores.
Training details (losses, scores, etc.) are logged to the console, a log file,
and to Weights & Biases (wandb).


# Train with 1 input frames and 1 output frame.
python pore_net/unet_recon/bench.py \
  --data_dir ./data/tif_073_segmented_tifs_down8/ \
  --checkpoint_path ./out/UNET_recon_06-13_17:09:03_tif_073_segmented_tifs_down8_bs2_lr0.001_ep100/checkpoint_epoch_100.pth \
  --frame_start 150 \
  --frame_end 200 \
"""

import os
import argparse
import torch
from tqdm import tqdm
import pandas as pd
import wandb

from pore_net.unet_recon.dataset import PoreShapeDatasetUnet
from pore_net.unet import PoreScaleUNet
from pore_net.utils import get_dice_score
from datetime import datetime

n_channel_in = 1
n_channel_out = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Create CSV path in the same directory as checkpoint
    checkpoint_base_dir = os.path.dirname(args.checkpoint_path)
    checkpoint_base_dir = os.path.abspath(checkpoint_base_dir)
    csv_path = os.path.join(checkpoint_base_dir, "dice_scores.csv")
    print("checkpoint_base_dir", checkpoint_base_dir)

    # Use only the last folder of the data_dir to create a concise experiment name  # noqa E501
    dataset = PoreShapeDatasetUnet(
        data_dir=args.data_dir,
        frame_idxs=[i for i in range(args.frame_start, args.frame_end)],
    )

    model = PoreScaleUNet(n_channels=n_channel_in, n_classes=n_channel_out)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Initialize list to store results
    results = []

    for i in tqdm(range(len(dataset))):
        frame_idx = args.frame_start + i
        frame = dataset[i]
        frame = torch.from_numpy(frame).to(device).unsqueeze(0)

        with torch.no_grad():
            output = model(frame)

        # Convert to numpy for dice calculation
        pred = (output > 0.5).cpu().numpy().squeeze(0)
        gt = frame.cpu().numpy().squeeze(0)

        print("pred.shape", pred.shape)
        print("gt.shape", gt.shape)

        dice_score = get_dice_score(pred, gt)
        print(f"Frame {frame_idx} - Dice score: {dice_score}")

        # Append results
        results.append({"data_idx": frame_idx, "dice_score": dice_score})

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark 3D U-Net Model and calculate dice scores."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--frame_start",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--frame_end",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    main(args)
