#!/usr/bin/env python3
"""
This script trains a 3D U-Net model using the PoreShapeDataset.
Training details (losses, scores, etc.) are logged to the console, a log file,
and to Weights & Biases (wandb).


# Train with 2 input frames and 1 output frame.
python script/train_unet.py \
  --data_dir ./data/tif_073_segmented_tifs_down8/ \
  --vel_path ./data/073_final_down8_maxpool/ \
  --num_frames_in 2 \
  --num_frames_out 1 \
  --num_epochs 100 \
  --learning_rate 1e-3 \
  --fp16 \
  --batch_size 2 \
  --num_workers 4 \

# Train with 3 input frames and 1 output frame.
python script/train_unet.py \
  --data_dir ./data/tif_073_segmented_tifs_down8/ \
  --vel_path ./data/073_final_down8_maxpool/ \
  --num_frames_in 3 \
  --num_frames_out 1 \
  --num_epochs 100 \
  --learning_rate 1e-3 \
  --fp16 \
  --batch_size 2 \
  --num_workers 4 \

# Train with 4 input frames and 1 output frame.
python script/train_unet.py \
  --data_dir ./data/tif_073_segmented_tifs_down8/ \
  --vel_path ./data/073_final_down8_maxpool/ \
  --num_frames_in 4 \
  --num_frames_out 1 \
  --num_epochs 100 \
  --learning_rate 1e-3 \
  --fp16 \
  --batch_size 2 \
  --num_workers 4 \

# Train with 5 input frames and 1 output frame.
python script/train_unet.py \
  --data_dir ./data/tif_073_segmented_tifs_down8/ \
  --vel_path ./data/073_final_down8_maxpool/ \
  --num_frames_in 5 \
  --num_frames_out 1 \
  --num_epochs 100 \
  --learning_rate 1e-3 \
  --fp16 \
  --batch_size 2 \
  --num_workers 4 \

"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import wandb
import numpy as np

from pore_net.dataset import PoreShapeDataset
from pore_net.unet_recon import PoreScaleUNet

# Import AMP utilities for mixed precision training.
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime


def setup_logger(log_file):
    """
    Set up a logger to output messages to both console and a file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # Add handlers if not already added
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    else:
        logger.handlers.clear()
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    save_path,
    logger,
    use_fp16=False,  # new parameter to enable FP16 training
):
    model.to(device)
    best_val_loss = float("inf")

    # Initialize GradScaler if FP16 is enabled and device is CUDA.
    scaler = GradScaler() if use_fp16 and device.type == "cuda" else None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(
            train_loader,
            desc=f"Training Epoch {epoch+1}/{num_epochs}",
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if use_fp16 and device.type == "cuda":
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        dice_score = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if use_fp16 and device.type == "cuda":
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                # Use sigmoid to convert logits to probabilities.
                pred = (torch.sigmoid(outputs) > 0.5).float()
                dice_score += (2.0 * (pred * targets).sum()) / (
                    (pred + targets).sum() + 1e-8
                )

        val_loss /= len(val_loader.dataset)
        dice_score /= len(val_loader)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice Score: {dice_score:.4f}"  # noqa E501
        )

        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Dice Score": dice_score,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(
                os.path.dirname(save_path),
                f"checkpoint_epoch_{epoch+1}.pth",
            )
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")


def calculate_velocity_stats(vel_path):
    """Calculate velocity statistics from all velocity files"""
    vel_files = sorted([f for f in os.listdir(vel_path) if f.endswith(".npy")])
    all_velocities = []

    for vel_file in vel_files:
        velocity = np.load(os.path.join(vel_path, vel_file))
        # Only consider non-zero velocities
        valid_velocities = velocity[velocity != 0]
        if len(valid_velocities) > 0:
            all_velocities.extend(valid_velocities)

    if not all_velocities:
        return None

    all_velocities = np.array(all_velocities)
    return {
        "min": np.min(all_velocities),
        "max": np.max(all_velocities),
    }


def main(args):
    # Use only the last folder of the data_dir to create a concise experiment name  # noqa E501
    timestamp = datetime.now().strftime("%m-%d_%H:%M:%S")
    data_dir_name = os.path.basename(os.path.normpath(args.data_dir))
    experiment_name = f"{data_dir_name}_bs{args.batch_size}_lr{args.learning_rate}_ep{args.num_epochs}"  # noqa E501
    experiment_name = f"{timestamp}_{experiment_name}"
    if args.fp16:
        experiment_name += "_fp16"
    if args.normalize_velocity:
        experiment_name += "_normvel"
    # Add frame information to experiment name
    experiment_name += f"_in{args.num_frames_in}_out{args.num_frames_out}"

    # Create a folder under 'out' with the experiment name
    experiment_folder = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "out",
        experiment_name,
    )
    os.makedirs(experiment_folder, exist_ok=True)

    # Update save_path and log_file to be inside the experiment folder
    args.save_path = os.path.join(experiment_folder, "best_model.pth")
    args.log_file = os.path.join(experiment_folder, "training_log.txt")

    # Initialize wandb with the experiment name and configuration
    wandb.init(
        project="porescaleunet",
        name=experiment_name,
        config=vars(args),
    )  # noqa E501

    logger = setup_logger(args.log_file)
    logger.info("Starting training with the following arguments:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    # Calculate velocity statistics if normalization is enabled
    velocity_stats = None
    if args.normalize_velocity:
        logger.info("Calculating velocity statistics...")
        velocity_stats = calculate_velocity_stats(args.vel_path)
        if velocity_stats is None:
            logger.warning(
                "No valid velocity measurements found. Disabling velocity normalization."  # noqa E501
            )
            args.normalize_velocity = False
        else:
            logger.info(
                f"Velocity statistics: min={velocity_stats['min']:.4f}, max={velocity_stats['max']:.4f}"  # noqa E501
            )

    train_dataset = PoreShapeDataset(
        data_dir=args.data_dir,
        vel_path=args.vel_path,
        frame_idxs=[i for i in range(148)],
        rock_tif_path=args.rock_tif_path,
        rock_down_factor=args.down_factor,
        use_fp16=args.fp16,
        normalize_velocity=args.normalize_velocity,
        velocity_stats=velocity_stats,
        n_frames_in=args.num_frames_in,
        n_frames_out=args.num_frames_out,
    )

    val_dataset = PoreShapeDataset(
        data_dir=args.data_dir,
        vel_path=args.vel_path,
        frame_idxs=[i for i in range(148, 197)],
        rock_tif_path=args.rock_tif_path,
        rock_down_factor=args.down_factor,
        use_fp16=args.fp16,
        normalize_velocity=args.normalize_velocity,
        velocity_stats=velocity_stats,
        n_frames_in=args.num_frames_in,
        n_frames_out=args.num_frames_out,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    channel_per_frame = 3 + 1
    n_channel_in = args.num_frames_in * channel_per_frame + (
        1 if args.rock_tif_path else 0
    )
    n_channel_out = args.num_frames_out
    model = PoreScaleUNet(n_channels=n_channel_in, n_classes=n_channel_out)
    # Use BCEWithLogitsLoss which is safe for autocast.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=args.num_epochs,
        save_path=args.save_path,
        logger=logger,
        use_fp16=args.fp16,  # use FP16 training if specified
    )

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train 3D U-Net Model for PoreScale data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/tif_073_segmented_tifs_down8/",
    )
    parser.add_argument(
        "--vel_path",
        type=str,
        default="./data/073_final_down8_maxpool/",
    )
    parser.add_argument(
        "--rock_tif_path",
        type=str,
        default="../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif",  # noqa E501
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--down_factor", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--num_frames_in",
        type=int,
        default=2,
        help="Number of input frames per sample",
    )
    parser.add_argument(
        "--num_frames_out",
        type=int,
        default=1,
        help="Number of output frames per sample",
    )
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument(
        "--log_file",
        type=str,
        default="training_log.txt",
        help="Path to the log file.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 training",
    )
    parser.add_argument(
        "--normalize_velocity",
        action="store_true",
        default=False,
        help="Enable velocity normalization",
    )
    args = parser.parse_args()
    main(args)
