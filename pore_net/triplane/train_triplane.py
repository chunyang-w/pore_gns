#!/usr/bin/env python3
""" # noqa: E501
This script fix the decoder part of a Triplane network, train the triplane
network on validation sets to test its performance on testset.

The optimised weight will be saved in the same directory as the checkpoint path parent directory.

1. Init model, load decoder weights from a pretrained model
2. Fix the decoder weights, train the triplane on validation set
for each shape, train 40 epochs.
3. Save the triplane weights.

Example usage:
1. Train the triplane on later 50 frames
python pore_net/triplane/train_triplane.py \
    --checkpoint_path ./out/triplane_06_14_22_34_num_points_500000_lr_0.001_ds8/model_epoch_406_loss_0.20061655342578888.pt \
    --data_dir ./data/PC_073_segmented_tifs_pad_0_ds8near_thresh3.0 \
    --frame_start 150 \
    --frame_end 200 \
    --resolution 128 \
    --num_epochs 200 \
    --points_batch_size 500000 \
    --exp_id 73 \

2. Train the triplane on first 150 frames
python pore_net/triplane/train_triplane.py \
    --checkpoint_path ./out/triplane_06_14_22_34_num_points_500000_lr_0.001_ds8/model_epoch_406_loss_0.20061655342578888.pt \
    --data_dir ./data/PC_073_segmented_tifs_pad_0_ds8near_thresh3.0 \
    --frame_start 0 \
    --frame_end 150 \
    --resolution 128 \
    --num_epochs 200 \
    --points_batch_size 500000 \
    --exp_id 73 \

3. Train the triplane on first 10 frames - 072
python pore_net/triplane/train_triplane.py \
    --checkpoint_path ./out/triplane_06_14_22_34_num_points_500000_lr_0.001_ds8/model_epoch_406_loss_0.20061655342578888.pt \
    --data_dir ./data/PC_072_pad_0_ds8near_thresh3.0 \
    --frame_start 0 \
    --frame_end 10 \
    --resolution 128 \
    --num_epochs 800 \
    --points_batch_size 500000 \
    --exp_id 72 \

3. Train the triplane on latter 50 frames - 073 alternative training - this is based on the refined decoder
python pore_net/triplane/train_triplane.py \
    --checkpoint_path ./out/triplane_06_14_22_34_num_points_500000_lr_0.001_ds8/triplane_refine_0_150_epoch_190.pt \
    --data_dir ./data/PC_073_segmented_tifs_pad_0_ds8near_thresh3.0 \
    --frame_start 150 \
    --frame_end 200 \
    --resolution 128 \
    --num_epochs 200 \
    --points_batch_size 500000 \
    --exp_id 73 \
"""

import os
import argparse
import wandb
import torch
import torch.nn as nn
from pore_net.triplane.dataset import CloudPointDataset
from pore_net.triplane.model import MultiTriplane


device = "cuda" if torch.cuda.is_available() else "cpu"
use_wandb = True


def fit_triplane(model, idx, data, optimizer, num_epochs):
    epoch = 0
    X, Y = data
    X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    X, Y = X.to(device), Y.to(device)
    for epoch in range(num_epochs):
        loss_total = 0

        preds = model(
            idx, X
        )  # The output is (BS, M, 1), we only need the first channel  # noqa: E501

        loss = nn.BCEWithLogitsLoss()(preds, Y.reshape(Y.shape[0], Y.shape[1], -1))

        # # # DENSITY REG
        rand_coords = torch.rand_like(X) * 2 - 1
        rand_coords_offset = rand_coords + torch.randn_like(rand_coords) * 1e-2
        # rand_coords_offset = rand_coords + torch.randn_like(rand_coords) * 1e-1
        rand_coords, rand_coords_offset = rand_coords.to(device), rand_coords_offset.to(
            device
        )  # noqa: E501
        d_rand_coords = model(idx, rand_coords)
        d_rand_coords_offset = model(idx, rand_coords_offset)
        loss += (
            nn.functional.mse_loss(
                # d_rand_coords, d_rand_coords_offset) * 6e-1
                d_rand_coords,
                d_rand_coords_offset,
            )
            * 9e-1
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch += 1
        loss_total += loss
        print(
            f"epoch {epoch} \t frame_{idx} \t loss: {loss_total.item()}"
        )  # noqa: E501
        if use_wandb:
            if epoch % 10 == 0:
                wandb.log({f"frame_{idx}_loss": loss_total.item()}, step=epoch)


def main(args):
    # 1. int model
    num_frames = args.frame_end - args.frame_start
    print(f"num_frames: {num_frames}")
    model = MultiTriplane(
        resolution=args.resolution,
        num_objs=num_frames,
        input_dim=3,
        output_dim=1,
    )

    # Load weights into model
    # weights = torch.load(args.checkpoint_path)["model_state_dict"]
    weights = torch.load(args.checkpoint_path)

    decoder_sd = {
        k.replace("net.", ""): v for k, v in weights.items() if k.startswith("net.")
    }
    model.net.load_state_dict(decoder_sd)

    # Freeze decoder weights
    for param in model.net.parameters():
        param.requires_grad = False

    # Load dataset - load all 200 data, but only use the last 50 data for fitting  # noqa: E501
    # Triplanes
    dataset = CloudPointDataset(
        data_path=args.data_dir,
        num_near_points=args.points_batch_size // 2,
        num_voxel_points=args.points_batch_size // 2,
        subset_size=args.frame_end,
        replace=True if args.exp_id == 72 else False,
    )

    # Put model embeeding into trainable mode
    model.to(device)
    model.embeddings.train()

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Init wandb
    if use_wandb:
        wandb.init(
            project="triplane-autodecoder",
            name=f"triplane_fit_frame_{args.frame_start}_to_{args.frame_end}_exp_{args.exp_id}",
        )  # noqa: E501

    for idx, frame_idx in enumerate(range(args.frame_start, args.frame_end)):
        print(f"Training Triplane for frame {frame_idx}")
        # Load data
        data = dataset[frame_idx]

        # Train Triplane
        fit_triplane(model, idx, data, optimizer, args.num_epochs)  # noqa: E501
        # 0. Name the output path

    ckpt_dir = os.path.dirname(args.checkpoint_path)
    output_path = os.path.join(
        ckpt_dir, f"triplane_{args.frame_start}_{args.frame_end}_exp_{args.exp_id}.pt"
    )  # noqa: E501

    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=149)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--points_batch_size", type=int, default=500000)
    parser.add_argument("--exp_id", type=int, default=73)
    args = parser.parse_args()

    # Print arguments in a fancy format
    print("\n" + "=" * 50)
    print("Training Configuration:")
    print("=" * 50)
    # Get all arguments as dictionary
    arg_dict = vars(args)
    # Print each argument
    for arg_name, arg_value in arg_dict.items():
        # Special handling for frame range
        if arg_name == "frame_start":
            print(f"{'Frame Range:':<25} {arg_value} to {args.frame_end}")
            continue
        elif arg_name == "frame_end":
            continue
        print(f"{arg_name.replace('_',' ').title()+':':<25} {arg_value}")
    # Print device separately since it's not in args
    print(f"{'Device:':<25} {device}")
    print("=" * 50 + "\n")
    main(args)
