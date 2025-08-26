#!/usr/bin/env python3
""" # noqa: E501
This script fix the decoder part of a Triplane network, train the triplane
network on validation sets to test its performance on testset.

The optimised weight will be saved in the same directory as the checkpoint path parent directory.

1. Init model, load decoder triplane weights from a pretrained model
2. Fix the triplane weights, train the decoder only.
3. Save the model weights.


Example usage:
1. Train the triplane on later 50 frames
python pore_net/triplane/train_decoder_refine.py \
    --checkpoint_path ./out/triplane_06_14_22_34_num_points_500000_lr_0.001_ds8/triplane_0_150.pt \
    --data_dir ./data/PC_073_segmented_tifs_pad_0_ds8near_thresh3.0 \
    --frame_start 0 \
    --frame_end 150 \
    --resolution 128 \
    --num_epochs 200 \
    --points_batch_size 500000

"""
import os
import argparse
import wandb
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pore_net.triplane.dataset import CloudPointDataset
from pore_net.triplane.model import MultiTriplane
from pore_net.triplane.train_decoder import edr_loss


device = "cuda" if torch.cuda.is_available() else "cpu"
use_wandb = True


def main(args):
    # 1. int model
    num_frames = args.frame_end - args.frame_start
    # print(f"num_frames: {num_frames}")
    model = MultiTriplane(
        resolution=args.resolution,
        num_objs=num_frames,
        input_dim=3,
        output_dim=1,
    )

    # Load weights into model
    weights = torch.load(args.checkpoint_path)
    # print(weights.keys())
    # embeddings_sd = {
    #     k.replace("embeddings.", ""): v
    #     for k, v in weights.items()
    #     if k.startswith("embeddings.")
    # }
    # model.net.load_state_dict(embeddings_sd)

    model.load_state_dict(weights)

    # Freeze embeddings weights
    for param in model.embeddings.parameters():
        param.requires_grad = False

    # Load dataset - load all 200 data, but only use the last 50 data for fitting  # noqa: E501
    # Triplanes
    dataset = CloudPointDataset(
        data_path=args.data_dir,
        num_near_points=args.points_batch_size // 2,
        num_voxel_points=args.points_batch_size // 2,
        subset_size=args.frame_end,
    )

    # Put model net into trainable mode
    model.to(device)
    model.net.train()

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Init wandb
    if use_wandb:
        wandb.init(
            project="triplane-autodecoder",
            name=f"triplane_decoder_refine_frame_{args.frame_start}_to_{args.frame_end}",
        )  # noqa: E501
    ckpt_dir = os.path.dirname(args.checkpoint_path)

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        idx_array = torch.Tensor(np.array(list(range(len(dataset))))).long().to(device)
        idx_array = idx_array.reshape(-1, 1)
        # for idx, frame_idx in enumerate(range(args.frame_start, args.frame_end)):  # noqa: E501
        for obj_idx in tqdm(idx_array):
            # print(f"Refining decoder for frame {obj_idx}")
            # Load data
            data = dataset[obj_idx]

            X, Y = data
            X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
            X, Y = X.to(device), Y.to(device)

            loss_total = 0

            preds = model(
                obj_idx, X
            )  # The output is (BS, M, 1), we only need the first channel  # noqa: E501

            loss = nn.BCEWithLogitsLoss()(preds, Y.reshape(Y.shape[0], Y.shape[1], -1))
            loss_total += loss

            # # # DENSITY REG
            _, n_coords, _ = X.shape
            loss_edr = edr_loss(
                obj_idx,
                model,
                device,
                offset_distance=0.01,
                num_points=n_coords,
            )
            loss_total += loss_edr * 3e-1

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        print(
            f"epoch \t {epoch} \t loss_total: \t{loss_total.item()} \t loss: \t{loss.item()} \t loss_edr: \t{loss_edr.item()}"
        )  # noqa: E501
        if use_wandb:
            if epoch % 10 == 0:
                wandb.log(
                    {
                        "loss": loss_total.item(),
                        "loss_edr": loss_edr.item(),
                        "loss_decoder": loss.item(),
                    },
                    step=epoch,
                )  # noqa: E501
        if epoch % 10 == 0:
            output_path = os.path.join(
                ckpt_dir,
                f"triplane_refine_{args.frame_start}_{args.frame_end}_epoch_{epoch}.pt",
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
    args = parser.parse_args()

    # Print arguments in a fancy format
    print("\n" + "=" * 50)
    print("Refining decoder Configuration:")
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
