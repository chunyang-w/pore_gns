"""
This is the one that works well in our setting.

In comparison to train.py, this file is the original code from the NFD repo - which use a different model and
loss design.

The newer version use tanh function to clamp the triplane to [-1, 1], this version do a
regularization on the triplane to be in [-1, 1].

The implementation detail of the model weight is also different - use Parameter instead of Embedding to do stuff.

This implementation is closer to the NFD paper design.

Source Code from: https://github.com/JRyanShue/NFD/blob/main/nfd/triplane_decoder/train_decoder.py
Code for paper 3D Neural Field Generation using Triplane Diffusion, https://arxiv.org/abs/2211.16677  # noqa: E501

Modified by: Chunyang Wang
GitHub username: chunyang-w

Example usage (Run this under project root):

1. Train with 5M points, 1e-3 lr
python pore_net/triplane/train_decoder.py \
    --data_dir ./data/PC_073_segmented_tifs_pad_0_ds6near_thresh3.0 \
    --lr 1e-3 \
    --tv_reg \
    --l2_reg \
    --subset_size 150 \
    --points_batch_size 500000 \
    --edr_val 0.01 \
    --batch_size 1 \
    --checkpoint_path ./out \
    --log_every 20 \
    --val_every 100 \
    --save_every 600 \
    --resolution 128 \
    --channels 32 \

2. Train with 5M points, 1e-3 lr
python pore_net/triplane/train_decoder.py \
    --data_dir ./data/PC_073_segmented_tifs_pad_0_ds4near_thresh3.0 \
    --lr 1e-3 \
    --tv_reg \
    --l2_reg \
    --subset_size 150 \
    --points_batch_size 500000 \
    --edr_val 0.01 \
    --batch_size 1 \
    --checkpoint_path ./out \
    --log_every 20 \
    --val_every 100 \
    --save_every 600 \
    --resolution 256 \
    --channels 32 \


3. Train with 5M points, 1e-3 lr, downsampled 8
python pore_net/triplane/train_decoder.py \
    --data_dir ./data/PC_073_segmented_tifs_pad_0_ds8near_thresh3.0 \
    --lr 1e-3 \
    --tv_reg \
    --l2_reg \
    --subset_size 150 \
    --points_batch_size 500000 \
    --edr_val 0.01 \
    --batch_size 1 \
    --checkpoint_path ./out \
    --log_every 20 \
    --val_every 100 \
    --save_every 1000 \
    --resolution 128 \
    --channels 32 \

Modification to make:

+ [x] Modify ckpt path - output path
+ [x] What is edr_val - this is the offset distance for the explicit density regularization  # noqa: E501

TODO:
+ [x] Make more rigrous ouput name for each exp
+ [ ] Figure out the batch size specified in the dataset and model
"""

import os
import argparse
import wandb
import time
import torch
import numpy as np
from pore_net.triplane.dataset import CloudPointDataset
from pore_net.triplane.model import MultiTriplane

device = "cuda" if torch.cuda.is_available() else "cpu"


def edr_loss(
    obj_idx, auto_decoder, device="cuda", offset_distance=0.01, num_points=10000
):
    # num_points = 10000
    random_coords = (
        torch.rand(obj_idx.shape[0], num_points, 3).to(device) * 2 - 1
    )  # noqa: E501 sample from [-1, 1]
    offset_coords = (
        random_coords + torch.randn_like(random_coords) * offset_distance
    )  # noqa: E501 Make offset_magnitude bigger if you want smoother
    densities_initial = auto_decoder(obj_idx, random_coords)
    densities_offset = auto_decoder(obj_idx, offset_coords)
    density_smoothness_loss = torch.nn.functional.mse_loss(
        densities_initial, densities_offset
    )  # noqa: E501

    return density_smoothness_loss


def main():
    parser = argparse.ArgumentParser(description="Train auto-decoder.")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="directory to pull data from",
        required=True,
    )
    """
    One thing to note is that in each batch, we would have
    the scene_idx to be the same - we are training for one
    particular embedding/triplane for a single batch.
    """
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        required=False,
        help="number of scenes per batch",
    )
    parser.add_argument(
        "--points_batch_size",
        type=int,
        default=500000,
        required=False,
        help="number of points per scene, nss and uniform combined",
    )
    parser.add_argument("--log_every", type=int, default=20, required=False)
    parser.add_argument("--val_every", type=int, default=100, required=False)
    parser.add_argument("--save_every", type=int, default=200, required=False)
    parser.add_argument(
        "--load_ckpt_path",
        type=str,
        default=None,
        required=False,
        help="checkpoint to continue training from",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        required=False,
        help="learning rate",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./out",
        required=False,
        help="where to save model checkpoints",
    )
    parser.add_argument(
        "--tv_reg",
        default=False,
        required=False,
        action="store_true",
        help="Whether to use TV regularization.",
    )
    parser.add_argument(
        "--l2_reg",
        default=False,
        required=False,
        action="store_true",
        help="Whether to use L2 regularization.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        required=False,
        help="triplane resolution",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=32,
        required=False,
        help="triplane depth",
    )
    parser.add_argument(
        "--aggregate_fn",
        type=str,
        default="sum",
        required=False,
        help="function for aggregating triplane features",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        required=False,
        help="size of the dataset subset if we're training on a subset",
    )
    parser.add_argument(
        "--steps_per_batch",
        type=int,
        default=1,
        required=False,
        help="If specified, how many GD steps to run on a batch before moving on. To address I/O stuff.",  # noqa: E501
    )
    parser.add_argument(
        "--edr_val",
        type=float,
        default=None,
        required=False,
        help="If specified, use explicit density regularization with the specified offset distance value.",  # noqa: E501
    )
    parser.add_argument(
        "--use_tanh",
        default=False,
        required=False,
        action="store_true",
        help="Whether to use tanh to clamp triplanes to [-1, 1].",
    )
    args = parser.parse_args()

    # Print all arguments in a formatted way
    print("\n" + "=" * 50)
    print("Training with the following parameters:")
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: \t{value}")
    print("=" * 50 + "\n")

    # device = 'cpu'

    cur_time = time.strftime("%m_%d_%H_%M")
    exp_dir = os.path.join(
        os.path.abspath(args.checkpoint_path),
        f"triplane_{cur_time}_num_points_{args.points_batch_size}_lr_{args.lr}_ds8",  # noqa
    )

    os.makedirs(exp_dir, exist_ok=True)

    # When you load the entire dataset onto GPU memory...
    dataset = CloudPointDataset(
        data_path=args.data_dir,
        num_near_points=args.points_batch_size // 2,
        num_voxel_points=args.points_batch_size // 2,
        subset_size=args.subset_size,
    )

    print("Dataset loaded")
    sample = dataset[0]
    points, labels = sample
    print("length of dataset:", len(dataset))
    print("Points shape:", points.shape)
    print("Labels shape:", labels.shape)

    # Triplane auto-decoder
    # how_many_scenes = len(dataloader) * args.batch_size
    how_many_scenes = len(dataset)
    print(f"Number of scenes: {how_many_scenes}")

    auto_decoder = MultiTriplane(
        resolution=args.resolution,
        num_objs=how_many_scenes,
        input_dim=3,
        output_dim=1,
        num_channels=args.channels,
        device=device,
    ).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        params=auto_decoder.parameters(), lr=args.lr, betas=(0.9, 0.999)
    )

    if args.load_ckpt_path:
        checkpoint = torch.load(args.load_ckpt_path)
        auto_decoder.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(
            f"Loaded checkpoint from {args.load_ckpt_path}. Resuming training from epoch {epoch} with loss {loss}..."  # noqa: E501
        )

    auto_decoder.train()

    N_EPOCHS = 3000000  # A big number -- can easily do early stopping with Ctrl+C.
    step = 0

    wandb.init(project="triplane-autodecoder", name=exp_dir)

    for epoch in range(N_EPOCHS):
        print("-" * 80)
        print(f"EPOCH {epoch}...")
        print("-" * 80)
        # for (obj_idx, pts_sdf) in dataloader:
        # for (coordinates, gt_occupancies) in dataloader:
        idx_array = torch.Tensor(np.array(list(range(len(dataset))))).long().to(device)
        idx_array = idx_array.reshape(-1, args.batch_size)
        # print("idx_array shape:", idx_array.shape)
        for obj_idx in idx_array:
            total_loss = 0
            # print(f'Time to load data with CPU: {time.time() - load_start_time}')  # noqa: E501

            obj_idx, pts_sdf = obj_idx.int().to(device), dataset[obj_idx]

            coordinates, gt_occupancies = pts_sdf  # Modification by Chunyang Wang
            batch_size, n_coords, n_dims = coordinates.shape

            coordinates = torch.from_numpy(coordinates).float().to(device)
            gt_occupancies = (
                torch.from_numpy(gt_occupancies).float().to(device)
            )  # noqa: E501

            for _step in range(args.steps_per_batch):
                pred_occupancies = auto_decoder(obj_idx, coordinates)

                # BCE loss
                loss = loss_fn(
                    pred_occupancies,
                    gt_occupancies.reshape(
                        (gt_occupancies.shape[0], gt_occupancies.shape[1], -1)
                    ),
                )
                total_loss += loss
                # Explicit density regulation
                if args.edr_val is not None:
                    loss_edr = edr_loss(
                        obj_idx,
                        auto_decoder,
                        device,
                        offset_distance=args.edr_val,
                        num_points=n_coords,
                    )
                    total_loss += loss_edr * 3e-1
                if args.tv_reg:
                    loss_tv = auto_decoder.tvreg() * 1e-2
                    total_loss += loss_tv
                if args.l2_reg:
                    loss_l2_reg = auto_decoder.l2reg() * 1e-3
                    total_loss += loss_l2_reg

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                step += 1

            if not step % args.log_every:
                print(
                    f"Step {step}: Loss {loss.item()}, EDR {loss_edr.item()}, TV {loss_tv.item()}, L2 {loss_l2_reg.item()}, total {total_loss.item()}"  # noqa: E501
                )
                wandb.log({"total_loss": total_loss.item()})
                wandb.log({"loss": loss.item()})
                if args.edr_val is not None:
                    wandb.log({"edr_loss": loss_edr.item()})
                if args.tv_reg:
                    wandb.log({"tv_loss": loss_tv.item()})
                if args.l2_reg:
                    wandb.log({"l2_loss": loss_l2_reg.item()})

            if not step % args.save_every:
                print(f"Saving checkpoint at step {step}")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": auto_decoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),
                    },
                    f"{exp_dir}/model_epoch_{epoch}_loss_{loss.item()}.pt",
                )


if __name__ == "__main__":
    main()
