"""  # noqa: E501
We can confirm the pre-downsampled data is scaled to 0-1.

What we want to achieve with this script:
1. Inference the GNS model on the given dataset on the given period of time
2. Save the inference results in a csv file.

Variables of the script:
1. dataset_path: path to the dataset
2. model_path: path to the model
3. stat_path: path to the data statistics

Example usage:

1. 073,  ds_patch = 8, no-autoreg
python script/inference_GNS_Unet_noreg.py \
    --exp_id 73 \
    --ds_patch 8 \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --stats_path /gpfs/home/cw1722/particle/pore_net/data/supersmoothed_train_stats.pt \
    --file_name ../data/Velocity_smooth/073_final.csv \
    --model_path_gns ./out/GNS_noise_5_ds_8/latest_checkpoint.pt \
    --model_path_unet ./out/04-20_11:58:32_tif_073_segmented_tifs_down8_bs2_lr0.001_ep100_fp16_in2_out1/checkpoint_epoch_100.pth \
    --n_frames_in 2 \
    --n_frames_out 1 \
    --channel_per_frame 4 \
    --use_rock True \
    --tif_dir ./data/tif_073_segmented_tifs_down8/ \
    --phy_dir ./data/073_final_down8_maxpool/ \
    --tag "noreg" \
    --t_start 150 \
    --num_frames 30 \


1. 073,  ds_patch = 8, no-autoreg, on test set 30 frames
python script/inference_GNS_Unet_noreg.py \
    --exp_id 73 \
    --ds_patch 8 \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --stats_path /gpfs/home/cw1722/particle/pore_net/data/supersmoothed_train_stats.pt \
    --file_name ../data/Velocity_smooth/073_final.csv \
    --model_path_gns ./out/GNS_noise_5_ds_8/latest_checkpoint.pt \
    --model_path_unet ./out/04-20_11:58:32_tif_073_segmented_tifs_down8_bs2_lr0.001_ep100_fp16_in2_out1/checkpoint_epoch_100.pth \
    --n_frames_in 2 \
    --n_frames_out 1 \
    --channel_per_frame 4 \
    --use_rock True \
    --tif_dir ./data/tif_073_segmented_tifs_down8/ \
    --phy_dir ./data/073_final_down8_maxpool/ \
    --tag "noreg_testset120" \
    --t_start 120 \
    --num_frames 30 \

1. 072,  ds_patch = 8, no-autoreg
python script/inference_GNS_Unet_noreg.py \
    --exp_id 72 \
    --ds_patch 8 \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --stats_path /gpfs/home/cw1722/particle/pore_net/data/supersmoothed_train_stats_072.pt \
    --file_name ../data/Velocity_smooth/072_final.csv \
    --model_path_gns ./out/GNS_noise_5_ds_8/latest_checkpoint.pt \
    --model_path_unet ./out/04-20_11:58:32_tif_073_segmented_tifs_down8_bs2_lr0.001_ep100_fp16_in2_out1/checkpoint_epoch_100.pth \
    --n_frames_in 2 \
    --n_frames_out 1 \
    --channel_per_frame 4 \
    --use_rock True \
    --tif_dir ./data/tif_072_down8_interpolated/ \
    --phy_dir ./data/072_final_down8_maxpool/ \
    --tag "noreg" \
    --t_start 150 \
    --num_frames 30 \

1. 073,  ds_patch = 8, no-autoreg, benchmarking the inference time
python script/inference_GNS_Unet_noreg.py \
    --exp_id 73 \
    --ds_patch 8 \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --stats_path /gpfs/home/cw1722/particle/pore_net/data/supersmoothed_train_stats.pt \
    --file_name ../data/Velocity_smooth/073_final.csv \
    --model_path_gns ./out/GNS_noise_5_ds_8/latest_checkpoint.pt \
    --model_path_unet ./out/04-20_11:58:32_tif_073_segmented_tifs_down8_bs2_lr0.001_ep100_fp16_in2_out1/checkpoint_epoch_100.pth \
    --n_frames_in 2 \
    --n_frames_out 1 \
    --channel_per_frame 4 \
    --use_rock True \
    --tif_dir ./data/tif_073_segmented_tifs_down8/ \
    --phy_dir ./data/073_final_down8_maxpool/ \
    --tag "noreg_timing" \
    --t_start 150 \
    --num_frames 30 \
    --benchtime \

"""

# particle/pore_net/data/073_final_down8_maxpool/phy_info_frame_5.npy
import os
import glob
import time
import torch
import tifffile
import argparse
import numpy as np
import pandas as pd

from os import path
from skimage import io
from natsort import natsorted
from tqdm import tqdm
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
from types import SimpleNamespace
from functools import reduce

from pore_net.GNS import GNS
from pore_net.utils import (
    load_tif,
    extract_patches,
    down_sample,
    gns_out_to_grid,
    max_pool3d_numpy,
)
from pore_net.unet import PoreScaleUNet
from pore_net.dataset import PoreShapeDataset
from pore_net.utils import down_sample, normalize_velocity_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_node_features = 21
num_edge_features = 7
output_dim = 3
hidden_dim = 128
config = {
    "num_layers": 10,
    "PE": "fourier_feature",
    "image_encoder": "cnn",
    "image_size": 32,
}
config = SimpleNamespace(**config)

"""
What we want to achieve with this script:
1. Inference the GNS model on the given dataset on the given period of time
2. Save the inference results in a csv file.
"""


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ds_patch", type=int, required=True, help="Patch size for downsampling"
)
parser.add_argument(
    "--pore_path",
    type=str,
    required=True,
    help="Path to pore geometry tif file",
)
parser.add_argument(
    "--stats_path",
    type=str,
    required=True,
    help="Path to training statistics file",
)
parser.add_argument("--exp_id", type=int, required=True, help="Experiment ID")
parser.add_argument(
    "--file_name", type=str, required=True, help="Path to velocity CSV file"
)
parser.add_argument(
    "--model_path_gns", type=str, required=True, help="Path to GNS model checkpoint"
)
parser.add_argument(
    "--model_path_unet", type=str, required=True, help="Path to UNet model checkpoint"
)
parser.add_argument(
    "--n_frames_in", type=int, required=True, help="Number of frames in"
)
parser.add_argument(
    "--n_frames_out", type=int, required=True, help="Number of frames out"
)
parser.add_argument(
    "--channel_per_frame", type=int, required=True, help="Number of channels per frame"
)
parser.add_argument("--use_rock", type=bool, required=True, help="Whether to use rock")
parser.add_argument(
    "--tif_dir",
    type=str,
    required=True,
    help="Path to tif directory pre-downsampled, containing the npy files",  # noqa: E501
)
parser.add_argument(
    "--phy_dir", type=str, required=True, help="Path to physical directory"
)
parser.add_argument("--tag", type=str, default="", help="Tag for the experiment")
parser.add_argument("--t_start", type=int, default=150, help="Start frame")
parser.add_argument("--num_frames", type=int, default=30, help="Number of frames")
parser.add_argument(
    "--benchtime", action="store_true", help="Benchmark the inference time"
)

args = parser.parse_args()


def unnormalize(to_unnormalize, mean_vec, std_vec):
    return to_unnormalize * std_vec + mean_vec


def normalize(
    to_normalize,
    mean_vec,
    std_vec,
):
    normalized_tensor = (to_normalize - mean_vec) / std_vec
    return normalized_tensor


def compute_edge_attr_with_radius_graph(coords_tensor, radius=32):

    edge_index = radius_graph(
        coords_tensor.type(torch.float32),
        r=radius,
        loop=True,
        max_num_neighbors=64,
    )
    src, dst = edge_index[0], edge_index[1]

    delta = coords_tensor[dst] - coords_tensor[src]

    norms = torch.linalg.norm(delta, dim=1).unsqueeze(1)
    edge_attr_tensor = torch.cat([delta, norms], dim=1)

    return edge_index, edge_attr_tensor


def construct_starting_graph(total_oil_df, idx, C=5, rollout_step=30):
    v_oil = np.concatenate(
        [
            (total_oil_df[[f"vz_t{i}", f"vy_t{i}", f"vx_t{i}"]]).to_numpy()
            for i in range(idx - 1, idx - C - 1, -1)
        ],
        axis=1,
    )
    gt_vel = []
    gt_pos = []
    for t in range(rollout_step):
        gt_vel_t = np.concatenate(
            [
                (
                    total_oil_df[
                        [
                            f"vz_t{idx+t}",
                            f"vy_t{idx+t}",
                            f"vx_t{idx+t}",
                        ]
                    ]
                ).to_numpy()
            ],
            axis=1,
        )
        gt_pos_t = np.concatenate(
            [
                (total_oil_df[[f"z_t{idx+t}", f"y_t{idx+t}", f"x_t{idx+t}"]]).to_numpy()
                + 50
            ],
            axis=1,
        )
        gt_vel.append(gt_vel_t)
        gt_pos.append(gt_pos_t)

    p_t = np.concatenate(
        [
            total_oil_df[[f"{axis}_t{i}" for axis in ["z", "y", "x"]]].to_numpy()
            for i in range(idx, idx - 2, -1)
        ],
        axis=1,
    )

    oil_node_features = np.concatenate((p_t, v_oil), axis=1)
    p_t = torch.tensor(p_t, dtype=torch.float32)
    edge_index, edge_attr_tensor = compute_edge_attr_with_radius_graph(p_t)

    particle_id = np.array(total_oil_df.index.unique())

    return Data(
        x=torch.tensor(oil_node_features, dtype=torch.float32),
        edge_index=edge_index.to(dtype=torch.long),
        edge_attr=edge_attr_tensor,
        gt_vel=torch.tensor(np.array(gt_vel), dtype=torch.float32),
        gt_pos=torch.tensor(np.array(gt_pos), dtype=torch.float32),
        p_t=p_t,
        t_start=idx,
        particle_id=particle_id,
    )


def pred_to_csv(frame_start, num_frames, pred_pos, pred_vel, pred_id):
    """
    Transform predictions to csv format.
    frame_start: the start frame of the predictions
    num_frames: the number of frames of the predictions
    pred_pos: the predicted positions, of shape (num_frames, num_particles, 3)
    pred_vel: the predicted velocities, of shape (num_frames, num_particles, 3)
    pred_id: the predicted particle ids, of shape (num_particles,)
    """
    data_list = []
    for i in range(num_frames):
        frame = frame_start + i
        for p_idx in range(len(pred_id)):
            particle_id = pred_id[p_idx]
            z, y, x = pred_pos[i, p_idx]
            vz, vy, vx = pred_vel[i, p_idx]
            data_list.append(
                {
                    "frame": frame,
                    "particle": particle_id,
                    "z": z,
                    "y": y,
                    "x": x,
                    "vz": vz,
                    "vy": vy,
                    "vx": vx,
                }
            )
    df_pred = pd.DataFrame(data_list)
    df_pred = df_pred.sort_values(["particle", "frame"])
    return df_pred


if __name__ == "__main__":
    ds_patch = args.ds_patch
    pore_path = args.pore_path
    PATH_stats = args.stats_path
    exp_id = args.exp_id
    tif_dir = args.tif_dir
    file_name = args.file_name
    model_path_gns = args.model_path_gns
    model_path_unet = args.model_path_unet
    n_frames_in = args.n_frames_in
    n_frames_out = args.n_frames_out
    channel_per_frame = args.channel_per_frame
    use_rock = args.use_rock
    tag = args.tag
    t_start = args.t_start
    num_frames = args.num_frames

    model_gns_name = model_path_gns.split("/")[-2]
    # TODO: more nameing convention
    output_dir = path.join(
        path.dirname(path.dirname(path.abspath(__file__))),
        "out",
        f"{exp_id}_GNS_{model_gns_name}_UNET_framein{n_frames_in}_framesout{n_frames_out}_channel{channel_per_frame}_frame_{args.t_start}_{args.t_start + args.num_frames}_tag_{tag}",  # noqa: E501
    )
    print(output_dir)
    print("\n" + "#" * 80)
    print("#" + " " * 30 + "GNS Inference Started" + " " * 29 + "#")
    print("#" * 80 + "\n")

    print("=" * 40 + " Parameters " + "=" * 40)
    for arg in vars(args):
        print(f"{arg + ':':<30} {getattr(args, arg)}")
    print(f"{'Device:':<30} {device}")
    print(f"{'Number of node features:':<30} {num_node_features}")
    print(f"{'Number of edge features:':<30} {num_edge_features}")
    print(f"{'Output dimension:':<30} {output_dim}")
    print(f"{'Hidden dimension:':<30} {hidden_dim}")
    for key, value in vars(config).items():
        print(f"{key + ':':<30} {value}")
    print("\n" + "=" * 90 + "\n")

    # =============== GNS data ===============
    print("Loading GNS data")
    print("Load data from csv file:", file_name)
    df = pd.read_csv(file_name)
    pore_data = tifffile.imread(pore_path)
    pore_data = pore_data[:, 50:-50, 50:-50] // 255
    pore_data = torch.tensor(pore_data, dtype=torch.short)
    pore_mask = pore_data == 1
    pore_mask = down_sample(pore_mask, ds_patch)
    print("Loaded pore mask with shape:", pore_mask.shape)

    # =============== Unet data ===============

    tif_names = natsorted(glob.glob(args.tif_dir + "*.npy"))
    # print(tif_names)
    phy_names = natsorted(glob.glob(args.phy_dir + "*.npy"))
    # print(phy_names)
    tif_names = tif_names[
        t_start - args.n_frames_in : t_start + num_frames
    ]  # noqa: E501
    # TODO: check if this is correct here we minus 2 because due to interpolation, the previous 2 frames are excluded  # noqa: E501
    phy_names = phy_names[
        t_start - 2 - args.n_frames_in : t_start + num_frames - 2
    ]  # noqa: E501
    print("tif and phy files loaded ... print first and last names")
    print(len(tif_names), len(phy_names))
    print(tif_names[0], phy_names[0])
    print(tif_names[-1], phy_names[-1])

    # TODO: Check 072 alternative implementation
    rock_raw = io.imread(args.pore_path)
    rock = rock_raw // 255
    rock = down_sample(rock, args.ds_patch)
    rock = rock[None, None]
    rock = torch.tensor(rock, dtype=torch.float32).to(device)
    print("Rock data loaded ... print shape")
    print("Rock loaded, shape:", rock.shape)
    tif_arr = []
    phy_arr = []
    ground_arr = []
    pred_arr = []
    # Appending starting sequence to boost n-input auto-reg inference:
    for i in range(n_frames_in):
        tif_in_t0 = np.load(tif_names[i])
        # tif_in_t0 = io.imread(tif_names[i])
        # tif_in_t0 = down_sample(tif_in_t0, args.ds_patch)
        tif_in_t0 = tif_in_t0[None, None]
        tif_in_t0 = torch.tensor(tif_in_t0, dtype=torch.float32).to(device)
        tif_arr.append(tif_in_t0)
        print(f"appending {i}-th tif, shape: {tif_in_t0.shape}")
        # Load phy
        phy_in_t0 = np.load(phy_names[i])
        phy_in_t0 = phy_in_t0
        phy_in_t0 = phy_in_t0[None]
        phy_in_t0 = torch.tensor(phy_in_t0, dtype=torch.float32).to(device)
        phy_arr.append(phy_in_t0)
        print(f"appending {i}-th vel, shape:  {phy_in_t0.shape}")

    print("tif and phy data loaded ... print shapes")

    # =============== GNS model ===============
    print("Load GNS model from:", model_path_gns)
    model_gns = GNS(
        num_node_features,
        num_edge_features,
        hidden_dim,
        output_dim,
        config,
    ).to(device)
    model_gns.load_state_dict(
        torch.load(model_path_gns, map_location=torch.device(device))[
            "model_state_dict"
        ]
    )
    model_gns.eval()

    # =============== UNet model ===============
    print("Load UNet model from:", args.model_path_unet)
    channel_per_frame = 3 + 1

    n_channel_in = channel_per_frame * n_frames_in + (1 if use_rock else 0)
    n_channel_out = n_frames_out
    model_unet = PoreScaleUNet(n_channels=n_channel_in, n_classes=n_channel_out)

    model_unet.load_state_dict(torch.load(args.model_path_unet, map_location=device))
    model_unet.to(device)
    model_unet.eval()

    train_stats = torch.load(PATH_stats)
    (
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ) = train_stats
    mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge = (
        mean_vec_x.to(device),
        std_vec_x.to(device),
        mean_vec_edge.to(device),
        std_vec_edge.to(device),
    )
    mean_vec_y, std_vec_y = mean_vec_y.to(device), std_vec_y.to(device)

    # for t_center in range(150, 151):
    t_center = t_start
    rollout_start, rollout_end = t_center - 5, t_center + num_frames
    df_list = []
    for i in np.arange(rollout_start, rollout_end, 1):
        df_i = df[df["frame"] == i]
        df_i = (
            df_i[["frame", "particle", "z", "y", "x", "vz", "vy", "vx"]]
            .set_index("particle")
            .add_suffix(f"_t{i}")
        )

        df_list.append(df_i)

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on="particle", how="inner"),
        df_list,
    )
    particle = merged_df.index.unique().tolist()

    auto_input_data = construct_starting_graph(
        merged_df, t_center, rollout_step=num_frames
    )

    auto_out = auto_input_data.gt_vel.numpy()
    auto_t = auto_input_data.t_start

    auto_particles = auto_input_data.particle_id

    auto_input_data.to(device)

    positions_list = [
        auto_input_data.x.reshape(-1, 7, 3)[:, 1, :].clone(),
        auto_input_data.x.reshape(-1, 7, 3)[:, 0, :].clone(),
    ]

    past_velocity = auto_input_data.x[:, 6:].clone()

    velocity_pred_list = []

    idx_start = 0
    idx_end = n_frames_in

    with torch.no_grad():
        # for i in tqdm(np.arange(t_center, rollout_end)):
        for i in np.arange(t_center, rollout_end):
            idx = i - t_center
            print(f"** Processing frame {i}")
            # ================= Load tif data for GNS =================
            # should be like [[153, 183, 183]]
            tif_data = tif_arr[-1].squeeze(0).squeeze(0)
            # tif_data[rock[0, 0, :, :, :] == 1] = -1
            # tif_data = tif_data[:, 6:-6, 6:-6]  # TODO: this is because GNS used cropped tif data  # noqa: E501
            # make a copy of the tif data to avoid modifying the original data
            tif_data = tif_data[:, 6:-6, 6:-6].clone()
            # TODO: Check 072 alternative implementation
            if exp_id == 72:
                tif_data_pad = torch.zeros(
                    (153, 195 - 12, 195 - 12), dtype=torch.float32
                ).to(device)
                tif_data_pad[64:, :, :] = tif_data
                tif_data = tif_data_pad
            # add -1 mask to identify rock area
            tif_data[rock[0, 0, :, 6:-6, 6:-6] == 1] = -1
            # print("tif_data shape:", tif_data.shape)
            # should be like [12882, 4, 4, 4]
            auto_input_data.image_3D = extract_patches(
                positions=auto_input_data.x[:, :3],
                tif_data=tif_data,
                ds_patch=ds_patch,
            )
            auto_input_data.image_3D = auto_input_data.image_3D.to(device)
            # print("GNS input data shape:", auto_input_data.image_3D.shape)
            # exit(0)
            # ================= GNS inference =================
            if args.benchtime:
                start_time_gns = time.time()
            velocity_pred = model_gns(
                auto_input_data,
                mean_vec_x,
                std_vec_x,
                mean_vec_edge,
                std_vec_edge,
            )
            if args.benchtime:
                end_time_gns = time.time()

            # ================= UNet inference =================
            tif_in_t0 = torch.cat(tif_arr[idx_start:idx_end], dim=1).to(device)
            phy_in_t0 = torch.cat(phy_arr[idx_start:idx_end], dim=1).to(device)
            if exp_id == 72:
                rock_in = rock[:, :, 64:, :, :]
            elif exp_id == 73:
                rock_in = rock
            data_in_t0 = torch.cat(
                [rock_in, tif_in_t0, phy_in_t0],
                dim=1,
            )
            data_in = data_in_t0
            data_in = data_in.to(device)

            #  model inference for unet
            if args.benchtime:
                start_time_unet = time.time()
            out = model_unet(data_in)
            if args.benchtime:
                end_time_unet = time.time()

            if args.benchtime:
                print(
                    f"Time taken for GNS inference: {end_time_gns - start_time_gns} seconds"
                )
                print(
                    f"Time taken for UNet inference: {end_time_unet - start_time_unet} seconds"
                )

            out = (out > 0).to(torch.float32)
            out = out.to(
                torch.int8
            ).detach()  # Gives shape (1, 128, 128, 128)  # noqa: E501

            # ================= Post processing - velocity pred, position calc, update graph =================  # noqa: E501
            velocity_pred_unnormalised = unnormalize(
                velocity_pred, mean_vec_y, std_vec_y
            )
            velocity_pred_list.append(velocity_pred_unnormalised.clone().cpu())

            # Get ground truth velocity and use this as rollout basis instead of model's output
            velocity_gt = auto_input_data.gt_vel[idx]
            velocity_gt_unnormalised = unnormalize(velocity_gt, mean_vec_y, std_vec_y)

            # p_t_plus_one = (
            #     2 * velocity_pred_unnormalised + positions_list[-2]
            # )
            p_t_plus_one = 2 * velocity_gt_unnormalised + positions_list[-2]
            positions_list.append(p_t_plus_one)

            past_velocity = torch.roll(past_velocity, shifts=3, dims=1)
            past_velocity[:, :3] = velocity_gt_unnormalised.clone()
            # past_velocity[:, :3] = velocity_pred_unnormalised.clone()

            auto_input_data.x[:, 6:] = past_velocity.clone()
            auto_input_data.x[:, :3] = positions_list[-1]
            auto_input_data.x[:, 3:6] = positions_list[-2]

            edge_index, edge_attribute = compute_edge_attr_with_radius_graph(
                auto_input_data.x[:, :6]
            )
            auto_input_data.edge_index = edge_index.clone().detach()
            auto_input_data.edge_attr = edge_attribute.clone().detach()

            # ================= Time marching - update history info =================  # noqa: E501
            # Update Phy grid
            # phy_info_path = phy_names[idx_end]
            # phy_info_t1 = np.load(phy_info_path)
            # phy_info_t1 = torch.tensor(phy_info_t1).to(torch.float32)
            # phy_info_t1 = phy_info_t1.unsqueeze(0)
            # phy_info_t1 = phy_info_t1.to(device)
            grid_unbatched_no_channel_size = rock_raw.shape
            print(
                "grid_unbatched_no_channel_size shape:", grid_unbatched_no_channel_size
            )  # noqa: E501

            # Here do no-reg, append ground truth tif to tif_arr
            print("loading phy_in_t0 at frame", idx + n_frames_in)
            phy_in_t1 = np.load(phy_names[idx + n_frames_in])
            phy_in_t1 = phy_in_t1
            phy_in_t1 = phy_in_t1[None]
            phy_in_t1 = torch.tensor(phy_in_t1, dtype=torch.float32).to(device)
            # phy_arr.append(phy_in_t0)
            # phy_info_t1 = gns_out_to_grid(
            #     velocity_pred_list[-1],  # shape [12882, 3]
            #     positions_list[-1],  # shape [12882, 3]
            #     grid_size=grid_unbatched_no_channel_size,  # shape [128, 128, 128] or something like that  # noqa: E501
            # )
            # phy_info_t1 = max_pool3d_numpy(phy_info_t1, args.ds_patch)
            # phy_info_t1 = torch.from_numpy(phy_info_t1).unsqueeze(0)
            # phy_info_t1 = phy_info_t1.to(device)
            if exp_id == 72:
                print("phy_info_t1 shape:", phy_in_t1.shape)
                phy_in_t1 = phy_in_t1[:, :, 64:, :, :]
            phy_arr.append(phy_in_t1)
            # Update tif data
            print("out shape:", out.shape)
            # Here do no-reg, append ground truth tif to tif_arr
            tif_in_t1 = np.load(tif_names[idx + n_frames_in])
            tif_in_t1 = tif_in_t1[None, None]
            tif_in_t1 = torch.tensor(tif_in_t1, dtype=torch.float32).to(device)
            tif_arr.append(tif_in_t1)

            # Update prediction arr
            pred_arr.append(out)
            # Update ground truth tif
            target_path = tif_names[
                idx_end
            ]  # Idx_end points to the frame we are now aiming to predict  # noqa: E501
            # target = io.imread(target_path)
            target = np.load(target_path)
            # target = down_sample(target, args.ds_patch)
            target = (
                target == 1
            )  # TODO: this is only for 073, for 072 the grayscale is different  # noqa: E501
            target = target[None, None, :, :, :]
            target = torch.from_numpy(target)
            ground_arr.append(target)

            # ================= Counter increment =================
            idx_start += 1
            idx_end += 1
            # break
    print("Roll out finished")

    # ================= Transform predictions to csv format =================
    print("Transforming predictions to csv format")
    pred_id = auto_input_data.particle_id
    pred_pos = np.array([p.cpu().numpy() for p in positions_list])
    pred_vel = np.array([p.cpu().numpy() for p in velocity_pred_list])
    df_pred = pred_to_csv(
        frame_start=t_center,
        num_frames=rollout_end - t_center,
        # num_frames=1,
        pred_pos=pred_pos,
        pred_vel=pred_vel,
        pred_id=pred_id,
    )
    print("Predictions transformed to csv format")

    print("Saving predictions to csv file")
    os.makedirs(output_dir, exist_ok=True)
    df_pred.to_csv(
        path.join(output_dir, f"{exp_id}_t{t_center}-{rollout_end}.csv"),
    )
    print("Predictions saved to csv file")
    print("=" * 40 + " Inference Finished " + "=" * 40)
    print("#" * 80 + "\n")

    # ================= Save predictions to voxel file =================
    print("Saving predictions to voxel file")
    # gt = np.concat(ground_arr, axis=1)
    # pd = np.concat(pred_arr, axis=1)
    gt = torch.cat(ground_arr, dim=1).squeeze(0)
    pd = torch.cat(pred_arr, dim=1).squeeze(0)
    gt = gt.cpu().numpy()
    pd = pd.cpu().numpy()
    print("gt shape:", gt.shape)
    print("pd shape:", pd.shape)

    np.save(path.join(output_dir, f"{exp_id}_t{t_center}-{rollout_end}_pd.npy"), pd)
    np.save(path.join(output_dir, f"{exp_id}_t{t_center}-{rollout_end}_gt.npy"), gt)
    print("Predictions saved to voxel file")
    print("=" * 40 + " Inference Finished " + "=" * 40)
    print("#" * 80 + "\n")
    exit(0)
