"""
What we want to achieve with this script:
1. Inference the GNS model on the given dataset on the given period of time
2. Save the inference results in a csv file.

Variables of the script:
1. dataset_path: path to the dataset
2. model_path: path to the model
3. stat_path: path to the data statistics

Example usage:

1. 073,  ds_patch = 2, autoreg
python script/inference_GNS.py \
    --exp_id 73 \
    --ds_patch 2 \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --stats_path /gpfs/home/cw1722/particle/pore_net/data/supersmoothed_train_stats.pt \
    --tif_dir ../data/Segmentations/073_segmented_tifs \
    --file_name ../data/Velocity_smooth/073_final.csv \
    --model_path /gpfs/home/cw1722/particle/pore_net/out/GNS_noise_5/latest_checkpoint.pt \
    --tag "autoreg" \
    --t_start 150 \
    --num_frames 30 \

2. 073,  ds_patch = 8, autoreg
python script/inference_GNS.py \
    --exp_id 73 \
    --ds_patch 8 \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --stats_path /gpfs/home/cw1722/particle/pore_net/data/supersmoothed_train_stats.pt \
    --tif_dir ../data/Segmentations/073_segmented_tifs \
    --file_name ../data/Velocity_smooth/073_final.csv \
    --model_path /gpfs/home/cw1722/particle/pore_net/out/GNS_noise_5_ds_8/latest_checkpoint.pt \
    --tag "autoreg" \
    --t_start 150 \
    --num_frames 30 \

3. 073,  ds_patch = 8, autoreg, model only trained on 073 first 75% frames
python script/inference_GNS.py \
    --exp_id 73 \
    --ds_patch 8 \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --stats_path /gpfs/home/cw1722/particle/pore_net/data/supersmoothed_train_stats.pt \
    --tif_dir ../data/Segmentations/073_segmented_tifs \
    --file_name ../data/Velocity_smooth/073_final.csv \
    --model_path /gpfs/home/cw1722/particle/pore_net/out/GNS_noise_5_073/latest_checkpoint.pt \
    --tag "partial_073" \
    --t_start 150 \
    --num_frames 30 \

3. 072,  ds_patch = 8, autoreg, model only trained on 073 first 75% frames
python script/inference_GNS.py \
    --exp_id 72 \
    --ds_patch 8 \
    --pore_path ../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif \
    --stats_path /gpfs/home/cw1722/particle/pore_net/data/supersmoothed_train_stats.pt \
    --tif_dir ../data/Segmentations/072 \
    --file_name ../data/Velocity_smooth/072_final.csv \
    --model_path /gpfs/home/cw1722/particle/pore_net/out/GNS_noise_5_073/latest_checkpoint.pt \
    --tag "partial_073_teston_072" \
    --t_start 150 \
    --num_frames 30 \
"""

import os
import torch
import tifffile
import argparse
import numpy as np
import pandas as pd

from os import path
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
from types import SimpleNamespace
from functools import reduce

from pore_net.GNS import GNS
from pore_net.utils import load_tif, extract_patches, down_sample

device = "cuda:0"

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
    "--tif_dir",
    type=str,
    required=True,
    help="Directory containing segmented tif files",
)
parser.add_argument(
    "--file_name", type=str, required=True, help="Path to velocity CSV file"
)
parser.add_argument(
    "--model_path", type=str, required=True, help="Path to model checkpoint"
)
parser.add_argument("--tag", type=str, default="", help="Tag for the experiment")
parser.add_argument("--t_start", type=int, default=150, help="Start frame")
parser.add_argument("--num_frames", type=int, default=30, help="Number of frames")

args = parser.parse_args()

ds_patch = args.ds_patch
pore_path = args.pore_path
PATH_stats = args.stats_path
exp_id = args.exp_id
tif_dir = args.tif_dir
file_name = args.file_name
model_path = args.model_path
tag = args.tag
t_start = args.t_start
num_frames = args.num_frames

model_name = model_path.split("/")[-2]
output_dir = path.join(
    path.dirname(path.dirname(path.abspath(__file__))),
    "data",
    f"GNS_pred_exp_{exp_id}_model_{model_name}_tag_{tag}",
)


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
    print("\n" + "#" * 80)
    print("#" + " " * 30 + "GNS Inference Started" + " " * 29 + "#")
    print("#" * 80 + "\n")

    print("=" * 40 + " Parameters " + "=" * 40)
    print(f"{'Device:':<30} {device}")
    print(f"{'Number of node features:':<30} {num_node_features}")
    print(f"{'Number of edge features:':<30} {num_edge_features}")
    print(f"{'Output dimension:':<30} {output_dim}")
    print(f"{'Hidden dimension:':<30} {hidden_dim}")
    print(f"{'Number of GNN layers:':<30} {config.num_layers}")
    print(f"{'Positional encoding:':<30} {config.PE}")
    print(f"{'Image encoder:':<30} {config.image_encoder}")
    print(f"{'Image patch size:':<30} {config.image_size}")
    print(f"{'Downsampling factor:':<30} {ds_patch}")
    print("\n" + "=" * 90 + "\n")

    print("Load data from csv file:", file_name)
    df = pd.read_csv(file_name)

    pore_data = tifffile.imread(pore_path)
    pore_data = pore_data[:, 50:-50, 50:-50] // 255
    pore_data = torch.tensor(pore_data, dtype=torch.short)
    pore_mask = pore_data == 1
    pore_mask = down_sample(pore_mask, ds_patch)
    print("Loaded pore mask with shape:", pore_mask.shape)

    print("Load model from:", model_path)
    model = GNS(
        num_node_features,
        num_edge_features,
        hidden_dim,
        output_dim,
        config,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))["model_state_dict"]
    )
    model.eval()

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

    auto_input_data = construct_starting_graph(merged_df, t_center)

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

    with torch.no_grad():
        for i in np.arange(t_center, rollout_end):
            print(f"length of positions list: {len(positions_list)}")
            tif_data = load_tif(
                t_idx=i,
                tif_dir=tif_dir,
                pore_mask=pore_mask,
                exp_id=exp_id,
                ds_patch=ds_patch,
            )
            auto_input_data.image_3D = extract_patches(
                positions=auto_input_data.x[:, :3],
                tif_data=tif_data,
                ds_patch=ds_patch,
            )

            velocity_pred = model(
                auto_input_data,
                mean_vec_x,
                std_vec_x,
                mean_vec_edge,
                std_vec_edge,
            )
            velocity_pred_unnormalised = unnormalize(
                velocity_pred, mean_vec_y, std_vec_y
            )
            velocity_pred_list.append(velocity_pred_unnormalised.clone().cpu())

            p_t_plus_one = 2 * velocity_pred_unnormalised + positions_list[-2]
            positions_list.append(p_t_plus_one)

            past_velocity = torch.roll(past_velocity, shifts=3, dims=1)
            past_velocity[:, :3] = velocity_pred_unnormalised.clone()

            auto_input_data.x[:, 6:] = past_velocity.clone()
            auto_input_data.x[:, :3] = positions_list[-1]
            auto_input_data.x[:, 3:6] = positions_list[-2]

            edge_index, edge_attribute = compute_edge_attr_with_radius_graph(
                auto_input_data.x[:, :6]
            )
            auto_input_data.edge_index = edge_index.clone().detach()
            auto_input_data.edge_attr = edge_attribute.clone().detach()
    print("Roll out finished")

    print("Transforming predictions to csv format")
    pred_id = auto_input_data.particle_id
    pred_pos = np.array([p.cpu().numpy() for p in positions_list])
    pred_vel = np.array([p.cpu().numpy() for p in velocity_pred_list])
    df_pred = pred_to_csv(
        frame_start=t_center,
        num_frames=rollout_end - t_center,
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
