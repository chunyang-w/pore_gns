import glob
import torch
import os
import numpy as np
import pandas as pd  # noqa

from skimage import io
from torch.utils.data import Dataset, DataLoader  # noqa
from pore_net.utils import (
    down_sample,
)  # ensure this function is available
from pore_net.utils import (
    normalize_velocity_data,
)  # ensure this function is available
from natsort import natsorted
from tqdm import tqdm  # noqa
from functools import reduce
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data


class CustomDatasetPredictVelocity_no_synthetic_autoregressive(Dataset):
    def __init__(
        self,
        df,
        radius,
        C,
        autoregressive_step,
        exp_id=73,
        get_acc=False,
    ):
        self.radius = radius
        self.C = C
        self.autoregressive_step = autoregressive_step
        self.df = df
        self.get_acc = get_acc
        print(f"Radius: {self.radius}; Include {self.C} velocities")

    def __len__(self):
        return len(self.image_paths) - 5

    def __getitem__(self, idx):
        self.idx = idx
        df_list = [
            self._load_particle_data(i)
            for i in range(idx - self.C, idx + self.autoregressive_step)
        ]
        return self._construct_graph(df_list, idx)

    def _load_particle_data(self, i):
        df = self.df[self.df["frame"] == i]
        df = (
            df[["frame", "particle", "z", "y", "x", "vz", "vy", "vx"]]
            .set_index("particle")
            .add_suffix(f"_t{i}")
        )
        return df

    def _construct_graph(self, df_list, idx):
        merged_df = reduce(
            lambda left, right: pd.merge(
                left, right, on="particle", how="inner"
            ),  # noqa: E501
            df_list,
        )
        merged_df.reset_index(inplace=True)
        total_oil_df = merged_df

        self.merged_df = merged_df

        v_oil = np.concatenate(
            [
                (total_oil_df[[f"vz_t{i}", f"vy_t{i}", f"vx_t{i}"]]).to_numpy()
                for i in range(idx - 1, idx - self.C - 1, -1)
            ],
            axis=1,
        )
        # print('input vt', [*range(idx-1, idx - self.C-1, -1)])

        y_oil_nodes = []

        for t in range(self.autoregressive_step):
            # print('output v t+idx', t+idx)
            y_oil_nodes_t = np.concatenate(
                [
                    (
                        total_oil_df[
                            [
                                f"vz_t{idx+t}",
                                f"vy_t{idx+t}",
                                f"vx_t{idx+t}",
                            ]
                        ]  # noqa: E501
                    ).to_numpy()
                ],
                axis=1,
            )
            y_oil_nodes.append(y_oil_nodes_t)

        p_t = np.concatenate(
            [
                total_oil_df[
                    [f"{axis}_t{i}" for axis in ["z", "y", "x"]]
                ].to_numpy()  # noqa: E501
                for i in range(idx, idx - 2, -1)
            ],
            axis=1,
        )

        # print('p_t', [*range(idx, idx - 2, -1)])

        particle_id = total_oil_df["particle"].to_numpy()

        if self.get_acc:
            # print("get acceleration")
            # print("shape of v_oil_t0: ", v_oil.shape)
            # This is of shape (14252, 15), 15 is 3 velocities across 5 frames
            v_oil_t0 = v_oil[:, :-3]
            v_oil_t1 = v_oil[:, 3:]
            acc = v_oil_t0 - v_oil_t1

        feat_list = [p_t, v_oil]
        if self.get_acc:
            feat_list = [p_t, v_oil, acc]

        oil_node_features = np.concatenate(feat_list, axis=1)
        edge_index_list, edge_attr = self._compute_edges(p_t)

        return Data(
            x=torch.tensor(oil_node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index_list, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=torch.tensor(np.array(y_oil_nodes), dtype=torch.float32),
            particle_id=particle_id,
            t=idx,
            exp_id=73,
        )

    def _compute_edges(self, coordinates):
        coords_tensor = torch.tensor(coordinates, dtype=torch.float)
        edge_index = radius_graph(
            coords_tensor,
            r=self.radius,
            loop=True,
            max_num_neighbors=64,
        )
        edge_attr = []
        for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            delta = coords_tensor[j] - coords_tensor[i]
            edge_attr.append(
                [*delta.tolist(), torch.linalg.norm(delta).item()]
            )  # noqa: E501

        return edge_index.tolist(), edge_attr


class PoreShapeDataset(Dataset):
    def __init__(
        self,
        data_dir,
        vel_path,
        frame_idxs,
        transform=None,
        rock_tif_path=None,
        rock_down_factor=8,
        drop_start=2,
        drop_end=2,
        rock_slicer=None,
        use_fp16=True,  # default set to use float16 precision
        normalize_velocity=False,  # whether to normalize velocity
        velocity_stats=None,  # pre-calculated velocity statistics
        n_frames_in=2,  # the number of frames to use as input
        n_frames_out=1,  # the number of frames to use as output
    ):
        self.transform = transform
        self.vel_path = vel_path
        self.frame_idxs = frame_idxs
        self.use_fp16 = use_fp16
        self.normalize_velocity = normalize_velocity
        self.velocity_stats = velocity_stats

        # Load velocity (npy) and tif file paths
        self.vel_names = natsorted(glob.glob(os.path.join(vel_path, "*.npy")))
        self.tif_names = natsorted(glob.glob(os.path.join(data_dir, "*.npy")))[
            drop_start:-drop_end
        ]

        self.n_frames_in = n_frames_in
        self.n_frames_out = n_frames_out

        # Ensure the number of frames matches the number of TIF files
        assert len(self.vel_names) == len(
            self.tif_names
        ), f"Mismatch between number of frames and TIF files: {len(self.vel_names)} vel frames vs {len(self.tif_names)} TIF files."  # noqa E501

        print(
            "data at idx: [0] \t",
            self.vel_names[0],
            self.tif_names[0],
        )
        print(
            "data at idx: [-1] \t",
            self.vel_names[-1],
            self.tif_names[-1],
        )

        # Load rock tif if provided
        if rock_tif_path is not None:
            down = rock_down_factor
            self.rock_tif = (io.imread(rock_tif_path) // 255) == 1
            if rock_slicer is not None:
                self.rock_tif = self.rock_tif[rock_slicer]
            self.rock_tif = self.down_sample(self.rock_tif, down)[
                None, :, :, :
            ]  # noqa: E501
            self.rock_tif = (
                self.rock_tif.astype(np.float32)
                if not self.use_fp16
                else self.rock_tif.astype(np.float16)
            )
            print("rock tif loaded, shape: ", self.rock_tif.shape)
        else:
            self.rock_tif = None

        # Check that the number of frames matches the number of TIF files.
        num_frames = len(self.vel_names)
        num_tifs = len(self.tif_names)
        assert (
            num_frames == num_tifs
        ), f"Mismatch between number of frames and TIF files: {num_frames} frames vs {num_tifs} TIF files."  # noqa E501

        if self.normalize_velocity and self.velocity_stats is None:
            print(
                "Warning: normalize_velocity is True but velocity_stats not provided. Velocity will not be normalized."  # noqa E501
            )
            self.normalize_velocity = False

        print("dataset init done.")

    def down_sample(self, cube, factor):
        return down_sample(cube, factor)

    def normalize_velocity_data(self, velocity):
        """Normalize velocity data using pre-calculated statistics"""
        if not self.normalize_velocity or self.velocity_stats is None:
            return velocity

        velocity = normalize_velocity_data(velocity, self.velocity_stats)
        return velocity

    def __len__(self):
        # Assuming two frames per data sample
        return len(self.frame_idxs) // (self.n_frames_in + self.n_frames_out)

    def __getitem__(self, idx):
        # Get the input and output frame indices
        seq_len = self.n_frames_in + self.n_frames_out
        idxs_in = [idx * seq_len + i for i in range(self.n_frames_in)]
        idxs_out = [
            idx * seq_len + self.n_frames_in + i
            for i in range(self.n_frames_out)  # noqa: E501
        ]
        # print(idx_in, idx_out)

        tif_in_path = [self.tif_names[idx_in] for idx_in in idxs_in]
        tif_out_path = [self.tif_names[idx_out] for idx_out in idxs_out]
        phy_in_path = [self.vel_names[idx_in] for idx_in in idxs_in]

        tif_in_data = []
        tif_out_data = []
        phy_in_data = []

        for i in range(self.n_frames_in):
            # Load and downsample TIF images
            tif_in_data.append(np.load(tif_in_path[i])[None, :, :, :])
            # Load physical info (npy file) and normalize if requested
            phy_info = np.load(phy_in_path[i])
            if self.normalize_velocity:
                phy_info = self.normalize_velocity_data(phy_info)
            phy_in_data.append(phy_info)

        tif_in = np.concatenate(tif_in_data, axis=0)
        phy_in = np.concatenate(phy_in_data, axis=0)

        # Concatenate rock tif (if available), TIF image, and physical info along the channel axis  # noqa E501
        if self.rock_tif is not None:
            in_data = np.concatenate([self.rock_tif, tif_in, phy_in], axis=0)
        else:
            in_data = np.concatenate([tif_in, phy_info], axis=0)

        for i in range(self.n_frames_out):
            tif_out_data.append(np.load(tif_out_path[i])[None, :, :, :])
        tif_out = np.concatenate(tif_out_data, axis=0)

        out_data = tif_out
        # Convert data to the chosen precision
        if self.use_fp16:
            in_data = in_data.astype(np.float16)
            out_data = out_data.astype(np.float16)
        else:
            in_data = in_data.astype(np.float32)
            out_data = out_data.astype(np.float32)

        return in_data, out_data


if __name__ == "__main__":
    rock_tif_path = "../data/rock/001_064_RobuGlass3_rec_16bit_abs_ShiftedDown18Left7_compressed.tif"  # noqa E501

    dataset = PoreShapeDataset(
        data_dir="./data/tif_073_segmented_tifs_down8/",
        vel_path="./data/073_final_down8_maxpool/",
        frame_idxs=[i for i in range(10, 70)],
        rock_tif_path=rock_tif_path,
        transform=None,
        rock_down_factor=8,
        # rock_slicer=(slice(510, None), slice(None, None), slice(None, None)),
    )
    print("dataset length: ", len(dataset))
    for i in range(5):
        # dataset[i]
        d_in, d_out = dataset[i]
        print(d_in.shape, d_out.shape)
