"""
Benchmark for vecset model on dice score.

Written by: Chunyang Wang
GitHub username: chunyang-w
Example usage:
python pore_net/vecset/vis_cmp.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch
import numpy as np
import pyvista as pv
from skimage import measure
import pore_net.vecset.model as models_ae
from pore_net.vecset.dataset import PoreShapeDataset
import argparse

save_fig = False
num_frames = 250
data_idx = 190

data_path = "/Users/chunyang/projects/particle/ditto/data/073_segmented_tifs_pad_size20_down10dist_thresh5/"
checkpoints = "/Users/chunyang/Downloads/checkpoint-3350.pth"
model_name = "ae_d512_m512"
density = 100


def get_mesh(verts, faces, n_iter=10):
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    faces_pv = faces_pv.flatten()
    mesh = pv.PolyData(var_inp=verts, faces=faces_pv)
    mesh = mesh.smooth(n_iter=n_iter, relaxation_factor=0.5)
    return mesh


if __name__ == "__main__":
    # Add argparse.Namespace to safe globals
    torch.serialization.add_safe_globals([argparse.Namespace])
    dataset = PoreShapeDataset(  # Sample points from all points
        data_dir=data_path, transform=None, split="val"
    )
    # Plot ground truth
    dataset_gt = PoreShapeDataset(  # All points are retained
        data_dir=data_path,
        sampling=False,
        split="val",
        transform=None,
    )

    model = models_ae.__dict__[model_name]()
    model.eval()
    model.load_state_dict(
        torch.load(checkpoints, map_location="cpu", weights_only=False)["model"],
        strict=True,
    )

    gap = 2.0 / density
    x = np.linspace(-1, 1, density)
    y = np.linspace(-1, 1, density)
    z = np.linspace(-1, 1, density)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid_pred = (
        torch.from_numpy(np.stack([zv, yv, xv]).astype(np.float32))
        .view(3, -1)
        .transpose(0, 1)[None]
    )

    model.eval()

    sample_pred = dataset[data_idx]
    points_pred, labels_pred, surface_pred = sample_pred

    points_pred = points_pred.unsqueeze(0)
    labels_pred = labels_pred.unsqueeze(0)
    surface_pred = surface_pred.unsqueeze(0)
    # print(surface_pred.shape)

    with torch.no_grad():
        pred = model(surface_pred, grid_pred)["logits"]

    sample_gt = dataset_gt[data_idx]
    points_gt, labels_gt, surface_gt = sample_gt
    labels_gt = labels_gt.numpy()

    # Show mesh comparison
    scale_pred = None
    scale_pred = (161, 161, 127)
    nomalise = True

    cube_pred = (pred > -0).numpy().reshape(density, density, density)
    # pad the cube to make sure the surface is closed
    cube_pred = np.pad(cube_pred, 5, mode="constant")
    cube_gt = labels_gt.reshape(127, 161, 161).transpose(1, 2, 0)

    verts_pred, faces_pred, _, _ = measure.marching_cubes(cube_pred, level=0)
    verts_gt, faces_gt, _, _ = measure.marching_cubes(cube_gt, level=0)

    if scale_pred is not None:
        verts_pred = verts_pred * scale_pred

    if nomalise:
        verts_pred = verts_pred / np.max(verts_pred)
        verts_gt = verts_gt / np.max(verts_gt)

    mesh_pred = get_mesh(verts_pred, faces_pred)
    mesh_gt = get_mesh(verts_gt, faces_gt)

    # Plot
    p = pv.Plotter(
        title="Surface Reconstruction", shape=(1, 2), window_size=[1000, 500]
    )

    p.subplot(0, 0)
    p.add_mesh(mesh_pred, color="lightgreen")
    p.add_text("Prediction", font_size=20)

    p.subplot(0, 1)
    p.add_mesh(mesh_gt, color="lightblue")
    p.add_text("Ground Truth", font_size=20)

    p.link_views()

    if not save_fig:
        p.show()

    elif save_fig:
        p.open_gif("output/compare_move_camera_.gif", fps=24)
        p.camera.elevation = p.camera.elevation - 10
        p.camera.zoom(1.2)
        for i in range(num_frames):
            p.camera.azimuth = p.camera.azimuth - 1
            p.write_frame()
        p.close()
