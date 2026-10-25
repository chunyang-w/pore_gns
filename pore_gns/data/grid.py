"""The phase grid and what tracers read from it (patches) or write to it (velocity grid)."""

import tifffile
import torch

from ..constants import DS, MARGIN, PATCH


def subsample(a, f=DS):
    """Every f-th voxel, trailing remainder dropped."""
    d, h, w = (s // f * f for s in a.shape[-3:])
    return a[..., :d:f, :h:f, :w:f]


def crop(a, m=MARGIN, f=DS):
    """Grid -> tracer frame."""
    z, y, x = (s // f for s in m)
    D, H, W = a.shape[-3:]
    return a[..., z:D - z, y:H - y, x:W - x]


def load_rock(path):
    """Solid mask at grid resolution: True where the rock tif is non-zero."""
    return torch.from_numpy(subsample(tifffile.imread(path) > 0))


def extract_patches(phase, p, size=PATCH, f=DS):
    """size^3 cells of the phase grid (tracer frame) around each tracer, kept inside the grid."""
    lo = torch.div(p, f, rounding_mode="floor").long() - size // 2
    lo = torch.minimum(lo.clamp(min=0), torch.tensor(phase.shape, device=p.device) - size)
    o = torch.arange(size, device=p.device)
    oz, oy, ox = torch.meshgrid(o, o, o, indexing="ij")
    return phase[lo[:, 0, None, None, None] + oz, lo[:, 1, None, None, None] + oy, lo[:, 2, None, None, None] + ox]


def velocity_grid(v, p, shape, f=DS, m=MARGIN):
    """Tracer velocities on the U-Net grid, channels (vx, vy, vz): per cell the component of largest
    magnitude among its tracers, zero where there is none. Tracers outside the grid are dropped."""
    c = torch.div(p + torch.tensor(m, device=p.device), f, rounding_mode="floor").long()
    keep = ((c >= 0) & (c < torch.tensor(shape, device=p.device))).all(1)
    c, v = c[keep], v[keep]
    lin = (c[:, 0] * shape[1] + c[:, 1]) * shape[2] + c[:, 2]
    n = shape[0] * shape[1] * shape[2]
    out = torch.zeros(3, n, device=p.device)
    for k, ch in enumerate((2, 1, 0)):
        hi = torch.zeros(n, device=p.device).scatter_reduce(0, lin, v[:, ch], "amax")
        lo = torch.zeros(n, device=p.device).scatter_reduce(0, lin, v[:, ch], "amin")
        out[k] = torch.where(hi.abs() >= lo.abs(), hi, lo)
    return out.view(3, *shape)
