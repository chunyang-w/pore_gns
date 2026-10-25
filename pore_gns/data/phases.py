"""Phase frames and velocity grids -> U-Net pairs."""

import glob
import re

import numpy as np
import torch

from .grid import velocity_grid


class PhaseFrames(torch.utils.data.Dataset):
    """[rock, phase_{t-2}, phase_{t-1}, grid_{t-2}, grid_{t-1}] (9 channels) -> phase_t.

    phase_dir holds one .npy per frame at grid resolution (oil 1, else 0), the frame number before .npy.
    The velocity grids come from the tracers of the same frames.
    """

    def __init__(self, phase_dir, rock, tracers, frames=()):
        self.files = {int(re.findall(r"(\d+)\.npy$", f)[0]): f for f in glob.glob(f"{phase_dir}/*.npy")}
        self.rock, self.tracers, self.frames = rock.float()[None], tracers, list(frames)

    def phase(self, t):
        return torch.from_numpy(np.load(self.files[t])).float()[None]

    def grid(self, t):
        p, v = self.tracers.frame(t)
        return velocity_grid(v, p, self.rock.shape[1:])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        t = self.frames[i]
        x = torch.cat([self.rock, self.phase(t - 2), self.phase(t - 1), self.grid(t - 2), self.grid(t - 1)])
        return x, self.phase(t)
