"""One coupled step per CT frame (0.25 s).

The U-Net predicts the phase at t from frames t-2 and t-1; the graph network predicts the tracer
velocities at t from the kinematic history and that phase; the velocities give the positions at t+1
and the velocity grid the U-Net reads next. Each part saturates the GPU, so the step is their sum.
"""

import time
from typing import NamedTuple

import torch

from .data import crop, extract_patches, radius_edges, velocity_grid
from .models import fold_bn

CL = torch.channels_last_3d


class State(NamedTuple):
    """What a step needs from the past."""

    rock: torch.Tensor     # bool (D, H, W) at grid resolution
    p: list                # [p_{t-1}, p_t], voxels in the tracer frame
    v_hist: torch.Tensor   # [N, 15]: v_{t-1} .. v_{t-5}
    phases: list           # [phase_{t-2}, phase_{t-1}], (1, 1, D, H, W) fp16
    grids: list            # velocity grids of the same frames, (1, 3, D, H, W) fp16


class Rollout:
    def __init__(self, gns, unet, state, compile=True):
        self.shape = tuple(state.rock.shape)
        self.rock, self.rock_crop = state.rock[None, None].half(), crop(state.rock)
        self.p, self.vh, self.phases, self.grids = list(state.p), state.v_hist, list(state.phases), list(state.grids)
        self.ei, self.ea = radius_edges(torch.cat([self.p[-1], self.p[-2]], 1))
        unet = fold_bn(unet).half().to(memory_format=CL)  # inference form: no batch-norm passes, fp16, NDHWC
        self.unet = torch.compile(unet) if compile else unet
        self.gns = torch.compile(gns.eval(), dynamic=True) if compile else gns.eval()
        torch.backends.cudnn.benchmark = True

    def unet_input(self):
        return torch.cat([self.rock, *self.phases[-2:], *self.grids[-2:]], 1).contiguous(memory_format=CL)

    def gns_part(self, phase):
        """v_t from the kinematic history and phase_t; then p_{t+1} = p_{t-1} + 2 v_t, the edges of the
        next step and the velocity grid of t."""
        phase = crop(phase[0, 0]).float().masked_fill(self.rock_crop, -1.0)
        patches = extract_patches(phase, self.p[-1])
        x = torch.cat([self.p[-1], self.p[-2], self.vh], 1)
        with torch.autocast("cuda", torch.float16):
            v = self.gns(x, self.ei, self.ea, patches).float()
        p_next = self.p[-2] + 2 * v
        ei, ea = radius_edges(torch.cat([p_next, self.p[-1]], 1))
        return v, p_next, ei, ea, velocity_grid(v, self.p[-1], self.shape).half()[None]

    @torch.no_grad()
    def step(self):
        phase = (self.unet(self.unet_input()) > 0).half()
        v, p_next, ei, ea, grid = self.gns_part(phase)
        self.phases.append(phase)
        self.grids.append(grid)
        self.p.append(p_next)
        self.vh = torch.cat([v, self.vh[:, :-3]], 1)
        self.ei, self.ea = ei, ea
        return v, p_next, phase

    @torch.no_grad()
    def parts(self, n=10):
        """Each part alone, synchronised: (U-Net, graph network with patches, edges and grid) in seconds."""
        def timeit(fn):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n
        x, phase = self.unet_input(), self.phases[-1]
        return timeit(lambda: self.unet(x)), timeit(lambda: self.gns_part(phase))
