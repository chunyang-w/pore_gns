"""Coupled rollout, timed per frame; Dice against measured frames when an experiment is given.

python -m pore_gns.demo                                        # random inputs at the size of Exp. alpha, random weights
python -m pore_gns.demo --data D --gns G --stats S --unet U     # measured frames and trained weights
"""

import argparse
import time

import numpy as np
import torch

from .inputs import measured, synthetic
from .metrics import dice
from .models import GNS, UNet, load_gns
from .rollout import Rollout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="experiment folder; random inputs when absent")
    ap.add_argument("--gns", help="graph-network checkpoint")
    ap.add_argument("--stats", help="its normalisation statistics")
    ap.add_argument("--unet", help="U-Net state dict")
    ap.add_argument("--base", type=int, default=16, help="U-Net base filters (64 in the paper)")
    ap.add_argument("--t", type=int, default=150, help="first frame predicted")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--eager", action="store_true", help="skip torch.compile")
    a = ap.parse_args()
    dev = torch.device("cuda")

    gns = (load_gns(a.gns, a.stats) if a.gns else GNS()).to(dev)
    unet = UNet(base=a.base)
    if a.unet:
        unet.load_state_dict(torch.load(a.unet, map_location="cpu"))
    state, frames = measured(a.data, a.t, dev) if a.data else (synthetic(dev), None)
    roll = Rollout(gns, unet.to(dev), state, compile=not a.eager)
    print(f"{len(state.v_hist)} tracers, grid {roll.shape}, {roll.ei.shape[1]} edges")

    times = []
    for k in range(a.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        v, p_next, phase = roll.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        line = f"frame {a.t + k}: {times[-1]:.4f} s, {roll.ei.shape[1]} edges, |v| {v.norm(dim=1).mean():.3f} vox/frame"
        if frames is not None:
            line += f", Dice {dice(phase[0, 0], frames.phase(a.t + k)[0].to(dev)):.4f}"
        print(line, flush=True)

    s = np.array(times[1:])
    u, g = roll.parts()
    print(f"\nstep {s.mean():.4f} +- {s.std():.4f} s over steps 1-{a.steps - 1} (step 0: {times[0]:.1f} s, warm-up)")
    print(f"U-Net {u:.4f} s, graph network with patches, edges and grid {g:.4f} s")


if __name__ == "__main__":
    main()
