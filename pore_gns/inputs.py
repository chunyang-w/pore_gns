"""Initial states: random stand-ins at the size of Exp. alpha, or measured frames of an experiment."""

import torch
import torch.nn.functional as F

from .constants import DS, HISTORY, MARGIN
from .data import PhaseFrames, TracerGraphs, load_rock, velocity_grid
from .rollout import State


def synthetic(dev, n=12882, shape=(153, 195, 195), seed=0):
    """Tracers in pore-sized clusters, blobby rock (~65 % solid) and oil."""
    g = torch.Generator(dev).manual_seed(seed)
    rand = lambda *s: torch.rand(*s, generator=g, device=dev)  # noqa: E731
    randn = lambda *s: torch.randn(*s, generator=g, device=dev)  # noqa: E731
    extent = torch.tensor(shape, device=dev) * DS - 2 * torch.tensor(MARGIN, device=dev)
    centres = rand(600, 3) * extent
    p = (centres[torch.randint(600, (n,), generator=g, device=dev)] + 15 * randn(n, 3)).clamp(torch.zeros(3, device=dev), extent - 1)
    vh = 0.3 * randn(n, 3).repeat(1, HISTORY) + 0.05 * randn(n, 3 * HISTORY)
    p1, p2 = p - vh[:, :3], p - vh[:, :3] - vh[:, 3:6]
    blob = lambda: F.avg_pool3d(rand(1, 1, *shape), 9, stride=1, padding=4)[0, 0]  # noqa: E731
    f = blob()
    rock = f > f.mean() - 0.4 * f.std()
    phases = []
    for _ in range(2):
        f = blob()
        phases.append(((f > f.mean() + f.std()) & ~rock).half()[None, None])
    grids = [velocity_grid(vh[:, 3:6], p2, shape).half()[None], velocity_grid(vh[:, :3], p1, shape).half()[None]]
    return State(rock, [p1, p], vh, phases, grids)


def measured(data, t, dev):
    """Experiment folder (velocity.csv, rock.tif, tif_view/*_<frame>.npy) at frame t; also its PhaseFrames."""
    tracers = TracerGraphs(f"{data}/velocity.csv")
    rock = load_rock(f"{data}/rock.tif").to(dev)
    frames = PhaseFrames(f"{data}/tif_view", rock, tracers)
    _, w = tracers.window(t)
    x = TracerGraphs.features(w.to(dev))
    phases = [frames.phase(k).to(dev).half()[None] for k in (t - 2, t - 1)]
    grids = [frames.grid(k).to(dev).half()[None] for k in (t - 2, t - 1)]
    return State(rock, [x[:, 3:6], x[:, :3]], x[:, 6:], phases, grids), frames
