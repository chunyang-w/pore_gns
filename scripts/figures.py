"""README figures from a measured rollout: docs/rollout.png (Dice per frame, time per step by part,
16 against 64 base filters) and docs/slice.png (measured against predicted phase after the rollout).

python scripts/figures.py --data D --gns G --stats S --unet U
"""

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from pore_gns import DS, MARGIN, Rollout, UNet, dice, load_gns, measured  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
BLUE, ORANGE, AQUA, ROCK = "#2a78d6", "#eb6834", "#1baf7a", "#d9d7d1"
DOCS = Path(__file__).resolve().parents[1] / "docs"


def axes_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


@torch.no_grad()
def run(gns, unet, state, frames, t, steps):
    """Rollout of `steps` frames from t: (per-frame Dice, per-step seconds), last v, p_{t+1}, phase."""
    roll = Rollout(gns, unet, state)
    rec = []
    for k in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        v, p, phase = roll.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rec.append((dice(phase[0, 0], frames.phase(t + k)[0].to(phase.device)), dt))
    return roll, np.array(rec), v, p, phase


def rollout_figure(t, dices, parts, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw=dict(width_ratios=[3, 2], wspace=0.35))
    frames = t + np.arange(len(dices))
    a.plot(frames, dices, color=BLUE, lw=2, marker="o", ms=4, mec="white", mew=0.8)
    for k in (0, len(dices) - 1):
        a.annotate(f"{dices[k]:.4f}", (frames[k], dices[k]), (0, 8), textcoords="offset points", ha="center", fontsize=8, color=INK)
    a.set(xlabel="frame", ylabel="Dice against the measured phase", ylim=(0.98, 0.995))
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_title("Exp. alpha, 30-step rollout", fontsize=9, color=INK, loc="left")

    labels = list(parts)
    u = np.array([parts[k][0] for k in labels]) * 1e3
    g = np.array([parts[k][1] for k in labels]) * 1e3
    y = np.arange(len(labels))
    gap = 0.01 * (u + g).max()
    b.barh(y, u, height=0.5, color=BLUE)
    b.barh(y, g, left=u + gap, height=0.5, color=ORANGE)
    for yi, tot in zip(y, u + g):
        b.text(tot + 3 * gap, yi, f"{tot:.0f} ms", va="center", fontsize=8, color=INK)
    b.set_yticks(y, labels)
    b.set_ylim(1.6, -0.6)
    b.set_xlim(0, (u + g).max() * 1.25)
    b.set_xlabel("time per step, ms")
    b.grid(axis="x", color=GRID, lw=0.6)
    b.set_axisbelow(True)
    b.set_title("U-Net base filters", fontsize=9, color=INK, loc="left")
    b.legend(handles=[Patch(color=BLUE, label="U-Net"), Patch(color=ORANGE, label="graph network, patches, edges, grid")],
             frameon=False, fontsize=7, loc="upper right")
    for ax in (a, b):
        axes_style(ax)
    fig.savefig(path, dpi=160, bbox_inches="tight")


def slice_figure(meas, pred, rock, p, t_last, steps, path):
    """Slice of largest measured oil area: measured, predicted (tracers within one cell as dots), disagreement."""
    z = int(meas.sum((1, 2)).argmax())
    c = np.floor_divide(p + np.array(MARGIN), DS)
    near = np.abs(c[:, 0] - z) <= 1
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.4), gridspec_kw=dict(wspace=0.05))

    def image(oil, extra=()):
        img = np.ones((*oil.shape, 3))
        img[rock[z]] = matplotlib.colors.to_rgb(ROCK)
        img[oil] = matplotlib.colors.to_rgb(BLUE)
        for mask, col in extra:
            img[mask] = matplotlib.colors.to_rgb(col)
        return img

    axs[0].imshow(image(meas[z]))
    axs[0].set_title(f"measured, frame {t_last}", fontsize=9, color=INK, loc="left")
    axs[1].imshow(image(pred[z]))
    axs[1].scatter(c[near, 2], c[near, 1], s=3, c=INK, lw=0)
    axs[1].set_title(f"predicted after {steps} steps", fontsize=9, color=INK, loc="left")
    axs[2].imshow(image(np.zeros_like(meas[z]), [(pred[z] & ~meas[z], ORANGE), (meas[z] & ~pred[z], AQUA)]))
    axs[2].set_title("disagreement", fontsize=9, color=INK, loc="left")
    axs[2].legend(handles=[Patch(color=ORANGE, label="predicted only"), Patch(color=AQUA, label="measured only")],
                  frameon=False, fontsize=7, loc="lower right")
    for ax in axs:
        ax.set_axis_off()
    fig.text(0.5, 0.02, f"z slice {z} of {meas.shape[0]} (22 um cells): rock grey, oil blue; dots: tracers within one cell of the slice", ha="center", fontsize=8, color=MUTED)
    fig.savefig(path, dpi=160, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--gns", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--unet", required=True, help="16-base-filter state dict")
    ap.add_argument("--t", type=int, default=150)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    dev = torch.device("cuda")
    DOCS.mkdir(exist_ok=True)

    gns = load_gns(a.gns, a.stats).to(dev)
    unet = UNet(base=16)
    unet.load_state_dict(torch.load(a.unet, map_location="cpu"))
    state, frames = measured(a.data, a.t, dev)
    roll, rec, v, p, phase = run(gns, unet.to(dev), state, frames, a.t, a.steps)
    parts = {"16": roll.parts()}
    p_last = roll.p[-2].cpu().numpy()  # tracers at the last predicted frame (p is one frame ahead)
    print(f"Dice {rec[0, 0]:.4f} -> {rec[-1, 0]:.4f}; step {rec[1:, 1].mean():.4f} s; parts {parts['16']}")

    del roll
    parts["64 (paper)"] = Rollout(gns, UNet(base=64).to(dev), state).parts()  # timing only: weights do not matter
    print(f"64 base filters: parts {parts['64 (paper)']}")
    rollout_figure(a.t, rec[:, 0], parts, DOCS / "rollout.png")

    t_last = a.t + a.steps - 1
    slice_figure(frames.phase(t_last)[0].numpy() > 0.5, phase[0, 0].bool().cpu().numpy(), state.rock.cpu().numpy(),
                 p_last, t_last, a.steps, DOCS / "slice.png")


if __name__ == "__main__":
    main()
