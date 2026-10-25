"""Tracer trajectories -> graphs for the graph network."""

import numpy as np
import pandas as pd
import torch

from ..constants import HISTORY, RADIUS
from .graph import radius_edges


class TracerGraphs:
    """Tracer CSV (frame, particle, z, y, x, vz, vy, vx) -> one graph per frame t.

    Node: p_t, p_{t-1}, v_{t-1..t-5}. Edge: see radius_edges. Target: v_t = (p_{t+1} - p_{t-1}) / 2,
    so the next position is p_{t-1} + 2 v_t. A graph holds the tracers seen at every frame t-5..t.
    """

    def __init__(self, csv, radius=RADIUS, history=HISTORY):
        df = pd.read_csv(csv, usecols=["frame", "particle", "z", "y", "x", "vz", "vy", "vx"])
        self.t0, t1 = int(df.frame.min()), int(df.frame.max())
        code, self.ids = pd.factorize(df.particle)
        self.table = np.full((len(self.ids), t1 - self.t0 + 1, 6), np.nan, np.float32)  # [particle, frame, zyx vzvyvx]
        self.table[code, df.frame.values - self.t0] = df[["z", "y", "x", "vz", "vy", "vx"]].values
        self.radius, self.history = radius, history

    def frame(self, t):
        """(p, v) of every tracer seen at frame t."""
        a = self.table[:, t - self.t0]
        a = torch.from_numpy(a[~np.isnan(a[:, 0])])
        return a[:, :3], a[:, 3:]

    def window(self, t):
        """ids and [N, history + 1, 6] of the tracers seen at every frame t-history..t."""
        k = t - self.t0
        w = self.table[:, k - self.history:k + 1]
        ok = ~np.isnan(w).any(axis=(1, 2))
        return self.ids[ok], torch.from_numpy(w[ok])

    @staticmethod
    def features(w):
        """Node features from a window: p_t, p_{t-1}, then v_{t-1} down to v_{t-history}."""
        return torch.cat([w[:, -1, :3], w[:, -2, :3], w[:, :-1, 3:].flip(1).flatten(1)], 1)

    def __getitem__(self, t):
        ids, w = self.window(t)
        x = self.features(w)
        ei, ea = radius_edges(x[:, :6], self.radius)
        return dict(x=x, edge_index=ei, edge_attr=ea, y=w[:, -1, 3:], particle=ids)

    def stats(self, frames):
        """Mean and std of node features, edge attributes and targets over frames, in GNS.STATS order."""
        g = [self[t] for t in frames]
        out = []
        for k in ("x", "edge_attr", "y"):
            a = torch.cat([d[k] for d in g])
            out += [a.mean(0), a.std(0)]
        return out
