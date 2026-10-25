"""Graph network: encode-process-decode on the tracer graph."""

import torch
import torch.nn as nn


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o), nn.LayerNorm(o))


class PatchEncoder(nn.Module):
    """4^3 phase patch (rock -1, water 0, oil 1) -> hidden vector."""

    def __init__(self, hidden):
        super().__init__()

        def block(i, o):
            return [nn.Conv3d(i, o, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(o)]

        self.conv_layers = nn.Sequential(
            *block(1, 16), nn.MaxPool3d(2), *block(16, 32), nn.MaxPool3d(2),
            *block(32, 64), *block(64, 128), nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(128, hidden)

    def forward(self, patches):
        return self.fc(self.conv_layers(patches[:, None]).flatten(1))


class ProcessorLayer(nn.Module):
    """One round of message passing; nodes and edges both carry a residual."""

    def __init__(self, h):
        super().__init__()
        self.edge_mlp = mlp(3 * h, h, h)
        self.node_mlp = mlp(2 * h, h, h)

    def forward(self, x, edge_index, e):
        src, dst = edge_index
        e = e + self.edge_mlp(torch.cat([x[dst], x[src], e], 1))
        agg = torch.zeros_like(x).index_add_(0, dst, e)  # sum of incoming messages
        return x + self.node_mlp(torch.cat([x, agg], 1)), e


class GNS(nn.Module):
    """Node features: p_t, p_{t-1}, v_{t-1..t-5} (21). Edge features: delta of the 6-d coordinate and its
    norm (7). Output: v_t in voxels per frame. Inputs and output are standardised with the training
    statistics held as buffers (identity until set)."""

    STATS = ("mean_x", "std_x", "mean_e", "std_e", "mean_y", "std_y")

    def __init__(self, hidden=128, layers=10, node_dim=21, edge_dim=7):
        super().__init__()
        self.node_encoder = mlp(node_dim, hidden, hidden)
        self.cnn_encoder = PatchEncoder(hidden)
        self.edge_encoder = mlp(edge_dim, hidden, hidden)
        self.processor = nn.ModuleList(ProcessorLayer(hidden) for _ in range(layers))
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        for name, n in zip(self.STATS, (node_dim, node_dim, edge_dim, edge_dim, 3, 3)):
            self.register_buffer(name, torch.ones(n) if name.startswith("std") else torch.zeros(n))

    def forward(self, x, edge_index, edge_attr, patches):
        h = self.node_encoder((x - self.mean_x) / self.std_x) + self.cnn_encoder(patches)
        e = self.edge_encoder((edge_attr - self.mean_e) / self.std_e)
        for layer in self.processor:
            h, e = layer(h, edge_index, e)
        return self.decoder(h) * self.std_y + self.mean_y


def load_gns(ckpt, stats, **kw):
    """Checkpoint (weights only) plus its stats file, a list in GNS.STATS order."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    sd.update(zip(GNS.STATS, torch.load(stats, map_location="cpu")))
    gns = GNS(**kw)
    gns.load_state_dict(sd)
    return gns
