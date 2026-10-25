"""The tracer graph."""

import torch
from torch_cluster import radius_graph

from ..constants import RADIUS


def radius_edges(c, r=RADIUS, k=64):
    """Directed edges between tracers within r in the 6-d coordinate [p_t, p_{t-1}], self loops included,
    at most k neighbours per tracer; attributes c[dst] - c[src] and its norm."""
    c = c.float().contiguous()
    ei = radius_graph(c, r=r, loop=True, max_num_neighbors=k)
    d = c[ei[1]] - c[ei[0]]
    return ei, torch.cat([d, d.norm(dim=1, keepdim=True)], 1)
