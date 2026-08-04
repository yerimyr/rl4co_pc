from __future__ import annotations

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from rl4co.models.nn.env_embeddings import env_init_embedding


def pc_dense_edge_features(td: TensorDict) -> Tensor:
    """Build dense PC edge features with shape [B, N, N, F].

    The generator already exposes edge_features, but W and compat
    are important PC relations as well. Keeping this conversion in one place
    makes it clear which edge tensors the edge-aware encoder consumes.
    """

    features = [td["edge_features"].float()]
    for key in ("W", "compat"):
        if key in td.keys():
            features.append(td[key].float().unsqueeze(-1))
    return torch.cat(features, dim=-1)


def pc_edge_mask(td: TensorDict, include_self_loops: bool = False) -> Tensor:
    """Return the valid directed PC edge mask [B, N, N]."""

    if "assembly_adj" in td.keys():
        mask = td["assembly_adj"].bool().clone()
    elif "relation_valid" in td.keys():
        mask = td["relation_valid"].bool().clone()
    else:
        n = td["node_features"].size(-2)
        mask = torch.ones(*td.batch_size, n, n, dtype=torch.bool, device=td.device)

    if "valid_part_mask" in td.keys():
        valid = td["valid_part_mask"].bool()
        mask = mask & valid.unsqueeze(-1) & valid.unsqueeze(-2)

    if not include_self_loops:
        n = mask.size(-1)
        eye = torch.eye(n, dtype=torch.bool, device=mask.device)
        mask = mask & ~eye
    return mask


def pc_tensordict_to_edge_index(
    td: TensorDict, include_self_loops: bool = False
) -> list[tuple[Tensor, Tensor]]:
    """Convert a batched PC TensorDict to per-instance edge_index/edge_attr.

    This helper is useful for validation or for plugging PC into PyG-style
    encoders later. The dense encoder below uses the dense representation to
    avoid a hard dependency on torch_geometric.
    """

    edge_attr_dense = pc_dense_edge_features(td)
    mask = pc_edge_mask(td, include_self_loops=include_self_loops)
    graphs = []
    for b in range(mask.size(0)):
        edge_index = mask[b].nonzero(as_tuple=False).transpose(0, 1).contiguous()
        edge_attr = edge_attr_dense[b][mask[b]]
        graphs.append((edge_index, edge_attr))
    return graphs


class PCEdgeMessagePassingLayer(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.node_update = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, edge_h: Tensor, edge_mask: Tensor) -> Tensor:
        h_i = h.unsqueeze(-2).expand_as(edge_h)
        h_j = h.unsqueeze(-3).expand_as(edge_h)
        msg = self.message(torch.cat([h_i, h_j, edge_h], dim=-1))
        msg = msg * torch.sigmoid(self.gate(edge_h))
        msg = msg * edge_mask.unsqueeze(-1).float()

        degree = edge_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        agg = msg.sum(dim=-2) / degree
        h = self.norm1(h + self.dropout(self.node_update(agg)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class PCEdgeAwareEncoder(AutoregressiveEncoder):
    """PC encoder that injects edge relations before the standard AM decoder.

    Inputs:
        node_features: [B, N, F_node]
        edge_features/W/compat: [B, N, N, ...]

    Output:
        h: [B, N, embed_dim], compatible with AttentionModelDecoder.
        init_h: [B, N, embed_dim], used by RL4CO's constructive policy cache.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        env_name: str = "pc",
        init_embedding: nn.Module | None = None,
        edge_input_dim: int = 9,
        dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.init_embedding = (
            env_init_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )
        self.edge_embedding = nn.Linear(edge_input_dim, embed_dim)
        self.layers = nn.ModuleList(
            [PCEdgeMessagePassingLayer(embed_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, td: TensorDict, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        init_h = self.init_embedding(td)
        edge_h = self.edge_embedding(pc_dense_edge_features(td).to(init_h.device))
        edge_mask = pc_edge_mask(td).to(init_h.device)
        if mask is not None:
            edge_mask = edge_mask & mask.bool()

        h = init_h
        for layer in self.layers:
            h = layer(h, edge_h, edge_mask)
        return h, init_h
