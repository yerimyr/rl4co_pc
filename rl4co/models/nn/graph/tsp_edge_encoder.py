from __future__ import annotations

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.models.common.constructive import AutoregressiveEncoder


def tsp_distance_matrix(td: TensorDict) -> Tensor:
    """Build pairwise Euclidean distance matrix from TSP coordinates.

    The original TSP instance still stores city coordinates in ``locs``. This
    helper is the preprocessing step for the edge-distance experiment:
    ``locs [B, N, 2] -> distances [B, N, N, 1]``.
    """

    locs = td["locs"].float()
    return torch.cdist(locs, locs, p=2).unsqueeze(-1)


class TSPDistanceMessagePassingLayer(nn.Module):
    """Message passing layer that uses edge-distance embeddings only."""

    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.weight_score = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
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
        pair_h = torch.cat([h_i, h_j, edge_h], dim=-1)

        msg = self.message(pair_h)
        msg = msg * torch.sigmoid(self.gate(edge_h))
        msg = msg * edge_mask.unsqueeze(-1).float()

        scores = self.weight_score(pair_h)
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~edge_mask.unsqueeze(-1), mask_value)
        weights = torch.softmax(scores, dim=-2)
        weights = weights * edge_mask.unsqueeze(-1).float()
        weights = weights / weights.sum(dim=-2, keepdim=True).clamp_min(1e-9)

        agg = (weights * msg).sum(dim=-2)
        h = self.norm1(h + self.dropout(self.node_update(agg)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class TSPDistanceMatrixEncoder(AutoregressiveEncoder):
    """TSP encoder for the edge-distance input ablation.

    Standard RL4CO TSP AM embeds city coordinates directly:
    ``locs [B, N, 2] -> node embeddings [B, N, D]``.

    This encoder keeps the same TSP environment, decoder, action mask, reward,
    and REINFORCE training loop, but changes the encoder-side input:

    1. Build pairwise distance matrix from coordinates.
    2. Treat each edge distance as a scalar edge feature.
    3. Build node embeddings by aggregating learned edge-distance messages.

    The output shape remains ``[B, N, embed_dim]`` so the standard
    AttentionModel decoder can be reused without changes.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.edge_embedding = nn.Linear(1, embed_dim)
        self.node_seed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.init_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.layers = nn.ModuleList(
            [
                TSPDistanceMessagePassingLayer(embed_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, td: TensorDict, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        distances = tsp_distance_matrix(td)
        edge_h = self.edge_embedding(distances)

        n = edge_h.size(-2)
        eye = torch.eye(n, dtype=torch.bool, device=edge_h.device)
        edge_mask = ~eye.expand(*edge_h.shape[:-3], n, n)
        if mask is not None:
            edge_mask = edge_mask & mask.bool()

        degree = edge_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        init_h = (edge_h * edge_mask.unsqueeze(-1).float()).sum(dim=-2) / degree
        init_h = self.init_projection(init_h) + self.node_seed.expand_as(init_h)

        h = init_h
        for layer in self.layers:
            h = layer(h, edge_h, edge_mask)
        return h, init_h


class TSPDistanceMatrixInitEmbedding(nn.Module):
    """Initial embedding that replaces coordinate features with distance rows.

    This keeps the standard AttentionModelEncoder unchanged. The only changed
    part is the feature passed into its first embedding step:

    ``locs [B, N, 2] -> distance matrix [B, N, N] -> embedding [B, N, D]``.
    """

    def __init__(self, embed_dim: int, linear_bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_bias = linear_bias
        self.init_embed: nn.Linear | None = None

    def _get_or_create_projection(self, num_nodes: int, device: torch.device) -> nn.Linear:
        if self.init_embed is None or self.init_embed.in_features != num_nodes:
            self.init_embed = nn.Linear(num_nodes, self.embed_dim, self.linear_bias).to(device)
        return self.init_embed

    def forward(self, td: TensorDict) -> Tensor:
        distance_rows = tsp_distance_matrix(td).squeeze(-1)
        projection = self._get_or_create_projection(
            distance_rows.size(-1), distance_rows.device
        )
        return projection(distance_rows)


class TSPEdgeListDistanceInitEmbedding(nn.Module):
    """Initial embedding from a compressed TSP edge list.

    This implements the representation requested for the TSP ablation:

    ``locs [B, N, 2] -> edge distances [B, E, 1]``

    where each row is one undirected edge ``(i, j)`` and the only feature is
    its distance. Since the standard AM decoder still chooses among cities, the
    edge embeddings are aggregated back to one embedding per city:

    ``edge embeddings [B, E, D] -> node embeddings [B, N, D]``.
    """

    def __init__(self, embed_dim: int, aggregation: str = "mean"):
        super().__init__()
        if aggregation not in {"mean", "sum"}:
            raise ValueError("aggregation must be one of {'mean', 'sum'}")
        self.embed_dim = embed_dim
        self.aggregation = aggregation
        self.edge_embed = nn.Linear(1, embed_dim)
        self.node_seed = nn.Parameter(torch.zeros(1, 1, embed_dim))

    @staticmethod
    def _edge_index(num_nodes: int, device: torch.device) -> tuple[Tensor, Tensor]:
        return torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)

    def forward(self, td: TensorDict) -> Tensor:
        locs = td["locs"].float()
        batch_size, num_nodes, _ = locs.shape
        src, dst = self._edge_index(num_nodes, locs.device)

        distances = torch.norm(locs[:, src] - locs[:, dst], dim=-1, keepdim=True)
        edge_h = self.edge_embed(distances)

        node_h = edge_h.new_zeros(batch_size, num_nodes, self.embed_dim)
        node_h.index_add_(1, src, edge_h)
        node_h.index_add_(1, dst, edge_h)

        if self.aggregation == "mean":
            degree = edge_h.new_full((num_nodes,), max(num_nodes - 1, 1))
            node_h = node_h / degree.view(1, num_nodes, 1)

        return node_h + self.node_seed.expand(batch_size, num_nodes, self.embed_dim)
