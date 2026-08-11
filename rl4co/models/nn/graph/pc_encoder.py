from __future__ import annotations

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.zoo.matnet.encoder import MatNetLayer


def pc_dense_edge_features(td: TensorDict, include_connection: bool = False) -> Tensor:
    """Build dense PC edge features with shape [B, N, N, F].

    Edge attributes describe pair relations. The current edge embedding consumes
    edge_features and W by default; ablations can additionally include the
    connection matrix as a feature instead of using it as a hard mask.
    Older PC datasets may still include assembly_adj as the first edge feature;
    that redundant channel is stripped here for backward compatibility.
    """

    edge_features = td["edge_features"].float()
    if edge_features.size(-1) == 7:
        edge_features = edge_features[..., 1:]

    features = [edge_features]
    if "W" in td.keys():
        features.append(td["W"].float().unsqueeze(-1))
    if include_connection and "assembly_adj" in td.keys():
        features.append(td["assembly_adj"].float().unsqueeze(-1))
    return torch.cat(features, dim=-1)


def pc_edge_mask(
    td: TensorDict,
    include_self_loops: bool = False,
    use_compat_mask: bool = True,
    use_message_mask: bool = True,
) -> Tensor:
    """Return the valid directed PC edge mask [B, N, N].

    ``compat`` is intentionally not required in the TensorDict. When
    ``use_compat_mask`` is enabled, the compatibility mask is derived from the
    raw PC constraints so the same dataset can be used for with/without-compat
    ablations.
    """

    n = td["node_features"].size(-2)
    if not use_message_mask:
        mask = torch.ones(*td.batch_size, n, n, dtype=torch.bool, device=td.device)
    elif "assembly_adj" in td.keys():
        mask = td["assembly_adj"].bool().clone()
    elif "relation_valid" in td.keys():
        mask = td["relation_valid"].bool().clone()
    else:
        mask = torch.ones(*td.batch_size, n, n, dtype=torch.bool, device=td.device)

    if "valid_part_mask" in td.keys():
        valid = td["valid_part_mask"].bool()
        mask = mask & valid.unsqueeze(-1) & valid.unsqueeze(-2)
    else:
        valid = None

    if use_message_mask and use_compat_mask:
        if "compat" in td.keys():
            compat = td["compat"].bool()
        else:
            compat = pc_raw_compatibility_mask(td)
        mask = mask & compat

    if not include_self_loops:
        n = mask.size(-1)
        eye = torch.eye(n, dtype=torch.bool, device=mask.device)
        mask = mask & ~eye
    return mask


def pc_raw_compatibility_mask(td: TensorDict) -> Tensor:
    """Compute pair compatibility from raw PC tensors without storing compat.

    A pair is compatible for representation-level message passing when the two
    parts can belong to the same group under the pairwise PC constraints:
    material variation, maintenance difference, relative motion, and standard
    part isolation. Connectivity is handled separately by ``pc_edge_mask``.
    """

    if "node_features" not in td.keys():
        raise KeyError("pc_raw_compatibility_mask requires node_features")

    n = td["node_features"].size(-2)
    batch_shape = td.batch_size
    device = td["node_features"].device

    compat = torch.ones(*batch_shape, n, n, dtype=torch.bool, device=device)

    for key in ("mat_var", "maint_diff", "rel_motion"):
        if key in td.keys():
            compat = compat & ~td[key].bool()

    if "valid_part_mask" in td.keys():
        valid = td["valid_part_mask"].bool()
        compat = compat & valid.unsqueeze(-1) & valid.unsqueeze(-2)
    else:
        valid = torch.ones(*batch_shape, n, dtype=torch.bool, device=device)

    if "isstandard" in td.keys():
        standard = td["isstandard"].eq(1) & valid
        standard_pair = standard.unsqueeze(-1) | standard.unsqueeze(-2)
        compat = compat & ~standard_pair

    return compat


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
    def __init__(
        self, embed_dim: int, dropout: float = 0.0, aggregation: str = "mean"
    ):
        super().__init__()
        if aggregation not in {"mean", "weighted"}:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. Expected 'mean' or 'weighted'."
            )
        self.aggregation = aggregation
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

        if self.aggregation == "weighted":
            scores = self.weight_score(pair_h)
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~edge_mask.unsqueeze(-1), mask_value)
            weights = torch.softmax(scores, dim=-2)
            weights = weights * edge_mask.unsqueeze(-1).float()
            weights = weights / weights.sum(dim=-2, keepdim=True).clamp_min(1e-9)
            agg = (weights * msg).sum(dim=-2)
        else:
            degree = edge_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            agg = msg.sum(dim=-2) / degree

        h = self.norm1(h + self.dropout(self.node_update(agg)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class PCEdgeAwareEncoder(AutoregressiveEncoder):
    """PC encoder that injects edge relations before the standard AM decoder.

    Inputs:
        node_features: [B, N, F_node]
        edge_features/W: [B, N, N, ...]
        assembly_adj: [B, N, N], used as the message passing mask
        compat: optional legacy mask. If missing, it is derived from raw
            constraints when use_compat_mask=True.

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
        edge_input_dim: int = 7,
        dropout: float = 0.0,
        use_compat_mask: bool = True,
        use_message_mask: bool = True,
        include_connection_feature: bool = False,
        exclude_sep_from_encoder: bool = False,
        aggregation: str = "mean",
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
        self.use_compat_mask = use_compat_mask
        self.use_message_mask = use_message_mask
        self.include_connection_feature = include_connection_feature
        self.exclude_sep_from_encoder = exclude_sep_from_encoder
        self.sep_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = nn.ModuleList(
            [
                PCEdgeMessagePassingLayer(
                    embed_dim, dropout=dropout, aggregation=aggregation
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, td: TensorDict, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        init_h = self.init_embedding(td)
        edge_h = self.edge_embedding(
            pc_dense_edge_features(
                td, include_connection=self.include_connection_feature
            ).to(init_h.device)
        )
        edge_mask = pc_edge_mask(
            td,
            use_compat_mask=self.use_compat_mask,
            use_message_mask=self.use_message_mask,
        ).to(init_h.device)
        if mask is not None:
            edge_mask = edge_mask & mask.bool()

        if self.exclude_sep_from_encoder:
            part_init_h = init_h[..., 1:, :]
            edge_h = edge_h[..., 1:, 1:, :]
            edge_mask = edge_mask[..., 1:, 1:]
            h = part_init_h
        else:
            part_init_h = None
            h = init_h

        for layer in self.layers:
            h = layer(h, edge_h, edge_mask)

        if self.exclude_sep_from_encoder:
            sep_h = self.sep_embedding.expand(*h.shape[:-2], 1, h.size(-1))
            h = torch.cat([sep_h, h], dim=-2)
            init_h = torch.cat([sep_h, part_init_h], dim=-2)
        return h, init_h


class PCMatNetEncoder(AutoregressiveEncoder):
    """MatNet-style PC encoder using pairwise PC relation matrices.

    The wrapper converts PC tensors into MatNet row/column embeddings and a dense
    relation matrix, then returns a single [B, N, D] embedding tensor so the
    standard AttentionModel decoder can be reused.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        env_name: str = "pc",
        init_embedding: nn.Module | None = None,
        edge_input_dim: int = 8,
        feedforward_hidden: int = 512,
        normalization: str = "batch",
        include_connection_feature: bool = True,
        use_message_mask: bool = False,
        exclude_sep_from_encoder: bool = False,
        bias: bool = False,
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
        self.row_projection = nn.Linear(embed_dim, embed_dim)
        self.col_projection = nn.Linear(embed_dim, embed_dim)
        self.relation_projection = nn.Sequential(
            nn.Linear(edge_input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        self.layers = nn.ModuleList(
            [
                MatNetLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=bias,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.Linear(2 * embed_dim, embed_dim)
        self.include_connection_feature = include_connection_feature
        self.use_message_mask = use_message_mask
        self.exclude_sep_from_encoder = exclude_sep_from_encoder
        self.sep_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def _encode_matnet(self, td: TensorDict) -> tuple[Tensor, Tensor, Tensor]:
        init_h = self.init_embedding(td)
        if self.exclude_sep_from_encoder:
            part_init_h = init_h[..., 1:, :]
        else:
            part_init_h = None

        row_h = self.row_projection(init_h)
        col_h = self.col_projection(init_h)
        relation_features = pc_dense_edge_features(
            td, include_connection=self.include_connection_feature
        ).to(init_h.device)

        if self.exclude_sep_from_encoder:
            row_h = row_h[..., 1:, :]
            col_h = col_h[..., 1:, :]
            relation_features = relation_features[..., 1:, 1:, :]

        relation_matrix = self.relation_projection(relation_features).squeeze(-1)
        attn_mask = None
        if self.use_message_mask:
            attn_mask = pc_edge_mask(
                td, use_compat_mask=False, use_message_mask=True
            ).to(init_h.device)
            if self.exclude_sep_from_encoder:
                attn_mask = attn_mask[..., 1:, 1:]

        for layer in self.layers:
            row_h, col_h = layer(row_h, col_h, relation_matrix, attn_mask=attn_mask)
        matnet_h = self.output_projection(torch.cat([row_h, col_h], dim=-1))

        if self.exclude_sep_from_encoder:
            sep_h = self.sep_embedding.expand(
                *matnet_h.shape[:-2], 1, matnet_h.size(-1)
            )
            matnet_h = torch.cat([sep_h, matnet_h], dim=-2)
            init_h = torch.cat([sep_h, part_init_h], dim=-2)
        return matnet_h, init_h, relation_matrix

    def forward(self, td: TensorDict, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        h, init_h, _ = self._encode_matnet(td)
        return h, init_h


class PCNodeFFNLayer(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor) -> Tensor:
        return self.norm(h + self.dropout(self.ffn(h)))


class PCSplitHybridEncoder(AutoregressiveEncoder):
    """Split PC encoder: node attributes via a node branch, pair relations via MatNet."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        env_name: str = "pc",
        init_embedding: nn.Module | None = None,
        edge_input_dim: int = 8,
        feedforward_hidden: int = 512,
        normalization: str = "batch",
        include_connection_feature: bool = True,
        use_message_mask: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
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
        self.node_layers = nn.ModuleList(
            [PCNodeFFNLayer(embed_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.matnet_encoder = PCMatNetEncoder(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            env_name=env_name,
            init_embedding=self.init_embedding,
            edge_input_dim=edge_input_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            include_connection_feature=include_connection_feature,
            use_message_mask=use_message_mask,
            bias=bias,
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, td: TensorDict, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        init_h = self.init_embedding(td)
        node_h = init_h
        for layer in self.node_layers:
            node_h = layer(node_h)
        matnet_h, _, _ = self.matnet_encoder._encode_matnet(td)
        h = self.norm(node_h + self.fusion(torch.cat([node_h, matnet_h], dim=-1)))
        return h, init_h
