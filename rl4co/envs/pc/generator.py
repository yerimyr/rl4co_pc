from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tensordict import TensorDict

from rl4co.envs.common.utils import Generator


@dataclass
class FPIGeneratorParams:
    num_parts: int = 4
    max_num_parts: int | None = None
    material_types: int = 3
    L_low: float = 5.0
    L_high: float = 120.0
    W_low: float = 5.0
    W_high: float = 55.0
    H_low: float = 2.0
    H_high: float = 24.0
    p_maint_H: float = 0.10
    p_standard: float = 0.10
    p_relative_motion: float = 0.10
    p_extra_edge: float = 0.50
    topology_mode: str = "mixed"
    p_chain: float = 0.10
    p_star: float = 0.10
    p_tree: float = 0.10
    p_two_module_bridge: float = 0.35
    p_dense_clustered: float = 0.60
    p_sparse_random: float = 0.10
    build_limit_L: float = 1000.0
    build_limit_W: float = 1000.0
    build_limit_H: float = 500.0


class FPIGenerator(Generator):
    """General-graph FPI instance generator for part consolidation.

    Node 0 is reserved as the SEP token. Real parts occupy nodes 1..N.
    Variable-size instances are padded to max_num_parts and expose valid_part_mask.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p = FPIGeneratorParams(**kwargs)
        self.min_num_parts = int(self.p.num_parts)
        self.max_num_parts = int(self.p.max_num_parts or self.p.num_parts)
        if self.max_num_parts < self.min_num_parts:
            raise ValueError("max_num_parts must be >= num_parts")

        self.num_parts = self.max_num_parts
        self.num_nodes = self.max_num_parts + 1
        self.topology_names = [
            "chain",
            "star",
            "tree",
            "two_module_bridge",
            "dense_clustered",
            "sparse_random",
        ]
        self.node_feat_dim = self.p.material_types + 3 + 1 + 1 + 1
        self.edge_feat_dim = 1 + 1 + 3 + 1 + 1
        self.build_limit = torch.tensor(
            [self.p.build_limit_L, self.p.build_limit_W, self.p.build_limit_H],
            dtype=torch.float32,
        )

    def _add_undirected_edge(self, adj: torch.Tensor, i: int, j: int) -> None:
        if i != j:
            adj[i, j] = True
            adj[j, i] = True

    def _connect_sequence(self, adj: torch.Tensor, nodes: list[int]) -> None:
        for idx in range(len(nodes) - 1):
            self._add_undirected_edge(adj, nodes[idx], nodes[idx + 1])

    def _sample_topology_id(self, device: torch.device) -> int:
        mode = self.p.topology_mode
        if mode != "mixed":
            if mode not in self.topology_names:
                raise ValueError(f"Unknown topology_mode: {mode}")
            return self.topology_names.index(mode)

        probs = torch.tensor(
            [
                self.p.p_chain,
                self.p.p_star,
                self.p.p_tree,
                self.p.p_two_module_bridge,
                self.p.p_dense_clustered,
                self.p.p_sparse_random,
            ],
            dtype=torch.float32,
            device=device,
        )
        probs = probs / probs.sum().clamp_min(1e-8)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _build_chain_adjacency(self, n: int, device: torch.device) -> torch.Tensor:
        adj = torch.zeros((n, n), dtype=torch.bool, device=device)
        self._connect_sequence(adj, torch.randperm(n, device=device).tolist())
        return adj

    def _build_star_adjacency(self, n: int, device: torch.device) -> torch.Tensor:
        adj = torch.zeros((n, n), dtype=torch.bool, device=device)
        center = int(torch.randint(0, n, (1,), device=device).item())
        for node in range(n):
            self._add_undirected_edge(adj, center, node)
        return adj

    def _build_tree_adjacency(self, n: int, device: torch.device) -> torch.Tensor:
        adj = torch.zeros((n, n), dtype=torch.bool, device=device)
        nodes = torch.randperm(n, device=device)
        for i in range(1, n):
            child = int(nodes[i].item())
            parent = int(nodes[int(torch.randint(0, i, (1,), device=device).item())].item())
            self._add_undirected_edge(adj, parent, child)
        return adj

    def _build_two_module_bridge_adjacency(self, n: int, device: torch.device) -> torch.Tensor:
        adj = torch.zeros((n, n), dtype=torch.bool, device=device)
        order = torch.randperm(n, device=device).tolist()
        split = min(max(1, n // 2), n - 1)
        left, right = order[:split], order[split:]

        self._connect_sequence(adj, left)
        self._connect_sequence(adj, right)
        for group in (left, right):
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if not adj[group[i], group[j]] and torch.rand(1, device=device).item() < 0.92:
                        self._add_undirected_edge(adj, group[i], group[j])

        self._add_undirected_edge(adj, left[-1], right[0])
        for i in left:
            for j in right:
                if not adj[i, j] and torch.rand(1, device=device).item() < 0.22:
                    self._add_undirected_edge(adj, i, j)
        return adj

    def _build_dense_clustered_adjacency(self, n: int, device: torch.device) -> torch.Tensor:
        adj = torch.zeros((n, n), dtype=torch.bool, device=device)
        order = torch.randperm(n, device=device).tolist()
        clusters = [list(map(int, split.tolist())) for split in np.array_split(order, 2) if len(split)]

        for cluster in clusters:
            self._connect_sequence(adj, cluster)
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    if not adj[cluster[i], cluster[j]] and torch.rand(1, device=device).item() < 0.98:
                        self._add_undirected_edge(adj, cluster[i], cluster[j])

        for idx in range(len(clusters) - 1):
            self._add_undirected_edge(adj, clusters[idx][-1], clusters[idx + 1][0])
            if torch.rand(1, device=device).item() < 0.80:
                self._add_undirected_edge(adj, clusters[idx][0], clusters[idx + 1][-1])
            for src in clusters[idx]:
                for dst in clusters[idx + 1]:
                    if not adj[src, dst] and torch.rand(1, device=device).item() < 0.32:
                        self._add_undirected_edge(adj, src, dst)
        return adj

    def _build_sparse_random_connected_adjacency(self, n: int, device: torch.device) -> torch.Tensor:
        adj = self._build_tree_adjacency(n, device)
        extra_prob = min(self.p.p_extra_edge, 0.12)
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i, j] and torch.rand(1, device=device).item() < extra_prob:
                    self._add_undirected_edge(adj, i, j)
        return adj

    def _build_adjacency_from_topology(
        self, topology_id: int, n: int, device: torch.device
    ) -> torch.Tensor:
        builders = [
            self._build_chain_adjacency,
            self._build_star_adjacency,
            self._build_tree_adjacency,
            self._build_two_module_bridge_adjacency,
            self._build_dense_clustered_adjacency,
            self._build_sparse_random_connected_adjacency,
        ]
        return builders[topology_id](n, device)

    def _sample_embeddedness_weights(self, adj: torch.Tensor) -> torch.Tensor:
        n = int(adj.shape[0])
        weights = torch.zeros((n, n), dtype=torch.float32, device=adj.device)
        if n <= 1:
            return weights

        common_neighbors = torch.matmul(adj.float(), adj.float())
        edge_mask = torch.triu(adj, diagonal=1)
        if not bool(edge_mask.any().item()):
            return weights

        sampled = (common_neighbors[edge_mask] + torch.rand_like(common_neighbors[edge_mask])) / float(
            max(n - 1, 1)
        )
        weights[edge_mask] = sampled
        return weights + weights.transpose(0, 1)

    def _generate(self, batch_size) -> TensorDict:
        device = torch.device("cpu")
        B = int(batch_size[0])
        N = self.max_num_parts

        material_all = torch.full((B, N + 1), -1, dtype=torch.long, device=device)
        maint_all = torch.full((B, N + 1), -1, dtype=torch.long, device=device)
        std_all = torch.full((B, N + 1), -1, dtype=torch.long, device=device)
        size_all = torch.zeros((B, N + 1, 3), dtype=torch.float32, device=device)
        pos_all = torch.zeros((B, N + 1, 1), dtype=torch.float32, device=device)
        node_features = torch.zeros(
            (B, N + 1, self.node_feat_dim), dtype=torch.float32, device=device
        )
        W = torch.zeros((B, N + 1, N + 1), dtype=torch.float32, device=device)
        assembly_adj = torch.zeros((B, N + 1, N + 1), dtype=torch.bool, device=device)
        mat_var_all = torch.zeros((B, N + 1, N + 1), dtype=torch.float32, device=device)
        maint_diff_all = torch.zeros((B, N + 1, N + 1), dtype=torch.float32, device=device)
        rel_motion_all = torch.zeros((B, N + 1, N + 1), dtype=torch.float32, device=device)
        stack_all = torch.zeros((B, N + 1, N + 1, 3), dtype=torch.float32, device=device)
        edge_features = torch.zeros(
            (B, N + 1, N + 1, self.edge_feat_dim), dtype=torch.float32, device=device
        )
        compat = torch.ones((B, N + 1, N + 1), dtype=torch.bool, device=device)
        relation_valid = torch.zeros((B, N + 1, N + 1), dtype=torch.bool, device=device)
        relation_consistent = torch.ones((B,), dtype=torch.bool, device=device)
        topology_id = torch.zeros((B,), dtype=torch.long, device=device)
        num_parts = torch.zeros((B,), dtype=torch.long, device=device)
        material_type_count = torch.ones((B,), dtype=torch.long, device=device)
        valid_part_mask = torch.zeros((B, N + 1), dtype=torch.bool, device=device)
        valid_part_mask[:, 0] = True
        build_limit = self.build_limit.to(device).unsqueeze(0).repeat(B, 1)

        for b in range(B):
            n = int(torch.randint(self.min_num_parts, self.max_num_parts + 1, (1,)).item())
            num_parts[b] = n
            valid_part_mask[b, 1 : n + 1] = True
            eye = torch.eye(n, dtype=torch.bool, device=device)

            material_type_count[b] = self.p.material_types
            material = torch.randint(0, self.p.material_types, (n,), device=device)
            size = torch.stack(
                [
                    torch.rand((n,), device=device) * (self.p.L_high - self.p.L_low) + self.p.L_low,
                    torch.rand((n,), device=device) * (self.p.W_high - self.p.W_low) + self.p.W_low,
                    torch.rand((n,), device=device) * (self.p.H_high - self.p.H_low) + self.p.H_low,
                ],
                dim=-1,
            ).float()
            maintfreq = (torch.rand((n,), device=device) < self.p.p_maint_H).long()
            isstandard = (torch.rand((n,), device=device) < self.p.p_standard).long()

            topo_id = self._sample_topology_id(device)
            topology_id[b] = topo_id
            adj_parts = self._build_adjacency_from_topology(topo_id, n, device)
            degree = adj_parts.sum(dim=-1).float()
            pos1d = (degree / degree.max().clamp_min(1.0)).unsqueeze(-1)

            mat_var = (material.unsqueeze(-1) != material.unsqueeze(-2)) & adj_parts
            stack_size_full = size.unsqueeze(-2) + size.unsqueeze(-3)
            maint_diff = (maintfreq.unsqueeze(-1) != maintfreq.unsqueeze(-2)) & adj_parts
            rel = torch.rand((n, n), device=device) < self.p.p_relative_motion
            rel = (torch.triu(rel, diagonal=1) | torch.triu(rel, diagonal=1).transpose(-1, -2)) & ~eye
            rel_motion = rel & adj_parts

            stack_size = stack_size_full * adj_parts.unsqueeze(-1).float()
            stack_ok = (stack_size_full <= self.build_limit.to(device).view(1, 1, 3)).all(dim=-1)
            standard_pair_block = isstandard.unsqueeze(-1).bool() | isstandard.unsqueeze(-2).bool()
            compat_parts = (
                adj_parts & ~mat_var & ~maint_diff & ~rel_motion & stack_ok & ~standard_pair_block
            ) | eye

            mat_oh = torch.nn.functional.one_hot(
                material, num_classes=self.p.material_types
            ).float()
            part_node_features = torch.cat(
                [
                    mat_oh,
                    size,
                    maintfreq.float().unsqueeze(-1),
                    isstandard.float().unsqueeze(-1),
                    pos1d.float(),
                ],
                dim=-1,
            )
            part_edge_features = torch.cat(
                [
                    adj_parts.float().unsqueeze(-1),
                    mat_var.float().unsqueeze(-1),
                    stack_size,
                    maint_diff.float().unsqueeze(-1),
                    rel_motion.float().unsqueeze(-1),
                ],
                dim=-1,
            )

            material_all[b, 1 : n + 1] = material
            maint_all[b, 1 : n + 1] = maintfreq
            std_all[b, 1 : n + 1] = isstandard
            size_all[b, 1 : n + 1, :] = size
            pos_all[b, 1 : n + 1, :] = pos1d
            node_features[b, 1 : n + 1, :] = part_node_features
            W[b, 1 : n + 1, 1 : n + 1] = self._sample_embeddedness_weights(adj_parts)
            assembly_adj[b, 1 : n + 1, 1 : n + 1] = adj_parts
            mat_var_all[b, 1 : n + 1, 1 : n + 1] = mat_var.float()
            maint_diff_all[b, 1 : n + 1, 1 : n + 1] = maint_diff.float()
            rel_motion_all[b, 1 : n + 1, 1 : n + 1] = rel_motion.float()
            stack_all[b, 1 : n + 1, 1 : n + 1, :] = stack_size.float()
            edge_features[b, 1 : n + 1, 1 : n + 1, :] = part_edge_features
            compat[b, 1 : n + 1, 1 : n + 1] = compat_parts
            relation_valid[b, 1 : n + 1, 1 : n + 1] = adj_parts

        return TensorDict(
            {
                "node_features": node_features,
                "edge_features": edge_features,
                "topology_id": topology_id,
                "num_parts": num_parts,
                "material_type_count": material_type_count,
                "valid_part_mask": valid_part_mask,
                "material": material_all,
                "size": size_all,
                "maintfreq": maint_all,
                "isstandard": std_all,
                "pos1d": pos_all,
                "W": W,
                "assembly_adj": assembly_adj,
                "mat_var": mat_var_all,
                "stack_size": stack_all,
                "maint_diff": maint_diff_all,
                "rel_motion": rel_motion_all,
                "compat": compat,
                "relation_valid": relation_valid,
                "relation_consistent": relation_consistent,
                "build_limit": build_limit,
            },
            batch_size=batch_size,
        )

