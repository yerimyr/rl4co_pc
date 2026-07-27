from __future__ import annotations

import random

import torch

from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.pc.env import PartConsolidationEnv
from rl4co.envs.pc.generator import FPIGenerator


class PartConsolidationImprovementEnv(PartConsolidationEnv):
    """PC improvement environment using split/merge actions.

    State starts from a random feasible grouping. At each step, the policy selects
    one local modification:
        0..E-1: split the corresponding part-pair edge
        E..2E-1: merge the corresponding part-pair edge

    Rewards are terminal-only. Intermediate rewards are zero; the final reward is
    the PC score of the final grouping.
    """

    name = "pc_improvement"

    def __init__(
        self,
        generator: FPIGenerator = None,
        generator_params: dict = {},
        improvement_steps: int = 20,
        random_group_new_group_prob: float = 0.60,
        seed: int | None = None,
        **kwargs,
    ):
        if generator is None:
            generator = FPIGenerator(**generator_params)
        max_parts = generator.num_parts
        self._pair_i, self._pair_j = self._build_pair_tensors(max_parts)
        self.num_pairs = int(self._pair_i.numel())
        self.num_actions = 2 * self.num_pairs
        super().__init__(generator=generator, generator_params=generator_params, **kwargs)
        self.improvement_steps = int(improvement_steps)
        self.random_group_new_group_prob = float(random_group_new_group_prob)
        self._py_rng = random.Random(seed)
        self._make_spec(self.generator)

    @staticmethod
    def _build_pair_tensors(max_parts: int) -> tuple[Tensor, Tensor]:
        pairs = [(i, j) for i in range(1, max_parts + 1) for j in range(i + 1, max_parts + 1)]
        if not pairs:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        pair_i, pair_j = zip(*pairs)
        return torch.tensor(pair_i, dtype=torch.long), torch.tensor(pair_j, dtype=torch.long)

    def _reset(self, td: TensorDict | None = None, batch_size=None) -> TensorDict:
        device = td.device
        batch_size = torch.Size(batch_size)
        out = td.clone()
        out.update(
            {
                "selected": torch.zeros(
                    (*batch_size, self.num_nodes), dtype=torch.bool, device=device
                ),
                "current_group_mask": torch.zeros(
                    (*batch_size, self.num_nodes), dtype=torch.bool, device=device
                ),
                "first_node": torch.zeros((*batch_size,), dtype=torch.int64, device=device),
                "current_node": torch.zeros((*batch_size,), dtype=torch.int64, device=device),
            }
        )
        edge_bits = self._random_initial_edge_bits(out)
        group_id = self.edge_bits_to_group_id(edge_bits, out)
        reward = self.score_edge_bits(edge_bits, out)

        out.update(
            {
                "edge_bits": edge_bits,
                "group_id": group_id,
                "i": torch.zeros((*out.batch_size, 1), dtype=torch.int64, device=out.device),
                "done": torch.zeros((*out.batch_size, 1), dtype=torch.bool, device=out.device),
                "reward": torch.zeros((*out.batch_size, 1), dtype=torch.float32, device=out.device),
                "init_reward": reward,
                "current_reward": reward,
                "best_reward": reward,
            }
        )
        out["action_mask"] = self.get_action_mask(out)
        return out

    def _step(self, td: TensorDict) -> TensorDict:
        action = td["action"].long().view(-1)
        edge_bits = td["edge_bits"].clone()
        action_mask = self.get_action_mask(td)
        active = ~td["done"].view(-1)
        B = edge_bits.size(0)
        rows = torch.arange(B, device=edge_bits.device)
        action_safe = action.clamp(0, self.num_actions - 1)
        valid_action = active & action.ge(0) & action.lt(self.num_actions)
        valid_action = valid_action & action_mask[rows, action_safe]
        pair_idx = torch.where(
            action_safe < self.num_pairs,
            action_safe,
            action_safe - self.num_pairs,
        )
        split_action = action_safe < self.num_pairs
        edge_bits[rows[valid_action], pair_idx[valid_action]] = ~split_action[valid_action]
        edge_bits = self.canonicalize_edge_bits_batch(edge_bits, td)

        group_id = self.edge_bits_to_group_id(edge_bits, td)
        current_reward = self.score_edge_bits(edge_bits, td)
        best_reward = torch.maximum(td["best_reward"].view(-1), current_reward.view(-1))
        step_idx = td["i"] + 1
        done = step_idx.ge(self.improvement_steps)
        reward = torch.where(done.view(-1), current_reward.view(-1), torch.zeros_like(current_reward.view(-1)))

        td.update(
            {
                "edge_bits": edge_bits,
                "group_id": group_id,
                "i": step_idx,
                "done": done,
                "reward": reward.view(-1, 1),
                "current_reward": current_reward.view(-1, 1),
                "best_reward": best_reward.view(-1, 1),
            }
        )
        td["action_mask"] = self.get_action_mask(td)
        return td

    def _make_spec(self, generator: FPIGenerator):
        PartConsolidationEnv._make_spec(self, generator)
        self.observation_spec.update(
            {
                "edge_bits": Unbounded(shape=(self.num_pairs,), dtype=torch.bool),
                "group_id": Unbounded(shape=(generator.num_nodes,), dtype=torch.int64),
                "init_reward": Unbounded(shape=(1,), dtype=torch.float32),
                "current_reward": Unbounded(shape=(1,), dtype=torch.float32),
                "best_reward": Unbounded(shape=(1,), dtype=torch.float32),
                "action_mask": Unbounded(shape=(self.num_actions,), dtype=torch.bool),
            }
        )
        self.action_spec = Bounded(shape=(1,), dtype=torch.int64, low=0, high=self.num_actions)

    def get_action_mask(self, td: TensorDict) -> Tensor:
        edge_bits = td["edge_bits"].bool()
        B = edge_bits.size(0)
        device = edge_bits.device
        pair_i = self._pair_i.to(device)
        pair_j = self._pair_j.to(device)

        valid = td["valid_part_mask"].bool()
        assembly = td["assembly_adj"].bool()
        pair_valid = valid[:, pair_i] & valid[:, pair_j] & assembly[:, pair_i, pair_j]
        reach = self.edge_bits_to_connectivity(edge_bits, td)
        same_group = reach[:, pair_i, pair_j]

        split_mask = edge_bits & pair_valid & self._split_bridge_mask(edge_bits, td, reach)
        merge_mask = (~edge_bits) & pair_valid & (~same_group) & self._merge_feasible_mask(td, reach)
        mask = torch.cat([split_mask, merge_mask], dim=-1)

        done = td["done"].view(-1).bool() if "done" in td.keys() else torch.zeros(B, dtype=torch.bool, device=device)
        mask[done] = False
        no_action = ~mask.any(dim=-1)
        mask[no_action, 0] = True
        return mask

    def _split_bridge_mask(self, edge_bits: Tensor, td: TensorDict, reach: Tensor) -> Tensor:
        B, E = edge_bits.shape
        N = self.num_nodes
        device = edge_bits.device
        pair_i = self._pair_i.to(device)
        pair_j = self._pair_j.to(device)
        adj = self.edge_bits_to_adjacency(edge_bits, td)
        adj_without = adj[:, None, :, :].expand(B, E, N, N).clone()
        batch_idx = torch.arange(B, device=device)[:, None].expand(B, E)
        edge_idx = torch.arange(E, device=device)[None, :].expand(B, E)
        u_idx = pair_i[None, :].expand(B, E)
        v_idx = pair_j[None, :].expand(B, E)
        adj_without[batch_idx, edge_idx, u_idx, v_idx] = False
        adj_without[batch_idx, edge_idx, v_idx, u_idx] = False

        valid = td["valid_part_mask"].bool()
        eye = torch.eye(N, dtype=torch.bool, device=device).view(1, 1, N, N)
        valid_pair = valid[:, None, :, None] & valid[:, None, None, :]
        conn = (adj_without | (eye & valid_pair)).float()
        valid_pair_f = valid_pair.float()
        for _ in range(N):
            conn = torch.bmm(conn.view(B * E, N, N), conn.view(B * E, N, N)).view(B, E, N, N)
            conn = (conn > 0).float() * valid_pair_f
        connected_after_removal = conn.bool()[batch_idx, edge_idx, u_idx, v_idx]
        return reach[:, pair_i, pair_j] & ~connected_after_removal

    def _merge_feasible_mask(self, td: TensorDict, reach: Tensor) -> Tensor:
        device = reach.device
        pair_i = self._pair_i.to(device)
        pair_j = self._pair_j.to(device)
        comp_i = reach[:, pair_i, :]
        comp_j = reach[:, pair_j, :]
        merged = comp_i | comp_j

        cardinality = merged.sum(dim=-1)
        size_sum = torch.einsum("ben,bnd->bed", merged.float(), td["size"].float())
        size_ok = size_sum.le(td["build_limit"].float()[:, None, :]).all(dim=-1)
        standard_ok = ~(merged & td["isstandard"].bool()[:, None, :]).any(dim=-1) | cardinality.le(1)
        bad_pair = td["mat_var"].bool() | td["maint_diff"].bool() | td["rel_motion"].bool()
        no_bad_pair = ~(
            merged[:, :, :, None] & merged[:, :, None, :] & bad_pair[:, None, :, :]
        ).any(dim=(-1, -2))
        return size_ok & standard_ok & no_bad_pair

    def _get_reward(self, td: TensorDict, actions: Tensor) -> Tensor:
        return self.score_edge_bits(td["edge_bits"], td)

    def score_edge_bits(self, edge_bits: Tensor, td: TensorDict) -> Tensor:
        same_component = self.edge_bits_to_connectivity(edge_bits, td).float()
        w = td["W"].float()
        strengths = w.sum(dim=-1)
        two_m = strengths.sum(dim=-1).clamp_min(self._reward_eps)
        observed = (w * same_component).sum(dim=(-1, -2))
        expected = (
            strengths[:, :, None] * strengths[:, None, :] * same_component
        ).sum(dim=(-1, -2)) / two_m
        q_observed = observed / two_m
        q_expected = self._modularity_gamma * expected / two_m
        return ((q_observed - q_expected) * float(self._objective_scale)).view(-1, 1)

    def edge_bits_to_adjacency(self, edge_bits: Tensor, td: TensorDict) -> Tensor:
        B = edge_bits.size(0)
        device = edge_bits.device
        pair_i = self._pair_i.to(device)
        pair_j = self._pair_j.to(device)
        adj = torch.zeros((B, self.num_nodes, self.num_nodes), dtype=torch.bool, device=device)
        valid_edge = edge_bits.bool() & td["assembly_adj"].bool()[:, pair_i, pair_j]
        adj[:, pair_i, pair_j] = valid_edge
        adj[:, pair_j, pair_i] = valid_edge
        return adj

    def edge_bits_to_connectivity(self, edge_bits: Tensor, td: TensorDict) -> Tensor:
        B = edge_bits.size(0)
        N = self.num_nodes
        device = edge_bits.device
        valid = td["valid_part_mask"].bool()
        valid_pair = valid[:, :, None] & valid[:, None, :]
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        conn = (self.edge_bits_to_adjacency(edge_bits, td) | (eye & valid_pair)).float()
        valid_pair_f = valid_pair.float()
        for _ in range(N):
            conn = torch.bmm(conn, conn)
            conn = (conn > 0).float() * valid_pair_f
        return conn.bool()

    def edge_bits_to_groups(self, edge_bits: Tensor, td: TensorDict) -> list[list[list[int]]]:
        edge_bits = edge_bits.bool()
        pair_i = self._pair_i.to(edge_bits.device)
        pair_j = self._pair_j.to(edge_bits.device)
        all_groups = []
        for b in range(edge_bits.size(0)):
            valid_nodes = torch.nonzero(td["valid_part_mask"][b, 1:].bool(), as_tuple=False).flatten()
            nodes = (valid_nodes + 1).tolist()
            parent = {node: node for node in nodes}

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, c: int) -> None:
                ra, rb = find(a), find(c)
                if ra != rb:
                    parent[rb] = ra

            for e, (u_t, v_t) in enumerate(zip(pair_i, pair_j)):
                if bool(edge_bits[b, e].item()):
                    u, v = int(u_t.item()), int(v_t.item())
                    if u in parent and v in parent:
                        union(u, v)

            groups: dict[int, list[int]] = {}
            for node in nodes:
                groups.setdefault(find(node), []).append(node)
            all_groups.append([sorted(group) for group in groups.values()])
        return all_groups

    def edge_bits_to_group_id(self, edge_bits: Tensor, td: TensorDict) -> Tensor:
        reach = self.edge_bits_to_connectivity(edge_bits, td)
        B, N, _ = reach.shape
        node_ids = torch.arange(N, dtype=torch.long, device=edge_bits.device)
        large = torch.full((B, N, N), N + 1, dtype=torch.long, device=edge_bits.device)
        reps = torch.where(reach, node_ids.view(1, 1, N), large).min(dim=-1).values
        reps = torch.where(td["valid_part_mask"].bool(), reps, torch.full_like(reps, -1))
        reps[:, 0] = 0
        return reps

    def canonicalize_edge_bits(self, edge_bits: Tensor, td: TensorDict, batch_idx: int) -> Tensor:
        groups = self.edge_bits_to_groups(edge_bits.unsqueeze(0), td[batch_idx : batch_idx + 1])[0]
        return self.groups_to_edge_bits([groups], td[batch_idx : batch_idx + 1])[0]

    def canonicalize_edge_bits_batch(self, edge_bits: Tensor, td: TensorDict) -> Tensor:
        reach = self.edge_bits_to_connectivity(edge_bits, td)
        pair_i = self._pair_i.to(edge_bits.device)
        pair_j = self._pair_j.to(edge_bits.device)
        pair_valid = (
            td["valid_part_mask"].bool()[:, pair_i]
            & td["valid_part_mask"].bool()[:, pair_j]
            & td["assembly_adj"].bool()[:, pair_i, pair_j]
        )
        return reach[:, pair_i, pair_j] & pair_valid

    def groups_to_edge_bits(self, groups: list[list[list[int]]], td: TensorDict) -> Tensor:
        device = td.device
        out = torch.zeros((len(groups), self.num_pairs), dtype=torch.bool, device=device)
        pair_i = self._pair_i.to(device)
        pair_j = self._pair_j.to(device)
        for b, groups_b in enumerate(groups):
            group_id = {}
            for gid, group in enumerate(groups_b):
                for node in group:
                    group_id[int(node)] = gid
            assembly = td["assembly_adj"][b].bool()
            for e, (u_t, v_t) in enumerate(zip(pair_i, pair_j)):
                u, v = int(u_t.item()), int(v_t.item())
                out[b, e] = (
                    group_id.get(u, -1) == group_id.get(v, -2) and bool(assembly[u, v].item())
                )
        return out

    def _random_initial_edge_bits(self, td: TensorDict) -> Tensor:
        groups = []
        for b in range(td.batch_size[0]):
            nodes = torch.nonzero(td["valid_part_mask"][b, 1:].bool(), as_tuple=False).flatten().add(1).tolist()
            self._py_rng.shuffle(nodes)
            groups_b: list[list[int]] = []
            for node in nodes:
                feasible_targets = []
                for idx, group in enumerate(groups_b):
                    trial = sorted(group + [node])
                    if self._group_feasible(
                        trial,
                        td["size"][b],
                        td["build_limit"][b],
                        td["isstandard"][b],
                        td["mat_var"][b],
                        td["maint_diff"][b],
                        td["rel_motion"][b],
                        td["assembly_adj"][b],
                    ):
                        feasible_targets.append(idx)
                if not feasible_targets or self._py_rng.random() < self.random_group_new_group_prob:
                    groups_b.append([node])
                else:
                    groups_b[self._py_rng.choice(feasible_targets)].append(node)
            groups.append([sorted(group) for group in groups_b])
        return self.groups_to_edge_bits(groups, td)

    def _groups_feasible(self, groups: list[list[int]], td: TensorDict, batch_idx: int) -> bool:
        return all(
            self._group_feasible(
                group,
                td["size"][batch_idx],
                td["build_limit"][batch_idx],
                td["isstandard"][batch_idx],
                td["mat_var"][batch_idx],
                td["maint_diff"][batch_idx],
                td["rel_motion"][batch_idx],
                td["assembly_adj"][batch_idx],
            )
            for group in groups
        )
