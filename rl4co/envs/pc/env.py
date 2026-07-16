from __future__ import annotations

import torch

from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.pc.evaluator import DEFAULT_MODULARITY_GAMMA, DEFAULT_OBJECTIVE_SCALE
from rl4co.envs.pc.generator import FPIGenerator


class PartConsolidationEnv(RL4COEnvBase):
    """Part Consolidation environment with SEP + part-selection actions.

    Actions:
        0: close the current open group and start a new one.
        1..N: add that part to the current open group.

    Reward is terminal only and uses the modularity-style PC objective.
    """

    name = "pc"

    def __init__(
        self,
        generator: FPIGenerator = None,
        generator_params: dict = {},
        min_group_size_before_sep: int = 1,
        allow_fallback: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = FPIGenerator(**generator_params)
        self.generator = generator
        self.min_group_size_before_sep = int(min_group_size_before_sep)
        self.allow_fallback = bool(allow_fallback)
        self.num_nodes = self.generator.num_nodes
        self.max_parts = self.generator.num_parts
        self.node_feat_dim = self.generator.node_feat_dim
        self.edge_feat_dim = self.generator.edge_feat_dim
        self._reward_eps = 1e-8
        self._modularity_gamma = DEFAULT_MODULARITY_GAMMA
        self._objective_scale = DEFAULT_OBJECTIVE_SCALE
        self._make_spec(self.generator)

    def _reset(self, td: TensorDict | None = None, batch_size=None) -> TensorDict:
        device = td.device
        batch_size = torch.Size(batch_size)
        valid_part_mask = td["valid_part_mask"].bool()
        selected = torch.zeros((*batch_size, self.num_nodes), dtype=torch.bool, device=device)
        selected[..., 0] = True
        current_group_mask = torch.zeros_like(selected)
        current_node = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        first_node = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        out = td.clone()
        out.update(
            {
                "selected": selected,
                "current_group_mask": current_group_mask,
                "first_node": first_node,
                "current_node": current_node,
                "i": i,
                "done": torch.zeros((*batch_size, 1), dtype=torch.bool, device=device),
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32, device=device),
            }
        )
        out["action_mask"] = self.get_action_mask(out)
        # Keep padded or invalid part slots selected from the policy perspective.
        out["selected"] = out["selected"] | ~valid_part_mask
        out["selected"][..., 0] = True
        out["action_mask"] = self.get_action_mask(out)
        return out

    def _step(self, td: TensorDict) -> TensorDict:
        action = td["action"].long().view(*td.batch_size)
        done = td["done"].clone()
        active = ~done.squeeze(-1)
        sep_action = active & action.eq(0)
        part_action = active & action.gt(0)

        selected = td["selected"].clone()
        current_group_mask = td["current_group_mask"].clone()
        current_node = td["current_node"].clone()
        first_node = td["first_node"].clone()

        if part_action.any():
            selected[part_action] = selected[part_action].scatter(
                -1, action[part_action].unsqueeze(-1), True
            )
            current_group_mask[part_action] = current_group_mask[part_action].scatter(
                -1, action[part_action].unsqueeze(-1), True
            )
            current_node = torch.where(part_action, action, current_node)
            first_node = torch.where((td["i"].squeeze(-1) == 0) & part_action, action, first_node)

        if sep_action.any():
            current_group_mask[sep_action] = False
            current_node = torch.where(sep_action, torch.zeros_like(current_node), current_node)

        remaining = td["valid_part_mask"][..., 1:].bool() & ~selected[..., 1:]
        done = done | (~remaining.any(dim=-1, keepdim=True))

        td.update(
            {
                "selected": selected,
                "current_group_mask": current_group_mask,
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "done": done,
                "reward": torch.zeros_like(done, dtype=torch.float32),
            }
        )
        td["action_mask"] = self.get_action_mask(td)
        return td

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        selected = td["selected"].bool()
        current_group = td["current_group_mask"].bool()
        valid = td["valid_part_mask"].bool()
        size = td["size"].float()
        build_limit = td["build_limit"].float()
        assembly_adj = td["assembly_adj"].bool()
        isstandard = td["isstandard"].bool()
        bad_pair = td["mat_var"].bool() | td["maint_diff"].bool() | td["rel_motion"].bool()

        B, N = selected.shape
        candidate_eye = torch.eye(N, dtype=torch.bool, device=selected.device).unsqueeze(0)
        candidate_groups = current_group[:, None, :] | candidate_eye

        candidate_size = torch.einsum("ban,bnd->bad", candidate_groups.float(), size)
        size_ok = candidate_size.le(build_limit[:, None, :]).all(dim=-1)
        cardinality = candidate_groups.sum(dim=-1)
        standard_ok = ~(candidate_groups & isstandard[:, None, :]).any(dim=-1) | cardinality.le(1)
        no_bad_pair = ~(
            candidate_groups[:, :, :, None]
            & candidate_groups[:, :, None, :]
            & bad_pair[:, None, :, :]
        ).any(dim=(-1, -2))

        group_nonempty = current_group.any(dim=-1)
        connected_to_group = (
            current_group[:, None, :, None]
            & candidate_eye[:, :, None, :]
            & assembly_adj[:, None, :, :]
        ).any(dim=(-1, -2))
        connected_ok = ~group_nonempty[:, None] | connected_to_group

        part_mask = valid & ~selected & size_ok & standard_ok & no_bad_pair & connected_ok
        part_mask[:, 0] = False

        remaining_after = (valid[:, 1:] & ~selected[:, 1:]).any(dim=-1)
        sep_mask = (
            current_group[:, 1:].sum(dim=-1).ge(self.min_group_size_before_sep)
            & remaining_after
        )

        mask = part_mask
        mask[:, 0] = sep_mask

        done = td["done"].squeeze(-1).bool()
        if done.any():
            mask[done] = False
            mask[done, 0] = True
        return mask

    def _make_spec(self, generator: FPIGenerator):
        self.observation_spec = Composite(
            node_features=Unbounded(
                shape=(generator.num_nodes, generator.node_feat_dim), dtype=torch.float32
            ),
            edge_features=Unbounded(
                shape=(generator.num_nodes, generator.num_nodes, generator.edge_feat_dim),
                dtype=torch.float32,
            ),
            W=Unbounded(shape=(generator.num_nodes, generator.num_nodes), dtype=torch.float32),
            valid_part_mask=Unbounded(shape=(generator.num_nodes,), dtype=torch.bool),
            selected=Unbounded(shape=(generator.num_nodes,), dtype=torch.bool),
            current_group_mask=Unbounded(shape=(generator.num_nodes,), dtype=torch.bool),
            first_node=Unbounded(shape=(1,), dtype=torch.int64),
            current_node=Unbounded(shape=(1,), dtype=torch.int64),
            i=Unbounded(shape=(1,), dtype=torch.int64),
            action_mask=Unbounded(shape=(generator.num_nodes,), dtype=torch.bool),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_nodes,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        if self.check_solution:
            self.check_solution_validity(td, actions)
        groups = self.actions_to_groups(actions, td)
        return self._terminal_reward_components(groups, td, actions.device)["Q_gamma"]

    def reward_metrics_from_actions(
        self, actions: torch.Tensor, td: TensorDict
    ) -> dict[str, torch.Tensor]:
        return self._terminal_reward_components(self.actions_to_groups(actions, td), td, actions.device)

    def actions_to_groups(self, actions: torch.Tensor, td: TensorDict) -> list[list[list[int]]]:
        B, T = actions.shape
        valid_part_mask = td["valid_part_mask"].bool()
        out = []
        for b in range(B):
            seen = set()
            groups_b: list[list[int]] = []
            current: list[int] = []
            for t in range(T):
                action = int(actions[b, t].item())
                if action == 0:
                    if current:
                        groups_b.append(current)
                        current = []
                    continue
                if action >= self.num_nodes:
                    continue
                if not bool(valid_part_mask[b, action].item()) or action in seen:
                    continue
                current.append(action)
                seen.add(action)
            if current:
                groups_b.append(current)
            out.append(groups_b)
        return out

    def _terminal_reward_components(
        self, groups: list[list[list[int]]], td: TensorDict, device: torch.device
    ) -> dict[str, torch.Tensor]:
        B = len(groups)
        feasible = torch.zeros((B,), dtype=torch.float32, device=device)
        infeasible_solution = torch.zeros((B,), dtype=torch.float32, device=device)
        infeasible_groups = torch.zeros((B,), dtype=torch.float32, device=device)
        num_groups = torch.tensor([len(g) for g in groups], dtype=torch.float32, device=device)
        total_internal_strength = torch.zeros((B,), dtype=torch.float32, device=device)
        feasible_pair_count = torch.zeros((B,), dtype=torch.float32, device=device)

        for b, groups_b in enumerate(groups):
            infeasible = False
            for group in groups_b:
                total_internal_strength[b] += self._group_internal_strength(group, td["W"][b])
                feasible_pair_count[b] += self._group_feasible_pair_count(group, td["compat"][b])
                if not self._group_feasible(
                    group,
                    td["size"][b],
                    td["build_limit"][b],
                    td["isstandard"][b],
                    td["mat_var"][b],
                    td["maint_diff"][b],
                    td["rel_motion"][b],
                    td["assembly_adj"][b],
                ):
                    infeasible = True
                    infeasible_groups[b] += 1.0
            infeasible_solution[b] = float(infeasible)
            feasible[b] = float(not infeasible)

        q_gamma, q_observed, q_expected = self._group_modularity(groups, td["W"].to(device), device)
        return {
            "feasible": feasible,
            "infeasible_solution": infeasible_solution,
            "infeasible_groups": infeasible_groups,
            "num_groups": num_groups,
            "total_internal_strength": total_internal_strength,
            "feasible_pair_count": feasible_pair_count,
            "normalized_internal_strength": total_internal_strength
            / torch.clamp(feasible_pair_count, min=1.0),
            "Q_gamma": q_gamma,
            "Q_observed": q_observed,
            "Q_expected": q_expected,
        }

    def _group_modularity(
        self, groups: list[list[list[int]]], w: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_gamma = torch.zeros((len(groups),), dtype=torch.float32, device=device)
        q_observed = torch.zeros_like(q_gamma)
        q_expected = torch.zeros_like(q_gamma)

        for b, groups_b in enumerate(groups):
            wb = w[b].float()
            strengths = wb.sum(dim=-1)
            two_m = strengths.sum().clamp_min(self._reward_eps)
            observed = torch.tensor(0.0, dtype=torch.float32, device=device)
            expected = torch.tensor(0.0, dtype=torch.float32, device=device)
            for group in groups_b:
                if not group:
                    continue
                idx = torch.tensor(group, dtype=torch.long, device=device)
                observed = observed + wb.index_select(0, idx).index_select(1, idx).sum()
                group_strength = strengths.index_select(0, idx).sum()
                expected = expected + (group_strength * group_strength) / two_m
            q_observed[b] = observed / two_m
            q_expected[b] = self._modularity_gamma * expected / two_m
            q_gamma[b] = q_observed[b] - q_expected[b]

        scale = float(self._objective_scale)
        return q_gamma * scale, q_observed * scale, q_expected * scale

    def _group_internal_strength(self, group: list[int], w: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=w.device)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                total = total + w[group[i], group[j]]
        return total

    def _group_feasible_pair_count(self, group: list[int], compat: torch.Tensor) -> torch.Tensor:
        count = torch.tensor(0.0, device=compat.device)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                count = count + float(bool(compat[group[i], group[j]].item()))
        return count

    def _group_feasible(
        self,
        group: list[int],
        size: torch.Tensor,
        build_limit: torch.Tensor,
        isstandard: torch.Tensor,
        mat_var: torch.Tensor,
        maint_diff: torch.Tensor,
        rel_motion: torch.Tensor,
        assembly_adj: torch.Tensor,
    ) -> bool:
        if not group:
            return True
        if len(group) >= 2 and isstandard[group].bool().any():
            return False
        if not torch.all(size[group].sum(dim=0) <= build_limit):
            return False
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if (
                    bool(mat_var[a, b].item())
                    or bool(maint_diff[a, b].item())
                    or bool(rel_motion[a, b].item())
                ):
                    return False
        visited = {group[0]}
        stack = [group[0]]
        while stack:
            cur = stack.pop()
            for nxt in group:
                if bool(assembly_adj[cur, nxt].item()) and nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        return len(visited) == len(group)

    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor) -> None:
        valid = td["valid_part_mask"][:, 1:].bool()
        for b, groups_b in enumerate(self.actions_to_groups(actions, td)):
            flattened = [node for group in groups_b for node in group]
            expected = set(torch.nonzero(valid[b], as_tuple=False).flatten().add(1).tolist())
            assert set(flattened) == expected, "Invalid PC solution: missing or extra part"
            assert len(flattened) == len(set(flattened)), "Invalid PC solution: repeated part"

    def replace_selected_actions(
        self,
        cur_actions: torch.Tensor,
        new_actions: torch.Tensor,
        selection_mask: torch.Tensor,
    ) -> torch.Tensor:
        cur_actions[selection_mask] = new_actions[selection_mask]
        return cur_actions
