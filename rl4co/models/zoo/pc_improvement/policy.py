from __future__ import annotations

import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class PCImprovementPolicy(nn.Module):
    """Policy over PC split/merge improvement actions."""

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        env_name: str = "pc_improvement",
        improvement_steps: int | None = None,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
    ):
        super().__init__()
        self.env_name = env_name
        self.improvement_steps = improvement_steps
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type
        self.init_embedding = env_init_embedding("pc", {"embed_dim": embed_dim})
        self.group_embedding = nn.Embedding(512, embed_dim)
        layers = []
        input_dim = 4 * embed_dim + 11
        for idx in range(num_layers):
            layers.append(nn.Linear(input_dim if idx == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 2))
        self.pair_mlp = nn.Sequential(*layers)

    def forward(
        self,
        td: TensorDict,
        env: str | RL4COEnvBase = None,
        phase: str = "train",
        return_actions: bool = True,
        return_embeds: bool = False,
        only_return_embed: bool = False,
        actions=None,
        **decoding_kwargs,
    ) -> dict:
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        steps = self.improvement_steps or env.improvement_steps
        log_likelihood = torch.zeros(td.batch_size, dtype=torch.float32, device=td.device)
        entropy = torch.zeros_like(log_likelihood)
        selected_actions = []
        step_log_likelihoods = []
        final_embeds = None
        return_sum_log_likelihood = decoding_kwargs.pop("return_sum_log_likelihood", True)

        for step in range(steps):
            logits = self._logits(td, env)
            mask = td["action_mask"].bool()
            mask_value = torch.finfo(logits.dtype).min
            log_probs = torch.log_softmax(logits.masked_fill(~mask, mask_value), dim=-1)
            probs = log_probs.exp()
            final_embeds = self._node_embeddings(td)

            if actions is not None:
                action = actions[:, step].long()
            elif decode_type == "greedy":
                action = probs.argmax(dim=-1)
            elif decode_type == "sampling":
                action = torch.distributions.Categorical(probs=probs).sample()
            else:
                raise ValueError(f"Unsupported decode_type for PCImprovementPolicy: {decode_type}")

            selected_actions.append(action)
            step_log_likelihood = log_probs.gather(1, action.view(-1, 1)).squeeze(-1)
            step_log_likelihoods.append(step_log_likelihood)
            log_likelihood = log_likelihood + step_log_likelihood
            entropy = entropy - (probs * log_probs).sum(dim=-1)

            td.set("action", action)
            td = env.step(td)["next"]

        reward = td["reward"].view(-1)
        if return_sum_log_likelihood:
            out_log_likelihood = log_likelihood
        else:
            out_log_likelihood = torch.stack(step_log_likelihoods, dim=-1)

        out = {
            "reward": reward,
            "log_likelihood": out_log_likelihood,
            "entropy": entropy / max(steps, 1),
            "init_reward": td["init_reward"].view(-1),
            "current_reward": td["current_reward"].view(-1),
            "best_reward": td["best_reward"].view(-1),
        }
        if return_actions:
            out["actions"] = torch.stack(selected_actions, dim=-1)
        if return_embeds:
            out["embeds"] = final_embeds.detach()
        if only_return_embed:
            return {"embeds": final_embeds.detach()}
        return out

    def _node_embeddings(self, td: TensorDict) -> torch.Tensor:
        node_h = self.init_embedding(td)
        group_id = td["group_id"].clamp_min(0).clamp_max(511)
        return node_h + self.group_embedding(group_id)

    def _logits(self, td: TensorDict, env: RL4COEnvBase) -> torch.Tensor:
        h = self._node_embeddings(td)
        pair_i = env._pair_i.to(td.device)
        pair_j = env._pair_j.to(td.device)
        hi = h.index_select(1, pair_i)
        hj = h.index_select(1, pair_j)
        edge_features = td["edge_features"][:, pair_i, pair_j, :]
        w = td["W"][:, pair_i, pair_j].unsqueeze(-1)
        assembly = td["assembly_adj"][:, pair_i, pair_j].float().unsqueeze(-1)
        compat = td["compat"][:, pair_i, pair_j].float().unsqueeze(-1)
        same_group = (
            td["group_id"][:, pair_i].eq(td["group_id"][:, pair_j])
            & td["group_id"][:, pair_i].ge(0)
        ).float().unsqueeze(-1)
        pair_context = torch.cat(
            [hi, hj, torch.abs(hi - hj), hi * hj, edge_features, w, assembly, compat, same_group],
            dim=-1,
        )
        pair_logits = self.pair_mlp(pair_context)
        split_logits = pair_logits[..., 0]
        merge_logits = pair_logits[..., 1]
        return torch.cat([split_logits, merge_logits], dim=-1)
