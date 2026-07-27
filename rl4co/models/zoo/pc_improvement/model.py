import torch
import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.rl import PPO
from rl4co.models.rl import REINFORCE
from rl4co.models.zoo.pc_improvement.policy import PCImprovementPolicy


class PCImprovementModel(REINFORCE):
    """REINFORCE model wrapper for PC split/merge improvement."""

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module = None,
        policy_kwargs: dict = {},
        baseline: str = "no",
        **kwargs,
    ):
        if policy is None:
            policy = PCImprovementPolicy(env_name=env.name, **policy_kwargs)
        super().__init__(env, policy, baseline=baseline, **kwargs)


class PCImprovementCritic(nn.Module):
    """Value function for the PC improvement state."""

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.init_embedding = env_init_embedding("pc", {"embed_dim": embed_dim})
        self.group_embedding = nn.Embedding(512, embed_dim)
        input_dim = embed_dim + 2
        layers = []
        for idx in range(num_layers):
            layers.append(nn.Linear(input_dim if idx == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.value_head = nn.Sequential(*layers)

    def forward(self, td):
        node_h = self.init_embedding(td)
        group_id = td["group_id"].clamp_min(0).clamp_max(511)
        h = node_h + self.group_embedding(group_id)
        valid = td["valid_part_mask"].float().unsqueeze(-1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        current_reward = td["current_reward"].float().view(-1, 1)
        init_reward = td["init_reward"].float().view(-1, 1)
        value_input = torch.cat([pooled, current_reward, init_reward], dim=-1)
        return self.value_head(value_input)


class PCImprovementPPOModel(PPO):
    """PPO model wrapper for PC split/merge improvement."""

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module = None,
        critic: nn.Module = None,
        policy_kwargs: dict = {},
        critic_kwargs: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = PCImprovementPolicy(env_name=env.name, **policy_kwargs)
        if critic is None:
            critic_defaults = {
                "embed_dim": policy_kwargs.get("embed_dim", 128),
                "hidden_dim": policy_kwargs.get("hidden_dim", 256),
            }
            critic_defaults.update(critic_kwargs)
            critic = PCImprovementCritic(**critic_defaults)
        super().__init__(env, policy, critic=critic, **kwargs)
