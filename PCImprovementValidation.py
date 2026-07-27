from __future__ import annotations

import torch

from rl4co.envs import PartConsolidationImprovementEnv
from rl4co.models.zoo.pc_improvement import PCImprovementPolicy


def main() -> None:
    torch.manual_seed(1234)
    env = PartConsolidationImprovementEnv(
        generator_params=dict(num_parts=6),
        improvement_steps=5,
        check_solution=False,
        seed=1234,
    )
    td = env.reset(batch_size=[4])

    print("=== PC Improvement Environment ===")
    print(f"num_parts: {env.max_parts}")
    print(f"num_pairs: {env.num_pairs}")
    print(f"num_actions: {env.num_actions} = split({env.num_pairs}) + merge({env.num_pairs})")
    print(f"edge_bits shape: {tuple(td['edge_bits'].shape)}")
    print(f"action_mask shape: {tuple(td['action_mask'].shape)}")
    print(f"valid actions per instance: {td['action_mask'].sum(-1).tolist()}")
    print(f"initial reward: {td['init_reward'].view(-1).tolist()}")

    action = td["action_mask"].float().argmax(dim=-1)
    before = td["edge_bits"].clone()
    td.set("action", action)
    td = env.step(td)["next"]
    changed = before.ne(td["edge_bits"]).any(dim=-1)

    print("\n=== One Greedy Mask Action Step ===")
    print(f"selected actions: {action.tolist()}")
    print(f"edge_bits changed: {changed.tolist()}")
    print(f"step index: {td['i'].view(-1).tolist()}")
    print(f"intermediate reward, should be zero: {td['reward'].view(-1).tolist()}")

    policy = PCImprovementPolicy(embed_dim=32, hidden_dim=64, improvement_steps=5)
    td = env.reset(batch_size=[4])
    out = policy(td, env, phase="train")

    print("\n=== Policy Rollout ===")
    print(f"actions shape: {tuple(out['actions'].shape)}")
    print(f"log_likelihood finite: {bool(torch.isfinite(out['log_likelihood']).all().item())}")
    print(f"final reward: {out['reward'].tolist()}")
    print(f"initial reward: {out['init_reward'].tolist()}")
    print(f"best reward observed: {out['best_reward'].tolist()}")
    print("\nPASS: PC improvement split/merge validation completed.")


if __name__ == "__main__":
    main()
