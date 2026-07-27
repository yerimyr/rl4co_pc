import torch

from rl4co.envs import PartConsolidationImprovementEnv
from rl4co.models import PCImprovementModel, PCImprovementPPOModel
from rl4co.models.zoo.pc_improvement import PCImprovementPolicy
from rl4co.utils import RL4COTrainer


def test_pc_improvement_env_reset_and_step():
    env = PartConsolidationImprovementEnv(
        generator_params=dict(num_parts=6), improvement_steps=3, check_solution=False
    )
    td = env.reset(batch_size=[2])

    assert td["edge_bits"].shape == (2, env.num_pairs)
    assert td["group_id"].shape == (2, env.num_nodes)
    assert td["action_mask"].shape == (2, env.num_actions)
    assert td["action_mask"].any(dim=-1).all()

    action = td["action_mask"].float().argmax(dim=-1)
    td.set("action", action)
    next_td = env.step(td)["next"]

    assert next_td["edge_bits"].shape == (2, env.num_pairs)
    assert next_td["i"].eq(1).all()
    assert torch.isfinite(next_td["current_reward"]).all()


def test_pc_improvement_policy_rollout():
    env = PartConsolidationImprovementEnv(
        generator_params=dict(num_parts=6), improvement_steps=3, check_solution=False
    )
    td = env.reset(batch_size=[2])
    policy = PCImprovementPolicy(embed_dim=32, hidden_dim=64, improvement_steps=3)
    out = policy(td, env, phase="train")

    assert out["actions"].shape == (2, 3)
    assert out["reward"].shape == (2,)
    assert out["log_likelihood"].shape == (2,)
    assert torch.isfinite(out["reward"]).all()
    assert torch.isfinite(out["log_likelihood"]).all()


def test_pc_improvement_reinforce_training():
    env = PartConsolidationImprovementEnv(
        generator_params=dict(num_parts=6), improvement_steps=3, check_solution=False
    )
    model = PCImprovementModel(
        env,
        policy_kwargs=dict(embed_dim=32, hidden_dim=64, improvement_steps=3),
        baseline="no",
        train_data_size=8,
        val_data_size=8,
        test_data_size=8,
        batch_size=2,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator="cpu")
    trainer.fit(model)
    trainer.test(model)


def test_pc_improvement_ppo_training():
    env = PartConsolidationImprovementEnv(
        generator_params=dict(num_parts=6), improvement_steps=3, check_solution=False
    )
    model = PCImprovementPPOModel(
        env,
        policy_kwargs=dict(embed_dim=32, hidden_dim=64, improvement_steps=3),
        critic_kwargs=dict(embed_dim=32, hidden_dim=64),
        train_data_size=8,
        val_data_size=8,
        test_data_size=8,
        batch_size=2,
        mini_batch_size=2,
        ppo_epochs=1,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator="cpu")
    trainer.fit(model)
    trainer.test(model)
