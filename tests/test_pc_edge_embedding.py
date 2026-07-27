import torch

from rl4co.envs import PartConsolidationEnv
from rl4co.models import AttentionModelPolicy
from rl4co.models.nn.graph.pc_encoder import (
    PCEdgeAwareEncoder,
    pc_dense_edge_features,
    pc_tensordict_to_edge_index,
)


def _pc_tensordict(batch_size=2, num_parts=20):
    env = PartConsolidationEnv(generator_params=dict(num_parts=num_parts))
    data = env.generator(batch_size=[batch_size])
    return env, env.reset(data)


def test_pc_edge_conversion_shapes():
    env, td = _pc_tensordict(batch_size=2, num_parts=20)
    dense_edge_attr = pc_dense_edge_features(td)
    graphs = pc_tensordict_to_edge_index(td)

    assert dense_edge_attr.shape[:3] == (2, env.generator.num_nodes, env.generator.num_nodes)
    assert len(graphs) == 2
    for edge_index, edge_attr in graphs:
        assert edge_index.shape[0] == 2
        assert edge_attr.shape[0] == edge_index.shape[1]
        assert edge_attr.shape[-1] == dense_edge_attr.shape[-1]


def test_pc_edge_encoder_output_changes_when_edges_change():
    _, td = _pc_tensordict(batch_size=2, num_parts=20)
    encoder = PCEdgeAwareEncoder(embed_dim=64, num_layers=1, env_name="pc")

    h1, init_h1 = encoder(td)

    td_edge_changed = td.clone()
    td_edge_changed["W"] = td_edge_changed["W"] + 0.25 * td_edge_changed[
        "assembly_adj"
    ].float()
    td_edge_changed["edge_features"] = td_edge_changed["edge_features"].clone()
    td_edge_changed["edge_features"][..., 0] = 1.0 - td_edge_changed["edge_features"][..., 0]

    h2, init_h2 = encoder(td_edge_changed)

    assert h1.shape == h2.shape == (2, 21, 64)
    assert init_h1.shape == init_h2.shape == (2, 21, 64)
    assert torch.isfinite(h1).all()
    assert torch.isfinite(h2).all()
    assert torch.allclose(init_h1, init_h2)
    assert not torch.allclose(h1, h2)


def test_pc_edge_logits_change_when_edges_change():
    env, td = _pc_tensordict(batch_size=2, num_parts=20)
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=64,
        num_heads=4,
        encoder=PCEdgeAwareEncoder(embed_dim=64, num_layers=1, env_name=env.name),
    )

    h1, _ = policy.encoder(td)
    td1, _, cache1 = policy.decoder.pre_decoder_hook(td, env, h1)
    logits1, mask1 = policy.decoder(td1, cache1)

    td_edge_changed = td.clone()
    td_edge_changed["W"] = td_edge_changed["W"] + 0.25 * td_edge_changed[
        "assembly_adj"
    ].float()
    td_edge_changed["edge_features"] = td_edge_changed["edge_features"].clone()
    td_edge_changed["edge_features"][..., 0] = 1.0 - td_edge_changed["edge_features"][..., 0]

    h2, _ = policy.encoder(td_edge_changed)
    td2, _, cache2 = policy.decoder.pre_decoder_hook(td_edge_changed, env, h2)
    logits2, mask2 = policy.decoder(td2, cache2)

    assert logits1.shape == logits2.shape == (2, 21)
    assert torch.equal(mask1, mask2)
    assert not torch.allclose(logits1, logits2)
