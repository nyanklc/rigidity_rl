"""agent_loader.py -- rebuilding a run from its manifest.
Needs trained checkpoints, which are gitignored, so these skip on a fresh clone."""
import os
import pytest

import agent_loader
from conftest import ROOT, find_checkpoint, requires_artifacts


def test_missing_manifest_raises_a_helpful_error(make_env):
    e = make_env(n=4, domains="R^2")
    with pytest.raises(FileNotFoundError, match="manifest"):
        agent_loader.load_agent("definitely_not_a_real_run_name", e, e, device="cpu")


def test_backbone_depth_is_read_from_the_state_dict():
    sd = {"gnn.conv1.weight": None, "gnn.conv2.weight": None, "head.0.bias": None}
    assert agent_loader.backbone_depth(sd) == 2
    assert agent_loader.backbone_depth({"head.0.bias": None}) is None


def test_rebuild_backbone_changes_the_depth():
    from policy.gnn_backbone import GNNBackboneEquivariant

    class Holder:
        device = "cpu"
    h = Holder()
    h.gnn = GNNBackboneEquivariant(8, 6, 16, num_layers=3)
    assert h.gnn.num_layers == 3
    agent_loader.rebuild_backbone(h, 2)
    assert h.gnn.num_layers == 2
    agent_loader.rebuild_backbone(h, None)      # no-op
    assert h.gnn.num_layers == 2


def test_shapes_of_summarises_a_state_dict():
    import torch
    sd = {"a": torch.zeros(2, 3), "b": torch.zeros(5)}
    assert agent_loader.shapes_of(sd) == {"a": (2, 3), "b": (5,)}


def test_list_checkpoints_returns_a_mapping():
    got = agent_loader.list_checkpoints()
    assert isinstance(got, dict)
    for algo, names in got.items():
        assert algo in ("PPO", "DQN", "DDQN")
        assert all(isinstance(x, str) for x in names)


def test_the_legacy_shape_sniffing_loader_is_gone():
    """Dropped deliberately: a manifest is required now."""
    for gone in ("load_agent_legacy", "infer_architecture", "match_model_class",
                 "resolve_algorithm", "algorithm_from_roles"):
        assert not hasattr(agent_loader, gone), gone


@pytest.mark.slow
@requires_artifacts
def test_load_run_rebuilds_a_usable_agent():
    found = find_checkpoint(prefer=("phase4_ppo_equi_n8_R3_at300k",))
    if not found:
        pytest.skip("no usable manifest+checkpoint pair")
    name, env_name, algo = found
    if not os.path.exists(os.path.join(ROOT, "environments", f"{env_name}.json")):
        pytest.skip("environment config for this run is not present")
    agent, wrapped, raw, info = agent_loader.load_run(name, env_name, device="cpu")
    assert info["algorithm"] == algo
    roles = set(agent.models)
    assert roles & {"policy", "q_network"}
    obs, _ = wrapped.reset()
    from probe import deterministic_action
    scores, action = deterministic_action(agent, obs)
    assert scores.shape[-1] == wrapped.action_space.n
    assert 0 <= int(action.reshape(-1)[0]) < wrapped.action_space.n
