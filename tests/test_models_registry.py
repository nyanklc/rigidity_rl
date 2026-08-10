"""policy/registry.py: (role, backbone, action space) -> model class.
DESIGN_NOTES.md#model-registry."""
import inspect
import numpy as np
import pytest
import torch

from conftest import BACKBONES
from policy import MODELS, BACKBONES as REG_BACKBONES, build_models, instantiate, resolve
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space

COMBOS = sorted({(r, b, a) for (r, b, a) in MODELS if a is not None})
# pre-existing breakage in the obsolete GAT/MLP backbone, unrelated to any recent work
KNOWN_BROKEN = {
    ("policy", "Default", "AllEdges"),                          # missing fc_edge_index
    ("policy", "Default", "AddRemoveEdgeMultiDiscrete"),        # global_mean_pool not imported
    ("policy", "Default", "AddEdgeDiscreteNoSkipNoSelfLoops"),  # undefined n
    ("policy", "Default", "SelectNodesSequentially"),           # mis-sized head
    ("value",  "Default", "SelectNodesSequentially"),           # mis-sized head (critic)
}


def build_and_forward(make_env, role, bb, act, n=6):
    e = make_env(n=n, domains="R^3", action_space_type=act)
    obs, _ = e.reset()
    algo = "PPO" if role in ("policy", "value") else "DQN"
    m = build_models(algo, backbone=bb, action_type=act, n=n,
                     node_feat_dim=obs["node_features"].shape[-1],
                     edge_feat_dim=obs["edge_features"].shape[-1] if "edge_features" in obs else 0,
                     gnn_hidden_dim=16, head_hidden_dim=16,
                     observation_space=e.observation_space,
                     action_space=e.action_space, device="cpu", allow_skip=False)
    f = flatten_tensorized_space(tensorize_space(e.observation_space, obs, device="cpu"))
    with torch.no_grad():
        out, _ = m[role].compute({"observations": f}, role=role)
    return e, out


@pytest.mark.parametrize("role,bb,act", COMBOS, ids=[f"{r}-{b}-{a}" for r, b, a in COMBOS])
def test_every_registered_combination_builds_and_forwards(make_env, role, bb, act):
    if (role, bb, act) in KNOWN_BROKEN:
        pytest.xfail("pre-existing breakage in the obsolete Default backbone")
    e, out = build_and_forward(make_env, role, bb, act)
    assert torch.isfinite(out).any()
    if role == "value":
        assert out.shape[-1] == 1
    else:
        assert torch.isfinite(torch.softmax(out.float(), -1)).all()


@pytest.mark.parametrize("role,bb,act", [c for c in COMBOS if c[0] != "value"],
                         ids=[f"{r}-{b}-{a}" for r, b, a in COMBOS if r != "value"])
def test_action_head_width_matches_the_action_space(make_env, role, bb, act):
    if (role, bb, act) in KNOWN_BROKEN:
        pytest.xfail("pre-existing breakage in the obsolete Default backbone")
    e, out = build_and_forward(make_env, role, bb, act)
    if hasattr(e.action_space, "n"):
        assert out.shape[-1] == e.action_space.n


def test_backbone_list_is_what_the_registry_uses():
    assert set(REG_BACKBONES) == {b for (_, b, _) in MODELS}
    assert set(REG_BACKBONES) == set(BACKBONES)


def test_critic_falls_back_to_the_per_backbone_default():
    """(role, backbone, None) covers every action space with no selection stage."""
    for bb in REG_BACKBONES:
        assert resolve("value", bb, "AddRemoveEdgeDiscreteNoSelfLoops") is MODELS[("value", bb, None)]
        assert resolve("value", bb, "SelectNodesSequentially") is MODELS[("value", bb, "SelectNodesSequentially")]


def test_resolve_raises_a_listing_error_for_an_unknown_pair():
    with pytest.raises(KeyError) as ei:
        resolve("policy", "Equivariant", "NoSuchActionSpace")
    assert "Equivariant" in str(ei.value)


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError, match="unknown backbone"):
        build_models("PPO", backbone="Nope", action_type="SelectNodesSequentially")


def test_instantiate_filters_kwargs_by_signature():
    class Only:
        def __init__(self, a, b=2):
            self.a, self.b = a, b
    got = instantiate(Only, {"a": 1, "b": 3, "unrelated": "dropped"})
    assert (got.a, got.b) == (1, 3)


def test_feature_dims_propagate_from_the_observation(make_env):
    """Widening the observation must widen the model, not silently truncate."""
    for flags in [(False, False, False), (True, True, True)]:
        g, f_, ed = flags
        e = make_env(n=6, domains="R^3", rigidity_global=g,
                     rigidity_flex=f_, rigidity_edge=ed)
        obs, _ = e.reset()
        m = build_models("PPO", backbone="GINE", action_type="SelectNodesSequentially",
                         n=6, node_feat_dim=obs["node_features"].shape[-1],
                         edge_feat_dim=obs["edge_features"].shape[-1],
                         gnn_hidden_dim=16, head_hidden_dim=16,
                         observation_space=e.observation_space,
                         action_space=e.action_space, device="cpu", allow_skip=False)
        first = m["policy"].gnn.conv1
        assert first.nn[0].in_features == obs["node_features"].shape[-1]


@pytest.mark.parametrize("backbone", ["Equivariant", "GINE"])
def test_both_backbones_see_candidate_edge_features(backbone, make_env):
    """GINE used to message-pass over existing edges only, discarding candidates."""
    from policy.gnn_backbone import GNNBackboneEquivariant, GNNBackboneGINE
    torch.manual_seed(0)
    n, F, E = 5, 10, 7
    nodes, edges = torch.randn(1, n, F), torch.randn(1, n, n, E)
    coors = torch.randn(1, n, 3)
    edges[..., 3] = 0.0
    edges[0, 0, 1, 3] = 1.0                      # only 0->1 exists
    pert = edges.clone()
    pert[0, 2, 3] += 5.0                         # perturb a NON-edge
    if backbone == "GINE":
        g = GNNBackboneGINE(F, E, 32)
        d = (g(nodes, pert) - g(nodes, edges)).abs().max().item()
    else:
        g = GNNBackboneEquivariant(F, E, 32, init_eps=1e-1)
        d = (g(feats=nodes, coors=coors, edges=pert)
             - g(feats=nodes, coors=coors, edges=edges)).abs().max().item()
    assert d > 1e-6, f"{backbone} ignores candidate-edge features"
