"""Action masking and skip_enabled.

The Phase 4 collapse: real logits drifted to -1e23 while masked entries sat at a
finite -1e9 sentinel, so masking inverted and argmax started picking *invalid*
actions -- which were free no-ops, making the failure absorbing.
"""
import inspect
import numpy as np
import pytest
import torch

from conftest import BACKBONES
from policy import build_models, MODELS
from policy.gnn_backbone import MASK_VALUE, unmask_if_all_masked
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space

# flat Discrete action spaces that carry a skip, and where its index sits
SKIP_INDEX = {
    "SelectNodesSequentially": lambda n: n,
    "AddRemoveEdgeDiscreteNoSelfLoops": lambda n: 2 * (n * n - n),
    "AddEdgeDiscreteNoSelfLoops": lambda n: n * n - n,
    "DecideOnEdge": lambda n: 2,
}
SKIPPABLE = [(role, bb, act) for (role, bb, act) in MODELS
             if act in SKIP_INDEX and role != "value"]


def obs_and_model(make_env, role, bb, act, n=6, allow_skip=True):
    e = make_env(n=n, domains="R^3", action_space_type=act)
    obs, _ = e.reset()
    algo = "PPO" if role == "policy" else "DQN"
    m = build_models(algo, backbone=bb, action_type=act, n=n,
                     node_feat_dim=obs["node_features"].shape[-1],
                     edge_feat_dim=obs["edge_features"].shape[-1] if "edge_features" in obs else 0,
                     gnn_hidden_dim=16, head_hidden_dim=16,
                     observation_space=e.observation_space,
                     action_space=e.action_space, device="cpu", allow_skip=allow_skip)
    f = flatten_tensorized_space(tensorize_space(e.observation_space, obs, device="cpu"))
    return e, list(m.values())[0], f


def test_mask_value_is_scale_free():
    """A finite sentinel can be outranked by drifting logits; -inf cannot."""
    assert MASK_VALUE == float("-inf")


def test_unmask_guard_prevents_nan():
    y = torch.full((1, 3), MASK_VALUE)
    out = unmask_if_all_masked(y)
    assert torch.isfinite(out).all()
    assert torch.isfinite(torch.softmax(out, -1)).all()
    x = torch.tensor([[1.0, 2.0, MASK_VALUE]])
    assert torch.equal(unmask_if_all_masked(x), x)


@pytest.mark.parametrize("backbone", ["Equivariant", "GINE", "Default"])
def test_masked_action_cannot_win_argmax_after_logit_drift(make_env, backbone):
    """The exact Phase 4 regression, reproduced by forcing the drift."""
    n = 6
    act = "AddRemoveEdgeDiscreteNoSelfLoops"
    E = n * n - n
    e, mod, _ = obs_and_model(make_env, "policy", backbone, act, n, allow_skip=False)
    e.network.edges = ~np.eye(n, dtype=bool)          # complete: every ADD invalid
    f = flatten_tensorized_space(tensorize_space(e.observation_space, e._get_obs(), device="cpu"))
    with torch.no_grad():
        mod.head[-1].bias.data -= 1e12                # drive real logits far down
        scores, _ = mod.compute({"observations": f}, role="policy")
    a = int(scores.flatten().argmax())
    assert E <= a < 2 * E, "argmax picked a masked ADD at the complete graph"
    assert torch.isfinite(torch.softmax(scores.float(), -1)).all()


@pytest.mark.parametrize("role,bb,act", SKIPPABLE,
                         ids=[f"{r}-{b}-{a}" for r, b, a in SKIPPABLE])
def test_model_declares_allow_skip(role, bb, act):
    assert "allow_skip" in inspect.signature(MODELS[(role, bb, act)].__init__).parameters


@pytest.mark.parametrize("role,bb,act", SKIPPABLE,
                         ids=[f"{r}-{b}-{a}" for r, b, a in SKIPPABLE])
def test_output_width_matches_the_action_space(make_env, role, bb, act):
    e, mod, f = obs_and_model(make_env, role, bb, act, allow_skip=False)
    with torch.no_grad():
        scores, _ = mod.compute({"observations": f}, role=role)
    assert scores.shape[-1] == e.action_space.n


@pytest.mark.parametrize("role,bb,act", SKIPPABLE,
                         ids=[f"{r}-{b}-{a}" for r, b, a in SKIPPABLE])
def test_skip_is_masked_when_disabled_and_finite_when_enabled(make_env, role, bb, act):
    n = 6
    si = SKIP_INDEX[act](n)
    _, mod_off, f_off = obs_and_model(make_env, role, bb, act, n, allow_skip=False)
    with torch.no_grad():
        off, _ = mod_off.compute({"observations": f_off}, role=role)
    assert torch.isinf(off.flatten()[si]) and off.flatten()[si] < 0

    _, mod_on, f_on = obs_and_model(make_env, role, bb, act, n, allow_skip=True)
    with torch.no_grad():
        on, _ = mod_on.compute({"observations": f_on}, role=role)
    assert torch.isfinite(on.flatten()[si])


@pytest.mark.parametrize("role,bb,act", [c for c in SKIPPABLE if c[0] == "q_network"],
                         ids=[f"{b}-{a}" for r, b, a in SKIPPABLE if r == "q_network"])
def test_random_act_never_offers_skip_when_disabled(make_env, role, bb, act):
    """DQN explores with random_act, so masking compute() alone is not enough."""
    n = 6
    si = SKIP_INDEX[act](n)
    _, mod, f = obs_and_model(make_env, role, bb, act, n, allow_skip=False)
    with torch.no_grad():
        picks = {int(mod.random_act({"observations": f}, role=role)[0].reshape(-1)[0])
                 for _ in range(400)}
    assert si not in picks


@pytest.mark.parametrize("act", ["SelectNodesSequentially", "AddRemoveEdgeDiscreteNoSelfLoops"])
@pytest.mark.parametrize("backbone", BACKBONES)
def test_rollout_executes_no_skip_when_disabled(make_env, act, backbone):
    n = 6
    e, mod, _ = obs_and_model(make_env, "policy", backbone, act, n, allow_skip=False)
    obs = e._get_obs()
    for _ in range(60):
        f = flatten_tensorized_space(tensorize_space(e.observation_space, obs, device="cpu"))
        with torch.no_grad():
            scores, _ = mod.compute({"observations": f}, role="policy")
        obs, *_ = e.step(int(scores.flatten().argmax()))
        assert e.last_action_kind != "skip"
