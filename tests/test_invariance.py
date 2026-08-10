"""Similarity invariance of the observation and of the policy.
CLAUDE.md "Invariance", THEORY.md sections 3 and 11."""
import numpy as np
import pytest
import torch

from conftest import ALL_DOMAINS, ORIENTED_DOMAINS, RD_DOMAINS, LOOSE_TOL
from policy import build_models
from policy.gnn_backbone import GNNBackboneEquivariant
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space

INVARIANT_CHANNELS = ["node_features", "edge_features", "adj"]


def snapshot(e):
    brm = e.network.extended_bearing_rigidity_matrix()
    e.compute_rigidity_features(brm, np.linalg.matrix_rank(brm), False)
    return {k: np.array(v, dtype=float) for k, v in e._get_obs().items()}


def transformed(e, kind, planar):
    if kind == "translate":
        e.network.translate_network([3.1, -2.4, 0.0 if planar else 1.7])
    elif kind == "rotate":
        e.network.rotate_network([0, 0, 1] if planar else [0.3, 0.5, 0.81], 0.9)
    else:
        e.network.scale_network(2.7)
    return snapshot(e)


def rigid_env(make_env, domain, n=6):
    return make_env(n=n, domains=[domain] * n, graph_features=True,
                    rigidity_global=True, rigidity_flex=True, rigidity_edge=True)


@pytest.mark.parametrize("domain", ALL_DOMAINS)
@pytest.mark.parametrize("kind", ["translate", "scale"])
def test_translation_and_scaling_leave_every_channel_alone(make_env, domain, kind):
    e = rigid_env(make_env, domain)
    e.reset()
    before = snapshot(e)
    after = transformed(e, kind, domain in ("R^2", "R^2xS^1"))
    for k in before:
        assert np.abs(before[k] - after[k]).max() < 1e-7, f"{domain} {kind} {k}"


@pytest.mark.parametrize("domain", ORIENTED_DOMAINS)
def test_rotation_leaves_every_invariant_channel_alone_in_oriented_domains(make_env, domain):
    """Bearings are R_i^T p_hat here, so the frame rotates with the world."""
    e = rigid_env(make_env, domain)
    e.reset()
    before = snapshot(e)
    after = transformed(e, "rotate", domain == "R^2xS^1")
    for k in INVARIANT_CHANNELS:
        assert np.abs(before[k] - after[k]).max() < 1e-7, f"{domain} {k}"


@pytest.mark.parametrize("domain", RD_DOMAINS)
def test_rotation_moves_the_bearings_in_Rd_known_limitation(make_env, domain):
    """R^d agents have no frame, so bearings are global-frame vectors.

    Asserted as an EXPECTED violation: if the observation is ever made
    rotation-invariant in R^d this fails and must be updated deliberately.
    See CLAUDE.md known issue 7 for the options.
    """
    e = rigid_env(make_env, domain)
    e.reset()
    before = snapshot(e)
    after = transformed(e, "rotate", domain == "R^2")
    assert np.abs(before["edge_features"][..., :3]
                  - after["edge_features"][..., :3]).max() > 1e-3
    # everything that is not a bearing stays invariant even here
    assert np.abs(before["node_features"] - after["node_features"]).max() < 1e-7
    assert np.abs(before["adj"] - after["adj"]).max() < 1e-7


def test_coordinates_rotate_by_design(make_env):
    """EGNN consumes coords only through ||x_i - x_j||^2, so this is fine."""
    e = rigid_env(make_env, "R^3")
    e.reset()
    before = snapshot(e)
    after = transformed(e, "rotate", False)
    assert np.abs(before["coord_features"] - after["coord_features"]).max() > 1e-3


def test_egnn_feats_ignore_a_rotation_of_the_coordinates():
    g = GNNBackboneEquivariant(10, 7, 32)
    x, c, ed = torch.randn(1, 6, 10), torch.randn(1, 6, 3), torch.randn(1, 6, 6, 7)
    th = 0.9
    R = torch.tensor([[np.cos(th), -np.sin(th), 0],
                      [np.sin(th), np.cos(th), 0], [0, 0, 1.0]], dtype=torch.float32)
    assert torch.allclose(g(feats=x, coors=c, edges=ed),
                          g(feats=x, coors=c @ R.T, edges=ed), atol=1e-6)


def logits_under_rotation(make_env, domain, backbone, init_eps):
    n = 6
    e = rigid_env(make_env, domain, n)
    obs, _ = e.reset()
    m = build_models("PPO", backbone=backbone, action_type="SelectNodesSequentially",
                     n=n, node_feat_dim=obs["node_features"].shape[-1],
                     edge_feat_dim=obs["edge_features"].shape[-1],
                     gnn_hidden_dim=32, head_hidden_dim=32,
                     observation_space=e.observation_space,
                     action_space=e.action_space, device="cpu", allow_skip=False)
    if backbone == "Equivariant":
        # the 1e-3 default makes an untrained EGNN numerically blind to edge
        # features, which fakes invariance; trained weights sit near 1e-1
        m["policy"].gnn = GNNBackboneEquivariant(
            obs["node_features"].shape[-1] + 1, obs["edge_features"].shape[-1],
            32, init_eps=init_eps)

    def go():
        f = flatten_tensorized_space(tensorize_space(e.observation_space, snapshot(e), device="cpu"))
        with torch.no_grad():
            return m["policy"].compute({"observations": f}, role="policy")[0]

    a = go()
    e.network.rotate_network([0, 0, 1] if domain in ("R^2", "R^2xS^1") else [0.3, 0.5, 0.81], 0.9)
    b = go()
    # masked entries are -inf in both; inf - inf is nan, so compare the finite ones
    finite = torch.isfinite(a) & torch.isfinite(b)
    assert finite.any(), "every logit was masked"
    return (a[finite] - b[finite]).abs().max().item()


@pytest.mark.parametrize("domain", ORIENTED_DOMAINS)
@pytest.mark.parametrize("backbone", ["Equivariant", "GINE"])
def test_policy_logits_are_rotation_invariant_in_oriented_domains(make_env, domain, backbone):
    assert logits_under_rotation(make_env, domain, backbone, 1e-1) < LOOSE_TOL


@pytest.mark.parametrize("backbone", ["Equivariant", "GINE"])
def test_policy_logits_are_rotation_dependent_in_Rd(make_env, backbone):
    """The consequence of the known limitation, measured at trained-scale weights."""
    assert logits_under_rotation(make_env, "R^3", backbone, 1e-1) > 1e-4


def test_untrained_egnn_fakes_invariance_at_the_default_init(make_env):
    """Guards the testing trap: at init_eps=1e-3 the EGNN cannot see edge features."""
    blind = logits_under_rotation(make_env, "R^3", "Equivariant", 1e-3)
    awake = logits_under_rotation(make_env, "R^3", "Equivariant", 1e-1)
    assert blind < 1e-6 < awake
