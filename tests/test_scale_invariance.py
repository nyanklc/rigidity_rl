"""Nothing the policy sees may scale with n.

A policy trained at one size cannot transfer if its inputs or its internal
activations grow with n. Both happened: EGNN/GINE aggregated with sum/add over
n-1 dense neighbours, the flex features carried a sqrt(n) that assumed a fixed
flex dimension, degree and common-neighbour counts were raw, and the initial-graph
sampler's spread grew like n^4. The pair channels are bounded to [0, 1] by their
own per-pair normalisation, which is what keeps them flat here.
"""
import numpy as np
import pytest
import torch

from conftest import ALL_DOMAINS
from policy.gnn_backbone import GNNBackboneEquivariant, GNNBackboneGINE

SIZES = [8, 16]          # n=32 costs ~10x per step; covered by the slow test below


def rich_env(make_env, n, domain):
    return make_env(n=n, domains=[domain] * n,
                    action_space_type="AddRemoveEdgeDiscreteNoSelfLoops",
                    graph_features=False, random_graph_with_mean_min_edges=True,
                    rigidity_global=True, rigidity_flex=True, rigidity_edge=True)


def channel_mean(make_env, n, domain, key, sl, reps=12):
    e = rich_env(make_env, n, domain)
    vals = []
    for _ in range(reps):
        e.reset()
        for _ in range(4):
            e.step(e.action_space.sample())
        a = e._get_obs()[key]
        vals.append(a.reshape(-1, a.shape[-1])[:, sl])
    return float(np.concatenate(vals).mean())


# (key, slice, label) for the channels that used to drift
DRIFTING = [
    ("node_features", slice(5, 7), "degree"),
    ("node_features", slice(10, 11), "flex_mag"),
    ("edge_features", slice(6, 7), "add_gain"),
    ("edge_features", slice(8, 9), "add_rank"),
]


@pytest.mark.parametrize("key,sl,label", DRIFTING, ids=[d[2] for d in DRIFTING])
def test_feature_scale_does_not_drift_with_n(make_env, key, sl, label):
    got = {n: channel_mean(make_env, n, "R^3", key, sl) for n in SIZES}
    lo, hi = min(got.values()), max(got.values())
    assert hi / max(lo, 1e-9) < 1.6, f"{label} drifts with n: {got}"


def test_m_over_m_req_is_centred_on_one(make_env):
    """The sampler must pose the same relative difficulty at every n."""
    for n in SIZES:
        e = rich_env(make_env, n, "R^3")
        ratio = []
        for _ in range(60):
            e.reset()
            ratio.append(int(e.network.edges.sum()) / e.m_req)
        assert 0.75 < float(np.mean(ratio)) < 1.35, f"n={n}: {np.mean(ratio):.2f}"


def test_initial_edge_spread_is_proportional_not_quadratic(make_env):
    """sd/mean must stay roughly constant; it used to grow like n^2."""
    rel = {}
    for n in SIZES:
        e = rich_env(make_env, n, "R^3")
        ms = []
        for _ in range(80):
            e.reset()
            ms.append(int(e.network.edges.sum()))
        rel[n] = float(np.std(ms)) / max(float(np.mean(ms)), 1e-9)
    assert max(rel.values()) / max(min(rel.values()), 1e-9) < 2.0, rel


def _trained_scale(g, std=0.15):
    """An untrained backbone is numerically blind to this: init_eps makes the
    pooled message ~1e-10 of the node residual, so sum and mean agree to 3
    decimals and the test passes under either. Same trap as the invariance
    tests.
    """
    for m in g.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    return g


def backbone_activations(backbone, sizes=(8, 16, 32, 64), **kw):
    F, E, out = 11, 8, {}
    for n in sizes:
        torch.manual_seed(0)
        g = (GNNBackboneEquivariant(F, E, 128, **kw) if backbone == "Equivariant"
             else GNNBackboneGINE(F, E, 128, **kw))
        _trained_scale(g)
        x, c, e = torch.randn(1, n, F), torch.randn(1, n, 3), torch.randn(1, n, n, E)
        with torch.no_grad():
            h = g(feats=x, coors=c, edges=e) if backbone == "Equivariant" else g(x, e)
        out[n] = float(h.abs().mean())
    return out


@pytest.mark.parametrize("backbone", ["Equivariant", "GINE"])
def test_backbone_activations_do_not_scale_with_n(backbone):
    out = backbone_activations(backbone)
    rel = {n: v / out[8] for n, v in out.items()}
    assert max(rel.values()) < 1.6, f"{backbone} activations scale with n: {rel}"


def test_sum_pooling_is_what_the_mean_default_prevents():
    """Guards the guard: if this stops blowing up, the test above went vacuous."""
    out = backbone_activations("Equivariant", m_pool="sum", update_coors=False)
    assert out[64] / out[8] > 10, f"sum pooling no longer drifts: {out}"


def test_egnn_coordinate_update_stays_off():
    """m_pool does not cover this path -- the coordinate update is a hardcoded
    sum over j, and its result re-enters the next layer via rel_dist."""
    # egnn_pytorch keeps no update_coors attribute; it drops coors_mlp instead
    assert GNNBackboneEquivariant(11, 8, 32).conv1.coors_mlp is None
    out = backbone_activations("Equivariant", update_coors=True)
    assert out[64] / out[8] > 10, f"coordinate drift no longer shows: {out}"


def test_aggregation_is_degree_normalized():
    """sum/add over n-1 dense neighbours is what made activations scale."""
    assert GNNBackboneEquivariant(11, 8, 32).conv1.m_pool_method == "mean"
    assert GNNBackboneGINE(11, 8, 32).conv1.aggr == "mean"


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_flex_features_are_comparable_across_domains(make_env, domain):
    """Different rank_K and deficits must not change the feature's scale."""
    v = channel_mean(make_env, 6, domain, "node_features", slice(10, 11), reps=20)
    assert 0.3 < v < 1.8, f"{domain}: flex_mag mean {v:.3f}"


def test_legacy_presets_keep_the_raw_counts(make_env):
    """Old checkpoints must still see the scales they were trained on."""
    e = make_env(n=8, domains="R^3", state_score_type="Weighted",
                 obs_space_type="DictEquivariantNodeFeaturesAndAdjAndSelection")
    e.reset()
    obs = e._get_obs()
    assert np.array_equal(obs["node_features"][:, 5:7], e.network.get_degree_features())


@pytest.mark.slow
@pytest.mark.parametrize("key,sl,label", DRIFTING, ids=[d[2] for d in DRIFTING])
def test_feature_scale_holds_up_to_n32(make_env, key, sl, label):
    got = {n: channel_mean(make_env, n, "R^3", key, sl, reps=8) for n in (8, 16, 32)}
    lo, hi = min(got.values()), max(got.values())
    assert hi / max(lo, 1e-9) < 1.7, f"{label} drifts with n: {got}"
