"""The Dict observation, its flags, and the legacy presets.
DESIGN_NOTES.md#dict-observation, #all-pairs-bearings, #rigidity-features."""
import numpy as np
import pytest

from conftest import ALL_DOMAINS
from environment import OBS_PRESETS, OBS_BACKBONE


@pytest.mark.parametrize("domain", ALL_DOMAINS)
@pytest.mark.parametrize("n", [4, 6, 8])
def test_observation_matches_the_declared_space(make_env, domain, n):
    e = make_env(n=n, domains=[domain] * n)
    obs, _ = e.reset()
    assert set(obs) == set(e.observation_space.spaces)
    for k, v in obs.items():
        assert e.observation_space[k].shape == np.shape(v), k


def test_edge_exists_channel_equals_the_adjacency(make_env):
    e = make_env(n=7, domains="R^3")
    obs, _ = e.reset()
    assert np.array_equal(obs["edge_features"][..., 3], obs["adj"])


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_bearings_are_unit_norm_on_every_ordered_pair(make_env, domain):
    n = 6
    e = make_env(n=n, domains=[domain] * n)
    obs, _ = e.reset()
    norms = np.linalg.norm(obs["edge_features"][..., :3], axis=-1)
    off = ~np.eye(n, dtype=bool)
    assert np.allclose(norms[off], 1.0, atol=1e-9)
    assert np.allclose(np.diag(norms), 0.0)


def test_candidate_bearing_flag_changes_content_not_shape(make_env):
    n = 8
    for flag in (True, False):
        e = make_env(n=n, domains="R^3", include_candidate_bearings=flag)
        e.reset()
        E = np.zeros((n, n), dtype=bool)
        E[0, 1] = E[1, 2] = E[2, 3] = True
        e.network.edges = E
        obs = e._get_obs()
        nz = int((np.linalg.norm(obs["edge_features"][..., :3], axis=-1) > 1e-9).sum())
        assert obs["edge_features"].shape == (n, n, 7)
        assert nz == (n * (n - 1) if flag else 3)


def test_coordinates_are_pose_normalized(make_env):
    e = make_env(n=8, domains="R^3")
    obs, _ = e.reset()
    c = obs["coord_features"]
    assert np.abs(c.mean(axis=0)).max() < 1e-9
    assert abs(np.sqrt((c ** 2).sum(-1).mean()) - 1.0) < 1e-9


def test_graph_features_flag_drops_the_centralities(make_env):
    lean = make_env(n=6, domains="R^3", graph_features=False)
    full = make_env(n=6, domains="R^3", graph_features=True)
    o_l, _ = lean.reset()
    o_f, _ = full.reset()
    assert o_l["node_features"].shape[-1] == o_f["node_features"].shape[-1] - 3
    assert o_l["edge_features"].shape[-1] == o_f["edge_features"].shape[-1] - 1


@pytest.mark.parametrize("flags,dn,de", [
    ((False, False, False), 0, 0),
    ((True, False, False), 3, 0),
    ((True, True, False), 4, 1),
    ((True, True, True), 4, 2),
])
def test_rigidity_flags_add_the_expected_channels(make_env, flags, dn, de):
    g, f, ed = flags
    base = make_env(n=6, domains="R^3")
    o0, _ = base.reset()
    e = make_env(n=6, domains="R^3",
                 rigidity_global=g, rigidity_flex=f, rigidity_edge=ed)
    o, _ = e.reset()
    assert o["node_features"].shape[-1] == o0["node_features"].shape[-1] + dn
    assert o["edge_features"].shape[-1] == o0["edge_features"].shape[-1] + de


def test_rigidity_flags_off_leaves_the_observation_untouched(make_env):
    a = make_env(n=6, domains="R^3")
    b = make_env(n=6, domains="R^3", rigidity_global=False,
                 rigidity_flex=False, rigidity_edge=False)
    np.random.seed(3); oa, _ = a.reset()
    np.random.seed(3); ob, _ = b.reset()
    for k in oa:
        assert np.array_equal(oa[k], ob[k]), k


def test_legacy_equivariant_preset_is_byte_exact(make_env):
    """A checkpoint fed rescaled inputs would be silently mis-evaluated."""
    e = make_env(n=8, domains="R^3", state_score_type="Weighted",
                 obs_space_type="DictEquivariantNodeFeaturesAndAdjAndSelection")
    e.reset()
    net = e.network
    obs = e._get_obs()
    expect_node = np.concatenate([net.get_domain_features(),
                                  net.get_degree_features(),
                                  net.get_closeness_centrality_features(),
                                  net.get_eigenvector_centrality_features(),
                                  net.get_node_betweenness_features()], axis=-1)
    expect_edge = np.concatenate([net.get_bearing_features(),
                                  net.get_edge_betweenness_features(),
                                  net.get_edge_reciprocity_features(),
                                  net.get_common_neighbors_features()], axis=-1)
    assert np.array_equal(obs["node_features"], expect_node)
    assert np.array_equal(obs["edge_features"], expect_edge)
    # raw, NOT pose-normalized -- this is the part that silently breaks checkpoints
    assert np.array_equal(obs["coord_features"], net.get_position_features())
    assert obs["edge_features"].shape[-1] == 6


@pytest.mark.parametrize("preset", sorted(OBS_PRESETS))
def test_every_obs_preset_builds_and_steps(make_env, preset):
    act = "DecideOnEdge" if "Proposal" in preset else "SelectNodesSequentially"
    e = make_env(n=6, domains="R^3", obs_space_type=preset, action_space_type=act)
    obs, _ = e.reset()
    assert set(obs) == set(e.observation_space.spaces)
    for _ in range(3):
        obs, *_ = e.step(e.action_space.sample())
    for k, v in obs.items():
        assert e.observation_space[k].shape == np.shape(v), k


def test_unknown_obs_type_raises(make_env):
    with pytest.raises(ValueError, match="unknown obs_type"):
        make_env(n=4, domains="R^3", obs_space_type="NoSuchObs")


def test_legacy_names_map_to_a_backbone():
    """A pre-merge config implied its GNN; the training scripts honour that."""
    assert OBS_BACKBONE["DictEquivariantNodeFeaturesAndAdjAndSelection"] == "Equivariant"
    assert OBS_BACKBONE["DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection"] == "GINE"
    assert "Dict" not in OBS_BACKBONE      # the current type implies nothing
