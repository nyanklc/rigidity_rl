"""Environment contract: reset/step, episode constants, config round-trip."""
import json
import numpy as np
import pytest

from conftest import ALL_DOMAINS, RANK_K_FORMULA, C_MAX, TERMINATIONS, config_dict
from environment import Environment
from rigidity import extended_bearing_rigidity_matrix as B_of


def test_reset_returns_obs_and_info(make_env):
    e = make_env()
    obs, info = e.reset()
    assert isinstance(info, dict)
    assert set(obs) == set(e.observation_space.spaces)


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_episode_constants_are_recomputed_each_episode(make_env, domain):
    n = 6
    e = make_env(n=n, domains=[domain] * n)
    for _ in range(3):
        e.reset()
        assert e.rank_K == RANK_K_FORMULA[domain](n)
        assert e.c_max == C_MAX[domain]
        assert e.m_req >= 1
        assert e.rank_K_pos == np.linalg.matrix_rank(
            B_of(e.network.fully_connected())[:, :3 * n])


def test_best_state_tracking_is_consistent(make_env):
    e = make_env(n=6, domains="R^3", max_steps=40,
                 termination_condition_type="MaxSteps")
    e.reset()
    best = e.best_state_score
    for _ in range(40):
        e.step(e.action_space.sample())
        best = max(best, e.last_stats["score"])
    s = e.last_episode_stats
    assert abs(s["Best state score"] - best) < 1e-9
    assert s["Best state score"] >= s["Final state score"] - 1e-9
    assert 0 <= s["Best step"] <= s["Length"]


def test_rigidity_channels_describe_the_post_action_graph(make_env):
    """step() must build the observation AFTER the rigidity computation."""
    n = 6
    e = make_env(n=n, domains="R^3", rigidity_global=True, rigidity_flex=True)
    e.reset()
    e.network.edges = np.zeros((n, n), dtype=bool)
    for i in range(4):
        e.network.edges[i, i + 1] = True
    for _ in range(8):
        obs, *_ = e.step(e.action_space.sample())
        rank = np.linalg.matrix_rank(e.network.extended_bearing_rigidity_matrix())
        expected = (e.rank_K - rank) / e.rank_K
        assert abs(obs["node_features"][0, -4] - expected) < 1e-9


@pytest.mark.parametrize("term", TERMINATIONS)
def test_every_termination_condition_runs(make_env, term):
    e = make_env(n=5, domains="R^3", termination_condition_type=term,
                 max_steps=15, truncate_enable=False)
    e.reset()
    for _ in range(30):
        _, _, t, tr, _ = e.step(e.action_space.sample())
        if t or tr:
            break
    assert e.last_episode_stats is not None or e.step_counter > 0


def test_max_steps_truncates_at_the_budget(make_env):
    e = make_env(n=5, domains="R^3", max_steps=12,
                 termination_condition_type="MaxSteps")
    e.reset()
    steps = 0
    while True:
        _, _, term, trunc, _ = e.step(e.action_space.sample())
        steps += 1
        if term or trunc:
            break
    assert steps == 12


def test_bandit_terminates_after_one_step(make_env):
    e = make_env(n=5, domains="R^3", termination_condition_type="Bandit")
    e.reset()
    _, _, term, _, _ = e.step(e.action_space.sample())
    assert term


def test_config_round_trips_through_load(tmp_path, env_config_file):
    path = env_config_file(n=6, domains="R^3", rigidity_flex=True,
                           graph_features=False, max_steps=17)
    e = Environment()
    e.load(path)
    assert e.n == 6
    assert e.max_steps == 17
    assert e.rigidity_flex and not e.graph_features
    assert not e.rigidity_global and not e.rigidity_edge
    obs, _ = e.reset()
    assert set(obs) == set(e.observation_space.spaces)


def test_config_defaults_when_new_keys_are_absent(tmp_path):
    """Older configs lack the newer flags; load() must default them."""
    cfg = config_dict()
    for k in ("include_candidate_bearings", "graph_features",
              "rigidity_global", "rigidity_flex", "rigidity_edge"):
        cfg.pop(k)
    p = tmp_path / "old.json"
    p.write_text(json.dumps(cfg))
    e = Environment()
    e.load(str(p))
    assert e.include_candidate_bearings is True
    assert e.graph_features is True
    assert not (e.rigidity_global or e.rigidity_flex or e.rigidity_edge)


def test_freeze_network_keeps_the_graph(make_env):
    e = make_env(n=6, domains="R^3")
    e.reset()
    e.freeze_network = True
    edges = e.network.edges.copy()
    positions = e.network.get_position_features().copy()
    e.reset()
    assert np.array_equal(e.network.edges, edges)
    assert np.allclose(e.network.get_position_features(), positions)


def test_reset_redraws_poses_and_edges_by_default(make_env):
    e = make_env(n=6, domains="R^3")
    e.reset()
    p0 = e.network.get_position_features().copy()
    e.reset()
    assert not np.allclose(e.network.get_position_features(), p0)


def test_last_stats_matches_a_fresh_computation(make_env):
    e = make_env(n=6, domains="R^3")
    e.reset()
    for _ in range(10):
        e.step(e.action_space.sample())
        brm = e.network.extended_bearing_rigidity_matrix()
        mbr, ibr, rank = e.network.is_MBR(rank_K=e.rank_K, brm=brm)
        assert e.last_stats["m"] == int(e.network.edges.sum())
        assert e.last_stats["rank"] == rank
        assert e.last_stats["is_IBR"] == bool(ibr)
        assert e.last_stats["is_MBR"] == bool(mbr)
