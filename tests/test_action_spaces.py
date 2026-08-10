"""Every action space, the index conventions, and action-kind classification."""
import numpy as np
import pytest

from conftest import ACTION_SPACES


@pytest.mark.parametrize("action", ACTION_SPACES)
def test_action_space_steps_without_error(make_env, action):
    e = make_env(n=6, domains="R^3", action_space_type=action)
    obs, _ = e.reset()
    for _ in range(20):
        obs, reward, term, trunc, _ = e.step(e.action_space.sample())
        assert np.isfinite(float(reward))
        assert isinstance(bool(term), bool) and isinstance(bool(trunc), bool)
    for k, v in obs.items():
        assert e.observation_space[k].shape == np.shape(v), k


@pytest.mark.parametrize("action", ACTION_SPACES)
def test_no_action_ever_creates_a_self_loop(make_env, action):
    e = make_env(n=6, domains="R^3", action_space_type=action)
    e.reset()
    for _ in range(60):
        e.step(e.action_space.sample())
        assert not np.any(np.diag(e.network.edges)), action


def test_addremove_decoder_matches_the_model_mask_convention(make_env):
    """The model masks index k assuming add-block then remove-block, row-major
    skipping the diagonal. If the env decodes k differently, the wrong action is
    masked and the policy is silently steered."""
    n = 6
    e = make_env(n=n, domains="R^3", action_space_type="AddRemoveEdgeDiscreteNoSelfLoops")
    e.reset()
    E = n * n - n

    def decode(idx):
        before = e.network.edges.copy()
        e.step(idx)
        diff = np.argwhere(e.network.edges != before)
        return None if len(diff) == 0 else tuple(diff[0])

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            t = j if j < i else j - 1
            k = i * (n - 1) + t
            e.network.edges = np.zeros((n, n), dtype=bool)   # add is valid
            assert decode(k) == (i, j), f"add index {k}"
            e.network.edges = np.ones((n, n), dtype=bool) & ~np.eye(n, dtype=bool)
            assert decode(k + E) == (i, j), f"remove index {k + E}"


def test_addremove_skip_is_the_last_index(make_env):
    n = 6
    e = make_env(n=n, domains="R^3", action_space_type="AddRemoveEdgeDiscreteNoSelfLoops")
    e.reset()
    before = e.network.edges.copy()
    e.step(e.action_space.n - 1)
    assert np.array_equal(e.network.edges, before)
    assert e.last_action_kind == "skip"


def test_action_kinds_for_select_nodes_sequentially(make_env):
    """select, add, select, remove, skip -- the first pick is protocol, not waste."""
    n = 6
    e = make_env(n=n, domains="R^3", action_space_type="SelectNodesSequentially")
    e.reset()
    e.network.edges = np.zeros((n, n), dtype=bool)
    kinds = []
    for a in [2, 3, 2, 3, n]:
        e.step(a)
        kinds.append(e.last_action_kind)
    assert kinds == ["select", "add", "select", "remove", "skip"]
    assert e.episode_accum["kinds"] == {"add": 1, "remove": 1, "noop": 0,
                                        "skip": 1, "select": 2}


def test_action_kinds_for_addremove(make_env):
    """add-new, add-existing (noop), remove, remove-absent (noop), skip."""
    n = 6
    e = make_env(n=n, domains="R^3", action_space_type="AddRemoveEdgeDiscreteNoSelfLoops")
    e.reset()
    e.network.edges = np.zeros((n, n), dtype=bool)
    E = n * n - n

    def idx(i, j, rm=False):
        t = j if j < i else j - 1
        return i * (n - 1) + t + (E if rm else 0)

    kinds = []
    for a in [idx(0, 1), idx(0, 1), idx(0, 1, True), idx(0, 1, True), 2 * E]:
        e.step(a)
        kinds.append(e.last_action_kind)
    assert kinds == ["add", "noop", "remove", "noop", "skip"]


def test_action_kind_fractions_sum_to_one(make_env):
    e = make_env(n=6, domains="R^3", action_space_type="AddRemoveEdgeDiscreteNoSelfLoops",
                 max_steps=25, termination_condition_type="MaxSteps")
    e.reset()
    for _ in range(25):
        e.step(e.action_space.sample())
    s = e.last_episode_stats
    total = sum(s[f"Actions/ {k} fraction"]
                for k in ("add", "remove", "noop", "skip", "select"))
    assert abs(total - 1.0) < 1e-9


def test_edit_efficiency_separates_monotone_from_oscillating(make_env):
    n = 6
    E = n * n - n

    def idx(i, j, rm=False):
        t = j if j < i else j - 1
        return i * (n - 1) + t + (E if rm else 0)

    def run(seq):
        e = make_env(n=n, domains="R^3",
                     action_space_type="AddRemoveEdgeDiscreteNoSelfLoops")
        e.reset()
        full = ~np.eye(n, dtype=bool)
        e.network.edges = full.copy()
        e.initial_m = int(full.sum())
        for a in seq:
            e.step(a)
        acc = e.episode_accum
        m1 = int(e.network.edges.sum())
        return abs(m1 - e.initial_m) / max(acc["edits"], 1)

    monotone = run([idx(0, 1, True), idx(0, 2, True), idx(0, 3, True), idx(0, 4, True)])
    oscillating = run([idx(1, 2, True), idx(1, 2), idx(1, 2, True), idx(1, 2)])
    assert monotone == pytest.approx(1.0)
    assert oscillating < 0.5
