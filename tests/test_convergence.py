"""What the control harness pins down."""
import copy

import numpy as np
import pytest

import convergence
from rigidity import extended_bearing_rigidity_matrix, node_dof_projectors
from scenario import random_scenario


def goal_at(n, domain, seed=0, complete=True):
    np.random.seed(seed)
    net, _ = random_scenario(n, domain)
    return net.fully_connected() if complete else net


@pytest.mark.parametrize("domain", ["R^3", "SE(3)", "R^2xS^1"])
def test_b_transpose_annihilates_the_frameworks_own_bearings(domain):
    """`B^T . goal` is the bearing-error gradient, not a missing subtraction: every
    3-row block projects orthogonal to its own bearing, so `B^T . current` is zero."""
    net = goal_at(5, domain)
    brm = extended_bearing_rigidity_matrix(net)
    cur = np.asarray(net.get_bearings())
    goal = cur + np.random.default_rng(0).normal(0, 0.3, cur.shape)
    assert np.linalg.norm(brm.T @ cur) < 1e-9
    assert np.allclose(brm.T @ goal, brm.T @ (goal - cur), atol=1e-9)


def test_the_gain_masks_differ_exactly_where_the_gains_do():
    """B's columns are [3n positions | 3n attitudes] and the controller tiles per
    agent, so the two agree only when lin == ang."""
    n = 4
    assert np.array_equal(convergence.gain_mask(n, 1.0, 1.0),
                          convergence.tiled_mask(n, 1.0, 1.0))
    aligned = convergence.gain_mask(n, 100.0, 1.0)
    assert not np.array_equal(aligned, convergence.tiled_mask(n, 100.0, 1.0))
    assert np.all(aligned[:3 * n] == 100.0) and np.all(aligned[3 * n:] == 1.0)


@pytest.mark.parametrize("domain", ["R^2", "R^2xS^1", "R^3"])
def test_the_start_is_displaced_only_where_the_agents_can_move(domain):
    """A planar agent cannot undo a displacement in z, so a start that carries one is a
    shape no controller can reach."""
    goal = goal_at(4, domain)
    start = convergence.perturbed_start(goal, np.random.default_rng(0), 0.2)
    for a, b in zip(goal.agents, start.agents):
        d = b.pose.position - a.pose.position
        S, _ = node_dof_projectors(a)
        assert np.allclose(d, S @ d)
        if domain in ("R^2", "R^2xS^1"):
            assert abs(d[2]) < 1e-12
    assert np.array_equal(start.edges, goal.edges)


def test_bearing_error_is_per_edge_not_summed():
    """The experiment varies the edge count, so a summed error would make a denser
    graph drive each edge lower to reach the same number."""
    goal = goal_at(4, "R^3")
    bearings = np.asarray(goal.get_bearings())
    assert convergence.bearing_rms(goal, bearings) == pytest.approx(0.0)

    m = len(bearings) // 3
    off = bearings.copy()
    off[0] += 0.1                                    # one component, whatever m is
    assert convergence.bearing_rms(goal, off) == pytest.approx(0.1 / np.sqrt(m))


def test_the_controller_drives_the_start_back_to_the_goal_shape():
    """The gate on everything else. Position error is against the goal SHAPE, since
    bearing control cannot observe translation or scale."""
    goal = goal_at(4, "R^3")
    rank_K = convergence.rank_K_of(goal)
    start = convergence.perturbed_start(goal, np.random.default_rng(0), 0.1)
    trace, hit = convergence.simulate(start, goal, rank_K,
                                      convergence.gain_mask(4, 1.0, 1.0),
                                      steps=40_000, sample_every=1000)
    assert hit is not None
    assert trace[-1]["bearing_rms"] < trace[0]["bearing_rms"]
    assert trace[-1]["pos_err"] < trace[0]["pos_err"]
    assert set(convergence.TRACE_FIELDS) - {"case", "method"} <= set(trace[0])


def test_one_edge_set_serves_both_networks(make_env):
    """The goal bearings and the controller's B are indexed by the same edges, so a
    method picks one topology and it is applied to both formations."""
    e = make_env(n=5, domains="R^3", action_space_type="AddRemoveEdgeDiscreteNoSelfLoops",
                 max_steps=40, termination_condition_type="MaxSteps")
    e.reset()
    goal = copy.deepcopy(e.network)
    rows, traces, topologies = convergence.measure(
        env=e, goal_networks=[goal], methods=["greedy", "complete"],
        rng=np.random.default_rng(0), steps=2000, sample_every=500)

    assert {r["method"] for r in rows} == {"greedy", "complete"}
    assert topologies[(0, "complete")].sum() == 5 * 4
    for r in rows:
        assert r["m"] == int(topologies[(0, r["method"])].sum())
        assert [t for t in traces if t["method"] == r["method"]]
    # every method saw the same goal and the same start
    assert len({round(r["pos_err_0"], 12) for r in rows}) == 1
    assert all(set(convergence.ROW_FIELDS) <= set(r) for r in rows)


def test_the_trajectory_carries_the_positions_it_visited():
    """The 3-D figure needs the path, not only the error, so every sample keeps the
    formation it was measured on."""
    goal = goal_at(4, "R^3")
    rank_K = convergence.rank_K_of(goal)
    start = convergence.perturbed_start(goal, np.random.default_rng(0), 0.1)
    trace, _ = convergence.simulate(start, goal, rank_K,
                                    convergence.gain_mask(4, 1.0, 1.0),
                                    steps=2000, sample_every=500)
    assert len(trace) >= 3
    for t in trace:
        assert t["positions"].shape == (4, 3)
    first = trace[0]["positions"]
    assert np.allclose(first, [a.pose.position for a in start.agents])
    assert not np.allclose(first, trace[-1]["positions"])


def test_normalised_removes_the_gauge_a_figure_has_to_remove():
    """Translation and scale are unobservable to bearing control, and the velocity is
    orthogonal to them, so a raw path keeps whatever offset its start had."""
    P = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0], [1.0, 1.0, 1.0]])
    N = convergence.normalised(P)
    assert np.allclose(N.mean(axis=0), 0.0)
    assert np.sqrt((N ** 2).sum(axis=1).mean()) == pytest.approx(1.0)
    # a translated and rescaled copy normalises to the same thing
    assert np.allclose(N, convergence.normalised(P * 3.7 + np.array([5.0, -2.0, 9.0])))


@pytest.mark.parametrize("domain,expected", [("R^3", False), ("R^2", False),
                                             ("SE(3)", True), ("R^2xS^1", True)])
def test_rotation_is_gauge_only_where_every_agent_carries_a_frame(domain, expected):
    """A frameless agent measures in the global frame, so rotating the world rotates
    its bearings and the rotation is observable."""
    assert convergence.rotation_is_gauge(goal_at(4, domain)) is expected


def test_a_run_that_never_reaches_the_tolerance_says_so():
    """`steps_to_tol` stays None rather than reporting the step budget, which would
    read as a convergence time."""
    goal = goal_at(4, "R^3")
    rank_K = convergence.rank_K_of(goal)
    start = convergence.perturbed_start(goal, np.random.default_rng(0), 0.1)
    trace, hit = convergence.simulate(start, goal, rank_K,
                                      convergence.gain_mask(4, 1.0, 1.0),
                                      steps=200, sample_every=100, tol=1e-12)
    assert hit is None and len(trace) >= 2
