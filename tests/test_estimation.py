"""Bearing noise, the shape estimator, and its agreement with the bound. 18."""
import copy

import numpy as np
import pytest

from conftest import ALL_DOMAINS, MIXES, PLANAR
import estimation as E
from rigidity import (estimation_error, estimation_error_blocks, estimation_error_of,
                      extended_bearing_rigidity_matrix as B_of, greedy_rigid_construction,
                      scaled_rigidity_matrix)
from scenario import random_scenario


def rigid_net(domains, n=8, seed=0):
    doms = [domains] * n if isinstance(domains, str) else list(domains)
    net, _ = random_scenario(len(doms), doms, edge_count=0)
    rank_K = int(np.linalg.matrix_rank(B_of(net.fully_connected())))
    greedy_rigid_construction(net, rank_K, np.random.default_rng(seed))
    return net, rank_K


# ---------------------------------------------------------------- the bearing map
@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_B_is_the_jacobian_of_the_estimator_bearing_map(domain):
    """B differentiates this module's bearing map, in every domain."""
    net, _ = rigid_net(domain, n=6)
    J, n, eps = B_of(net), net.n, 1e-6

    worst = 0.0
    for c in range(6 * n):
        d = np.zeros(6 * n)
        d[c] = 1.0
        d = E.restrict_to_dofs(net, d)
        if np.linalg.norm(d) == 0:
            continue
        fd = (E.true_bearings(E.apply_variation(net, d, eps))
              - E.true_bearings(E.apply_variation(net, d, -eps))) / (2 * eps)
        worst = max(worst, float(np.abs(J @ d - fd).max()))
    assert worst < 1e-7


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_true_bearings_are_unit_and_ordered_like_Bs_row_blocks(domain):
    net, _ = rigid_net(domain, n=6)
    b = E.true_bearings(net).reshape(-1, 3)
    assert b.shape[0] == int(net.edges.sum()) == B_of(net).shape[0] // 3
    assert np.allclose(np.linalg.norm(b, axis=1), 1.0)


# ---------------------------------------------------------------- the noise model
@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_noise_is_tangent_and_sigma_is_an_angle(domain):
    """The perturbed bearing stays a unit vector and turns by about sigma."""
    net, _ = rigid_net(domain, n=6)
    b = E.true_bearings(net).reshape(-1, 3)

    for sigma in (1e-3, 1e-2):
        z = E.perturb_bearings(net, sigma, np.random.default_rng(3)).reshape(-1, 3)
        assert np.allclose(np.linalg.norm(z, axis=1), 1.0)
        angle = np.arccos(np.clip(np.einsum("kd,kd->k", z, b), -1, 1))
        # two tangent DOFs, so E[angle^2] = 2 sigma^2 to first order
        assert 0.5 < float(np.sqrt((angle ** 2).mean()) / (np.sqrt(2) * sigma)) < 1.6


def test_zero_sigma_leaves_the_bearings_alone():
    net, _ = rigid_net("SE(3)", n=6)
    b = E.true_bearings(net)
    assert np.allclose(E.perturb_bearings(net, 0.0, np.random.default_rng(0)), b)


# ---------------------------------------------------------------- admissible DOFs
@pytest.mark.parametrize("domain", sorted(PLANAR))
def test_a_planar_agent_never_leaves_its_plane(domain):
    """A planar agent's z stays 0 through the solve."""
    net, rank_K = rigid_net(domain, n=6)
    z = E.perturb_bearings(net, 0.05, np.random.default_rng(1))
    est, _ = E.solve_shape(net, z)
    assert np.allclose([a.pose.position[2] for a in est.agents], 0.0, atol=1e-12)


def test_restrict_to_dofs_kills_exactly_Bs_zero_columns():
    net, _ = rigid_net(["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"] * 2)
    dead = np.abs(B_of(net)).max(axis=0) < 1e-12
    kept = E.restrict_to_dofs(net, np.ones(6 * net.n))
    assert np.allclose(kept[dead], 0.0)


# ---------------------------------------------------------------- the solver
@pytest.mark.parametrize("domain", ALL_DOMAINS + [MIXES[4]])
def test_exact_bearings_are_solved_to_machine_precision(domain):
    """Noiseless data from a displaced start solves to machine precision."""
    net, rank_K = rigid_net(domain, n=8)
    z = E.true_bearings(net)
    start = E.apply_variation(
        net, E.restrict_to_dofs(net, np.random.default_rng(2).normal(size=6 * net.n) * 0.01))

    est, info = E.solve_shape(net, z, init=start, iters=40)
    assert info["converged"]
    assert info["residual"] < 1e-9


def test_the_gauge_quotient_is_exact_in_Rd_and_second_order_elsewhere():
    """The gauge quotient is exact in R^d and second order in SE(3)."""
    for domain, exact in (("R^3", True), ("SE(3)", False)):
        net, rank_K = rigid_net(domain, n=8)
        z = E.true_bearings(net)
        direction = E.restrict_to_dofs(net, np.random.default_rng(5).normal(size=6 * net.n))
        direction /= np.linalg.norm(direction)

        errs = []
        for delta in (4e-2, 2e-2, 1e-2):
            est, _ = E.solve_shape(net, z, init=E.apply_variation(net, direction, delta),
                                   iters=40)
            errs.append(E.shape_error(net, est, rank_K)[0])

        if exact:
            assert max(errs) < 1e-9
        else:
            # halving delta must shrink the remainder by clearly more than half
            assert errs[0] / errs[1] > 2.5 and errs[1] / errs[2] > 2.5


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_shape_error_ignores_the_unobservable_gauge(domain):
    """A gauge motion scores zero error; a direction B sees does not."""
    net, rank_K = rigid_net(domain, n=6)
    Z_K = E.gauge_basis(net, rank_K)
    assert Z_K.shape[1] > 0

    g = Z_K @ np.random.default_rng(4).normal(size=Z_K.shape[1])
    g = g / np.linalg.norm(g) * 1e-4
    moved = E.apply_variation(net, g)

    assert E.shape_error(net, moved, rank_K)[0] < 1e-6
    # and a direction B does see is an error
    seen = E.restrict_to_dofs(net, np.random.default_rng(6).normal(size=6 * net.n))
    seen = seen - Z_K @ (Z_K.T @ seen)
    seen = seen / np.linalg.norm(seen) * 1e-4
    assert E.shape_error(net, E.apply_variation(net, seen), rank_K)[0] > 1e-6


# ---------------------------------------------------------------- analytic metrics
@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_estimation_error_matches_a_brute_force_pseudo_inverse(domain):
    net, rank_K = rigid_net(domain, n=6)
    Bs = scaled_rigidity_matrix(net)
    a_opt, e_opt, d_opt = estimation_error_of(net, rank_K)

    M = Bs.T @ Bs
    w_all = np.sort(np.linalg.eigvalsh(M))
    w = w_all[len(M) - rank_K:]

    # the cutoff has to be placed by hand: B^T B squares the condition number and
    # at the default rcond a gauge mode survives in SE(3)
    rcond = 0.5 * (w_all[len(M) - rank_K - 1] + w[0]) / w_all[-1]
    assert np.isclose(a_opt, np.trace(np.linalg.pinv(M, rcond=rcond)), rtol=1e-6)
    assert np.isclose(e_opt, 1.0 / w[0], rtol=1e-6)
    assert np.isclose(d_opt, -np.log(w).sum(), rtol=1e-6)


@pytest.mark.parametrize("domain", ALL_DOMAINS + [MIXES[4]])
def test_the_block_split_sums_back_to_the_whole_trace(domain):
    """a_pos + a_att equals a_opt, across the SVD and eigh paths."""
    net, rank_K = rigid_net(domain, n=8)
    a_opt, _, _ = estimation_error_of(net, rank_K)
    a_pos, a_att = estimation_error_blocks(scaled_rigidity_matrix(net), rank_K, net.n)
    assert np.isclose(a_pos + a_att, a_opt, rtol=1e-8)


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_estimation_error_is_similarity_invariant(domain):
    """The three criteria are unchanged by translation, rotation and scaling."""
    net, rank_K = rigid_net(domain, n=6)
    base = np.array(estimation_error_of(net, rank_K))

    moved = copy.deepcopy(net)
    moved.translate_network([3.0, -2.0, 1.5])
    axis = np.array([0.0, 0.0, 1.0]) if any(a.domain in PLANAR for a in net.agents) \
        else np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
    moved.rotate_network(axis, 0.7)
    moved.scale_network(2.7)

    assert np.allclose(np.array(estimation_error_of(moved, rank_K)), base, rtol=1e-6,
                       atol=1e-8)


@pytest.mark.parametrize("domain", ["R^2", "R^3"])
def test_Rd_has_no_attitude_error_to_predict(domain):
    """R^d has no attitude DOFs, so a_att is 0 and a_pos is the whole trace."""
    net, rank_K = rigid_net(domain, n=6)
    a_pos, a_att = estimation_error_blocks(scaled_rigidity_matrix(net), rank_K, net.n)
    a_opt, _, _ = estimation_error_of(net, rank_K)
    assert a_att < 1e-12
    assert np.isclose(a_pos, a_opt, rtol=1e-6)


def test_estimation_error_is_infinite_on_a_flexible_framework():
    net, _ = random_scenario(6, "R^3", edge_count=3)
    rank_K = int(np.linalg.matrix_rank(B_of(net.fully_connected())))
    assert not np.isfinite(estimation_error_of(net, rank_K)[0])
    assert not np.isfinite(estimation_error(np.zeros(0), rank_K)[0])


# ---------------------------------------------------------------- the two halves agree
@pytest.mark.parametrize("domain", ["R^3", "R^2xS^1", "SE(3)", MIXES[3]])
def test_measured_error_matches_the_cramer_rao_prediction(domain):
    """Measured RMS error matches the predicted one at small sigma."""
    net, rank_K = rigid_net(domain, n=8)
    pred_pos, pred_att = E.predicted_error(net, rank_K)

    for sigma in (1e-4, 1e-3):
        got = E.monte_carlo_error(net, sigma, trials=120,
                                  rng=np.random.default_rng(11), rank_K=rank_K)
        assert 0.9 < got["position"]["rms"] / (sigma * pred_pos) < 1.15
        if pred_att > 0:
            assert 0.9 < got["attitude"]["rms"] / (sigma * pred_att) < 1.15


def test_measured_error_is_linear_in_sigma():
    """Measured error is linear in sigma."""
    net, rank_K = rigid_net("R^3", n=8)
    small = E.monte_carlo_error(net, 1e-4, trials=120, rng=np.random.default_rng(12),
                                rank_K=rank_K)["position"]["rms"]
    big = E.monte_carlo_error(net, 1e-3, trials=120, rng=np.random.default_rng(12),
                              rank_K=rank_K)["position"]["rms"]
    assert 9.0 < big / small < 11.0


def test_a_collapsed_estimate_is_not_scored_as_a_perfect_one():
    """A collapsed or blown-up estimate scores infinite error, not zero."""
    net, rank_K = rigid_net("R^3", n=6)
    collapsed = copy.deepcopy(net)
    for agent in collapsed.agents:
        agent.pose.position = net.agents[0].pose.position.copy()
    assert not np.isfinite(E.shape_error(net, collapsed, rank_K)[0])

    blown_up = copy.deepcopy(net)
    for agent in blown_up.agents:
        agent.pose.position = agent.pose.position * 50.0
    assert not np.isfinite(E.shape_error(net, blown_up, rank_K)[0])


def test_a_stiffer_graph_estimates_better():
    """A denser graph measures a smaller shape error."""
    net, rank_K = rigid_net("R^3", n=8)
    dense = copy.deepcopy(net)
    dense.edges = ~np.eye(net.n, dtype=bool)

    kw = dict(trials=60, rng=np.random.default_rng(13), rank_K=rank_K)
    sparse_err = E.monte_carlo_error(net, 1e-3, **kw)["position"]["rms"]
    dense_err = E.monte_carlo_error(dense, 1e-3, **kw)["position"]["rms"]
    assert dense_err < sparse_err
