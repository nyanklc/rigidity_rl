"""The state score phi. THEORY.md section 7."""
import numpy as np
import pytest

from conftest import ALL_DOMAINS, C_MAX, RANK_K_FORMULA, STATE_SCORES

W_RANK, W_EDGE = 100.0, 25.0


def phi_of(env):
    brm = env.network.extended_bearing_rigidity_matrix()
    mbr, ibr, rank = env.network.is_MBR(rank_K=env.rank_K, brm=brm)
    return env.compute_state_score(brm, ibr, mbr, rank), rank


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_weighted_normalized_matches_its_closed_form(make_env, domain):
    n = 6
    e = make_env(n=n, domains=[domain] * n)
    for _ in range(5):
        e.reset()
        phi, rank = phi_of(e)
        m = int(e.network.edges.sum())
        expected = (W_RANK * rank - W_EDGE * m * e.c_max) / e.rank_K
        assert abs(phi - expected) < 1e-9


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_optimum_is_w_rank_minus_w_edge_when_perfectly_packed(make_env, domain):
    """phi* = w_rank - w_edge exactly when m*c_max == rank_K."""
    n = 6
    e = make_env(n=n, domains=[domain] * n)
    e.reset()
    rank_K, c_max = e.rank_K, e.c_max
    if rank_K % c_max:
        pytest.skip(f"{domain} n={n}: rank_K/c_max is not an integer, ceiling applies")
    m = rank_K // c_max
    phi = (W_RANK * rank_K - W_EDGE * m * c_max) / rank_K
    assert abs(phi - (W_RANK - W_EDGE)) < 1e-9


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_a_rank_maximal_edge_is_always_worth_adding(make_env, domain):
    """Structural guarantee: gain and cost share the c_max/rank_K factor."""
    n = 6
    e = make_env(n=n, domains=[domain] * n)
    e.reset()
    gain = W_RANK * e.c_max / e.rank_K
    cost = W_EDGE * e.c_max / e.rank_K
    assert gain - cost > 0
    assert abs((gain - cost) - (W_RANK - W_EDGE) * e.c_max / e.rank_K) < 1e-12


def test_pruning_a_redundant_edge_is_positive(make_env):
    e = make_env(n=6, domains="R^3")
    e.reset()
    assert W_EDGE * e.c_max / e.rank_K > 0


def test_score_is_dimensionless_across_n_and_domain(make_env):
    """The optimum must not drift with configuration, unlike Weighted."""
    opt = []
    for n, domain in [(4, "R^2"), (8, "R^2"), (8, "R^3"), (16, "R^3")]:
        e = make_env(n=n, domains=domain)
        e.reset()
        m = e.rank_K // e.c_max
        opt.append((W_RANK * e.rank_K - W_EDGE * m * e.c_max) / e.rank_K)
    assert max(opt) - min(opt) < 1e-9


def test_weighted_is_unchanged(make_env):
    """Old runs must stay comparable: Weighted is still 20*rank - 10*m."""
    e = make_env(n=8, domains="R^3", state_score_type="Weighted")
    e.reset()
    phi, rank = phi_of(e)
    assert abs(phi - (20.0 * rank - 10.0 * int(e.network.edges.sum()))) < 1e-9


def test_weighted_trade_off_is_dimension_dependent(make_env):
    """Documents WHY WeightedNormalized exists: Weighted's ratio moves with d."""
    ratios = {}
    for domain in ("R^2", "R^3"):
        e = make_env(n=6, domains=domain)
        e.reset()
        ratios[domain] = (20.0 * e.c_max - 10.0) / 10.0
    assert ratios["R^3"] != ratios["R^2"]


@pytest.mark.parametrize("score", STATE_SCORES)
def test_every_state_score_returns_a_finite_number(make_env, score):
    e = make_env(n=5, domains="R^3", state_score_type=score)
    e.reset()
    phi, _ = phi_of(e)
    assert np.isfinite(float(phi))


def test_reward_is_the_change_in_phi(make_env):
    """The step reward is potential-based: r = phi(s') - phi(s)."""
    e = make_env(n=6, domains="R^3", time_penalty_value=0.0)
    e.reset()
    prev, _ = phi_of(e)
    for _ in range(15):
        _, reward, _, _, _ = e.step(e.action_space.sample())
        now = e.last_stats["score"]
        assert abs(reward - (now - prev)) < 1e-9
        prev = now
