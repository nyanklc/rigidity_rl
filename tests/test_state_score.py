"""The state score phi."""
import copy
import numpy as np
import pytest

from conftest import ALL_DOMAINS, C_MAX, RANK_K_FORMULA, STATE_SCORES
from rigidity import rigidity_decomposition

W_RANK, W_EDGE = 100.0, 25.0


def phi_of(env):
    brm = env.network.extended_bearing_rigidity_matrix()
    mbr, ibr, rank = env.network.is_MBR(rank_K=env.rank_K, brm=brm)
    lam = rigidity_decomposition(brm, env.rank_K)[2]
    return env.compute_state_score(brm, ibr, mbr, rank, lam=lam), rank


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


# The rigidity margin term.
KAPPA = 0.9


def rigid_env(make_env, domain, n=6, kappa=KAPPA, **kw):
    """An env whose graph is complete, so the margin term is actually live."""
    e = make_env(n=n, domains=[domain] * n, stiffness_kappa=kappa, **kw)
    e.reset()
    e.network.edges = e.network.fully_connected().edges
    np.fill_diagonal(e.network.edges, False)
    e.stiffness_rng = np.random.default_rng(0)
    e.compute_episode_constants()
    return e


def one_edge(e):
    return W_EDGE * e.c_max / e.rank_K


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_kappa_zero_reproduces_the_rank_only_score(make_env, domain):
    """kappa = 0 must leave phi byte-identical, so old runs stay comparable."""
    n = 6
    for _ in range(3):
        e = rigid_env(make_env, domain, n=n, kappa=0.0)
        phi, rank = phi_of(e)
        m = int(e.network.edges.sum())
        assert abs(phi - (W_RANK * rank - W_EDGE * m * e.c_max) / e.rank_K) < 1e-9


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_margin_term_is_bounded_by_kappa_edges(make_env, domain):
    """The whole margin range is worth kappa edges -- that is what denominates kappa."""
    e = rigid_env(make_env, domain)
    phi, rank = phi_of(e)
    base = (W_RANK * rank - W_EDGE * int(e.network.edges.sum()) * e.c_max) / e.rank_K
    assert 0.0 <= phi - base <= KAPPA * one_edge(e) + 1e-12


def test_q_is_one_half_when_lambda_equals_stiffness_ref(make_env):
    e = rigid_env(make_env, "R^3")
    e.stiffness_ref = 1.0
    brm = e.network.extended_bearing_rigidity_matrix()
    base = (W_RANK * e.rank_K - W_EDGE * int(e.network.edges.sum()) * e.c_max) / e.rank_K
    got = e.compute_state_score(brm, True, False, e.rank_K, lam=1.0)
    assert abs((got - base) / (KAPPA * one_edge(e)) - 0.5) < 1e-12


def test_margin_is_gated_on_rigidity(make_env):
    """A flexible graph must never be charged or credited for margin."""
    e = rigid_env(make_env, "R^3")
    brm = e.network.extended_bearing_rigidity_matrix()
    base = (W_RANK * 3 - W_EDGE * int(e.network.edges.sum()) * e.c_max) / e.rank_K
    assert abs(e.compute_state_score(brm, False, False, 3, lam=1e9) - base) < 1e-12


def test_more_margin_scores_higher_at_the_same_edge_count(make_env):
    """The point of the term: among equally sparse graphs, prefer the stiffer one."""
    e = rigid_env(make_env, "R^3")
    brm = e.network.extended_bearing_rigidity_matrix()
    args = dict(is_IBR=True, is_MBR=False, rank_brm=e.rank_K)
    lo = e.compute_state_score(brm, lam=e.stiffness_ref / 10.0, **args)
    hi = e.compute_state_score(brm, lam=e.stiffness_ref * 10.0, **args)
    assert hi > lo


def _transformed_phi(e, kind, planar):
    if kind == "translate":
        e.network.translate_network([3.1, -2.4, 0.0 if planar else 1.7])
    elif kind == "rotate":
        e.network.rotate_network([0, 0, 1] if planar else [0.3, 0.5, 0.81], 0.9)
    else:
        e.network.scale_network(2.7)
    e.stiffness_rng = np.random.default_rng(0)      # same construction order, or stiffness_ref moves
    e.compute_episode_constants()
    return phi_of(e)[0]


@pytest.mark.parametrize("domain", ALL_DOMAINS)
@pytest.mark.parametrize("kind", ["translate", "rotate"])
def test_margin_phi_is_exactly_invariant_to_translation_and_rotation(make_env, domain, kind):
    e = rigid_env(make_env, domain)
    before = phi_of(e)[0]
    assert abs(_transformed_phi(e, kind, domain in ("R^2", "R^2xS^1")) - before) < 1e-9


@pytest.mark.parametrize("domain", ["R^2", "R^3"])
def test_margin_phi_is_exactly_scale_invariant_without_attitude_columns(make_env, domain):
    """In R^d every column of B carries 1/length, so a rescale cancels in lambda/stiffness_ref."""
    e = rigid_env(make_env, domain)
    before = phi_of(e)[0]
    assert abs(_transformed_phi(e, "scale", domain == "R^2") - before) < 1e-9


@pytest.mark.parametrize("domain", ["R^2xS^1", "R^3xS^1", "SE(3)"])
def test_margin_phi_is_only_approximately_scale_invariant_with_attitude(make_env, domain):
    """A rescale reweights B's position columns against its attitude columns, so
    lambda/stiffness_ref moves -- by at most ~7% of one edge.4
    """
    e = rigid_env(make_env, domain)
    before = phi_of(e)[0]
    drift = abs(_transformed_phi(e, "scale", domain == "R^2xS^1") - before)
    assert 0 < drift < 0.07 * KAPPA * one_edge(e)


def test_stiffness_ref_is_reproducible_from_the_seed(make_env):
    a = rigid_env(make_env, "R^3")
    b = make_env(n=6, domains=["R^3"] * 6, stiffness_kappa=KAPPA)
    b.network = a.network
    b.stiffness_rng = np.random.default_rng(0)
    b.compute_episode_constants()
    assert a.stiffness_ref == b.stiffness_ref > 0


def test_enabling_stiffness_does_not_move_the_instance_stream(make_env):
    """stiffness_ref's construction must draw from a private rng, not the global stream
    instances come from.
    """
    def edges_after_two_resets(kappa):
        np.random.seed(7)
        e = make_env(n=6, domains=["R^3"] * 6, stiffness_kappa=kappa)
        e.reset()
        e.reset()
        return e.network.edges.copy(), np.array(
            [a.pose.position for a in e.network.agents], dtype=float)

    e0, p0 = edges_after_two_resets(0.0)
    e9, p9 = edges_after_two_resets(KAPPA)
    assert np.array_equal(e0, e9)
    assert np.allclose(p0, p9)


def test_stiffness_ref_is_the_same_for_the_same_poses(make_env):
    """phi has to be a function of the state. The reference construction draws from
    a private rng, so it is reseeded per episode; without that every restore of one
    instance scores under a different phi."""
    e = make_env(n=6, domains=["R^3"] * 6, stiffness_kappa=KAPPA)
    e.reset()
    net = copy.deepcopy(e.network)
    e.freeze_network = True
    refs = []
    for _ in range(4):
        e.network = copy.deepcopy(net)
        e.reset()
        refs.append(e.stiffness_ref)
    assert len(set(refs)) == 1 and refs[0] > 0


# The selectable spectral functional.
SPECTRAL = ["eigenvalue", "trace", "logdet"]


def spectral_env(make_env, functional, domain="R^3", n=6, kappa=2.0):
    e = make_env(n=n, domains=[domain] * n,
                 state_score_type="WeightedNormalizedSpectral",
                 spectral_functional=functional, stiffness_kappa=kappa)
    e.reset()
    return e


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_eigenvalue_mode_is_the_weighted_normalized_score(make_env, domain):
    """The new branch must reproduce the old one exactly at functional=eigenvalue."""
    n = 6
    for seed in range(3):
        old = make_env(n=n, domains=[domain] * n, stiffness_kappa=2.0)
        new = make_env(n=n, domains=[domain] * n,
                       state_score_type="WeightedNormalizedSpectral",
                       spectral_functional="eigenvalue", stiffness_kappa=2.0)
        np.random.seed(seed)
        old.reset()
        edges = old.network.edges.copy()

        new.network = copy.deepcopy(old.network)
        new.compute_episode_constants()
        old.network.edges = edges.copy()
        new.network.edges = edges.copy()

        assert phi_of(old)[0] == phi_of(new)[0]


@pytest.mark.parametrize("functional", SPECTRAL)
def test_kappa_zero_collapses_every_functional_to_the_same_score(make_env, functional):
    e = spectral_env(make_env, functional, kappa=0.0)
    phi, rank = phi_of(e)
    m = int(e.network.edges.sum())
    assert abs(phi - (W_RANK * rank - W_EDGE * m * e.c_max) / e.rank_K) < 1e-9


@pytest.mark.parametrize("functional", SPECTRAL)
def test_the_spectral_bonus_is_bounded_by_kappa_edges(make_env, functional):
    """0 < bonus < kappa * one_edge on a rigid graph, for every functional."""
    kappa = 2.0
    e = spectral_env(make_env, functional, kappa=kappa)
    e.network.edges = e.network.fully_connected().edges
    np.fill_diagonal(e.network.edges, False)
    e.compute_episode_constants()

    phi, rank = phi_of(e)
    m = int(e.network.edges.sum())
    bonus = phi - (W_RANK * rank - W_EDGE * m * e.c_max) / e.rank_K
    assert 0.0 < bonus < kappa * one_edge(e)


def test_an_unknown_functional_is_refused(make_env):
    with pytest.raises(ValueError):
        make_env(n=5, domains="R^3", state_score_type="WeightedNormalizedSpectral",
                 spectral_functional="nonsense")
