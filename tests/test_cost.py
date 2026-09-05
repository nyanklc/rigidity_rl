"""What the counters count, and that counting changes nothing."""
import inspect

import numpy as np
import pytest

import cost
import outputs as E
import rigidity as R


@pytest.fixture(autouse=True)
def clear_counts():
    cost.COUNTS.clear()
    yield
    cost.COUNTS.clear()


def test_counted_returns_exactly_what_it_wrapped():
    def f(a, b, *, c=3):
        return a + b + c
    g = cost.counted(f)
    assert g(1, 2, c=10) == f(1, 2, c=10) == 13


def test_counted_keeps_the_name_the_docstring_and_the_source():
    """manifest.py archives sources and agent_loader replays them by name."""
    def f():
        """what f does."""
    g = cost.counted(f)
    assert g.__name__ == "f" and g.__doc__ == "what f does."
    assert "def f()" in inspect.getsource(g)


def test_one_call_is_one_tally():
    @cost.counted
    def widget():
        pass
    widget()
    widget()
    assert cost.COUNTS["widget"] == 2


def test_the_meter_reports_deltas_not_absolutes():
    cost.tally("thing", 5)
    with cost.Meter() as m:
        cost.tally("thing", 2)
    assert m.counts == {"thing": 2}
    assert cost.COUNTS["thing"] == 7


def test_reporting_diverts_to_its_own_bucket_and_restores(make_env):
    env = make_env(n=4, domains="R^2")
    env.reset()
    with cost.Meter() as m:
        with cost.reporting():
            env.network.extended_bearing_rigidity_matrix()
        env.network.extended_bearing_rigidity_matrix()
    assert m.counts["extended_bearing_rigidity_matrix"] == 1
    assert m.counts["_reporting"] == 1


def test_reporting_nests(make_env):
    with cost.Meter() as m:
        with cost.reporting():
            with cost.reporting():
                cost.tally("inner")
            cost.tally("outer")
        cost.tally("free")
    assert m.counts == {"_reporting": 2, "free": 1}


def test_the_leaf_total_does_not_double_count_a_nested_call(make_env):
    """is_MBR calls the primitives that do the work; only those are leaves."""
    env = make_env(n=5, domains="R^3")
    env.reset()
    brm = env.network.extended_bearing_rigidity_matrix()
    with cost.Meter() as m:
        env.network.is_MBR(rank_K=env.rank_K, brm=brm)
    assert m.counts["is_MBR"] == 1
    assert "is_MBR" not in cost.LEAVES
    # the total sees what is_MBR called, and never is_MBR itself
    assert m.total() == sum(v for k, v in m.counts.items() if k in cost.LEAVES)
    assert m.total() < sum(m.counts.values())


def test_every_leaf_really_is_one(make_env):
    """A leaf that calls another counted primitive would double count in the total.

    Checked by calling each one and seeing what else it tallies, rather than by
    trusting the list -- which is how `eigenvalues` got in there, since it builds B.
    """
    e = make_env(n=6, domains="R^3")
    e.reset()
    net = e.network
    B = net.extended_bearing_rigidity_matrix()
    rank, _, lam = R.rigidity_decomposition(B, e.rank_K)
    L = R.characteristic_length(net)
    Z, _, w, V = R.nullspace_and_softest(B, int(rank))
    Zs = R.nullspace_in_scaled_units(Z, net.n, L)
    B_K = R.extended_bearing_rigidity_matrix(net.fully_connected())
    Z_K = R.nullspace_in_scaled_units(R.nullspace(B_K, e.rank_K), net.n, L)

    calls = {
        "extended_bearing_rigidity_matrix": lambda: R.extended_bearing_rigidity_matrix(net),
        "rigidity_decomposition": lambda: R.rigidity_decomposition(B, e.rank_K),
        "nullspace": lambda: R.nullspace(B, int(rank)),
        "nullspace_and_softest": lambda: R.nullspace_and_softest(B, int(rank)),
        "error_covariance": lambda: R.error_covariance(B, e.rank_K),
        "estimation_error_blocks": lambda: R.estimation_error_blocks(B, e.rank_K, net.n),
        "removal_costs": lambda: R.removal_costs(B, net, int(e.rank_K), lam=lam, w=w, V=V,
                                                 c_max=e.c_max),
        "candidate_gain": lambda: R.candidate_gain(net, Zs, length_scale=L),
        "edge_block_ranks": lambda: R.edge_block_ranks(B),
        "flex_space": lambda: R.flex_space(Zs, Z_K),
        "nullspace_in_scaled_units": lambda: R.nullspace_in_scaled_units(Z, net.n, L),
        "is_IBR_explicit": lambda: R.is_IBR_explicit(B, e.rank_K),
    }
    assert set(calls) == set(cost.LEAVES)
    for name, call in calls.items():
        with cost.Meter() as m:
            call()
        assert m.counts == {name: 1}, (name, m.counts)


def test_a_composite_is_not_in_the_leaf_list(make_env):
    """Network.eigenvalues builds B, so counting both would count that build twice."""
    e = make_env(n=5, domains="R^3")
    e.reset()
    with cost.Meter() as m:
        e.network.eigenvalues()
    assert m.counts["eigenvalues"] == 1
    assert m.counts["extended_bearing_rigidity_matrix"] == 1
    assert "eigenvalues" not in cost.LEAVES
    assert m.total() == 1


def test_every_leaf_and_column_has_a_stated_operation():
    """The report legend is what stops a bare count being read as work."""
    import report
    for name in cost.LEAVES:
        assert name in cost.OPERATION, name
    for key, _ in report.COST_COLUMNS:
        assert key in cost.OPERATION, key


def test_a_greedy_run_is_counted_where_it_actually_spends(make_env):
    """n(n-1) rescorings per improvement step, each a build and a decomposition."""
    env = make_env(n=5, domains="R^3", max_steps=60)
    env.reset()
    with cost.Meter() as m:
        res = E.run_greedy(env, max_steps=60, verbose=False)
    n = env.network.n
    phi = m.counts["score_network"]
    assert phi == n * (n - 1) * (res["work"] + 1) + 1
    assert m.counts["extended_bearing_rigidity_matrix"] == phi
    assert m.ms > 0
