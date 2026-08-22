"""Reference values every change is measured against, and report.py's maths."""
import copy

import numpy as np
import pytest

import report
from baselines import run_greedy, score_network, run_random


def test_greedy_reaches_the_exact_optimum_at_n4_R2(make_env):
    """phi* = w_rank - w_edge = 75.00 with 5 edges."""
    e = make_env(n=4, domains="R^2", max_steps=60,
                 termination_condition_type="MaxSteps")
    scores, edges = [], []
    for _ in range(8):
        e.reset()
        res = run_greedy(e, max_steps=60, verbose=False)
        scores.append(res["score"])
        edges.append(res["m"])
    assert all(abs(s - 75.0) < 1e-9 for s in scores), scores
    assert all(m == 5 for m in edges), edges


def test_greedy_is_rigid_and_minimal_at_n4_R2(make_env):
    e = make_env(n=4, domains="R^2", max_steps=60,
                 termination_condition_type="MaxSteps")
    for _ in range(5):
        e.reset()
        res = run_greedy(e, max_steps=60, verbose=False)
        assert res["is_IBR"] and res["is_MBR"]


def test_greedy_is_at_least_as_good_as_random_and_far_cheaper(make_env):
    """Both scored on best-state-visited from the SAME graph, so random can tie on
    small graphs -- the real separation is how many edits it takes to get there.

    The restore between methods is the point: without it run_random continues from
    the graph run_greedy just optimised and can only improve on it, which makes the
    comparison meaningless (and eventually fails, when random finds the two-edit
    swap greedy cannot). baselines.py deep-copies for the same reason.
    """
    e = make_env(n=6, domains="R^3", max_steps=80,
                 termination_condition_type="MaxSteps")
    g, r, gw, rw = [], [], [], []
    for _ in range(5):
        e.reset()
        instance = copy.deepcopy(e.network)
        e.freeze_network = True

        res_g = run_greedy(e, max_steps=80, verbose=False)

        e.network = copy.deepcopy(instance)
        e.reset()
        res_r = run_random(e, steps=80)

        e.freeze_network = False
        g.append(res_g["score"]); r.append(res_r["score"])
        gw.append(res_g["work"]); rw.append(res_r["work"])
    assert np.mean(g) >= np.mean(r) - 1e-9
    assert np.mean(gw) < np.mean(rw)


def test_score_network_agrees_with_the_environment(make_env):
    e = make_env(n=6, domains="R^3")
    e.reset()
    score, ibr, mbr, rank, m = score_network(e, need_mbr=True)
    assert m == int(e.network.edges.sum())
    brm = e.network.extended_bearing_rigidity_matrix()
    mbr2, ibr2, rank2 = e.network.is_MBR(rank_K=e.rank_K, brm=brm)
    assert (ibr, mbr, rank) == (ibr2, mbr2, rank2)


def test_gmean_and_gsd_on_known_values():
    assert report._gmean([1.0, 10.0, 100.0]) == pytest.approx(10.0)
    assert report._gmean([4.0, 4.0]) == pytest.approx(4.0)
    assert report._gsd([4.0, 4.0]) == pytest.approx(1.0, abs=1e-9)
    assert report._gmean([]) is None


def test_aggregate_filters_zero_margins_out_of_the_geometric_mean():
    """A non-rigid network has margin exactly 0, which no geometric mean can take.
    aggregate() is what filters; _gmean itself assumes positive input."""
    rows = [dict(episode=i, method="learned", m=10.0, score=70.0, is_IBR=True,
                 is_MBR=False, min_eig=v, work=1, best_at=1)
            for i, v in enumerate([0.0, 1.0, 100.0])]
    agg = report.aggregate(rows)["learned"]
    assert agg["min_eig_n"] == 3            # all three seen by the arithmetic mean
    assert agg["min_eig_n_pos"] == 2        # only two enter the geometric one
    assert agg["min_eig_gmean"] == pytest.approx(10.0)


def test_aggregate_means_match_a_hand_computation():
    rows = [
        {"episode": 0, "method": "learned", "m": 10.0, "score": 70.0, "is_IBR": True,
         "is_MBR": False, "min_eig": 1e-2, "work": 5, "best_at": 3},
        {"episode": 1, "method": "learned", "m": 12.0, "score": 60.0, "is_IBR": True,
         "is_MBR": True, "min_eig": 1e-4, "work": 7, "best_at": 5},
    ]
    row = report.aggregate(rows)["learned"]
    assert row["episodes"] == 2
    assert row["edges_mean"] == pytest.approx(11.0)
    assert row["score_mean"] == pytest.approx(65.0)
    assert row["rigid_pct"] == pytest.approx(100.0)
    assert row["minimal_pct"] == pytest.approx(50.0)
    assert row["work_mean"] == pytest.approx(6.0)
    assert row["min_eig_gmean"] == pytest.approx(1e-3)


@pytest.mark.slow
def test_brute_force_optimum_matches_greedy_at_n4_R2(make_env):
    """Exhaustive search must not beat greedy's 75.00 / 5 edges."""
    import itertools
    e = make_env(n=4, domains="R^2")
    e.reset()
    n = 4
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    best = -np.inf
    for k in range(1, 8):
        for sub in itertools.combinations(pairs, k):
            E = np.zeros((n, n), dtype=bool)
            for i, j in sub:
                E[i, j] = True
            e.network.edges = E
            best = max(best, score_network(e)[0])
    assert abs(best - 75.0) < 1e-9
