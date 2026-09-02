"""Reference values every change is measured against, and report.py's maths."""
import copy

import numpy as np
import pytest

import cost
import report
from evaluation import (_is_best, _percentile_of, decision_record, edit_landscape,
                        measure_noise, phi_landscape, run_anneal, run_degree, run_greedy,
                        run_initial, run_spectral, score_network, run_random)


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
    swap greedy cannot). evaluation.py deep-copies for the same reason.
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


DOMAINS = ["R^2", "R^3", "R^2xS^1", "R^3xS^1", "SE(3)"]


@pytest.mark.parametrize("domain", DOMAINS)
def test_spectral_is_greedy_at_kappa_zero(make_env, domain):
    """phi is affine in rank there, so the closed-form landscape IS greedy's landscape.

    The tie break matters: an addition and a redundant removal are an exact tie in the
    c_max = 2 domains, and greedy keeps the first in row-major order.
    """
    e = make_env(n=6, domains=domain, max_steps=80,
                 termination_condition_type="MaxSteps")
    for _ in range(3):
        e.reset()
        instance = copy.deepcopy(e.network)
        e.freeze_network = True

        g = run_greedy(e, max_steps=80, verbose=False)
        e.network = copy.deepcopy(instance)
        e.reset()
        s = run_spectral(e, max_steps=80, verbose=False)
        e.freeze_network = False

        assert abs(g["score"] - s["score"]) < 1e-9, (domain, g["score"], s["score"])
        assert g["m"] == s["m"] and g["work"] == s["work"]


def test_spectral_costs_less_than_greedy_for_the_same_answer(make_env):
    e = make_env(n=8, domains="R^3", max_steps=80,
                 termination_condition_type="MaxSteps")
    e.reset()
    instance = copy.deepcopy(e.network)
    e.freeze_network = True
    with cost.Meter() as mg:
        run_greedy(e, max_steps=80, verbose=False)
    e.network = copy.deepcopy(instance)
    e.reset()
    with cost.Meter() as ms:
        run_spectral(e, max_steps=80, verbose=False)
    e.freeze_network = False
    assert ms.total() < mg.total()
    assert ms.counts["score_network"] < mg.counts["score_network"]


def test_the_closed_form_landscape_refuses_a_score_it_cannot_express(make_env):
    e = make_env(n=5, domains="R^3", state_score_type="Weighted")
    e.reset()
    with pytest.raises(ValueError, match="closed-form"):
        phi_landscape(e)


@pytest.mark.parametrize("runner", ["degree", "anneal", "spectral"])
def test_the_new_baselines_end_rigid_and_report_their_graph(make_env, runner):
    """A row without `edges` silently vanishes from the formation figures."""
    e = make_env(n=6, domains="R^3", max_steps=80,
                 termination_condition_type="MaxSteps")
    rng = np.random.default_rng(0)
    for _ in range(3):
        e.reset()
        if runner == "degree":
            res = run_degree(e, rng, verbose=False)
        elif runner == "anneal":
            res = run_anneal(e, rng, budget=300, verbose=False)
        else:
            res = run_spectral(e, max_steps=80, verbose=False)
        assert res["is_IBR"], res
        assert res["edges"] is not None and res["edges"].sum() == res["m"]


def test_anneal_spends_exactly_its_budget_and_repeats_under_a_fixed_rng(make_env):
    e = make_env(n=6, domains="R^3", max_steps=80,
                 termination_condition_type="MaxSteps")
    e.reset()
    instance = copy.deepcopy(e.network)
    e.freeze_network = True
    with cost.Meter() as m:
        a = run_anneal(e, np.random.default_rng(7), budget=200, verbose=False)
    e.network = copy.deepcopy(instance)
    e.reset()
    b = run_anneal(e, np.random.default_rng(7), budget=200, verbose=False)
    e.freeze_network = False
    # one baseline score, then one per proposal
    assert m.counts["score_network"] == 201
    assert a["score"] == b["score"] and a["m"] == b["m"]


def test_degree_reads_nothing_but_degrees_and_the_rigidity_test(make_env):
    """No marginal ranks, no spectrum: that is what makes it the tier-1 reference."""
    e = make_env(n=6, domains="R^3", max_steps=80,
                 termination_condition_type="MaxSteps")
    e.reset()
    with cost.Meter() as m:
        run_degree(e, np.random.default_rng(0), verbose=False)
    for forbidden in ("candidate_gain", "removal_costs", "nullspace_and_softest",
                      "score_network"):
        assert m.counts.get(forbidden, 0) == 0, forbidden


def test_score_network_agrees_with_the_environment(make_env):
    e = make_env(n=6, domains="R^3")
    e.reset()
    score, ibr, mbr, rank, m = score_network(e, need_mbr=True)
    assert m == int(e.network.edges.sum())
    brm = e.network.extended_bearing_rigidity_matrix()
    mbr2, ibr2, rank2 = e.network.is_MBR(rank_K=e.rank_K, brm=brm)
    assert (ibr, mbr, rank) == (ibr2, mbr2, rank2)


def _panel_row(method, episode=0, is_IBR=True):
    return {"episode": episode, "method": method, "is_IBR": is_IBR,
            "edges": np.zeros((3, 3), dtype=bool), "m": 5}


def test_the_formation_figures_never_drop_the_policy():
    """The panel cap picks by FORMATION_PRIORITY, not by the table's order.

    `learned` sits second to last in METHOD_ORDER, so a cap applied in display order
    drops the one method the figures exist to show.
    """
    rows = [_panel_row(m) for m in report.METHOD_ORDER]
    drawn = [r["method"] for r in report._panel_rows(rows)]
    assert "learned" in drawn, drawn


def test_the_panel_cap_keeps_the_policy_even_when_it_bites():
    rows = [_panel_row(m) for m in report.METHOD_ORDER]
    assert len(rows) > 3
    saved = report.MAX_FORMATION_PANELS
    try:
        report.MAX_FORMATION_PANELS = 3
        drawn = [r["method"] for r in report._panel_rows(rows)]
        assert len(drawn) == 3
        assert "learned" in drawn, drawn
        # and the reader is told what went missing
        note = report._panel_cap_note(rows)
        assert note and "not drawn here" in note
    finally:
        report.MAX_FORMATION_PANELS = saved


def test_panels_are_drawn_in_the_table_order_they_were_not_chosen_in():
    """Selection is by priority; display stays METHOD_ORDER so the panels read as the table."""
    rows = [_panel_row(m) for m in ("learned", "greedy", "random")]
    drawn = [r["method"] for r in report._panel_rows(rows)]
    assert drawn == ["random", "greedy", "learned"]


def test_a_flexible_network_has_no_panel():
    """No rigidity, no error ellipsoid and no softest mode -- there is nothing to draw."""
    rows = [_panel_row("greedy"), _panel_row("initial", is_IBR=False)]
    assert [r["method"] for r in report._panel_rows(rows)] == ["greedy"]


def test_the_panel_grid_leaves_at_most_one_empty_cell_past_four():
    for count in range(5, 10):
        nrows, ncols = report._grid_for(count)
        assert nrows * ncols >= count
        assert nrows * ncols - count <= 1, (count, nrows, ncols)


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


# ---------------------------------------------------------------- the noise sweep
SIGMAS = [0.002, 0.01]


def test_measure_noise_fills_every_sigma_on_a_rigid_row(make_env):
    e = make_env(n=5, domains="R^3", max_steps=40)
    e.reset()
    row = run_greedy(e, max_steps=40, verbose=False)
    assert row["is_IBR"]

    measure_noise(e, row, SIGMAS, trials=8, rng=np.random.default_rng(0))
    assert sorted(row["noise"]) == SIGMAS
    assert all(v > 0 for v in row["noise"].values())
    assert row["pred_err"] > 0


def test_measure_noise_scales_with_sigma(make_env):
    e = make_env(n=5, domains="R^3", max_steps=40)
    e.reset()
    row = run_greedy(e, max_steps=40, verbose=False)
    measure_noise(e, row, [1e-4, 1e-3], trials=40, rng=np.random.default_rng(1))
    ratio = row["noise"][1e-3] / row["noise"][1e-4]
    assert 8.0 < ratio < 12.0


def test_measure_noise_skips_a_flexible_row(make_env):
    """A non-rigid graph has infinite error, so it carries no sweep at all."""
    e = make_env(n=6, domains="R^3", max_steps=40)
    e.reset()
    e.network.edges[:] = False
    e.network.edges[0, 1] = True
    row = run_initial(e)
    assert not row["is_IBR"]

    measure_noise(e, row, SIGMAS, trials=4, rng=np.random.default_rng(0))
    assert "noise" not in row


def test_the_sweep_reaches_the_table(make_env):
    e = make_env(n=5, domains="R^3", max_steps=40)
    rows = []
    for ep in range(2):
        e.reset()
        row = run_greedy(e, max_steps=40, verbose=False)
        measure_noise(e, row, SIGMAS, trials=6, rng=np.random.default_rng(ep))
        row["episode"] = ep
        rows.append(row)

    agg = report.aggregate(rows)
    assert sorted(agg["greedy"]["noise"]) == SIGMAS
    text = report.format_table(rows, {"environment": "t"}, brief=True)
    assert "MEASURED SHAPE ERROR UNDER BEARING NOISE" in text


def test_rows_without_a_sweep_leave_the_table_unchanged(make_env):
    e = make_env(n=5, domains="R^3", max_steps=40)
    e.reset()
    row = run_greedy(e, max_steps=40, verbose=False)
    row["episode"] = 0
    text = report.format_table([row], {"environment": "t"}, brief=True)
    assert "MEASURED SHAPE ERROR" not in text


def noisy_row(method, episode, noise, pred_err=8.0, failed=None):
    return dict(episode=episode, method=method, m=10, score=75.0, is_IBR=True,
                is_MBR=True, min_eig=1e-3, shape_err=8.0, work=5, best_at=5,
                noise=noise, noise_failed=failed or {}, pred_err=pred_err)


def test_a_blown_up_recovery_never_reaches_the_aggregate():
    """A trial whose estimate collapsed has infinite error; averaging it poisons
    every statistic and crashed the figures."""
    rows = [noisy_row("greedy", ep, {0.01: 0.08, 0.2: np.inf}) for ep in range(3)]
    agg = report.aggregate(rows)["greedy"]["noise"]
    assert 0.01 in agg and 0.2 not in agg
    assert np.isfinite(agg[0.01][0])


def test_a_non_finite_prediction_is_dropped_too():
    rows = [noisy_row("greedy", ep, {0.01: 0.08}, pred_err=np.inf) for ep in range(3)]
    assert report.aggregate(rows)["greedy"]["noise"] == {}


def test_the_figures_survive_non_finite_entries(tmp_path):
    rows = ([noisy_row("greedy", ep, {0.01: 0.08, 0.2: np.inf}) for ep in range(3)]
            + [noisy_row("constructive", ep, {0.01: np.nan}) for ep in range(3)])
    # neither may raise, and neither may draw a non-finite axis limit
    report.plot_prediction_check(str(tmp_path), rows, {"environment": "t"})
    report.plot_noise_sweep(str(tmp_path), rows, {"environment": "t"})


def test_the_table_marks_a_noise_level_where_recoveries_failed():
    rows = [noisy_row("greedy", ep, {0.2: 1.4}, failed={0.2: 0.2}) for ep in range(3)]
    text = report.format_table(rows, {"environment": "t"}, brief=True)
    assert "*" in text.split("MEASURED SHAPE ERROR")[1]


def test_measure_noise_skips_a_level_it_cannot_measure(make_env):
    """Every recovery blowing up leaves no number for that level, only the fact."""
    e = make_env(n=5, domains="R^3", max_steps=40)
    e.reset()
    row = run_greedy(e, max_steps=40, verbose=False)
    measure_noise(e, row, [1e-4, 3.0], trials=6, rng=np.random.default_rng(0))
    assert 1e-4 in row["noise"]
    assert all(np.isfinite(v) for v in row["noise"].values())


# ---------------------------------------------------------------- decision analysis
def test_the_landscape_covers_every_toggle_and_restores_the_graph(make_env):
    e = make_env(n=6, domains="R^3", max_steps=40)
    e.reset()
    before = e.network.edges.copy()
    land = edit_landscape(e)
    assert len(land) == 6 * 5
    assert np.array_equal(e.network.edges, before)


def test_percentile_is_midrank_so_ties_do_not_sink_the_best():
    """Several edits are often equally good; counting only strictly-worse ones would
    score picking one of them far below 100."""
    vals = [1.0] * 9 + [5.0]
    assert _percentile_of(vals, 5.0) == 95.0
    assert _percentile_of(vals, 1.0) == 45.0
    # a uniform tie is exactly chance
    assert _percentile_of([2.0] * 8, 2.0) == 50.0
    assert _percentile_of([1.0, 2.0], None) is None


def test_is_best_marks_the_optimum_including_ties():
    assert _is_best([1.0, 5.0, 5.0], 5.0)
    assert not _is_best([1.0, 5.0], 1.0)
    assert _is_best([1.0, None, 5.0], 5.0)


def test_a_record_ranks_the_best_and_worst_edit_apart(make_env):
    e = make_env(n=6, domains="R^3", max_steps=40)
    e.reset()
    land = edit_landscape(e)
    best = max(land, key=lambda k: land[k][0])
    worst = min(land, key=lambda k: land[k][0])

    hi = decision_record(land, best, "add")
    lo = decision_record(land, worst, "add")
    assert hi["phi_best"] and not lo["phi_best"]
    assert hi["phi_pct"] > lo["phi_pct"]
    assert hi["kind"] == "add"


def test_a_toggle_outside_the_landscape_records_nothing(make_env):
    e = make_env(n=6, domains="R^3", max_steps=40)
    e.reset()
    assert decision_record(edit_landscape(e), (0, 0), "add") is None


def test_the_error_ranking_is_absent_while_the_graph_is_flexible(make_env):
    """Shape error is infinite there, so no edit can be ranked by it."""
    e = make_env(n=6, domains="R^3", max_steps=40)
    e.reset()
    e.network.edges[:] = False
    e.network.edges[0, 1] = True
    land = edit_landscape(e)
    assert all(v[1] is None for v in land.values())
    rec = decision_record(land, (0, 2), "add")
    assert rec["err_pct"] is None and rec["phi_pct"] is not None


def test_the_decisions_figure_and_csv_round_trip(tmp_path):
    decisions = [dict(episode=0, step=t, kind="add" if t % 2 else "remove",
                      phi_pct=80.0, err_pct=50.0, share_pct=60.0,
                      phi_best=True, err_best=False, dphi=1.0, derr=0.1)
                 for t in range(6)]
    assert report.plot_decisions(str(tmp_path), decisions, {"environment": "t"})
    report.write_decisions(str(tmp_path), decisions)
    header = (tmp_path / "decisions.csv").read_text().splitlines()[0]
    assert header.split(",") == report.DECISION_FIELDS
    assert report.plot_decisions(str(tmp_path), [], {"environment": "t"}) is None
