"""The topology-property harness: the arithmetic traps, and two theorem checks.

Nothing here reads `environments/` or `train/`, so the fast suite passes on a clean
clone.
"""
import numpy as np
import pytest

from conftest import ALL_DOMAINS
import network as network_mod
import report
import repair
import rigidity as R
import topology as T
from scenario import random_scenario


# ------------------------------------------------------------ helpers

def net_with(positions, domains, edge_list):
    """A formation at given positions with a given edge set."""
    np.random.seed(0)
    n = len(positions)
    net, _ = random_scenario(n, list(domains), edge_count=0)
    for agent, p in zip(net.agents, positions):
        agent.pose.position = np.asarray(p, dtype=float)
    net.edges = np.zeros((n, n), dtype=bool)
    for i, j in edge_list:
        net.edges[i, j] = True
    return net


def rigid_network(n, domains, seed=0):
    """A minimal-ish rigid graph on random poses, and its rank_K."""
    np.random.seed(seed)
    net, _ = random_scenario(n, list(domains), edge_count=0)
    rank_K = int(np.linalg.matrix_rank(
        R.extended_bearing_rigidity_matrix(net.fully_connected())))
    repair.converge(net, rank_K, np.random.default_rng(seed))
    return net, rank_K


# ------------------------------------------------------------ reciprocity

def test_reciprocity_counts_edges_not_pairs():
    edges = np.zeros((3, 3), dtype=bool)
    edges[0, 1] = edges[1, 0] = edges[1, 2] = True
    m, recip_edges, recip_pairs, frac = T.reciprocity(edges)
    assert (m, recip_edges, recip_pairs) == (3, 2, 1)
    assert frac == pytest.approx(2 / 3)


def test_reciprocity_is_one_on_a_fully_reciprocal_graph():
    edges = np.ones((4, 4), dtype=bool)
    np.fill_diagonal(edges, False)
    assert T.reciprocity(edges)[3] == pytest.approx(1.0)


def test_expected_reciprocity_matches_uniform_draws():
    n, m, draws = 8, 14, 4000
    rng = np.random.default_rng(0)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    got = []
    for _ in range(draws):
        edges = np.zeros((n, n), dtype=bool)
        for k in rng.choice(len(pairs), size=m, replace=False):
            edges[pairs[k]] = True
        got.append(T.reciprocity(edges)[3])
    predicted = T.expected_reciprocity(n, m)
    se = float(np.std(got)) / np.sqrt(draws)
    assert abs(float(np.mean(got)) - predicted) < 3 * se


# ------------------------------------------------------------ degrees

def test_out_degree_is_the_row_sum_and_the_second_feature_column():
    net, _ = rigid_network(6, ["R^3"] * 6)
    out, inn = T.out_degrees(net.edges), T.in_degrees(net.edges)
    assert np.array_equal(out, net.edges.sum(axis=1))
    # get_degree_features stacks (in, out) despite the name order, which is the trap
    feats = net.get_degree_features()
    assert np.array_equal(out, feats[:, 1])
    assert np.array_equal(inn, feats[:, 0])


def test_degree_sums_agree_with_the_edge_count():
    net, _ = rigid_network(6, ["SE(3)"] * 6)
    m = int(net.edges.sum())
    assert T.out_degrees(net.edges).sum() == m
    assert T.in_degrees(net.edges).sum() == m


# ------------------------------------------------------------ bearing spread

def test_spread_is_zero_for_parallel_bearings_and_one_for_orthogonal():
    # a planar formation, so the spread is read in the two axes it spans
    positions = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0)]
    parallel = net_with(positions, ["R^2"] * 4, [(0, 1), (0, 2)])
    assert T.bearing_spread(parallel, 0, axes=T.spanned_axes(
        parallel.get_all_pairs_bearings_world()))[0] == pytest.approx(0.0, abs=1e-12)

    orthogonal = net_with(positions, ["R^2"] * 4, [(0, 1), (0, 3)])
    assert T.bearing_spread(orthogonal, 0, axes=T.spanned_axes(
        orthogonal.get_all_pairs_bearings_world()))[0] == pytest.approx(1.0)


def test_spread_is_none_below_two_outgoing_edges():
    net = net_with([(0, 0, 0), (1, 0, 0), (0, 1, 0)], ["R^3"] * 3, [(0, 1)])
    assert T.bearing_spread(net, 0) == (None, None)
    assert T.bearing_spread(net, 1) == (None, None)


def test_spread_is_the_same_in_the_body_frame():
    """The body frame is one rotation of all of agent i's own bearings, and singular
    values do not move under that. A mismatch means one bearing primitive is wrong."""
    net, _ = rigid_network(6, ["SE(3)"] * 6)
    world = net.get_all_pairs_bearings_world()
    body = net.get_all_pairs_bearings()
    for i in range(net.n):
        if int(net.edges[i].sum()) < 2:
            continue
        a = T.bearing_spread(net, i, bearings=world)
        b = T.bearing_spread(net, i, bearings=body)
        assert a[0] == pytest.approx(b[0], rel=1e-9, abs=1e-12)
        assert a[1] == pytest.approx(b[1], rel=1e-9, abs=1e-12)


# ------------------------------------------------------------ lengths

def test_edge_lengths_are_invariant_under_a_uniform_rescale():
    net, _ = rigid_network(6, ["R^3"] * 6)
    before = T.edge_lengths(net)
    for agent in net.agents:
        agent.pose.position = agent.pose.position * 7.0
    after = T.edge_lengths(net)
    assert set(before) == set(after)
    for e in before:
        assert after[e] == pytest.approx(before[e])


# ------------------------------------------------------------ Laman

def test_is_laman_on_two_graphs_with_known_answers():
    # n = 4, 2n - 3 = 5 edges, no subset over its budget
    good = np.zeros((4, 4), dtype=bool)
    for i, j in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]:
        good[i, j] = True
    assert T.is_laman(good) is True

    # K4 on 6 edges: the whole vertex set spans 6 > 2*4 - 3
    k4 = np.ones((4, 4), dtype=bool)
    np.fill_diagonal(k4, False)
    assert T.is_laman(k4) is False

    # 7 edges on 5 vertices is exactly 2n - 3, but K4 on {0,1,2,3} spends 6 of them on a
    # subset whose budget is 5, so the count alone does not settle it
    wrong = np.zeros((5, 5), dtype=bool)
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4)]:
        wrong[i, j] = True
    assert len(T.undirected(wrong)) == 2 * 5 - 3
    assert T.is_laman(wrong) is False


@pytest.mark.parametrize("n", [5, 6])
def test_laman_agrees_with_minimal_bearing_rigidity_in_the_plane(n):
    """In homogeneous R^2 a graph is minimally bearing rigid exactly when its underlying
    undirected graph is Laman. This is the external check on both sides."""
    for seed in range(25):
        net, rank_K = rigid_network(n, ["R^2"] * n, seed=seed)
        m = int(net.edges.sum())
        rigid = int(np.linalg.matrix_rank(
            R.extended_bearing_rigidity_matrix(net))) == rank_K
        assert rigid and m == 2 * n - 3
        # minimal in the plane means no edge is reciprocated: the reverse adds no rank
        assert T.reciprocity(net.edges)[1] == 0
        assert T.is_laman(net.edges) is True


# ------------------------------------------------------------ the floor

@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_every_rigid_graph_meets_the_out_degree_floor(domain):
    for seed in range(8):
        net, _ = rigid_network(6, [domain] * 6, seed=seed)
        out = T.out_degrees(net.edges)
        for i, agent in enumerate(net.agents):
            assert out[i] >= T.R_MIN[agent.domain], (
                f"{domain} agent {i} has out-degree {out[i]}, floor "
                f"{T.R_MIN[agent.domain]}")


def test_the_floor_holds_on_mixed_domains_too():
    rng = np.random.default_rng(0)
    for seed in range(8):
        domains = list(rng.choice(ALL_DOMAINS, size=6))
        net, _ = rigid_network(6, domains, seed=seed)
        out = T.out_degrees(net.edges)
        for i, agent in enumerate(net.agents):
            assert out[i] >= T.R_MIN[agent.domain]


# ------------------------------------------------------------ class ranks

def test_a_planar_to_planar_edge_carries_rank_one_and_the_rest_two():
    net, _ = rigid_network(8, ["R^2", "R^2", "R^2xS^1", "R^2xS^1",
                               "R^3", "R^3", "R^3xS^1", "SE(3)"])
    for (di, dj), c_k in T.class_ranks(net).items():
        expected = 1 if (di in T.PLANAR and dj in T.PLANAR) else 2
        assert c_k == expected, f"{di} -> {dj} has block rank {c_k}, expected {expected}"


# ------------------------------------------------------------ jaccard

def test_jaccard_on_known_overlaps():
    a, b = {(0, 1), (1, 2), (2, 3)}, {(0, 1), (1, 2), (2, 3)}
    assert T.jaccard(a, b) == (1.0, 3, 3)
    assert T.jaccard(a, {(3, 0)}) == (0.0, 0, 4)
    value, shared, union = T.jaccard(a, {(0, 1), (3, 0)})
    assert (shared, union) == (1, 4) and value == pytest.approx(0.25)


# ------------------------------------------------------------ bookkeeping

def test_describe_accounts_for_every_ordered_pair_and_every_edge():
    net, rank_K = rigid_network(8, ["R^2", "R^2", "R^2xS^1", "R^3",
                                    "R^3", "R^3xS^1", "SE(3)", "SE(3)"])
    row, nodes, pairs = T.describe(net, rank_K=rank_K,
                                   m_req=R.required_edge_count(net, rank_K=rank_K))
    assert T.consistency(row, pairs) == []
    assert sum(p["avail"] for p in pairs) == net.n * (net.n - 1)
    assert sum(p["edges"] for p in pairs) == row["m"]
    assert len(nodes) == net.n
    assert row["is_rigid"] and row["shape_err"] is not None


def test_m_req_agrees_with_the_closed_form_from_the_domain_mix():
    import sys
    sys.path.insert(0, "tools")
    from domain_closed_forms import predict
    domains = ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)", "R^3"]
    net, rank_K = rigid_network(6, domains)
    _, _, m_req = predict([a.domain for a in net.agents])
    assert R.required_edge_count(net, rank_K=rank_K) == m_req


# ------------------------------------------------------------ shared tables

def test_the_figure_domain_order_matches_the_observation_encoding():
    assert tuple(report.DOMAIN_ORDER) == network_mod.DOMAINS


def test_encode_edges_round_trips():
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = 6
        edges = rng.random((n, n)) < 0.3
        np.fill_diagonal(edges, False)
        assert np.array_equal(report.decode_edges(report.encode_edges(edges), n), edges)
    empty = np.zeros((4, 4), dtype=bool)
    assert report.encode_edges(empty) == ""
    assert np.array_equal(report.decode_edges("", 4), empty)
    assert report.encode_edges(None) is None
    assert report.decode_edges(None, 4) is None


# ------------------------------------------------------------ the calibration

def test_greedy_and_spectral_build_the_same_graph_at_kappa_zero(make_env):
    """At stiffness_kappa 0 the closed form spectral reads is exact, so the two hill
    climbs agree edge for edge. That cell reading 1.00 is what says the harness paired
    the instances rather than scoring each method on a different graph."""
    import copy
    import topology as T

    env = make_env(n=6, domains="R^3", state_score_type="WeightedNormalized")
    np.random.seed(0)
    instances = []
    for _ in range(4):
        env.freeze_network = False
        env.reset()
        instances.append(copy.deepcopy(env.network))

    rows, nodes, pairs, jac = T.measure(
        env=env, instances=instances, methods=["greedy", "spectral"], steps=40, seed=0)

    assert len(rows) == 8
    cells = [r for r in jac if {r["method_a"], r["method_b"]} == {"greedy", "spectral"}]
    assert len(cells) == 4
    for cell in cells:
        assert cell["jaccard"] == 1.0, cell
    for row in rows:
        assert row["is_rigid"] and row["floor_ok"]
        assert T.consistency(row, [p for p in pairs
                                   if p["episode"] == row["episode"]
                                   and p["method"] == row["method"]]) == []
