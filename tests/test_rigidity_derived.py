"""c_k, c_max, m_req, is_MBR, and the repair bound. 4-6, 19."""
import copy
import itertools
import numpy as np
import pytest

from conftest import ALL_DOMAINS, C_MAX, RANK_K_FORMULA
from rigidity import (extended_bearing_rigidity_matrix as B_of, edge_block_ranks,
                      greedy_rigid_construction, max_edge_rank, repair_edge_count,
                      required_edge_count, MBR_required_Rd, is_MBR)
from scenario import random_scenario


def net(n, domain, m):
    doms = [domain] * n if isinstance(domain, str) else list(domain)
    net_, _ = random_scenario(n, doms, edge_count=m)
    return net_


@pytest.mark.parametrize("domain,n", [("R^2", 4), ("R^2", 8), ("R^3", 8), ("R^3", 16)])
def test_every_edge_block_has_rank_d_minus_1(domain, n):
    """c_k is CONSTANT in homogeneous R^d -- which is why it is a dead feature there."""
    N = net(n, domain, m=2 * n)
    ranks = set(edge_block_ranks(B_of(N)))
    assert ranks == {C_MAX[domain]}


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_c_max_matches_the_expected_value(domain):
    n = 6
    N = net(n, domain, m=2 * n)
    assert max_edge_rank(N) == C_MAX[domain]


@pytest.mark.parametrize("domain", ALL_DOMAINS)
@pytest.mark.parametrize("n", [4, 6, 8, 16])
def test_m_req_equals_ceil_rank_K_over_c_max(domain, n):
    N = net(n, domain, m=n)
    rank_K = RANK_K_FORMULA[domain](n)
    assert required_edge_count(N) == -(-rank_K // C_MAX[domain])


@pytest.mark.parametrize("domain", ["R^2", "R^3"])
@pytest.mark.parametrize("n", [4, 5, 6, 8, 12, 16])
def test_m_req_agrees_with_the_closed_form(domain, n):
    d = 2 if domain == "R^2" else 3
    N = net(n, domain, m=n)
    assert required_edge_count(N) == MBR_required_Rd(n, d)


def test_m_req_depends_on_domain_not_just_dimension():
    """SE(3) needs far more edges than R^3 at the same n: 21 vs 10 at n=8."""
    assert required_edge_count(net(8, "SE(3)", m=8)) == 21
    assert required_edge_count(net(8, "R^3", m=8)) == 10


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_m_req_is_a_lower_bound_on_any_rigid_graph(domain):
    """Rank subadditivity: no rigid graph may use fewer than m_req edges."""
    n = 5
    N = net(n, domain, m=n)
    m_req = required_edge_count(N)
    rank_K = np.linalg.matrix_rank(B_of(N.fully_connected()))
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    rng = np.random.default_rng(0)
    for _ in range(40):
        k = int(rng.integers(1, m_req))          # strictly fewer than m_req
        E = np.zeros((n, n), dtype=bool)
        for idx in rng.choice(len(pairs), size=k, replace=False):
            E[pairs[idx]] = True
        N.edges = E
        assert np.linalg.matrix_rank(B_of(N)) < rank_K


def test_is_MBR_recognises_a_minimal_graph():
    n, domain = 6, "R^3"
    N = net(n, domain, m=1)
    rank_K = np.linalg.matrix_rank(B_of(N.fully_connected()))
    m_req = required_edge_count(N)
    # greedily build a rigid graph, then confirm minimality is reported at m_req
    E = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j or E.sum() >= m_req:
                continue
            E[i, j] = True
            N.edges = E
            if np.linalg.matrix_rank(B_of(N)) < min(rank_K, 2 * int(E.sum())):
                E[i, j] = False
    N.edges = E
    mbr, ibr, rank = N.is_MBR(rank_K=rank_K)
    if ibr:
        assert int(E.sum()) >= m_req
        assert mbr == (int(E.sum()) == m_req)


def test_block_ranks_can_be_passed_in_to_avoid_recomputation():
    N = net(6, "R^3", m=12)
    B = B_of(N)
    rank_K = np.linalg.matrix_rank(B_of(N.fully_connected()))
    a = N.is_MBR(rank_K=rank_K, brm=B)
    b = N.is_MBR(rank_K=rank_K, brm=B, block_ranks=edge_block_ranks(B))
    assert a == b


@pytest.mark.slow
@pytest.mark.parametrize("doms", [
    ["R^2"] * 4, ["R^3"] * 4,
    ["R^2", "R^2", "R^2xS^1", "R^2xS^1"],
    ["R^3", "R^3", "SE(3)", "SE(3)"],
    ["R^2", "R^3", "R^2xS^1", "SE(3)"],
])
def test_m_req_bound_is_tight_by_brute_force(doms):
    """Exhaustive: some edge set of exactly m_req edges really is rigid."""
    n = 4
    N = net(n, doms, m=1)
    rank_K = np.linalg.matrix_rank(B_of(N.fully_connected()))
    m_req = required_edge_count(N)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    for sub in itertools.combinations(pairs, m_req):
        E = np.zeros((n, n), dtype=bool)
        for i, j in sub:
            E[i, j] = True
        N.edges = E
        if np.linalg.matrix_rank(B_of(N)) == rank_K:
            return
    pytest.fail(f"no rigid graph with m_req={m_req} edges for {doms}")


# ---------------------------------------------------------------- the repair bound
def rigid_net(domains, n=8, seed=0):
    doms = [domains] * n if isinstance(domains, str) else list(domains)
    net_, _ = random_scenario(len(doms), doms, edge_count=0)
    rank_K = int(np.linalg.matrix_rank(B_of(net_.fully_connected())))
    greedy_rigid_construction(net_, rank_K, np.random.default_rng(seed))
    return net_, rank_K


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_a_rigid_graph_needs_no_repair(domain):
    net_, rank_K = rigid_net(domain, n=6)
    assert repair_edge_count(net_, rank_K=rank_K) == 0


@pytest.mark.parametrize("domain", ALL_DOMAINS + [["R^2", "R^3", "SE(3)", "R^3xS^1"]])
def test_repairing_the_empty_graph_is_building_it(domain):
    """From the empty graph the repair bound equals required_edge_count."""
    doms = [domain] * 6 if isinstance(domain, str) else list(domain)
    net_, _ = random_scenario(len(doms), doms, edge_count=0)
    net_.edges[:] = False
    rank_K = int(np.linalg.matrix_rank(B_of(net_.fully_connected())))
    assert repair_edge_count(net_, rank_K=rank_K) == required_edge_count(net_, rank_K=rank_K)


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_putting_back_what_was_removed_bounds_the_repair(domain):
    """Removing k edges leaves a graph the bound says is repairable in at most k."""
    net_, rank_K = rigid_net(domain, n=6)
    rng = np.random.default_rng(1)
    present = list(zip(*np.nonzero(net_.edges)))
    for k in (1, 2, 3):
        work = copy.deepcopy(net_)
        for idx in rng.choice(len(present), k, replace=False):
            i, j = present[idx]
            work.edges[i, j] = False
        assert repair_edge_count(work, rank_K=rank_K) <= k


def test_in_the_plane_the_bound_is_the_rank_deficit():
    """In R^2 the bound is exactly the rank deficit."""
    for seed in range(4):
        net_, rank_K = rigid_net("R^2", n=7, seed=seed)
        rng = np.random.default_rng(seed)
        present = list(zip(*np.nonzero(net_.edges)))
        for idx in rng.choice(len(present), 3, replace=False):
            i, j = present[idx]
            net_.edges[i, j] = False
        brm = B_of(net_)
        deficit = rank_K - int(np.linalg.matrix_rank(brm))
        assert rank_K == 2 * net_.n - 3
        assert repair_edge_count(net_, rank_K=rank_K, brmat=brm) == deficit


@pytest.mark.slow
@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_the_repair_bound_is_sound_against_brute_force(domain):
    """No edge set smaller than the bound restores rigidity."""
    import itertools
    n = 5
    rng = np.random.default_rng(0)
    checked = 0
    for trial in range(8):
        net_, _ = random_scenario(n, [domain] * n, edge_count=0)
        rank_K = int(np.linalg.matrix_rank(B_of(net_.fully_connected())))
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        net_.edges[:] = False
        for idx in rng.choice(len(pairs), int(rng.integers(2, 9)), replace=False):
            net_.edges[pairs[idx]] = True
        brm = B_of(net_)
        if np.linalg.matrix_rank(brm) >= rank_K:
            continue

        lb = repair_edge_count(net_, rank_K=rank_K, brmat=brm)
        absent = [(i, j) for i, j in pairs if not net_.edges[i, j]]
        for k in range(1, lb):
            for sub in itertools.combinations(absent, k):
                work = copy.deepcopy(net_)
                for i, j in sub:
                    work.edges[i, j] = True
                assert np.linalg.matrix_rank(B_of(work)) < rank_K, (
                    f"{domain}: {k} edges sufficed, bound said {lb}")
        checked += 1
    assert checked > 0
