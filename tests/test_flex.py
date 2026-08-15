"""The flex tensor and constraint power. THEORY.md section 9, validated in section 10."""
import numpy as np
import pytest

from conftest import ALL_DOMAINS, TOL
from conftest import ALL_DOMAINS
from rigidity import (extended_bearing_rigidity_matrix as B_of, flex_tensor,
                      flex_constraint_power, trivial_modes, rigidity_decomposition,
                      nullspace, nullspace_in_scaled_units, candidate_gain,
                      candidate_gain_reference, candidate_block,
                      flex_space, characteristic_length, rigidity_eigenvalue)
from scenario import random_scenario


def net(n, domain, m):
    doms = [domain] * n if isinstance(domain, str) else list(domain)
    net_, _ = random_scenario(n, doms, edge_count=m)
    return net_


def one_loose_node(n=8):
    """Nodes 0..n-2 fully connected (rigid); node n-1 held by a single bearing."""
    N = net(n, "R^3", m=1)
    E = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        for j in range(n - 1):
            if i != j:
                E[i, j] = True
    E[n - 1, 0] = True
    N.edges = E
    return N


def Pi_of(N):
    return flex_tensor(B_of(N), N.n, N.get_position_features())


def test_shape_is_n_by_n_by_3_by_3():
    N = net(6, "R^3", m=10)
    assert Pi_of(N).shape == (6, 6, 3, 3)


def test_shape_holds_on_an_empty_graph():
    """The early-return path used to give (n,3,3) -- add-only spaces start empty."""
    N = net(6, "R^3", m=1)
    N.edges = np.zeros((6, 6), dtype=bool)
    assert Pi_of(N).shape == (6, 6, 3, 3)


@pytest.mark.parametrize("m", [9, 12, 16])
def test_flex_dimension_equals_the_rank_deficit(m):
    """sum_i tr Pi[i,i] == dim F == rank_K_pos - rank(B_p)."""
    n = 8
    N = net(n, "R^3", m=m)
    B = B_of(N)
    rank_K_pos = np.linalg.matrix_rank(B_of(N.fully_connected())[:, :3 * n])
    deficit = rank_K_pos - np.linalg.matrix_rank(B[:, :3 * n])
    total = float(np.einsum("iidd->", Pi_of(N)))
    if deficit > 0:
        assert abs(total - deficit) < 1e-6
    else:
        assert abs(total - 1.0) < 1e-6      # rigid: the single weakest direction


def test_flex_space_is_orthogonal_to_the_trivial_modes():
    """eigh returns an arbitrary null-space basis, so they must be projected out."""
    N = one_loose_node()
    Pi = Pi_of(N)
    T = trivial_modes(N.get_position_features())          # (3n, 4), orthonormal
    P = Pi.transpose(0, 2, 1, 3).reshape(3 * N.n, 3 * N.n)
    assert np.allclose(T.T @ P, 0.0, atol=1e-8)


def test_flex_localises_on_a_singly_attached_node():
    N = one_loose_node()
    mag = np.sqrt(np.einsum("iidd->i", Pi_of(N)))
    assert int(np.argmax(mag)) == N.n - 1
    assert mag[-1] > 3 * np.median(mag[:-1])


def test_flex_localises_on_a_detached_node():
    N = one_loose_node()
    N.edges[N.n - 1, 0] = False
    mag = np.sqrt(np.einsum("iidd->i", Pi_of(N)))
    assert int(np.argmax(mag)) == N.n - 1


def test_flex_is_deterministic_across_calls():
    """Guards against an arbitrary eigenvector inside a degenerate eigenspace."""
    N = one_loose_node()
    vals = [np.einsum("iidd->i", Pi_of(N)) for _ in range(5)]
    for v in vals[1:]:
        assert np.allclose(v, vals[0], atol=TOL)


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_flex_scalars_are_rotation_invariant(domain):
    n = 6
    N = net(n, domain, m=2 * n)
    planar = domain in ("R^2", "R^2xS^1")
    before_mag = np.einsum("iidd->i", Pi_of(N))
    before_pow = flex_constraint_power(Pi_of(N), N.get_all_pairs_bearings_world())
    N.rotate_network([0, 0, 1] if planar else [0.3, 0.5, 0.81], 0.9)
    assert np.allclose(np.einsum("iidd->i", Pi_of(N)), before_mag, atol=1e-7)
    assert np.allclose(
        flex_constraint_power(Pi_of(N), N.get_all_pairs_bearings_world()),
        before_pow, atol=1e-7)


def test_constraint_power_is_positive_exactly_when_the_edge_raises_rank():
    """The ground-truth check: A[i,j] > 0 iff adding i->j increases rank(B)."""
    N = one_loose_node()
    n = N.n
    E = N.edges.copy()
    r0 = np.linalg.matrix_rank(B_of(N))
    A = flex_constraint_power(Pi_of(N), N.get_all_pairs_bearings_world())

    helps, useless = [], []
    for i in range(n):
        for j in range(n):
            if i == j or E[i, j]:
                continue
            N.edges = E.copy()
            N.edges[i, j] = True
            (helps if np.linalg.matrix_rank(B_of(N)) > r0 else useless).append(A[i, j])
    N.edges = E
    assert helps and useless
    # A is a square root, so a numerically-zero quadratic form (~1e-16) surfaces at
    # ~1e-8. Assert separation rather than an absolute floor.
    assert max(useless) < 1e-6, "an edge that does nothing scored nonzero"
    assert min(helps) > 1e-2, "an edge that raises rank scored ~zero"
    assert min(helps) > 1e4 * max(useless), "helpful and useless edges are not separated"


def test_constraint_power_uses_world_frame_bearings():
    """Body-frame bearings would make this rotation-dependent in oriented domains."""
    n = 6
    N = net(n, "SE(3)", m=12)
    body = N.get_all_pairs_bearings()
    world = N.get_all_pairs_bearings_world()
    assert not np.allclose(body, world), "SE(3) body and world bearings should differ"
    before = flex_constraint_power(Pi_of(N), world)
    N.rotate_network([0.3, 0.5, 0.81], 0.7)
    assert np.allclose(
        flex_constraint_power(Pi_of(N), N.get_all_pairs_bearings_world()),
        before, atol=1e-7)


# ------------------------------------------------- null-space features (WP2)

@pytest.mark.parametrize("domains", [[d] * 6 for d in ALL_DOMAINS] + [
    ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"],
    ["R^2"] * 3 + ["SE(3)"],
    ["R^2xS^1"] * 2 + ["R^3xS^1"] * 2,
])
def test_add_gain_is_exactly_the_addition_criterion(domains):
    """Edge i->j raises rank(B) iff b_ij Z != 0, and by rank(b_ij Z)."""
    n = len(domains)
    rng = np.random.default_rng(0)
    net, _ = random_scenario(n, list(domains))
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    E = np.zeros((n, n), dtype=bool)
    for k in rng.choice(len(pairs), size=int(0.4 * len(pairs)), replace=False):
        E[pairs[k]] = True
    net.edges = E

    rank_K = np.linalg.matrix_rank(B_of(net.fully_connected()))
    B = B_of(net)
    rank, _, _ = rigidity_decomposition(B, rank_K)
    Z = nullspace(B, rank)
    gain, rk = candidate_gain(net, Z)

    for i, j in pairs:
        if E[i, j]:
            continue
        net.edges = E.copy()
        net.edges[i, j] = True
        true_gain = np.linalg.matrix_rank(B_of(net)) - rank
        net.edges = E
        assert int(round(rk[i, j])) == true_gain, (i, j)
        assert (gain[i, j] > 1e-6) == (true_gain > 0), (i, j)
    assert gain.min() >= 0.0 and gain.max() <= 1.0 + 1e-12


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_nullspace_matches_an_svd_basis(domain):
    net, _ = random_scenario(6, domain, edge_count=9)
    B = B_of(net)
    rank_K = np.linalg.matrix_rank(B_of(net.fully_connected()))
    rank, _, _ = rigidity_decomposition(B, rank_K)
    Z = nullspace(B, rank)
    assert Z.shape == (B.shape[1], B.shape[1] - rank)
    assert np.abs(B @ Z).max() / max(np.abs(B).max(), 1e-12) < 1e-9
    ref = np.linalg.svd(B, full_matrices=True)[2][rank:].T
    assert np.abs(Z @ Z.T - ref @ ref.T).max() < 1e-8      # same subspace


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_flex_space_is_the_non_trivial_null_space(domain):
    """dim = rank_K - rank, i.e. ker(B_G) with the trivial variations removed."""
    net, _ = random_scenario(6, domain, edge_count=8)
    BK = B_of(net.fully_connected())
    rank_K = np.linalg.matrix_rank(BK)
    B = B_of(net)
    rank, _, _ = rigidity_decomposition(B, rank_K)
    L = characteristic_length(net)
    Z = nullspace_in_scaled_units(nullspace(B, rank), net.n, L)
    ZK = nullspace_in_scaled_units(nullspace(BK, rank_K), net.n, L)
    assert flex_space(Z, ZK).shape[1] == rank_K - rank


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_rigidity_decomposition_agrees_with_matrix_rank_and_eigenvalue(domain):
    net, _ = random_scenario(6, domain, edge_count=11)
    BK = B_of(net.fully_connected())
    rank_K = np.linalg.matrix_rank(BK)
    B = B_of(net)
    rank, _, lam = rigidity_decomposition(B, rank_K)
    assert rank == np.linalg.matrix_rank(B)
    assert abs(lam - rigidity_eigenvalue(net, rank_K=rank_K)) < 1e-9


def test_scaling_the_formation_leaves_the_null_space_features_alone():
    """B mixes 1/length with dimensionless columns, so the length unit is fixed
    to the formation's own size. Without that, ker(B) moves under scaling."""
    net, _ = random_scenario(6, "SE(3)", edge_count=9)
    BK = B_of(net.fully_connected())
    rank_K = np.linalg.matrix_rank(BK)

    def features(net_):
        B = B_of(net_)
        rank, _, _ = rigidity_decomposition(B, rank_K)
        L = characteristic_length(net_)
        Z = nullspace_in_scaled_units(nullspace(B, rank), net_.n, L)
        return candidate_gain(net_, Z, length_scale=L)[0]

    before = features(net)
    net.scale_network(7.5)
    assert np.abs(before - features(net)).max() < 1e-7


@pytest.mark.parametrize("domains", [[d] * 6 for d in ALL_DOMAINS] + [
    ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)", "R^3"],
])
def test_fast_candidate_gain_matches_the_readable_one(domains):
    """candidate_gain is a hand-expanded candidate_gain_reference; if the two ever
    disagree the expansion is wrong, which is how the sign error was found."""
    n = len(domains)
    rng = np.random.default_rng(1)
    net, _ = random_scenario(n, list(domains))
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    E = np.zeros((n, n), dtype=bool)
    for k in rng.choice(len(pairs), size=int(0.4 * len(pairs)), replace=False):
        E[pairs[k]] = True
    net.edges = E

    rank_K = np.linalg.matrix_rank(B_of(net.fully_connected()))
    B = B_of(net)
    rank, _, _ = rigidity_decomposition(B, rank_K)
    L = characteristic_length(net)
    Z = nullspace_in_scaled_units(nullspace(B, rank), n, L)

    g_fast, r_fast = candidate_gain(net, Z, length_scale=L)
    g_ref, r_ref = candidate_gain_reference(net, Z, length_scale=L)
    assert np.abs(g_fast - g_ref).max() < 1e-9
    assert np.array_equal(r_fast, r_ref)


def test_candidate_block_is_the_row_block_the_edge_would_append():
    """b_ij is literally B's rows for that edge, so B on {i->j} alone must be it."""
    net, _ = random_scenario(5, ["R^2", "R^3", "R^2xS^1", "R^3xS^1", "SE(3)"])
    E = np.zeros((5, 5), dtype=bool)
    E[0, 2] = E[3, 1] = True
    net.edges = E
    B = B_of(net)                       # rows in np.nonzero(edges) order
    for k, (i, j) in enumerate(zip(*np.nonzero(E))):
        b = candidate_block(net, i, j)
        assert np.abs(b - B[3 * k:3 * k + 3]).max() < 1e-12
