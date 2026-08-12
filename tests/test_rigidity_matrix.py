"""The extended bearing rigidity matrix and its null space. THEORY.md sections 2-3, 12."""
import copy

import numpy as np
import pytest
import quaternion

from conftest import (ALL_DOMAINS, ORIENTED_DOMAINS, RANK_K_FORMULA, PLANAR,
                      DOF_PER_AGENT, MIXES, max_rank_K)
from rigidity import (extended_bearing_rigidity_matrix as B_of, bearing_DOFs,
                      node_dof_projectors)
from scenario import random_scenario
from util import orthogonal_projection_matrix, skew_symmetric


def net(n, domain, m=None):
    # per-agent list rather than the homogeneous string: the string path rejects
    # "R^3xS^1" outright (see test_homogeneous_string_domain_accepts_every_domain)
    doms = [domain] * n if isinstance(domain, str) else list(domain)
    net_, _ = random_scenario(n, doms, edge_count=m if m is not None else max(n, 2 * n))
    return net_


def test_shape_is_3m_by_6n():
    N = net(6, "R^3", m=11)
    B = B_of(N)
    assert B.shape == (3 * int(N.edges.sum()), 6 * 6)


def test_one_three_row_block_per_edge_in_nonzero_order():
    """is_MBR and the block-rank features depend on this ordering."""
    N = net(5, "R^3", m=7)
    B = B_of(N)
    ii, jj = np.nonzero(N.edges)
    assert B.shape[0] == 3 * len(ii)
    for k, (i, j) in enumerate(zip(ii, jj)):
        block = B[3 * k:3 * (k + 1), :]
        # the block touches only its own two endpoints' position columns
        touched = {c // 3 for c in np.nonzero(np.abs(block[:, :15]).sum(axis=0))[0]}
        assert touched <= {int(i), int(j)}


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_translation_is_a_trivial_motion(domain):
    n = 6
    N = net(n, domain)
    B = B_of(N)
    for axis in range(3):
        v = np.zeros(6 * n)
        v[axis:3 * n:3] = 1.0            # same translation on every node
        assert np.allclose(B @ v, 0.0, atol=1e-9), f"{domain} axis {axis}"


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_uniform_scaling_is_a_trivial_motion(domain):
    """Scaling moves every node along its own bearing, which P(p_hat) kills."""
    n = 6
    N = net(n, domain)
    p = np.array([a.pose.position for a in N.agents])
    v = np.zeros(6 * n)
    v[:3 * n] = (p - p.mean(axis=0)).reshape(-1)
    assert np.allclose(B_of(N) @ v, 0.0, atol=1e-9)


@pytest.mark.parametrize("domain", ALL_DOMAINS)
@pytest.mark.parametrize("n", [4, 6, 8, 16])
def test_rank_K_matches_the_closed_form(domain, n):
    N = net(n, domain, m=1)
    rank_K = np.linalg.matrix_rank(B_of(N.fully_connected()))
    assert rank_K == RANK_K_FORMULA[domain](n)


def test_fully_connected_has_no_self_loops():
    N = net(5, "R^3")
    K = N.fully_connected()
    assert not np.any(np.diag(K.edges))
    assert int(K.edges.sum()) == 5 * 4
    assert B_of(K).shape[0] == 3 * 5 * 4


def test_rank_is_scale_invariant():
    """Bearings are unit vectors, so a uniform scaling cannot change the rank."""
    N = net(6, "R^3", m=10)
    r0 = np.linalg.matrix_rank(B_of(N))
    N.scale_network(7.5)
    assert np.linalg.matrix_rank(B_of(N)) == r0


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_homogeneous_string_domain_accepts_every_domain(domain):
    """Network.set_agents_domain_homogeneous accepts all five domains.

    This used to xfail for R^3xS^1, on the grounds that its branch was commented
    out and the call fell through to a bare quit(). The branch is present
    (network.py, set_agents_domain_homogeneous), the xfail was stale, and because
    pytest.xfail() short-circuits before the assertion it could never have gone
    green to report that. Asserted properly now.
    """
    net_, _ = random_scenario(4, domain, edge_count=4)
    assert all(a.domain == domain for a in net_.agents)


# --------------------------------------------------------------------- WP1
# Per-node DOF restriction. These are the tests that would have caught the
# heterogeneous bug: the pre-WP1 matrix attached the translational restriction to
# the edge, so a planar agent measuring a spatial one regained a z DOF and the
# matrix spent rank resisting a motion nobody can perform. ROADMAP.md#1.2.

def _perturb(net, delta, eps):
    """chi + eps*delta, with delta = [dp_0..dp_{n-1}, dw_0..dw_{n-1}] in R^(6n).

    dw is a world-frame angular variation, matching B's convention: the frame
    update is R_i <- exp([dw]_x) R_i.
    """
    out = copy.deepcopy(net)
    n = out.n
    for i, a in enumerate(out.agents):
        a.pose.position = a.pose.position + eps * delta[3 * i:3 * i + 3]
        dw = eps * delta[3 * n + 3 * i:3 * n + 3 * i + 3]
        th = np.linalg.norm(dw)
        if th > 0:
            K = skew_symmetric(dw / th)
            Rd = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
            R = quaternion.as_rotation_matrix(a.pose.orientation)
            a.pose.orientation = quaternion.from_rotation_matrix(Rd @ R)
    return out


def _bearings(net):
    """The function B differentiates: measured bearings in nonzero(edges) order."""
    ii, jj = np.nonzero(net.edges)
    return np.concatenate([net.agents[i].get_bearing(net.agents[j])
                           for i, j in zip(ii, jj)])


def _admissible_basis(net):
    """Orthonormal columns spanning the variations the agents can actually make."""
    n = net.n
    cols = []
    for i, a in enumerate(net.agents):
        S, P = node_dof_projectors(a)
        for M, off in ((S, 0), (P, 3 * n)):
            u, s, _ = np.linalg.svd(M)
            for c in range(int((s > 1e-9).sum())):
                v = np.zeros(6 * n)
                v[off + 3 * i:off + 3 * i + 3] = u[:, c]
                cols.append(v)
    return np.array(cols).T


@pytest.mark.parametrize("domains", [[d] * 6 for d in ALL_DOMAINS] + MIXES)
def test_matrix_is_the_numerical_jacobian_of_the_bearings(domains):
    """B @ delta must equal d(bearings)/dt for every admissible variation.

    The definition, checked directly by central differences. This validates the
    whole construction at once -- D_p, D_a, the incidence signs, and both DOF
    projectors -- in every domain and mix.
    """
    n = len(domains)
    net, _ = random_scenario(n, list(domains), edge_count=max(3, 2 * n))
    # a non-default rotation axis, so the v v^T projector is actually exercised
    for a in net.agents:
        if a.domain == "R^3xS^1":
            a.set_domain("R^3xS^1", rotation_axis=np.array([1.0, 2.0, -0.5]))

    B, A, eps = B_of(net), _admissible_basis(net), 1e-6
    rng = np.random.default_rng(0)
    for _ in range(10):
        d = A @ rng.standard_normal(A.shape[1])
        d /= np.linalg.norm(d)
        num = (_bearings(_perturb(net, d, eps)) - _bearings(_perturb(net, d, -eps))) / (2 * eps)
        assert np.linalg.norm(B @ d - num) / max(np.linalg.norm(num), 1e-12) < 1e-6


@pytest.mark.parametrize("domains", MIXES)
def test_rank_K_respects_the_dof_budget(domains):
    """rank_K cannot exceed sum(DOF) minus the trivial motions.

    The pre-WP1 matrix gave rank_K = 36 on the `mixed` composition against a
    budget of 33 -- i.e. zero trivial motions, which no framework has.
    """
    net, _ = random_scenario(len(domains), list(domains), edge_count=len(domains))
    rank_K = np.linalg.matrix_rank(B_of(net.fully_connected()))
    assert rank_K <= max_rank_K(domains)


@pytest.mark.parametrize("domains", MIXES)
def test_infeasible_coordinates_are_zero_columns(domains):
    """Michieletto Def. 13 counts infeasible variations as null columns of B."""
    n = len(domains)
    net, _ = random_scenario(n, list(domains), edge_count=n)
    net.edges = ~np.eye(n, dtype=bool)
    B = B_of(net)
    A = _admissible_basis(net)
    Q, _ = np.linalg.qr(A)
    inadmissible = np.eye(6 * n) - Q @ Q.T          # projector onto what cannot move
    assert np.abs(B @ inadmissible).max() < 1e-12
    assert A.shape[1] == sum(DOF_PER_AGENT[d] for d in domains)


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_matches_michieletto_table_I_on_homogeneous_networks(domain):
    """The per-node construction must reproduce the per-edge U_ij / V_ij exactly.

    Table I is the homogeneous case, where U_ij = S_i = S_j and V_ij = P_i, so the
    two constructions coincide -- and no homogeneous result may move because of
    WP1. bearing_DOFs is retained for exactly this comparison.
    """
    for n in (4, 6, 8):
        net, _ = random_scenario(n, domain, edge_count=max(2, n))
        p = [a.pose.position for a in net.agents]
        R = [a.pose.rotation_mat() for a in net.agents]
        ii, jj = np.nonzero(net.edges)
        m = len(ii)

        E, Eo = np.zeros((n, m)), np.zeros((n, m))
        U, V = np.zeros((3 * m, 3 * m)), np.zeros((3 * m, 3 * m))
        Dp, Da = np.zeros((3 * m, 3 * m)), np.zeros((3 * m, 3 * m))
        for k, (i, j) in enumerate(zip(ii, jj)):
            E[i, k], E[j, k], Eo[i, k] = -1, +1, -1
            U[3 * k:3 * k + 3, 3 * k:3 * k + 3], V[3 * k:3 * k + 3, 3 * k:3 * k + 3] = \
                bearing_DOFs(net.agents[i], net.agents[j])
            pij = p[j] - p[i]
            s = 1.0 / np.linalg.norm(pij)
            Dp[3 * k:3 * k + 3, 3 * k:3 * k + 3] = s * R[i].T @ orthogonal_projection_matrix(s * pij)
            Da[3 * k:3 * k + 3, 3 * k:3 * k + 3] = -R[i].T @ skew_symmetric(s * pij)

        table_I = np.hstack([Dp @ U @ np.kron(E, np.eye(3)).T,
                             Da @ V @ np.kron(Eo, np.eye(3)).T])
        assert np.abs(B_of(net) - table_I).max() < 1e-12


def test_rotation_axis_is_a_projector_not_a_row():
    """V_ij for R^3xS^1 must be v v^T, not [0; 0; v] laid out as rows.

    The two coincide at v = e3, the only axis in use, so nothing measured has ever
    depended on it -- but the parameter is exposed and the row form is wrong for
    every other axis. Michieletto Table I gives [0_{3x2} v], a column.
    """
    v = np.array([1.0, 2.0, -0.5])
    v /= np.linalg.norm(v)
    net, _ = random_scenario(5, "R^3xS^1", edge_count=8)
    for a in net.agents:
        a.set_domain("R^3xS^1", rotation_axis=v)

    _, P = node_dof_projectors(net.agents[0])
    assert np.allclose(P, np.outer(v, v))
    assert np.allclose(P @ v, v)                       # the free direction survives
    w = np.cross(v, [0.0, 0.0, 1.0])
    assert np.allclose(P @ w, 0.0)                     # anything perpendicular does not

    # and e3 is the special case where the old row form happened to be right
    for a in net.agents:
        a.set_domain("R^3xS^1", rotation_axis=np.array([0.0, 0.0, 1.0]))
    _, P_e3 = node_dof_projectors(net.agents[0])
    assert np.allclose(P_e3, np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float))
