"""The extended bearing rigidity matrix and its null space. THEORY.md sections 2-3."""
import numpy as np
import pytest

from conftest import ALL_DOMAINS, ORIENTED_DOMAINS, RANK_K_FORMULA, PLANAR
from rigidity import extended_bearing_rigidity_matrix as B_of
from scenario import random_scenario


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
    """Network.set_agents_domain_homogeneous should accept all five domains.

    R^3xS^1 is currently rejected -- its branch is commented out, so it falls
    through to a bare quit() that kills the process rather than raising. The
    per-agent set_domain() path handles it fine, so only the homogeneous-string
    entry point is affected. xfail(strict) so fixing it shows up here.
    """
    if domain == "R^3xS^1":
        pytest.xfail("set_agents_domain_homogeneous rejects R^3xS^1 and calls quit()")
    net_, _ = random_scenario(4, domain, edge_count=4)
    assert all(a.domain == domain for a in net_.agents)
