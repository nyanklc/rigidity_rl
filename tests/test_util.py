"""Geometry helpers in util.py."""
import numpy as np
import pytest

from util import (Pose, skew_symmetric, orthogonal_projection_matrix,
                  sample_gaussian, adj_to_edge_index)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_skew_symmetric_is_antisymmetric_and_is_the_cross_product(seed):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=3), rng.normal(size=3)
    S = skew_symmetric(a)
    assert np.allclose(S, -S.T)
    assert np.allclose(S @ b, np.cross(a, b))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_orthogonal_projection_matrix_properties(seed):
    """P = I - x x^T for unit x: symmetric, idempotent, kills x, rank 2."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=3)
    x /= np.linalg.norm(x)
    P = orthogonal_projection_matrix(x)
    assert np.allclose(P, P.T)
    assert np.allclose(P @ P, P)
    assert np.allclose(P @ x, 0.0)
    assert np.linalg.matrix_rank(P) == 2


def test_pose_euler_rotation_matrix_round_trip():
    for euler in ([0.0, 0.0, 0.0], [0.3, -0.7, 1.1]):
        p = Pose(np.zeros(3), np.array(euler))
        R = p.rotation_mat()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_pose_set_rotation_mat_round_trip():
    p = Pose(np.zeros(3), np.array([0.2, 0.4, -0.6]))
    R = p.rotation_mat()
    p.set_rotation_mat(R)
    assert np.allclose(p.rotation_mat(), R, atol=1e-10)


def test_sample_gaussian_shape_and_centre():
    vals = [float(sample_gaussian(10.0, 4.0, 8).item()) for _ in range(500)]
    assert abs(np.mean(vals) - 10.0) < 1.5


def test_adj_to_edge_index_matches_dense():
    adj = np.zeros((4, 4), dtype=bool)
    adj[0, 1] = adj[2, 3] = adj[3, 0] = True
    ei = np.asarray(adj_to_edge_index(adj))
    got = {(int(i), int(j)) for i, j in zip(ei[0], ei[1])}
    assert got == {(0, 1), (2, 3), (3, 0)}
