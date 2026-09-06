"""Frozen evaluation instances round-trip exactly."""
import numpy as np
import pytest

import benchmark
from conftest import ALL_DOMAINS
from rigidity import extended_bearing_rigidity_matrix as B_of


@pytest.fixture
def bench_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "DIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("domains", ["R^2", "R^3", "SE(3)",
                                     ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"]])
def test_instances_round_trip_exactly(domains, make_env, bench_dir):
    n = len(domains) if isinstance(domains, list) else 5
    env = make_env(n=n, domains=domains)
    benchmark.save(env, "b", instances=4, seed=0)
    nets, meta = benchmark.load("b")

    assert meta["instances"] == 4 and meta["n"] == n and len(nets) == 4
    for net in nets:
        assert [a.domain for a in net.agents] == (
            domains if isinstance(domains, list) else [domains] * n)

    # replaying the same seed must produce the same graphs
    np.random.seed(0)
    env.freeze_network = False
    for net in nets:
        env.reset()
        assert np.array_equal(env.network.edges, net.edges)
        for a, b in zip(env.network.agents, net.agents):
            assert np.allclose(a.pose.position, b.pose.position, atol=1e-12)
            assert np.allclose(a.pose.rotation_mat(), b.pose.rotation_mat(), atol=1e-12)


def test_rigidity_survives_the_round_trip(make_env, bench_dir):
    """The stored network must give the same rank as the one it came from."""
    env = make_env(n=6, domains=["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)", "R^2"])
    benchmark.save(env, "b", instances=3, seed=1)
    nets, _ = benchmark.load("b")

    np.random.seed(1)
    env.freeze_network = False
    for net in nets:
        env.reset()
        assert np.linalg.matrix_rank(B_of(env.network)) == np.linalg.matrix_rank(B_of(net))
        assert (np.linalg.matrix_rank(B_of(env.network.fully_connected()))
                == np.linalg.matrix_rank(B_of(net.fully_connected())))


def test_rotation_axes_survive_reset_and_the_round_trip(make_env, bench_dir):
    """env.rotation_axes is carried like env.domains; set_domain would reset it to e3."""
    axis = np.array([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)
    env = make_env(n=4, domains="R^3xS^1")
    env.rotation_axes = [axis] * 4                       # what a scenario file supplies

    env.reset()
    for a in env.network.agents:
        assert np.allclose(a.rotation_axis, axis)

    benchmark.save(env, "b", instances=2, seed=0)
    nets, _ = benchmark.load("b")
    for net in nets:
        for a in net.agents:
            assert a.rotation_axis is not None
            assert np.allclose(a.rotation_axis, axis)


def test_frameless_domains_store_a_null_axis(make_env, bench_dir):
    env = make_env(n=4, domains="R^3")
    benchmark.save(env, "b", instances=2, seed=0)
    nets, _ = benchmark.load("b")
    assert all(a.rotation_axis is None for net in nets for a in net.agents)


def test_digest_changes_with_content(make_env, bench_dir):
    env = make_env(n=4, domains="R^3")
    benchmark.save(env, "b", instances=2, seed=0)
    first = benchmark.digest("b")
    benchmark.save(env, "b", instances=2, seed=1)
    assert benchmark.digest("b") != first


def test_missing_benchmark_says_how_to_make_one(bench_dir):
    with pytest.raises(FileNotFoundError, match="uv run benchmark.py"):
        benchmark.load("does_not_exist")


def test_random_domain_instances_keep_their_own_mix(make_env, bench_dir):
    """One mix stamped on every instance would pair poses with somebody else's domains."""
    env = make_env(n=6, domains="R^3", random_domains=True)
    benchmark.save(env, "rd", instances=6, seed=0)
    nets, _ = benchmark.load("rd")

    mixes = [tuple(a.domain for a in net.agents) for net in nets]
    assert len(set(mixes)) > 1
    for net in nets:
        for a in net.agents:
            if a.domain in ("R^2", "R^2xS^1"):
                assert abs(a.pose.position[2]) < 1e-12
            assert (a.rotation_axis is None) == (a.domain != "R^3xS^1")

    benchmark.rotate("rd", "rd_rot", seed=1)
    rotated, _ = benchmark.load("rd_rot")
    assert [tuple(a.domain for a in net.agents) for net in rotated] == mixes
