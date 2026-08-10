"""scenario.py and the initial-graph sampler."""
import numpy as np
import pytest

from conftest import ALL_DOMAINS
from scenario import random_scenario, save_scenario, load_scenario, randomize_scenario


def test_random_scenario_honours_the_edge_count():
    for m in (0, 1, 7, 12):
        net, _ = random_scenario(5, ["R^3"] * 5, edge_count=m)
        assert int(net.edges.sum()) == m


def test_random_scenario_never_makes_self_loops():
    for _ in range(20):
        net, _ = random_scenario(6, ["R^3"] * 6, edge_count=15)
        assert not np.any(np.diag(net.edges))


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_random_scenario_assigns_the_requested_domains(domain):
    net, _ = random_scenario(5, [domain] * 5, edge_count=5)
    assert all(a.domain == domain for a in net.agents)


def test_planar_domains_stay_in_the_plane():
    for domain in ("R^2", "R^2xS^1"):
        net, _ = random_scenario(6, [domain] * 6, edge_count=6)
        assert np.allclose(net.get_position_features()[:, 2], 0.0)


def test_save_load_round_trip(tmp_path):
    net, _ = random_scenario(5, ["R^3", "SE(3)", "R^2", "R^3", "SE(3)"], edge_count=8)
    goal, _ = random_scenario(5, ["R^3"] * 5, edge_count=1)
    path = str(tmp_path / "s.json")
    save_scenario(path, net, goal)
    back, _ = load_scenario(path)
    assert np.allclose(back.get_position_features(), net.get_position_features())
    assert np.array_equal(back.edges, net.edges)
    assert [a.domain for a in back.agents] == [a.domain for a in net.agents]


def test_randomize_scenario_keeps_the_domain_mix(tmp_path):
    doms = ["R^3", "SE(3)", "R^2", "SE(3)"]
    net, _ = random_scenario(4, doms, edge_count=5)
    path = str(tmp_path / "s.json")
    save_scenario(path, net, net)
    fresh, _ = randomize_scenario(path)
    assert [a.domain for a in fresh.agents] == doms
    assert not np.allclose(fresh.get_position_features(), net.get_position_features())


@pytest.mark.parametrize("domain,expected_m_req", [
    ("R^3", 10), ("R^2", 13), ("SE(3)", 21), ("R^2xS^1", 20), ("R^3xS^1", 14)])
def test_initial_edge_count_is_centred_on_m_req_per_domain(make_env, domain, expected_m_req):
    """SE(3) must not be seeded with the R^d closed form -- it needs 21 edges, not 10.

    The discriminating assertion is the relative one: an absolute tolerance wide
    enough for the sampler's spread is also wide enough to hide the bug.
    """
    from rigidity import MBR_required_Rd
    n = 8
    e = make_env(n=n, domains=[domain] * n, random_graph_with_mean_min_edges=True)
    counts = []
    for _ in range(150):
        e.reset()
        counts.append(int(e.network.edges.sum()))
    mean = float(np.mean(counts))
    assert e.m_req == expected_m_req
    closed_form = MBR_required_Rd(n, 2 if domain in ("R^2", "R^2xS^1") else 3)
    if closed_form != e.m_req:
        assert abs(mean - e.m_req) < abs(mean - closed_form), (
            f"{domain}: mean {mean:.1f} is nearer the R^d closed form "
            f"({closed_form}) than m_req ({e.m_req})")
    assert abs(mean - e.m_req) < 0.3 * e.m_req + 2


def test_initial_edge_count_stays_within_bounds(make_env):
    n = 6
    e = make_env(n=n, domains="R^3", random_graph_with_mean_min_edges=True)
    for _ in range(50):
        e.reset()
        assert 1 <= int(e.network.edges.sum()) <= n * n - n


def test_network_transforms_preserve_the_edge_set():
    net, _ = random_scenario(6, ["R^3"] * 6, edge_count=10)
    edges = net.edges.copy()
    net.translate_network([1.0, 2.0, 3.0])
    net.rotate_network([0, 0, 1], 0.5)
    net.scale_network(2.0)
    assert np.array_equal(net.edges, edges)


def test_scale_network_scales_about_the_centroid():
    """It used to raise AttributeError (.x/.y/.z on a numpy array)."""
    net, _ = random_scenario(6, ["R^3"] * 6, edge_count=6)
    p0 = net.get_position_features().copy()
    c0 = p0.mean(axis=0)
    net.scale_network(3.0)
    p1 = net.get_position_features()
    assert np.allclose(p1.mean(axis=0), c0)
    assert np.allclose(p1 - c0, 3.0 * (p0 - c0))
