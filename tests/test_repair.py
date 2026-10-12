"""The repair harness: membership changes, damage, enumeration and the metrics.

Nothing here reads `train/` or `models/`; models are built from scratch so the fast
suite passes on a clean clone.
"""
import copy

import numpy as np
import pytest
import torch
from gymnasium import spaces

from conftest import ALL_DOMAINS
import rigidity as R
from network import Agent
from scenario import random_scenario
from util import Pose


# ------------------------------------------------------------ shared helpers

def rigid_network(n, domains, seed=0, edge_count=0):
    """A rigid graph on random poses, and its rank_K."""
    np.random.seed(seed)
    net, _ = random_scenario(n, domains, edge_count=edge_count)
    rank_K = int(np.linalg.matrix_rank(
        R.extended_bearing_rigidity_matrix(net.fully_connected())))
    R.greedy_rigid_repair(net, rank_K, rng=np.random.default_rng(seed))
    return net, rank_K


# ------------------------------------------------------------ de-duplication

@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_shape_error_of_matches_the_expression_it_replaced(domain):
    net, rank_K = rigid_network(6, [domain] * 6)
    a_opt, _, _ = R.estimation_error_of(net, rank_K)
    assert R.shape_error_of(net, rank_K) == float(np.sqrt(a_opt / net.n))


def test_absent_pairs_is_the_complement_of_the_edges():
    net, _ = rigid_network(6, ["R^3"] * 6)
    absent = set(R.absent_pairs(net))
    present = set(net.get_edge_list())
    everything = {(i, j) for i in range(6) for j in range(6) if i != j}
    assert absent | present == everything and not (absent & present)
    assert not any(i == j for i, j in absent)


# ------------------------------------------------------------ membership on Network

def test_remove_agent_deletes_the_row_and_the_column():
    net, _ = rigid_network(6, ["R^2", "R^3", "SE(3)", "R^3xS^1", "R^2xS^1", "R^3"])
    before, domains = net.edges.copy(), [a.domain for a in net.agents]
    net.remove_agent(2)
    keep = [0, 1, 3, 4, 5]
    assert net.n == 5 and len(net.agents) == 5
    assert net.edges.shape == (5, 5) and net.nr_max_edges == 25
    assert np.array_equal(net.edges, before[np.ix_(keep, keep)])
    assert [a.domain for a in net.agents] == [domains[i] for i in keep]


def test_add_agent_appends_an_isolated_node_and_moves_nothing():
    net, _ = rigid_network(6, ["R^3"] * 6)
    before = net.edges.copy()
    agent = Agent(Pose(np.array([0.3, -0.2, 0.5]), np.zeros(3)))
    agent.set_domain("SE(3)")
    idx = net.add_agent(agent)
    assert idx == 6 and net.n == 7 and net.nr_max_edges == 49
    assert np.array_equal(net.edges[:6, :6], before)
    assert net.edges[idx].sum() == 0 and net.edges[:, idx].sum() == 0


def test_remove_agent_rejects_an_index_that_is_not_there():
    net, _ = rigid_network(5, ["R^3"] * 5)
    with pytest.raises(IndexError):
        net.remove_agent(5)


# ------------------------------------------------------------ Environment resync

MEMBERSHIP_FLAGS = dict(rigidity_global=True, rigidity_quality=True, rigidity_flex=True,
                        rigidity_edge=True, rigidity_stiffness=True,
                        rigidity_removal=True, stiffness_kappa=2.0,
                        graph_features=False)


def test_resync_matches_a_freshly_built_environment(make_env):
    """A membership change must leave the env indistinguishable from one built at the
    new n, or the policy is handed spaces describing the network it used to have."""
    domains = ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)", "R^2", "R^3"]
    e = make_env(n=len(domains), domains=domains, **MEMBERSHIP_FLAGS)
    e.reset()
    e.freeze_network = True

    e.network.remove_agent(3)                       # drop the R^3xS^1 agent
    obs, _ = e.resync_after_membership_change(seed=0)
    expected = [d for k, d in enumerate(domains) if k != 3]

    fresh = make_env(n=len(expected), domains=expected, **MEMBERSHIP_FLAGS)
    fresh.reset()

    assert e.n == fresh.n == len(expected)
    assert e.domains == expected
    assert e.rotation_axes == [a.rotation_axis for a in e.network.agents]
    assert e.network.n == e.n and e.network.nr_max_edges == e.n ** 2
    assert e.action_space.n == fresh.action_space.n
    assert e.rank_K == fresh.rank_K and e.m_req == fresh.m_req and e.c_max == fresh.c_max
    for key in fresh.observation_space.spaces:
        assert e.observation_space[key].shape == fresh.observation_space[key].shape, key
        assert e.observation_space[key].shape == np.shape(obs[key]), key


def test_a_leave_join_round_trip_restores_the_environment(make_env):
    domains = ["R^2", "R^3", "SE(3)", "R^3xS^1", "R^2xS^1", "R^3"]
    e = make_env(n=len(domains), domains=domains, **MEMBERSHIP_FLAGS)
    e.reset()
    e.freeze_network = True
    rank_K, m_req, width = e.rank_K, e.m_req, e.action_space.n

    pose = copy.deepcopy(e.network.agents[3].pose)
    e.network.remove_agent(3)
    e.resync_after_membership_change()
    assert e.n == len(domains) - 1

    agent = Agent(pose)
    agent.set_domain("R^3xS^1")
    e.network.add_agent(agent)
    e.resync_after_membership_change()

    assert e.n == len(domains)
    assert (e.rank_K, e.m_req, e.action_space.n) == (rank_K, m_req, width)


def test_the_environment_steps_after_a_membership_change(make_env):
    e = make_env(n=7, domains=["R^3"] * 7, **MEMBERSHIP_FLAGS)
    e.reset()
    e.freeze_network = True
    e.network.remove_agent(0)
    e.resync_after_membership_change(seed=0)
    for _ in range(3):
        obs, _, _, _, _ = e.step(e.action_space.sample())
    for key in e.observation_space.spaces:
        assert e.observation_space[key].shape == np.shape(obs[key]), key


# ------------------------------------------------------------ the variable-n path

def spaces_at(n, node_dim=14, edge_dim=12):
    return spaces.Dict({
        "node_features": spaces.Box(-1, 1, (n, node_dim)),
        "coord_features": spaces.Box(-1, 1, (n, 3)),
        "edge_features": spaces.Box(-1, 1, (n, n, edge_dim)),
        "adj": spaces.Box(0, 1, (n, n)),
        "selection": spaces.Box(0, 1, (n,), dtype=int),
    })


def q_network_at(n, action_type, node_dim=14, edge_dim=12, backbone="GINE"):
    from policy import build_models
    width = 2 * n * n - 2 * n + 1 if "AddRemove" in action_type else n + 1
    return build_models("DQN", backbone=backbone, action_type=action_type, n=n,
                        node_feat_dim=node_dim, edge_feat_dim=edge_dim,
                        gnn_hidden_dim=32, head_hidden_dim=32,
                        observation_space=spaces_at(n, node_dim, edge_dim),
                        action_space=spaces.Discrete(width),
                        device="cpu", allow_skip=False)["q_network"]


LIVE_ACTIONS = ["AddRemoveEdgeDiscreteNoSelfLoops", "SelectNodesSequentially"]


@pytest.mark.parametrize("action_type", LIVE_ACTIONS)
@pytest.mark.parametrize("backbone", ["GINE", "Equivariant"])
def test_weights_do_not_depend_on_n(action_type, backbone):
    """Node join and leave rest on this: the parameters must be n-free, so the same
    weights can be rebuilt at another n."""
    a = q_network_at(8, action_type, backbone=backbone)
    b = q_network_at(11, action_type, backbone=backbone)
    sa = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    sb = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    assert sa == sb
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=True)
    assert not missing and not unexpected


@pytest.mark.parametrize("action_type", LIVE_ACTIONS)
@pytest.mark.parametrize("n", [7, 9, 12])
def test_the_action_width_follows_the_observation_space(action_type, n):
    """compute() reads n off node_features, which comes from the model's OWN
    observation_space. Hand it data at another n without replacing that space and it
    returns a wrongly-sized vector over misaligned inputs, with no exception -- which
    is why the harness must rebuild the space, not just the action space."""
    from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space

    src = q_network_at(8, action_type)
    model = q_network_at(n, action_type)
    model.load_state_dict(src.state_dict())

    space = spaces_at(n)
    flat = flatten_tensorized_space(
        tensorize_space(space, space.sample(), device="cpu")).float()
    with torch.no_grad():
        q, _ = model.compute({"observations": flat}, role="q_network")
    expected = 2 * n * n - 2 * n + 1 if "AddRemove" in action_type else n + 1
    assert q.shape[-1] == expected


def test_a_stale_observation_space_is_silently_wrong():
    """The failure this guards against. Documented, not tolerated: the harness must
    replace model.observation_space, and this pins that doing so is what fixes it."""
    from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space

    model = q_network_at(8, "AddRemoveEdgeDiscreteNoSelfLoops")
    space = spaces_at(9)
    flat = flatten_tensorized_space(
        tensorize_space(space, space.sample(), device="cpu")).float()
    with torch.no_grad():
        stale, _ = model.compute({"observations": flat}, role="q_network")
    assert stale.shape[-1] == 2 * 8 * 8 - 2 * 8 + 1        # the 8-agent width, on 9-agent data

    model.observation_space = space
    with torch.no_grad():
        fixed, _ = model.compute({"observations": flat}, role="q_network")
    assert fixed.shape[-1] == 2 * 9 * 9 - 2 * 9 + 1


@pytest.mark.parametrize("action_type", LIVE_ACTIONS)
def test_a_cpu_model_holds_no_cuda_tensors(action_type):
    model = q_network_at(8, action_type)
    off = [k for k, v in model.state_dict().items() if v.device.type != "cpu"]
    assert not off, off


def test_tensorize_space_needs_an_explicit_device():
    """skrl's default is None, which resolves to cuda when one is present, so an
    observation built without `device=` lands on a different device from a cpu model."""
    from skrl.utils.spaces.torch import tensorize_space
    space = spaces_at(6)
    assert tensorize_space(space, space.sample(), device="cpu")["adj"].device.type == "cpu"


# ------------------------------------------------------------ convergence

@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_converge_reaches_a_minimal_rigid_graph(domain):
    """Every method is damaged from the same converged graph, so it has to be rigid
    and it has to be minimal -- otherwise a 'repair' is repairing slack."""
    import repair
    np.random.seed(0)
    net, _ = random_scenario(6, [domain] * 6, edge_count=0)
    rank_K = int(np.linalg.matrix_rank(
        R.extended_bearing_rigidity_matrix(net.fully_connected())))
    repair.converge(net, rank_K, np.random.default_rng(0))
    assert repair._is_rigid(net, rank_K)
    for i, j in net.get_edge_list():                # nothing left to prune
        net.edges[i, j] = False
        assert not repair._is_rigid(net, rank_K), f"{domain}: {(i, j)} was redundant"
        net.edges[i, j] = True


def test_converge_with_slack_leaves_exactly_that_many_spare_edges():
    import repair
    net, rank_K = rigid_network(7, ["R^3"] * 7)
    repair.converge(net, rank_K, np.random.default_rng(0), slack=0)
    minimal = int(net.edges.sum())
    net2, _ = rigid_network(7, ["R^3"] * 7)
    repair.converge(net2, rank_K, np.random.default_rng(0), slack=3)
    assert int(net2.edges.sum()) == minimal + 3


# ------------------------------------------------------------ events

def converged(n=7, domains=None, slack=0, seed=0):
    import repair
    domains = domains or ["R^3"] * n
    np.random.seed(seed)
    net, _ = random_scenario(n, domains, edge_count=0)
    rank_K = int(np.linalg.matrix_rank(
        R.extended_bearing_rigidity_matrix(net.fully_connected())))
    repair.converge(net, rank_K, np.random.default_rng(seed), slack=slack)
    return net, rank_K


def test_none_changes_nothing():
    import repair
    net, rank_K = converged()
    before = net.edges.copy()
    rec = repair.apply_event(net, "none", 0, rank_K, repair.event_rng(0, 0, "none", 0))
    assert np.array_equal(net.edges, before) and rec["skipped"] is None
    assert R.repair_edge_count(net, rank_K=rank_K) == 0


@pytest.mark.parametrize("event", ["edge_drop_random", "edge_drop_critical",
                                   "edge_drop_adversarial"])
def test_edge_drops_remove_exactly_k_edges(event):
    import repair
    net, rank_K = converged()
    m0 = int(net.edges.sum())
    rec = repair.apply_event(net, event, 2, rank_K, repair.event_rng(0, 0, event, 2))
    assert rec["skipped"] is None
    assert int(net.edges.sum()) == m0 - 2 and len(rec["dropped"]) == 2
    # a minimal graph has no redundant edge, so any drop breaks it
    assert not repair._is_rigid(net, rank_K)
    assert R.repair_edge_count(net, rank_K=rank_K) > 0


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_redundant_drop_keeps_the_graph_rigid(domain):
    """The contract of the event. Redundancy is not additive, so the edges have to be
    taken one at a time with the costs recomputed, or two individually-spare edges can
    be jointly load bearing."""
    import repair
    net, rank_K = converged(6, [domain] * 6, slack=2)
    rec = repair.apply_event(net, "edge_drop_redundant", 2, rank_K,
                             repair.event_rng(0, 0, "edge_drop_redundant", 2))
    if rec["skipped"]:
        pytest.skip(rec["skipped"])
    assert repair._is_rigid(net, rank_K)
    assert R.repair_edge_count(net, rank_K=rank_K) == 0
    assert len(rec["dropped"]) == 2


def test_redundant_drop_skips_a_minimal_graph_rather_than_falling_back():
    import repair
    net, rank_K = converged(slack=0)
    before = net.edges.copy()
    rec = repair.apply_event(net, "edge_drop_redundant", 2, rank_K,
                             repair.event_rng(0, 0, "edge_drop_redundant", 2))
    assert rec["skipped"] and np.array_equal(net.edges, before)


def test_spurious_edges_are_added_and_rigidity_survives():
    import repair
    net, rank_K = converged()
    m0 = int(net.edges.sum())
    rec = repair.apply_event(net, "edge_add_spurious", 3, rank_K,
                             repair.event_rng(0, 0, "edge_add_spurious", 3))
    assert int(net.edges.sum()) == m0 + 3 and len(rec["added"]) == 3
    assert repair._is_rigid(net, rank_K)


def test_node_leave_drops_the_agent_and_maps_the_indices():
    import repair
    net, rank_K = converged()
    n0 = net.n
    rec = repair.apply_event(net, "node_leave", 0, rank_K,
                             repair.event_rng(0, 0, "node_leave", 0))
    v = rec["node_removed"]
    assert net.n == n0 - 1 and v is not None
    assert set(rec["index_map"]) == set(range(n0)) - {v}
    assert sorted(rec["index_map"].values()) == list(range(n0 - 1))
    assert net.edges.shape == (n0 - 1, n0 - 1)


def test_node_join_adds_an_isolated_agent_that_needs_edges():
    import repair
    net, rank_K = converged()
    n0, m0 = net.n, int(net.edges.sum())
    rec = repair.apply_event(net, "node_join", 0, rank_K,
                             repair.event_rng(0, 0, "node_join", 0))
    idx = rec["node_added"]
    assert net.n == n0 + 1 and int(net.edges.sum()) == m0
    assert net.edges[idx].sum() == 0 and net.edges[:, idx].sum() == 0
    assert rec["join_domain"] in repair.DOMAINS
    new_rank_K = int(np.linalg.matrix_rank(
        R.extended_bearing_rigidity_matrix(net.fully_connected())))
    assert R.repair_edge_count(net, rank_K=new_rank_K) > 0     # an isolated node is flexible


def test_pose_drift_moves_the_poses_and_keeps_planar_agents_in_plane():
    import repair
    domains = ["R^2", "R^2xS^1", "R^3", "SE(3)", "R^3xS^1", "R^2"]
    net, rank_K = converged(6, domains)
    before = net.get_position_features().copy()
    edges = net.edges.copy()
    repair.apply_event(net, "pose_drift", 0, rank_K,
                       repair.event_rng(0, 0, "pose_drift", 0), drift_sigma=0.2)
    after = net.get_position_features()
    assert np.array_equal(net.edges, edges)                    # topology untouched
    assert not np.allclose(after, before)
    for agent in net.agents:
        if agent.domain in ("R^2", "R^2xS^1"):
            assert abs(agent.pose.position[2]) < 1e-12


def test_damage_depends_only_on_the_instance_not_on_a_stream():
    """A shared generator would make the damage depend on which methods are run."""
    import repair
    out = []
    for _ in range(2):
        net, rank_K = converged()
        rec = repair.apply_event(net, "edge_drop_random", 2, rank_K,
                                 repair.event_rng(7, 3, "edge_drop_random", 2))
        out.append(sorted(rec["dropped"]))
    assert out[0] == out[1]

    net, rank_K = converged()
    other = repair.apply_event(net, "edge_drop_random", 2, rank_K,
                               repair.event_rng(7, 4, "edge_drop_random", 2))
    assert sorted(other["dropped"]) != out[0]


def test_an_unknown_event_raises():
    import repair
    net, rank_K = converged()
    with pytest.raises(ValueError):
        repair.apply_event(net, "meteor_strike", 1, rank_K, np.random.default_rng(0))


# ------------------------------------------------------------ enumeration

def test_every_enumerated_repair_is_rigid_distinct_and_the_right_size():
    import repair
    net, rank_K = converged(6, ["R^3"] * 6)
    repair.apply_event(net, "edge_drop_random", 2, rank_K,
                       repair.event_rng(0, 0, "edge_drop_random", 2))
    size = R.repair_edge_count(net, rank_K=rank_K)
    pop = repair.enumerate_repairs(net, rank_K, size)
    assert pop and len({sub for sub, _ in pop}) == len(pop)
    m0 = int(net.edges.sum())
    for sub, err in pop:
        assert len(sub) == size
        work = copy.deepcopy(net)
        for i, j in sub:
            work.edges[i, j] = True
        assert repair._is_rigid(work, rank_K)
        assert int(work.edges.sum()) == m0 + size
        assert np.isfinite(err) and err > 0


def test_the_size_agrees_with_an_exhaustive_minimum():
    """repair_edge_count is a bound; here it must also be the true minimum, or the
    population is not the population of MINIMUM repairs."""
    import repair
    from tools.repair_bound import true_minimum
    net, rank_K = converged(5, ["R^3"] * 5)
    repair.apply_event(net, "edge_drop_random", 2, rank_K,
                       repair.event_rng(0, 0, "edge_drop_random", 2))
    assert R.repair_edge_count(net, rank_K=rank_K) == true_minimum(net, rank_K, cap=4)


def test_enumeration_refuses_rather_than_truncating():
    """A truncated enumeration is lexicographically biased and a percentile against it
    is not a percentile, so over the cap it returns None."""
    import repair
    net, rank_K = converged(8, ["R^3"] * 8)
    repair.apply_event(net, "edge_drop_random", 2, rank_K,
                       repair.event_rng(0, 0, "edge_drop_random", 2))
    size = R.repair_edge_count(net, rank_K=rank_K)
    assert repair.enumerate_repairs(net, rank_K, size, cap=1) is None
    pop, tier = repair.repair_population(net, rank_K, size,
                                         np.random.default_rng(0), cap=1, want=25)
    assert tier == "sampled" and 0 < len(pop) <= 25


def test_sampled_repairs_are_valid_and_distinct():
    import repair
    net, rank_K = converged(8, ["R^3"] * 8)
    repair.apply_event(net, "edge_drop_random", 2, rank_K,
                       repair.event_rng(0, 0, "edge_drop_random", 2))
    size = R.repair_edge_count(net, rank_K=rank_K)
    pop = repair.sample_repairs(net, rank_K, size, np.random.default_rng(0), want=20)
    assert pop and len({sub for sub, _ in pop}) == len(pop)
    for sub, _ in pop:
        work = copy.deepcopy(net)
        for i, j in sub:
            work.edges[i, j] = True
        assert repair._is_rigid(work, rank_K)


# ------------------------------------------------------------ metrics

def test_the_percentile_puts_the_best_at_the_top_and_the_median_at_fifty():
    import repair
    errors = [1.0, 2.0, 3.0, 4.0, 5.0]
    # midrank, so the best of five distinct values is 90 rather than 100; `is_best` is
    # what says a choice was actually optimal
    assert repair.quality_percentile(errors, 1.0) == 90.0
    assert repair.quality_percentile(errors, 5.0) == 10.0
    assert repair.quality_percentile(errors, 3.0) == 50.0
    assert repair.is_best_repair(errors, 1.0) is True
    assert repair.is_best_repair(errors, 2.0) is False
    assert repair.quality_percentile(errors, None) is None
    assert repair.quality_percentile(errors, np.inf) is None


def test_ties_are_midranked_so_an_optimal_choice_is_not_sunk():
    import repair
    errors = [1.0, 1.0, 1.0, 5.0]
    assert repair.is_best_repair(errors, 1.0) is True
    assert repair.quality_percentile(errors, 1.0) == pytest.approx(62.5)


def test_edit_metrics_count_the_right_things():
    import repair
    n = 5
    pre = np.zeros((n, n), dtype=bool)
    for e in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        pre[e] = True
    brk = pre.copy()
    brk[0, 1] = False                                    # the event dropped 0->1
    post = brk.copy()
    post[0, 2] = True                                    # a different edge added
    post[3, 4] = False                                   # and a survivor removed
    dmg = {"index_map": {i: i for i in range(n)}}
    m = repair.edit_metrics(pre, brk, post, dmg)
    assert m["added"] == 1 and m["removed"] == 1
    assert m["churn"] == 2
    assert m["collateral"] == 1                          # (3,4) survived and was cut
    assert m["restoration"] == 0.0                       # (0,1) was not put back


def test_edit_metrics_map_the_indices_through_a_node_removal():
    """Without the map, every locality metric on a membership event compares two
    different labellings of the agents. Restoration is undefined there, since an edge
    to an agent that left cannot be put back."""
    import repair
    pre = np.zeros((4, 4), dtype=bool)
    for e in [(0, 1), (1, 2), (2, 3)]:
        pre[e] = True
    # agent 1 leaves: old 0->0, 2->1, 3->2, and edges touching 1 vanish
    brk = np.zeros((3, 3), dtype=bool)
    brk[(1, 2)] = True                                   # old (2,3)
    post = brk.copy()
    post[(0, 1)] = True                                  # a new edge
    dmg = {"index_map": {0: 0, 2: 1, 3: 2}}
    m = repair.edit_metrics(pre, brk, post, dmg)
    assert m["added"] == 1 and m["removed"] == 0 and m["collateral"] == 0
    assert m["restoration"] is None                      # (0,1),(1,2) died with the node


# ------------------------------------------------------------ measure, end to end

def test_measure_restores_the_graph_between_methods(make_env):
    """greedy_rigid_repair mutates its argument, so every runner has to work on a copy
    or the second method repairs the first one's answer."""
    import repair
    e = make_env(n=6, domains=["R^3"] * 6, stiffness_kappa=2.0, rigidity_global=True)
    nets = [rigid_network(6, ["R^3"] * 6, seed=s)[0] for s in (1, 2)]
    stored = [n.edges.copy() for n in nets]
    repair.measure(env=e, instances=nets, events=["edge_drop_random"], ks=(1,),
                   methods=("greedy_repair", "optimal_repair", "random_repair"), seed=0)
    for net, before in zip(nets, stored):
        assert np.array_equal(net.edges, before), "measure mutated the instance"


def test_optimal_repair_is_always_the_best_of_the_population(make_env):
    """The end-to-end invariant: if the harness ever mis-scores anything, this catches
    it, because optimal_repair is chosen as the argmin of the very population it is
    then placed against."""
    import repair
    e = make_env(n=6, domains=["R^3"] * 6, stiffness_kappa=2.0, rigidity_global=True)
    nets = [rigid_network(6, ["R^3"] * 6, seed=s)[0] for s in range(4)]
    rows, _ = repair.measure(env=e, instances=nets,
                             events=["edge_drop_random", "node_join"], ks=(2,),
                             methods=("greedy_repair", "optimal_repair"), seed=0)
    scored = [r for r in rows if r.get("method") == "optimal_repair"
              and r["pctile"] is not None]
    assert scored, "no scored optimal_repair row to check"
    for r in scored:
        assert r["is_best"] is True, r
        assert r["err_ratio_to_best"] == pytest.approx(1.0)
        assert r["added_over_bound"] == 0 and r["exact_size"] is True


def test_the_percentile_is_withheld_when_a_method_overspends(make_env):
    """More edges almost always means lower error, so a method that used more than the
    minimum is not in the enumerated population and must not be scored against it."""
    import repair
    e = make_env(n=6, domains=["R^3"] * 6, stiffness_kappa=2.0, rigidity_global=True)
    nets = [rigid_network(6, ["R^3"] * 6, seed=s)[0] for s in range(4)]
    rows, _ = repair.measure(env=e, instances=nets, events=["edge_drop_random"],
                             ks=(2,), methods=("optimal_repair", "random_repair"), seed=0)
    for r in rows:
        if r.get("method") is None or r["repair_bound"] == 0:
            continue
        if r["exact_size"] is False:
            assert r["pctile"] is None and r["is_best"] is None


def test_the_none_control_leaves_a_converged_graph_alone(make_env):
    """Nothing happened, so a repairer should do nothing. Anything else is a method
    that cannot leave a converged network be, which is what the control is for."""
    import repair
    e = make_env(n=6, domains=["R^3"] * 6, stiffness_kappa=2.0, rigidity_global=True)
    nets = [rigid_network(6, ["R^3"] * 6, seed=s)[0] for s in range(3)]
    rows, _ = repair.measure(env=e, instances=nets, events=["none"], ks=(0,),
                             methods=("greedy_repair", "random_repair"), seed=0)
    for r in rows:
        if r.get("method") is None:
            continue
        assert r["repaired"] and r["churn"] == 0 and r["repair_bound"] == 0
        assert r["err_ratio_to_pre"] == pytest.approx(1.0)


def test_every_row_carries_every_field_and_survives_a_csv_round_trip(make_env, tmp_path):
    import csv
    import repair
    e = make_env(n=6, domains=["R^3"] * 6, stiffness_kappa=2.0, rigidity_global=True)
    nets = [rigid_network(6, ["R^3"] * 6, seed=s)[0] for s in range(2)]
    rows, spread = repair.measure(env=e, instances=nets,
                                  events=["none", "edge_drop_random", "node_leave"],
                                  ks=(1,), methods=("greedy_repair", "greedy"), seed=0)
    assert rows
    for r in rows:
        assert set(r) == set(repair.ROW_FIELDS), set(r) ^ set(repair.ROW_FIELDS)

    path = tmp_path / "repair.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(repair.ROW_FIELDS))
        w.writeheader()
        w.writerows(rows)
    with open(path) as fh:
        back = list(csv.DictReader(fh))
    assert len(back) == len(rows)
    assert repair.format_table(rows, brief=True)
    assert repair.summarise(rows)


def test_enumeration_is_measurement_work_not_a_method_cost(make_env):
    """The population is instrumentation. Charging it to whichever method ran first
    would make the cost column meaningless."""
    import cost
    import repair
    net, rank_K = converged(6, ["R^3"] * 6)
    repair.apply_event(net, "edge_drop_random", 2, rank_K,
                       repair.event_rng(0, 0, "edge_drop_random", 2))
    size = R.repair_edge_count(net, rank_K=rank_K)
    with cost.Meter() as meter:
        repair.enumerate_repairs(net, rank_K, size)
    assert sum(meter.counts.get(k, 0) for k in cost.LEAVES) == 0
    assert meter.counts.get("_reporting", 0) > 0


def test_sync_observation_space_reaches_every_model():
    """The one assignment the variable-n path rests on. Without it a node event scores
    a wrongly-sized Q-vector over misaligned inputs and raises nothing."""
    import repair

    class FakeEnv:
        observation_space = spaces_at(11)

    class FakeAgent:
        def __init__(self):
            self.models = {"q_network": q_network_at(8, "AddRemoveEdgeDiscreteNoSelfLoops"),
                           "target_q_network": q_network_at(8, "AddRemoveEdgeDiscreteNoSelfLoops"),
                           "policy": None}

    agent = FakeAgent()
    assert agent.models["q_network"].observation_space["adj"].shape == (8, 8)
    repair.sync_observation_space(agent, FakeEnv())
    for key in ("q_network", "target_q_network"):
        assert agent.models[key].observation_space["adj"].shape == (11, 11)
    assert agent.models["policy"] is None            # a missing role must not raise


def test_install_resyncs_whenever_n_moves_not_just_on_a_membership_event(make_env):
    """After a node event the env's spaces stay at the new n, so installing a smaller
    network without resyncing hands the policy spaces for a network that is not there."""
    import repair
    e = make_env(n=7, domains=["R^3"] * 7, **MEMBERSHIP_FLAGS)
    e.reset()
    e.freeze_network = True

    big, _ = rigid_network(8, ["R^3"] * 8, seed=1)
    repair._install(e, big, membership=True)
    assert e.n == 8 and e.observation_space["adj"].shape == (8, 8)

    small, _ = rigid_network(6, ["R^3"] * 6, seed=2)
    repair._install(e, small)                        # membership NOT declared
    assert e.n == 6 and e.observation_space["adj"].shape == (6, 6)
    obs = e._get_obs()
    for key in e.observation_space.spaces:
        assert e.observation_space[key].shape == np.shape(obs[key]), key


def test_the_headline_rule_follows_the_event_class():
    import repair
    assert repair.headline_rule("edge_drop_random") == "first_rigid"
    assert repair.headline_rule("node_leave") == "first_rigid"
    assert repair.headline_rule("edge_drop_redundant") == "best_phi"
    assert repair.headline_rule("none") == "best_phi"


def test_a_policy_contributes_one_row_per_case_not_three():
    """A rollout reports three stopping rules; averaging them would produce a row that
    describes no protocol at all."""
    import repair
    rows = []
    for inst in (0, 1):
        for rule in ("final", "best_phi", "first_rigid"):
            rows.append(dict.fromkeys(repair.ROW_FIELDS) | {
                "instance": inst, "event": "edge_drop_random", "event_k": 2,
                "method": "policy", "stop_rule": rule, "repaired": True,
                "added": {"final": 9, "best_phi": 5, "first_rigid": 2}[rule]})
        rows.append(dict.fromkeys(repair.ROW_FIELDS) | {
            "instance": inst, "event": "edge_drop_random", "event_k": 2,
            "method": "greedy_repair", "stop_rule": "final", "repaired": True,
            "added": 2})
    picked = repair._headline(rows)
    assert len(picked) == 4                                  # 2 instances x 2 methods
    policy = [r for r in picked if r["method"] == "policy"]
    assert all(r["stop_rule"] == "first_rigid" and r["added"] == 2 for r in policy)
    summary = {s["method"]: s for s in repair.summarise(rows)}
    assert summary["policy"]["cases"] == 2 and summary["policy"]["added"] == 2.0


def test_a_non_breaking_event_reads_the_best_state_visited():
    """first_rigid is step 0 where rigidity survived, so it would say the method did
    nothing whatever it did."""
    import repair
    rows = [dict.fromkeys(repair.ROW_FIELDS) | {
        "instance": 0, "event": "none", "event_k": 0, "method": "policy",
        "stop_rule": rule, "repaired": True, "churn": churn}
        for rule, churn in (("final", 9), ("best_phi", 4), ("first_rigid", 0))]
    picked = repair._headline(rows)
    assert len(picked) == 1 and picked[0]["stop_rule"] == "best_phi"
    assert picked[0]["churn"] == 4
