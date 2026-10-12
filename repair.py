"""What happens to a network after something goes wrong, and how well each method fixes it.

`THEORY.md` §19 settles the *count*: `repair_edge_count` is a proven lower bound on the
edges needed to restore rigidity, and greedy attains it wherever `c_max = 1`. §19.4 says
the count is not where the difficulty is -- equally-sized minimum repairs differ by 4x-17x
in the shape error they leave, and marginal-rank greedy lands mid-distribution. So the
question this module asks is not "did it repair" but **which repair did it pick**, scored
as a percentile among every repair of the same size.

Two classes of event, because the metrics differ:

  rank-breaking     edge_drop_random / _critical / _adversarial, node_leave, node_join
                    -> the count and the percentile are defined
  non-breaking      edge_drop_redundant, edge_add_spurious, pose_drift, none
                    -> `repair_edge_count` is 0, so quality and locality carry it

The library, not the script: `experiments/repair.py` is what runs it.
"""
import copy
import itertools
import json
import math
import time

import numpy as np

import cost
from network import Agent
from rigidity import (absent_pairs, extended_bearing_rigidity_matrix, greedy_rigid_repair,
                      nullspace_and_softest, removal_costs, repair_edge_count,
                      rigidity_decomposition, shape_error_of, characteristic_length)
from util import Pose

# Rank-breaking events; for the rest `repair_edge_count` is 0 by construction and the
# COUNT family says nothing.
EVENTS = ("none", "edge_drop_random", "edge_drop_critical", "edge_drop_adversarial",
          "edge_drop_redundant", "edge_add_spurious", "node_leave", "node_join",
          "pose_drift")
BREAKING = frozenset({"edge_drop_random", "edge_drop_critical", "edge_drop_adversarial",
                      "node_leave", "node_join"})
MEMBERSHIP = frozenset({"node_leave", "node_join"})

DOMAINS = ("R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)")

# Enumerating every repair is only a distribution if it is complete, so the tier is
# decided from C(|absent|, size) BEFORE enumerating rather than by truncating.
ENUMERATION_CAP = 200_000
SAMPLE_WANT = 400
SAMPLE_MAX_DRAWS = 50_000
# a percentile over fewer than this many repairs is 0/50/100 and says nothing
MIN_POPULATION = 8


# ---------------------------------------------------------------- converge

def converge(net, rank_K, rng, slack=0, prune=True):
    """Bring a sampled instance to the converged state the event is applied to.

    A benchmark stores the graph the sampler drew, which is neither rigid nor sparse,
    so damaging it would measure repair from an arbitrary graph. Every method faces the
    same converged graph, which is what makes the comparison paired -- and it is a
    greedy construction, the same one THEORY 19.4 breaks, so the numbers are comparable
    to that table.

    `slack` adds redundant edges on top. edge_drop_redundant needs them by definition:
    a minimal graph has no redundant edge, and the honest reading of that is that a
    converged network usually has none either.
    """
    from rigidity import greedy_rigid_construction
    greedy_rigid_construction(net, rank_K, rng)
    # A single random-order construction lands a couple of edges above m_req, so prune
    # back to minimality before anything is damaged. Pruning by "does it stay rigid",
    # in random order, rather than by phi or by marginal rank: either of those would
    # make the pre-event graph one particular method's own fixed point.
    if prune:
        for i, j in [tuple(e) for e in rng.permutation(np.array(net.get_edge_list()))]:
            net.edges[i, j] = False
            if not _is_rigid(net, rank_K):
                net.edges[i, j] = True
    for _ in range(slack):
        absent = absent_pairs(net)
        if not absent:
            break
        net.edges[absent[int(rng.integers(len(absent)))]] = True
    return net


# ---------------------------------------------------------------- damage

def event_rng(seed, index, event, k):
    """Private to the damage, and a function of the instance rather than of a stream.

    A shared generator would make the damage depend on which methods are being run,
    which is the failure baselines.py's per-method generators exist to prevent.
    """
    return np.random.default_rng([int(seed), int(index), EVENTS.index(event), int(k)])


def _identity_map(n):
    return {i: i for i in range(n)}


def _drop(net, pairs):
    for i, j in pairs:
        net.edges[i, j] = False


def _apply(net, pairs):
    for i, j in pairs:
        net.edges[i, j] = True


def _edge_costs(net, rank_K):
    """(rank_lost, stiffness_lost) per pair, from one decomposition."""
    brm = extended_bearing_rigidity_matrix(net)
    rank, _, lam = rigidity_decomposition(brm, rank_K)
    _, _, w, V = nullspace_and_softest(brm, int(rank))
    lam_s = float(lam) if rank >= rank_K else 0.0
    return removal_costs(brm, net, int(rank_K), lam=lam_s, w=w, V=V, c_max=1)


def apply_event(net, event, k, rank_K, rng, *, drift_sigma=0.1, join_domain=None,
                adversarial_cap=2000):
    """Apply `event` to `net` in place. Returns the damage record.

    `skipped` carries a reason when the event cannot be applied to this instance --
    a strictly minimal graph has no redundant edge to drop, for one -- and the caller
    drops the instance rather than falling back to a different event.
    """
    n = net.n
    rec = {"event": event, "k": int(k), "dropped": [], "added": [],
           "node_removed": None, "node_added": None, "join_domain": None,
           "index_map": _identity_map(n), "skipped": None}

    if event == "none":
        return rec

    if event == "pose_drift":
        L = characteristic_length(net)
        for agent in net.agents:
            step = rng.normal(size=3) * drift_sigma * L
            if agent.domain in ("R^2", "R^2xS^1"):
                step[2] = 0.0                      # a planar agent cannot leave its plane
            agent.pose.position = agent.pose.position + step
        return rec

    if event == "edge_add_spurious":
        absent = absent_pairs(net)
        if len(absent) < k:
            rec["skipped"] = "fewer absent pairs than the event asks for"
            return rec
        chosen = [absent[i] for i in rng.choice(len(absent), k, replace=False)]
        for i, j in chosen:
            net.edges[i, j] = True
        rec["added"] = chosen
        return rec

    if event == "node_leave":
        if n <= 3:
            rec["skipped"] = "too few agents to remove one"
            return rec
        v = int(rng.integers(n))
        rec["dropped"] = [(i, j) for (i, j) in net.get_edge_list() if v in (i, j)]
        net.remove_agent(v)
        rec["node_removed"] = v
        rec["index_map"] = {i: (i if i < v else i - 1) for i in range(n) if i != v}
        return rec

    if event == "node_join":
        p = np.array([a.pose.position for a in net.agents], dtype=float)
        lo, hi = p.min(axis=0), p.max(axis=0)
        pos = lo + rng.random(3) * np.maximum(hi - lo, 1e-6)
        domain = join_domain or DOMAINS[int(rng.integers(len(DOMAINS)))]
        if domain in ("R^2", "R^2xS^1"):
            pos[2] = 0.0
        agent = Agent(Pose(pos, np.zeros(3)))
        agent.set_domain(domain)
        if domain in ("R^2xS^1", "R^3xS^1", "SE(3)"):
            agent.randomize_orientation()
        rec["node_added"] = net.add_agent(agent)
        rec["join_domain"] = domain
        return rec

    present = net.get_edge_list()
    if len(present) < k:
        rec["skipped"] = "fewer edges than the event asks for"
        return rec

    if event == "edge_drop_random":
        chosen = [present[i] for i in rng.choice(len(present), k, replace=False)]

    elif event == "edge_drop_critical":
        # The k edges carrying the most rank. On a minimal graph every edge carries
        # some, so this approaches uniform there; and the obvious tie-break is dead,
        # since removal_costs sets stiffness_lost to 1 for every rank-critical edge.
        rank_lost, _ = _edge_costs(net, rank_K)
        jitter = rng.random(len(present))
        order = sorted(range(len(present)),
                       key=lambda t: (-rank_lost[present[t]], jitter[t]))
        chosen = [present[t] for t in order[:k]]

    elif event == "edge_drop_redundant":
        # rigidity survives, so phi's rank term cannot motivate a repair and only the
        # conditioning term can. Undefined on a minimal graph -- skip, never fall back.
        # One edge at a time, re-evaluating: redundancy is not additive, and two
        # individually-redundant edges can be jointly load bearing.
        chosen = []
        for _ in range(k):
            rank_lost, stiffness_lost = _edge_costs(net, rank_K)
            spare = [e for e in net.get_edge_list() if rank_lost[e] <= 0]
            if not spare:
                rec["skipped"] = "no redundant edge left to drop"
                _apply(net, chosen)                # put back what was taken
                return rec
            jitter = {e: rng.random() for e in spare}
            pick = min(spare, key=lambda e: (-stiffness_lost[e], jitter[e]))
            net.edges[pick] = False
            chosen.append(pick)
        rec["dropped"] = chosen
        return rec

    elif event == "edge_drop_adversarial":
        # the k-subset that costs the most to put right, by the bound itself
        subsets = list(itertools.combinations(range(len(present)), k))
        if len(subsets) > adversarial_cap:
            idx = rng.choice(len(subsets), adversarial_cap, replace=False)
            subsets = [subsets[i] for i in idx]
        best, best_cost = None, -1
        for sub in subsets:
            trial = copy.deepcopy(net)
            _drop(trial, [present[t] for t in sub])
            c = repair_edge_count(trial, rank_K=rank_K)
            if c > best_cost:
                best, best_cost = sub, c
        chosen = [present[t] for t in best]

    else:
        raise ValueError(f"unknown event {event!r}; known: {list(EVENTS)}")

    _drop(net, chosen)
    rec["dropped"] = chosen
    return rec


# ---------------------------------------------------------------- the repair population

def _is_rigid(net, rank_K):
    brm = extended_bearing_rigidity_matrix(net)
    return int(np.linalg.matrix_rank(brm)) >= rank_K if brm.size else rank_K == 0


@cost.measurement
def enumerate_repairs(net, rank_K, size, *, cap=ENUMERATION_CAP):
    """Every size-`size` set of absent pairs that restores rigidity, or None.

    None rather than a partial list when the space is too large: a truncated
    enumeration is lexicographically biased, and a percentile against a biased sample
    is not a percentile.
    """
    absent = absent_pairs(net)
    if size <= 0 or size > len(absent):
        return []
    if math.comb(len(absent), size) > cap:
        return None
    out = []
    for sub in itertools.combinations(absent, size):
        work = copy.deepcopy(net)
        for i, j in sub:
            work.edges[i, j] = True
        if _is_rigid(work, rank_K):
            out.append((tuple(sub), shape_error_of(work, rank_K)))
    return out


@cost.measurement
def sample_repairs(net, rank_K, size, rng, *, want=SAMPLE_WANT, max_draws=SAMPLE_MAX_DRAWS):
    """Rejection-sample valid repairs of exactly `size`. Unbiased, unlike truncation."""
    absent = absent_pairs(net)
    if size <= 0 or size > len(absent):
        return []
    seen, out = set(), []
    for _ in range(max_draws):
        if len(out) >= want:
            break
        sub = tuple(sorted(absent[i] for i in rng.choice(len(absent), size, replace=False)))
        if sub in seen:
            continue
        seen.add(sub)
        work = copy.deepcopy(net)
        for i, j in sub:
            work.edges[i, j] = True
        if _is_rigid(work, rank_K):
            out.append((sub, shape_error_of(work, rank_K)))
    return out


def repair_population(net, rank_K, size, rng, *, cap=ENUMERATION_CAP, want=SAMPLE_WANT):
    """(repairs, tier). tier is exhaustive | sampled | bound."""
    if size <= 0:
        return [], "bound"
    full = enumerate_repairs(net, rank_K, size, cap=cap)
    if full is not None:
        return full, "exhaustive"
    return sample_repairs(net, rank_K, size, rng, want=want), "sampled"


# ---------------------------------------------------------------- metrics

def quality_percentile(errors, chosen):
    """Where `chosen` sits among `errors`. **100 = best**, 50 = no better than chance.

    Lower shape error is better, so the values are negated and baselines._percentile_of
    does the midrank arithmetic -- one implementation, and the same convention as
    every other percent column in the repo. THEORY §19.4 quotes the complement.
    """
    from baselines import _percentile_of
    vals = [e for e in errors if e is not None and np.isfinite(e)]
    if chosen is None or not np.isfinite(chosen) or len(vals) < 2:
        return None
    return _percentile_of([-v for v in vals], -chosen)


def is_best_repair(errors, chosen, tol=1e-9):
    from baselines import _is_best
    vals = [e for e in errors if e is not None and np.isfinite(e)]
    if chosen is None or not np.isfinite(chosen) or not vals:
        return None
    return _is_best([-v for v in vals], -chosen, tol=tol)


def _edge_set(edges):
    return {(int(i), int(j)) for i, j in zip(*np.nonzero(edges))}


def edit_metrics(pre_edges, brk_edges, post_edges, damage):
    """Count and locality, in the POST-event index space.

    A node event renumbers the agents, so the pre-event edge set is mapped through
    `damage["index_map"]` before anything is compared; otherwise every metric on a
    membership event compares two different labellings.
    """
    imap = damage["index_map"]
    pre = {(imap[i], imap[j]) for i, j in _edge_set(pre_edges)
           if i in imap and j in imap}
    brk, post = _edge_set(brk_edges), _edge_set(post_edges)

    dropped = pre - brk                       # what the event took away
    survived = pre & brk
    added = post - brk
    removed = brk - post
    return {
        "added": len(added),
        "removed": len(removed),
        # net edits that stuck, against the edits actually applied
        "churn": len(post ^ brk),
        # survived edges the method chose to delete
        "collateral": len(survived - post),
        # did it simply put back what was lost. A DIAGNOSTIC, not a score: 19.4 says
        # the best repair is usually not the original edges
        "restoration": (len(post & dropped) / len(dropped)) if dropped else None,
    }


# ---------------------------------------------------------------- methods

def run_greedy_repair(ctx):
    """The classical repair: add by marginal rank gain until rigid. THEORY 19.4's opponent."""
    net = copy.deepcopy(ctx["broken"])
    if _is_rigid(net, ctx["rank_K"]):
        return {"edges": net.edges.copy(), "note": "already rigid, no legal move"}
    greedy_rigid_repair(net, ctx["rank_K"], rng=ctx["rng"]("greedy_repair"))
    return {"edges": net.edges.copy()}


def run_random_repair(ctx):
    """Add absent pairs at random until rigid. The floor the percentile is read against."""
    net = copy.deepcopy(ctx["broken"])
    rank_K = ctx["rank_K"]
    rng = ctx["rng"]("random_repair")
    for _ in range(net.n * (net.n - 1)):
        if _is_rigid(net, rank_K):
            break
        absent = absent_pairs(net)
        if not absent:
            break
        net.edges[absent[int(rng.integers(len(absent)))]] = True
    return {"edges": net.edges.copy()}


def run_optimal_repair(ctx):
    """The best of the enumerated population: the 100 of the percentile scale."""
    pop = ctx["population"]
    if not pop:
        return None
    sub, _ = min(pop, key=lambda t: t[1])
    net = copy.deepcopy(ctx["broken"])
    _apply(net, sub)
    return {"edges": net.edges.copy()}


def _via_baseline(ctx, fn, **kw):
    """Run one of baselines.py's edit-space methods on the broken graph."""
    env = ctx["env"]
    _install(env, ctx["broken"])                  # freeze_network is on: bookkeeping only
    fn(env, verbose=False, **kw)
    return {"edges": env.network.edges.copy()}


def run_greedy(ctx):
    """phi hill-climb from the broken graph. The only classical method with a move on
    every event, and the one that reads kappa."""
    import baselines
    return _via_baseline(ctx, baselines.run_greedy, max_steps=ctx["steps"])


def run_spectral(ctx):
    import baselines
    return _via_baseline(ctx, baselines.run_spectral, max_steps=ctx["steps"])


def run_rebuild(ctx):
    """constructive from the EMPTY graph -- what a one-shot method must do after damage.

    The control that decides whether repair is a distinct problem at all: if rebuilding
    matches on shape error at the same edge count, it is not.
    """
    import baselines
    env = ctx["env"]
    _install(env, ctx["broken"])
    baselines.run_constructive(env, ctx["rng"]("rebuild"), restarts=ctx["restarts"],
                               verbose=False)
    return {"edges": env.network.edges.copy()}


CLASSICAL = {
    "greedy_repair": run_greedy_repair,
    "greedy": run_greedy,
    "spectral": run_spectral,
    "random_repair": run_random_repair,
    "optimal_repair": run_optimal_repair,
    "rebuild": run_rebuild,
}
DEFAULT_METHODS = ("greedy_repair", "greedy", "optimal_repair", "random_repair", "rebuild")



# ---------------------------------------------------------------- policies

def sync_observation_space(agent, env):
    """Point every model at the environment's CURRENT observation space.

    The weights are n-free, but `compute` slices the flat observation with the space
    the model is holding and then reads n off the result. Left at the training n it
    returns a wrongly-sized Q-vector over misaligned inputs and raises nothing, so a
    node event would silently score garbage. This one assignment is what makes the
    variable-n path work; `tests/test_repair.py` pins both halves.
    """
    for model in agent.models.values():
        if model is not None:
            model.observation_space = env.observation_space


def run_policy_repair(label, ctx):
    """One rollout, three answers.

    The live checkpoints have `skip_enabled: false`, so the policy must edit every
    step and "edits used" is meaningless without a stopping rule. Rather than pick
    one, record all three from the same trajectory:

      first_rigid  the first state that is rigid again -- deployable, needs no stop
                   action, and the only rule under which `added` is comparable to
                   repair_edge_count and the percentile is well defined
      best_phi     the highest-scoring state visited, with the edit count that
                   reached it (baselines.run_policy pairs `work` with best_edges,
                   which counts edits the reported graph does not contain)
      final        where it stopped. Meaningful for a stop-action checkpoint, an
                   oscillation diagnostic otherwise
    """
    import torch
    from baselines import deterministic_action

    env, agent = ctx["env"], ctx["models"][label]["agent"]
    wrapped = ctx["wrapped"]
    steps = ctx["steps"]
    _install(env, ctx["broken"])
    sync_observation_space(agent, env)
    obs, _ = wrapped.reset()                       # freeze_network: bookkeeping only
    agent.enable_models_training_mode(False)

    rank_K = ctx["rank_K"]
    first = None
    seen, edits = set(), 0
    before = env.network.edges.tobytes()
    if _is_rigid(env.network, rank_K):
        first = (env.network.edges.copy(), 0, 0)

    for t in range(steps):
        if ctx["policy_mode"] == "greedy":
            action = deterministic_action(agent, obs)
        else:
            with torch.no_grad():
                action, _ = agent.act(obs, states=wrapped.state(),
                                      timestep=t, timesteps=steps)
        obs, _, terminated, truncated, _ = wrapped.step(action)
        after = env.network.edges.tobytes()
        if after != before:
            edits += 1
            before = after
        if first is None and _is_rigid(env.network, rank_K):
            first = (env.network.edges.copy(), edits, t + 1)

        done = terminated.any().item() if torch.is_tensor(terminated) else terminated
        trunc = truncated.any().item() if torch.is_tensor(truncated) else truncated
        if done or trunc:
            break
        if ctx["policy_mode"] == "greedy":
            key = (after, env.selection.tobytes())
            if key in seen:
                break
            seen.add(key)

    out = {"final": {"edges": env.network.edges.copy(), "edits": edits},
           "best_phi": {"edges": env.best_edges.copy(), "edits": int(env.best_step)}}
    if first is not None:
        out["first_rigid"] = {"edges": first[0], "edits": first[1],
                              "steps_to_rigid": first[2]}
    return out

# ---------------------------------------------------------------- the run

ROW_FIELDS = (
    "instance", "event", "event_k", "skipped",
    "n_pre", "n_post", "rank_K_pre", "rank_K_post", "join_domain",
    "m_pre", "m_brk", "m_post", "m_req_post", "repair_bound",
    "tier", "n_enumerated",
    "method", "stop_rule", "repaired", "steps_to_rigid",
    "added", "removed", "edits", "added_over_bound", "m_over_m_req", "exact_size",
    "churn", "collateral", "restoration",
    "shape_err_pre", "shape_err_brk", "shape_err_post",
    "err_ratio_to_pre", "err_ratio_to_best", "pctile", "is_best",
    "cost_total", "ms",
)


def _finite(x):
    return x if (x is not None and np.isfinite(x)) else None


def _install(env, net, membership=False):
    """Put `net` into `env` and redo the episode bookkeeping.

    Keyed on the network rather than on the event: after a membership event the env's
    spaces stay at the new n, so the next instance's ordinary event would otherwise
    install a smaller network against spaces describing the larger one.
    """
    resync = membership or net.n != getattr(env, "n", None)
    env.network = copy.deepcopy(net)
    if resync:
        return env.resync_after_membership_change()
    return env.reset()


def measure(*, env, instances, events, ks, methods=DEFAULT_METHODS, models=None,
            wrapped=None, seed=0, steps=None, policy_mode="greedy",
            cap=ENUMERATION_CAP, want=SAMPLE_WANT, restarts=5, drift_sigma=0.1,
            join_domain=None, enumerate_population=True, progress=None):
    """One row per (instance, event, k, method, stop_rule); plus the populations.

    Returns (rows, spread). `spread` carries every enumerated repair's error so the
    figures redraw without paying for the enumeration again.
    """
    rows, spread = [], []
    env.freeze_network = True
    models = models or {}

    for index, pristine in enumerate(instances):
        for event in events:
            for k in (ks if event not in ("none", "pose_drift") else (0,)):
                if progress:
                    progress(f"instance {index + 1}/{len(instances)}  {event} k={k}")

                # ---- converge, then damage. Both from generators keyed on the
                # instance, so neither depends on which methods are being run.
                pre = copy.deepcopy(pristine)
                rank_K_pre = _rank_K(pre)
                converge(pre, rank_K_pre,
                         np.random.default_rng([seed, index, 977]),
                         slack=(k if event == "edge_drop_redundant" else 0))
                brk = copy.deepcopy(pre)
                rng = event_rng(seed, index, event, k)
                dmg = apply_event(brk, event, k, rank_K_pre, rng,
                                  drift_sigma=drift_sigma, join_domain=join_domain)
                base = {"instance": index, "event": event, "event_k": k,
                        "skipped": dmg["skipped"], "join_domain": dmg["join_domain"],
                        "n_pre": pre.n, "n_post": brk.n, "rank_K_pre": rank_K_pre}
                if dmg["skipped"]:
                    rows.append(dict.fromkeys(ROW_FIELDS) | base)
                    continue

                membership = event in MEMBERSHIP
                _install(env, brk, membership)
                rank_K_post, m_req_post = int(env.rank_K), int(env.m_req)
                bound = repair_edge_count(brk, rank_K=rank_K_post)
                err_pre = shape_error_of(pre, rank_K_pre)
                err_brk = shape_error_of(brk, rank_K_post)

                # ---- the population every method is placed against
                pop, tier = ([], "bound")
                if enumerate_population and bound > 0:
                    pop, tier = repair_population(
                        brk, rank_K_post, bound,
                        np.random.default_rng([seed, index, 991]), cap=cap, want=want)
                errors = [e for _, e in pop if np.isfinite(e)]
                best_err = min(errors) if errors else None

                base |= {"rank_K_post": rank_K_post, "m_req_post": m_req_post,
                         "m_pre": int(pre.edges.sum()), "m_brk": int(brk.edges.sum()),
                         "repair_bound": bound, "tier": tier,
                         "n_enumerated": len(errors),
                         "shape_err_pre": _finite(err_pre),
                         "shape_err_brk": _finite(err_brk)}

                ctx = {"env": env, "broken": brk, "rank_K": rank_K_post,
                       "population": pop, "steps": steps or int(env.max_steps),
                       "restarts": restarts, "policy_mode": policy_mode,
                       "models": models, "wrapped": wrapped,
                       "rng": lambda tag, i=index, e=event, kk=k: np.random.default_rng(
                           [seed, i, EVENTS.index(e), kk, sum(map(ord, tag))])}

                for name in methods:
                    runner = CLASSICAL.get(name)
                    answers = None
                    with cost.Meter() as meter:
                        if runner is not None:
                            out = runner(ctx)
                            answers = {"final": out} if out is not None else None
                        elif name in models:
                            answers = run_policy_repair(name, ctx)
                    if not answers:
                        continue
                    for rule, out in answers.items():
                        rows.append(_row(base, name, rule, out, dmg, pre, brk,
                                         rank_K_post, m_req_post, bound, errors,
                                         best_err, meter))
                # every method saw the same graph; leave the env on it
                _install(env, brk, membership)

                if pop:
                    spread.append({"instance": index, "event": event, "event_k": k,
                                   "size": bound, "tier": tier,
                                   "errors": errors,
                                   "picks": {r["method"]: r["shape_err_post"]
                                             for r in rows
                                             if r.get("instance") == index
                                             and r.get("event") == event
                                             and r.get("event_k") == k
                                             and r.get("stop_rule") in (None, "final",
                                                                        "first_rigid")
                                             and r.get("exact_size")}})
    return rows, spread


def _rank_K(net):
    return int(np.linalg.matrix_rank(
        extended_bearing_rigidity_matrix(net.fully_connected())))


def _row(base, method, rule, out, dmg, pre, brk, rank_K, m_req, bound, errors,
         best_err, meter):
    post = out["edges"]
    net = copy.deepcopy(brk)
    net.edges = post.copy()
    repaired = _is_rigid(net, rank_K)
    err_post = shape_error_of(net, rank_K) if repaired else np.inf
    edits = edit_metrics(pre.edges, brk.edges, post, dmg)

    # the percentile only means something against the population it was drawn from:
    # a method spending more than `bound` edges is not in it, and more edges almost
    # always means lower error
    exact = bool(edits["added"] == bound and edits["removed"] == 0) if bound > 0 else None
    usable = exact and len(errors) >= MIN_POPULATION
    row = dict.fromkeys(ROW_FIELDS) | base | {
        "method": method, "stop_rule": rule, "repaired": bool(repaired),
        "steps_to_rigid": out.get("steps_to_rigid"),
        "m_post": int(post.sum()),
        "added": edits["added"], "removed": edits["removed"],
        "edits": out.get("edits", edits["added"] + edits["removed"]),
        "added_over_bound": (edits["added"] - bound) if bound > 0 else None,
        "m_over_m_req": int(post.sum()) - m_req,
        "exact_size": exact,
        "churn": edits["churn"], "collateral": edits["collateral"],
        "restoration": edits["restoration"],
        "shape_err_post": _finite(err_post),
        "err_ratio_to_pre": (err_post / base["shape_err_pre"]
                             if base.get("shape_err_pre") and np.isfinite(err_post)
                             else None),
        "err_ratio_to_best": (err_post / best_err
                              if best_err and np.isfinite(err_post) else None),
        "pctile": quality_percentile(errors, err_post) if usable else None,
        "is_best": is_best_repair(errors, err_post) if usable else None,
        "cost_total": sum(meter.counts.get(k, 0) for k in cost.LEAVES),
        "ms": meter.ms,
    }
    return row


# ---------------------------------------------------------------- reporting

def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _gmean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None
            and np.isfinite(r[key]) and r[key] > 0]
    return float(np.exp(np.mean(np.log(vals)))) if vals else None


def headline_rule(event):
    """Which stopping rule the table reads for this event.

    A rank-breaking event has a moment of success, so `first_rigid` is both deployable
    and the only rule under which `added` is comparable to the bound. Where rigidity
    survives, `first_rigid` is step 0 and says nothing, so the best state visited is
    what there is to report. Classical methods stop when they are done and report
    `final` either way; every rule is in the csv.
    """
    return "first_rigid" if event in BREAKING else "best_phi"


def _headline(rows):
    """Exactly one row per (instance, event, k, method).

    The rule the event calls for where the method offers it, `final` otherwise --
    a classical method stops when it is done and reports only that.
    """
    best = {}
    for r in rows:
        if r.get("method") is None:
            continue
        key = (r["instance"], r["event"], r["event_k"], r["method"])
        want = headline_rule(r["event"])
        cur = best.get(key)
        if cur is None or (r["stop_rule"] == want and cur["stop_rule"] != want):
            best[key] = r
    return list(best.values())


def summarise(rows, stop_rule=None):
    """One record per (event, method), in the order the events were asked for.

    One stopping rule per event, or a policy's three answers would be averaged into
    one meaningless row.
    """
    out = []
    live = (_headline(rows) if stop_rule is None else
            [r for r in rows if r.get("method") is not None
             and r["stop_rule"] in (stop_rule, "final")])
    for event in dict.fromkeys(r["event"] for r in live):
        for method in dict.fromkeys(r["method"] for r in live if r["event"] == event):
            sel = [r for r in live if r["event"] == event and r["method"] == method]
            scored = [r for r in sel if r["pctile"] is not None]
            out.append({
                "event": event, "method": method, "cases": len(sel),
                "repaired_pct": 100.0 * np.mean([r["repaired"] for r in sel]),
                "added": _mean(sel, "added"),
                "removed": _mean(sel, "removed"),
                "over_bound": _mean(sel, "added_over_bound"),
                "m_over_m_req": _mean(sel, "m_over_m_req"),
                "churn": _mean(sel, "churn"),
                "collateral": _mean(sel, "collateral"),
                "restoration": _mean(sel, "restoration"),
                "pctile": _mean(scored, "pctile"),
                "scored": len(scored),
                "best_pct": (100.0 * np.mean([bool(r["is_best"]) for r in scored])
                             if scored else None),
                "err_ratio_to_best": _gmean(scored, "err_ratio_to_best"),
                "err_ratio_to_pre": _gmean(sel, "err_ratio_to_pre"),
                "ms": _mean(sel, "ms"),
            })
    return out


def _cell(v, fmt="{:.2f}"):
    return "-" if v is None else fmt.format(v)


def format_table(rows, context=None, brief=False):
    lines = ["=" * 108, "REPAIR AND REWIRING", "=" * 108]
    for key, val in (context or {}).items():
        lines.append(f"  {key:<12}: {val}")
    skipped = [r for r in rows if r.get("method") is None]
    if skipped:
        reasons = {}
        for r in skipped:
            reasons[r["skipped"]] = reasons.get(r["skipped"], 0) + 1
        for why, count in reasons.items():
            lines.append(f"  skipped     : {count} instance-events, {why}")
    lines.append("")
    lines.append("  policy rows use the stopping rule the event calls for -- first_rigid "
                 "where rigidity")
    lines.append("  broke, best visited where it survived. repair.csv carries all three.")
    lines.append("")
    head = (f"  {'event':<22}{'method':<15}{'n':>4}{'rigid':>7}{'added':>7}"
            f"{'remvd':>7}{'over':>7}{'churn':>7}{'collat':>7}{'pctile':>8}"
            f"{'best%':>7}{'x best':>8}{'x pre':>8}")
    lines += [head, "  " + "-" * (len(head) - 2)]
    for s in summarise(rows):
        lines.append(
            f"  {s['event']:<22}{s['method']:<15}{s['cases']:>4}"
            f"{s['repaired_pct']:>6.0f}%{_cell(s['added'], '{:.1f}'):>7}"
            f"{_cell(s['removed'], '{:.1f}'):>7}"
            f"{_cell(s['over_bound'], '{:+.1f}'):>7}{_cell(s['churn'], '{:.1f}'):>7}"
            f"{_cell(s['collateral'], '{:.1f}'):>7}"
            f"{_cell(s['pctile'], '{:.0f}'):>8}{_cell(s['best_pct'], '{:.0f}'):>7}"
            f"{_cell(s['err_ratio_to_best']):>8}{_cell(s['err_ratio_to_pre']):>8}")
    if brief:
        return "\n".join(lines)
    lines += ["", "  HOW TO READ IT", """
  Every method is handed the same converged graph and the same damage. The graph is
  a greedy construction pruned back to minimality in random order -- pruned by "does
  it stay rigid" rather than by phi or by marginal rank, so it is nobody's own fixed
  point.

  rigid    percent that are rigid again afterwards.
  added    edges added. `over` is that minus repair_edge_count, a PROVEN lower bound,
           so 0.0 means the method used the fewest edges that could possibly work.
  churn    edges that differ between the damaged graph and the answer, and collat is
           the subset that were untouched by the event and got removed anyway. A
           repair should be local; a method that rewires everything is rebuilding.
  pctile   where the answer sits among EVERY repair of the minimum size, 100 = best,
           50 = no better than picking one at random. Blank when the method did not
           use exactly that many edges, since more edges almost always means lower
           error and the comparison would flatter it. `best%` is how often it found
           the optimum outright, and `x best` how many times its shape error is.
  x pre    shape error against the graph BEFORE the event, so below 1 means better
           conditioned than the network started out. The only quality number defined
           for every event, so it is the one to read where `pctile` is blank. Note
           the pre-event graph is minimal and pruned in random order, which is often
           not a well conditioned choice -- so a good repair beating it is normal.

  The `none` row is the control: nothing happened, so churn should be 0. Anything
  else is a method that cannot leave a converged network alone.""".rstrip()]
    return "\n".join(lines)
