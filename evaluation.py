"""The evaluation run: everything the thesis reports about a policy, in one place.

Every method is scored with Environment.compute_state_score, i.e. the exact state
score phi the agent trains on, so the rows are directly comparable:

  initial  the graph the sampler produced (what the agent starts from)
  random   uniform random actions through env.step()  -> the floor for this action space
  greedy   hill-climbing on phi, one edge toggle at a time
  constructive  builds from the empty graph, keeping any edge that raises rank(B)
  learned  a trained policy, sampling actions (--policy-mode greedy for argmax instead)
  optimal  exhaustive search for the fewest-edge rigid graph (small n only)

greedy gets stuck exactly where RL should win: states where no single edit improves
phi but a swap of two does.

usage:
  uv run evaluation.py <environment_name> [--episodes N] [--model NAME] [--brute-force]
      [--steps K] [--tag NAME] [--device cpu|cuda] [--methods a,b,c] [--restarts K]
      [--policy-mode sample|greedy] [--benchmark NAME] [--noise-sweep 0.5,1,5]
      [--no-plots] [--plot-episodes N] [--brief] [--out-dir PATH] [--replay-env]

Writes one directory per run under runs_evaluation/: the table, per-episode results,
per-step trajectories, provenance, and every figure. See report.py.
"""

import argparse
import copy
import csv
import glob
import itertools
import json
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from environment import Environment
from rigidity import (rigidity_eigenvalue, rigidity_decomposition, greedy_rigid_construction,
                      greedy_rigid_repair, repair_edge_count, estimation_error_of,
                      extended_bearing_rigidity_matrix,
                      measurement_sensitivity, candidate_gain, characteristic_length,
                      is_IBR_explicit,
                      nullspace_and_softest, nullspace_in_scaled_units, removal_costs)
import benchmark
import cost
import estimation
import manifest
import report

MAX_BRUTE_FORCE_N = 5


# the only state scores whose value depends on the is_MBR flag
MBR_DEPENDENT_SCORES = {"RigidAndMinRigid", "MinRigid", "MinRigidAndMinEigenvalue"}


@cost.counted
def score_network(env, need_mbr=None):
    """(score, is_IBR, is_MBR, rank, m) for whatever edges env.network currently holds.

    is_MBR costs one rank computation *per edge* on top of the rank of the whole matrix.
    greedy evaluates n(n-1) candidates per improvement step, so that is skipped unless the
    configured state score actually reads the flag. At n=16 it is roughly half the cost.
    """
    brm = env.network.extended_bearing_rigidity_matrix()
    if need_mbr is None:
        need_mbr = env.state_score_type in MBR_DEPENDENT_SCORES

    # rank and lam from one SVD: phi needs lam once stiffness_kappa > 0
    if need_mbr:
        is_MBR, is_IBR, rank = env.network.is_MBR(rank_K=env.rank_K, brm=brm)
        lam = rigidity_decomposition(brm, env.rank_K)[2] if is_IBR else 0.0
    elif int(env.network.edges.sum()) == 0:
        is_MBR, is_IBR, rank, lam = False, False, 0, 0.0
    else:
        is_MBR = False
        rank, _, lam = rigidity_decomposition(brm, env.rank_K)
        is_IBR = rank == env.rank_K

    score = env.compute_state_score(brm, is_IBR, is_MBR, rank, lam=lam)
    return score, bool(is_IBR), bool(is_MBR), int(rank), int(env.network.edges.sum())


def result(method, score, is_IBR, is_MBR, m, work=0, best_at=0, min_eig=None,
           shape_err=None, edges=None, edges_are="final"):
    """One method's outcome on one instance.

    `work` counts graph modifications actually applied and `best_at` is the step the best
    graph was reached at -- the old single `steps` column meant different things per method.
    """
    return {
        "method": method,
        "score": float(score),
        "is_IBR": bool(is_IBR),
        "is_MBR": bool(is_MBR),
        "m": int(m),
        "work": int(work),
        "best_at": int(best_at),
        "min_eig": None if min_eig is None else float(min_eig),
        # RMS state error per radian of bearing noise; None while flexible
        "shape_err": None if shape_err is None else float(shape_err),
        # the graph this row reports, so --noise-sweep can measure it afterwards,
        # and whether that is where the method stopped or the best it ever saw
        "edges": None if edges is None else edges.copy(),
        "edges_are": edges_are,
    }


@cost.measurement
def measure_noise(env, row, sigmas, trials, rng):
    """Fill row["noise"][sigma] with the measured RMS shape error of its graph.

    Runs on the graph the row reports, which for a rollout is the best state
    visited rather than the one the episode stopped on.
    """
    if row.get("edges") is None or not row["is_IBR"]:
        return
    net = copy.deepcopy(env.network)
    net.edges = row["edges"].copy()

    pred = estimation.predicted_error(net, env.rank_K)[0]
    if not np.isfinite(pred):
        return

    row["noise"], row["noise_failed"] = {}, {}
    for sigma in sigmas:
        got = estimation.monte_carlo_error(net, sigma, trials=trials, rng=rng,
                                           rank_K=env.rank_K)
        rms = got["position"]["rms"]
        # a level where every recovery blew up carries no number, only the fact
        if np.isfinite(rms) and rms > 0:
            row["noise"][sigma] = float(rms)
        row["noise_failed"][sigma] = got["position"]["failed"]
    if not row["noise"]:
        del row["noise"]
        return
    row["pred_err"] = float(pred)


MAX_DECISION_ANALYSIS_N = 12
MAX_REPAIR_FIGURE_N = 8


@cost.measurement
def edit_landscape(env):
    """Every single-edge toggle from the current graph, scored two ways.

    {(i, j): (dphi, derr, node_share)} -- how much phi rises, how much the shape
    error falls, and how much of the current error the measuring agent carries.
    The policy optimises phi, not the error, so the two rankings are reported
    side by side rather than one standing in for the other.
    """
    n = env.network.n
    base_phi = score_network(env)[0]
    base_err = shape_error_of(env.network, env.rank_K)
    _, per_node = measurement_sensitivity(env.network, env.rank_K)
    total = per_node.sum()
    share = per_node / total if np.isfinite(total) and total > 0 else np.zeros(n)

    out = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            env.network.edges[i, j] = not env.network.edges[i, j]
            phi = score_network(env)[0]
            err = shape_error_of(env.network, env.rank_K)
            env.network.edges[i, j] = not env.network.edges[i, j]
            # a move off a finite error into an infinite one is not a ranking, it is
            # a disqualification, and vice versa
            d_err = (base_err - err) if (np.isfinite(err) and np.isfinite(base_err)) else None
            out[(i, j)] = (phi - base_phi, d_err, float(share[i]))
    return out


def _percentile_of(values, chosen):
    """Where `chosen` sits among `values`, as a percentage. 50 = no better than chance.

    Midrank, so ties count half: several edits are often equally good, and counting
    only strictly-worse alternatives would score picking one of them well below 100.
    `_is_best` is what says the choice was actually optimal.
    """
    vals = [v for v in values if v is not None]
    if chosen is None or len(vals) < 2:
        return None
    below = sum(v < chosen for v in vals)
    tied = sum(v == chosen for v in vals)
    return 100.0 * (below + 0.5 * tied) / len(vals)


def _is_best(values, chosen, tol=1e-9):
    vals = [v for v in values if v is not None]
    if chosen is None or not vals:
        return None
    return bool(chosen >= max(vals) - tol)


def decision_record(landscape, applied, kind):
    """One row per edit the policy applied: where its choice ranked."""
    if applied not in landscape:
        return None
    dphi, derr, share = landscape[applied]
    phis = [v[0] for v in landscape.values()]
    errs = [v[1] for v in landscape.values()]
    shares = [v[2] for v in landscape.values()]
    return {
        "kind": kind,
        "phi_pct": _percentile_of(phis, dphi),
        "err_pct": _percentile_of(errs, derr),
        "share_pct": _percentile_of(shares, share),
        "phi_best": _is_best(phis, dphi),
        "err_best": _is_best(errs, derr),
        "dphi": float(dphi),
        "derr": None if derr is None else float(derr),
    }
REPAIR_FIGURE_INSTANCES = 8


def shape_error_of(net, rank_K):
    a_opt, _, _ = estimation_error_of(net, rank_K)
    return float(np.sqrt(a_opt / net.n)) if np.isfinite(a_opt) else np.inf


@cost.measurement
def repair_spread(net, rank_K, rng, drop=2, cap=20000):
    """Every minimum-size repair of a broken copy of `net`, scored by shape error.

    {"errors": [...], "greedy": err, "size": k} or None when the instance does not
    produce a comparison -- already rigid after the break, or fewer than two repairs.
    """
    work = copy.deepcopy(net)
    present = list(zip(*np.nonzero(work.edges)))
    if len(present) <= drop:
        return None
    for idx in rng.choice(len(present), drop, replace=False):
        work.edges[present[idx]] = False
    if np.linalg.matrix_rank(extended_bearing_rigidity_matrix(work)) >= rank_K:
        return None

    size = repair_edge_count(work, rank_K=rank_K)
    n = work.n
    absent = [(i, j) for i in range(n) for j in range(n)
              if i != j and not work.edges[i, j]]

    errors = []
    for count, sub in enumerate(itertools.combinations(absent, size)):
        if count >= cap:
            break
        cand = copy.deepcopy(work)
        for i, j in sub:
            cand.edges[i, j] = True
        if np.linalg.matrix_rank(extended_bearing_rigidity_matrix(cand)) >= rank_K:
            errors.append(shape_error_of(cand, rank_K))
    errors = [e for e in errors if np.isfinite(e) and e > 0]
    if len(errors) < 2:
        return None

    picked = copy.deepcopy(work)
    _, added = greedy_rigid_repair(picked, rank_K, rng=rng)
    greedy_err = shape_error_of(picked, rank_K) if len(added) == size else None

    return {"errors": errors, "greedy": greedy_err, "size": int(size)}


def record(trace, method, episode, step, stats):
    """Append one point of a (episode, method) time series."""
    if trace is None or stats is None:
        return
    trace.append({"episode": episode, "method": method, "step": int(step),
                  "score": stats["score"], "edges": stats["m"], "rank": stats["rank"],
                  "rank_K": stats["rank_K"], "is_IBR": stats["is_IBR"],
                  "is_MBR": stats["is_MBR"], "min_eig": stats["min_eig"],
                  "shape_err": stats.get("shape_err")})


@cost.measurement
def stats_now(env, need_mbr=True):
    """Full per-step record for the graph currently in `env`."""
    score, is_IBR, is_MBR, rank, m = score_network(env, need_mbr=need_mbr)
    return {"score": score, "m": m, "rank": rank, "rank_K": int(env.rank_K),
            "is_IBR": is_IBR, "is_MBR": is_MBR,
            "min_eig": float(rigidity_eigenvalue(env.network, rank_K=env.rank_K)),
            "shape_err": env.shape_error_now() if hasattr(env, "shape_error_now") else None}


def step_stats(env, tracing):
    """What the env recorded this step, computed here if it is too old to provide it.

    --replay-env can hand us an archived Environment from before last_stats existed.
    """
    if not tracing:
        return None
    stats = getattr(env, "last_stats", None)
    return stats if stats is not None else stats_now(env)


# --------------------------------------------------------------------------------------
def run_initial(env, trace=None, episode=0):
    # one call per episode, so pay for the is_MBR flag rather than reporting a false one
    st = stats_now(env)
    record(trace, "initial", episode, 0, st)
    return result("initial", st["score"], st["is_IBR"], st["is_MBR"], st["m"],
                  min_eig=st["min_eig"], shape_err=st.get("shape_err"),
                  edges=env.network.edges)


def run_greedy(env, max_steps=200, verbose=True, trace=None, episode=0):
    """Repeatedly apply the single edge toggle that improves phi the most.

    Cost is n(n-1) phi evaluations per improvement step, so this is by far the most
    expensive baseline at large n -- drop it with --methods if you only want the rest.
    """
    n = env.network.n
    candidates = [(i, j) for i in range(n) for j in range(n) if i != j]

    score, is_IBR, is_MBR, _, m = score_network(env)
    steps = 0
    if trace is not None:
        record(trace, "greedy", episode, 0, stats_now(env))
    bar = tqdm(desc="    greedy", unit="edit", leave=False) if verbose else None

    for _ in range(max_steps):
        best_delta = 0.0
        best_move = None
        best_eval = None

        for (i, j) in candidates:
            existed = env.network.edge_exists(i, j)
            if existed:
                env.network.remove_edge(i, j)
            else:
                env.network.add_edge(i, j)

            cand = score_network(env)

            # revert
            if existed:
                env.network.add_edge(i, j)
            else:
                env.network.remove_edge(i, j)

            delta = cand[0] - score
            if delta > best_delta:
                best_delta = delta
                best_move = (i, j, existed)
                best_eval = cand

        if best_move is None:  # local optimum, nothing improves phi
            break

        i, j, existed = best_move
        if existed:
            env.network.remove_edge(i, j)
        else:
            env.network.add_edge(i, j)
        score, is_IBR, is_MBR, _, m = best_eval
        steps += 1
        if trace is not None:
            record(trace, "greedy", episode, steps, stats_now(env))
        if bar is not None:
            bar.update(1)
            bar.set_postfix(m=m, phi=f"{score:.0f}")

    if bar is not None:
        bar.close()

    # the search may have skipped the is_MBR flag; the reported row needs it
    st = stats_now(env)
    # greedy stops at its own best, so work and best@ coincide
    return result("greedy", st["score"], st["is_IBR"], st["is_MBR"], st["m"],
                  work=steps, best_at=steps, min_eig=st["min_eig"],
                  shape_err=st.get("shape_err"), edges=env.network.edges)


# ── spectral ──────────────────────────────────────────────────────────────────────────
# The same hill climb greedy runs, reading the landscape off the rigidity algebra instead
# of rescoring every toggle. phi is affine in rank, so a toggle moves it by exactly the
# rank it adds or costs -- which candidate_gain and removal_costs already return for all
# pairs at once. At stiffness_kappa = 0 that is exact and this is greedy; above it the
# addition term is a ranking prior rather than the true lambda.
CLOSED_FORM_SCORES = {"WeightedNormalized", "WeightedNormalizedSpectral"}


def phi_landscape(env, stiffness=True):
    """d(phi) for every single-edge toggle, in closed form. NaN on the diagonal.

    stiffness=False leaves the bonus out, giving the rank-only landscape that is exactly
    d(phi) at stiffness_kappa = 0.

    Raises on a state score whose closed form is not this one: the weights below are
    WeightedNormalized's own, and returning them for another score would be silently wrong.
    """
    if env.state_score_type not in CLOSED_FORM_SCORES:
        raise ValueError(f"no closed-form landscape for state score "
                         f"{env.state_score_type!r}; {sorted(CLOSED_FORM_SCORES)} only")
    w_rank, w_edge = 100.0, 25.0                      # environment.compute_state_score
    n = env.network.n
    kappa = float(getattr(env, "stiffness_kappa", 0.0))

    B = env.network.extended_bearing_rigidity_matrix()
    rank, _, lam = rigidity_decomposition(B, env.rank_K)
    L = characteristic_length(env.network)
    Z, v, w, V = nullspace_and_softest(B, int(rank))
    Zs = nullspace_in_scaled_units(Z, n, L)
    _, add_rk = candidate_gain(env.network, Zs, length_scale=L)
    # the stiffness half of removal_costs is an eigvalsh(6n) per redundant edge and is
    # not read at kappa = 0, which is the whole cost advantage over greedy
    rem_rk, rem_st = removal_costs(B, env.network, int(env.rank_K), lam=lam, w=w, V=V,
                                   c_max=env.c_max,
                                   need_stiffness=stiffness and kappa > 0)

    c = max(int(env.c_max), 1)
    rank_K = max(int(env.rank_K), 1)
    E = env.network.edges.astype(bool)
    D = np.where(E,
                 (-w_rank * (rem_rk * c) + w_edge * c) / rank_K,
                 (w_rank * add_rk - w_edge * c) / rank_K)

    if stiffness and kappa > 0 and rank >= env.rank_K and env.stiffness_ref > 0:
        # an addition's true lambda is not available from the current matrix, so this is
        # add_stiffness (a ranking prior) scaled onto the stiffness term's own budget.
        # Removals use remove_stiffness, which is exact.
        budget = kappa * w_edge * c / rank_K
        add_st = np.zeros((n, n))
        if v is not None and v.shape[1] == 1:
            vs = nullspace_in_scaled_units(v, n, L)
            add_st = candidate_gain(env.network, vs, length_scale=L)[0]
        D = D + np.where(E, -budget * rem_st,
                         budget * add_st / max(float(add_st.max()), 1e-12))

    np.fill_diagonal(D, np.nan)
    return D


def run_spectral(env, max_steps=200, shortlist=5, verbose=True, trace=None, episode=0):
    """Rank every toggle by the closed-form landscape, then verify the best few.

    Greedy pays n(n-1) rebuild-and-decompose per improvement step for a landscape the
    rigidity algebra already holds. This computes the landscape once and rescores only
    the `shortlist` best candidates, so it stops on a real improvement rather than a
    predicted one -- which matters at stiffness_kappa > 0, where the addition term is a
    ranking prior and not the true lambda. Both methods hill-climb the same phi.
    """
    steps = 0
    score = score_network(env)[0]
    if trace is not None:
        record(trace, "spectral", episode, 0, stats_now(env))
    bar = tqdm(desc="    spectral", unit="edit", leave=False) if verbose else None

    for _ in range(max_steps):
        D = phi_landscape(env)
        if not np.isfinite(D).any():
            break
        # stable, so an exact tie is broken by row-major order -- the same rule greedy's
        # `delta > best_delta` applies, and in R^3 adding a rank-1 edge and dropping a
        # redundant one *are* an exact tie
        flat = np.argsort(-np.where(np.isfinite(D), D, -np.inf), axis=None, kind="stable")
        best_delta, best_move = 0.0, None
        for k in flat[:shortlist]:
            i, j = np.unravel_index(k, D.shape)
            env.network.edges[i, j] = not env.network.edges[i, j]
            delta = score_network(env)[0] - score
            env.network.edges[i, j] = not env.network.edges[i, j]
            if delta > best_delta:
                best_delta, best_move = delta, (int(i), int(j))

        if best_move is None:                         # local optimum
            break
        i, j = best_move
        env.network.edges[i, j] = not env.network.edges[i, j]
        score += best_delta
        steps += 1
        if trace is not None:
            record(trace, "spectral", episode, steps, stats_now(env))
        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()

    st = stats_now(env)
    return result("spectral", st["score"], st["is_IBR"], st["is_MBR"], st["m"],
                  work=steps, best_at=steps, min_eig=st["min_eig"],
                  shape_err=st.get("shape_err"), edges=env.network.edges)


# ── anneal ────────────────────────────────────────────────────────────────────────────
def anneal_temperatures(env):
    """(T0, T1) in phi's own units, so the schedule transfers across n and domain.

    One edge is worth w_edge*c_max/rank_K, so a move of that size is accepted at T0
    with probability e^-1 and is essentially refused by T1.
    """
    one_edge = 25.0 * max(int(env.c_max), 1) / max(int(env.rank_K), 1)
    return one_edge, one_edge / 100.0


def run_anneal(env, rng, budget=None, verbose=True, trace=None, episode=0):
    """Simulated annealing over single-edge toggles, on the configured phi.

    Unlike greedy it assumes nothing about the objective, which is the point at
    stiffness_kappa > 0 where the stiffness is not submodular and greedy carries no
    guarantee. `budget` counts phi evaluations, so it is directly comparable to what
    greedy spent on the same instance. Scored on the best state visited.
    """
    n = env.network.n
    if budget is None:
        budget = 4 * n * (n - 1)
    T0, T1 = anneal_temperatures(env)
    cool = (T1 / T0) ** (1.0 / max(budget - 1, 1))

    score = score_network(env)[0]
    best_score, best_edges, best_at = score, env.network.edges.copy(), 0
    work, T = 0, T0
    if trace is not None:
        record(trace, "anneal", episode, 0, stats_now(env))
    bar = tqdm(total=budget, desc="    anneal", unit="eval", leave=False) if verbose else None

    for t in range(budget):
        i = int(rng.integers(n))
        j = int(rng.integers(n - 1))
        j += (j >= i)                                 # any ordered pair but (i, i)
        env.network.edges[i, j] = not env.network.edges[i, j]
        cand = score_network(env)[0]

        delta = cand - score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-12)):
            score = cand
            work += 1
            if score > best_score:
                best_score, best_edges, best_at = score, env.network.edges.copy(), work
            if trace is not None:
                record(trace, "anneal", episode, work, stats_now(env))
        else:
            env.network.edges[i, j] = not env.network.edges[i, j]

        T *= cool
        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()

    env.network.edges = best_edges
    st = stats_now(env)
    return result("anneal", st["score"], st["is_IBR"], st["is_MBR"], st["m"],
                  work=work, best_at=best_at, min_eig=st["min_eig"],
                  shape_err=st.get("shape_err"), edges=env.network.edges,
                  edges_are="best visited")


# ── degree ────────────────────────────────────────────────────────────────────────────
def run_degree(env, rng, verbose=True, trace=None, episode=0):
    """Add to the least-connected pair until rigid, then prune the busiest redundant edge.

    The only method here that reads nothing an agent could not know locally, apart from
    the rigidity test itself: no marginal ranks, no spectrum, just degrees. The tier-1
    reference for what a distributed rule could plausibly do.
    """
    n = env.network.n
    steps = 0
    if trace is not None:
        record(trace, "degree", episode, 0, stats_now(env))
    bar = tqdm(desc="    degree", unit="edit", leave=False) if verbose else None

    def is_rigid():
        brm = env.network.extended_bearing_rigidity_matrix()
        return bool(brm.size) and is_IBR_explicit(brm, env.rank_K)[0]

    def applied():
        nonlocal steps
        steps += 1
        if trace is not None:
            record(trace, "degree", episode, steps, stats_now(env))
        if bar is not None:
            bar.update(1)

    E = env.network.edges
    # add: whoever measures least, to whoever is measured least
    for _ in range(n * (n - 1)):
        if is_rigid():
            break
        out_deg, in_deg = E.sum(axis=1), E.sum(axis=0)
        cost_ij = out_deg[:, None] + in_deg[None, :]
        cost_ij = np.where(E, np.inf, cost_ij.astype(float))
        np.fill_diagonal(cost_ij, np.inf)
        if not np.isfinite(cost_ij).any():
            break
        best = np.argwhere(cost_ij == cost_ij.min())
        i, j = best[rng.integers(len(best))]
        E[i, j] = True
        applied()

    # prune: the busiest edge whose removal keeps the network rigid
    while True:
        out_deg, in_deg = E.sum(axis=1), E.sum(axis=0)
        order = sorted(((int(out_deg[i] + in_deg[j]), int(i), int(j))
                        for i, j in np.argwhere(E)), reverse=True)
        for _, i, j in order:
            E[i, j] = False
            if is_rigid():
                applied()
                break
            E[i, j] = True
        else:
            break

    if bar is not None:
        bar.close()

    st = stats_now(env)
    return result("degree", st["score"], st["is_IBR"], st["is_MBR"], st["m"],
                  work=steps, best_at=steps, min_eig=st["min_eig"],
                  shape_err=st.get("shape_err"), edges=env.network.edges)


def _construct_once(env, order, rng):
    """One restart, on env.network. (edges, additions in order, rank reached).

    The loop is `rigidity.greedy_rigid_construction`, shared with the stiffness
    reference. `order` is accepted for call compatibility and rebuilt there.
    """
    return greedy_rigid_construction(env.network, env.rank_K, rng)


def run_constructive(env, rng, restarts=20, verbose=True, trace=None, episode=0):
    """Randomized constructive greedy, best of `restarts` independent orders.

    The classical algorithm for this problem, and the one to beat: no rigidity theory
    beyond rank(B), no learning. Unlike every other method it starts from the **empty
    graph** rather than the initial one, because it is a construction and not an edit.
    """
    n = env.network.n
    order = [(i, j) for i in range(n) for j in range(n) if i != j]

    best = None
    bar = tqdm(total=restarts, desc="    constructive", unit="restart",
               leave=False) if verbose else None
    for _ in range(restarts):
        _, added, rank = _construct_once(env, order, rng)
        score, is_IBR, is_MBR, _, m = score_network(env)
        # among rigid graphs phi is monotone decreasing in m, so this picks fewest edges
        if rank == env.rank_K and (best is None or score > best[0]):
            best = (score, list(added))
        if bar is not None:
            bar.update(1)
            bar.set_postfix(m=m, best=f"{best[0]:.0f}" if best else "-")
    if bar is not None:
        bar.close()

    if best is None:                     # no restart reached rank_K
        env.network.edges = np.zeros((n, n), dtype=bool)
        st = stats_now(env)
        record(trace, "constructive", episode, 0, st)
        return result("constructive", st["score"], st["is_IBR"], st["is_MBR"], st["m"])

    # replay the winner, so only it pays for the per-step statistics
    added = best[1]
    E = np.zeros((n, n), dtype=bool)
    env.network.edges = E.copy()
    record(trace, "constructive", episode, 0, stats_now(env))
    for k, (i, j) in enumerate(added, start=1):
        E[i, j] = True
        env.network.edges = E.copy()
        record(trace, "constructive", episode, k, stats_now(env))

    st = stats_now(env)
    # monotone construction: it ends on its own best graph
    return result("constructive", st["score"], st["is_IBR"], st["is_MBR"], st["m"],
                  work=len(added), best_at=len(added), min_eig=st["min_eig"],
                  shape_err=st.get("shape_err"), edges=env.network.edges)


def rollout_result(method, env, work):
    """Best graph the rollout visited, not the one it happened to stop on."""
    return result(method, env.best_state_score, env.best_stats["is_IBR"],
                  env.best_stats["is_MBR"], env.best_stats["m"],
                  work=work, best_at=env.best_step,
                  min_eig=env.best_stats.get("min_eig"),
                  shape_err=env.best_stats.get("shape_err"),
                  edges=env.best_edges, edges_are="best visited")


def run_random(env, steps, trace=None, episode=0):
    """Uniform random actions. Scored on the best state visited, not the final one."""
    record(trace, "random", episode, 0, step_stats(env, trace is not None))
    work = 0
    before = env.network.edges.tobytes()
    for t in range(steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        after = env.network.edges.tobytes()
        if after != before:          # count only steps that actually changed the graph
            work += 1
            before = after
        record(trace, "random", episode, t + 1, step_stats(env, trace is not None))
        if terminated or truncated:
            break
    return rollout_result("random", env, work)


@cost.counted
def deterministic_action(agent, obs):
    """argmax over the model's own scores, for PPO and DQN alike.

    skrl's CategoricalMixin.act always *samples*, so a PPO agent asked to act behaves as it
    did during training. DQN's act already takes an argmax when no exploration scheduler is
    configured. Going through compute() directly makes both algorithms deterministic in the
    same way, and keeps the models' action masking intact.
    """
    role = "policy" if "policy" in agent.models else "q_network"
    with torch.no_grad():
        scores, _ = agent.models[role].compute({"observations": obs}, role=role)
    return torch.argmax(scores, dim=-1, keepdim=True)


def run_policy(agent, wrapped_env, raw_env, steps, mode="sample", trace=None, episode=0,
               decisions=None):
    """Roll out a trained policy, scored on the best state visited.

    mode="greedy"  the action the policy considers best -- what you would deploy
    mode="sample"  sampled actions, i.e. the policy used as a sampling-based search over the
                   horizon (PPO only; a DQN q-network has nothing to sample from)
    """
    agent.enable_models_training_mode(False)  # eval mode (skrl 2.x naming)
    obs, _ = wrapped_env.reset()  # freeze_network keeps the instance
    seen = set()
    record(trace, "learned", episode, 0, step_stats(raw_env, trace is not None))
    work = 0
    before = raw_env.network.edges.tobytes()

    for t in range(steps):
        landscape = edit_landscape(raw_env) if decisions is not None else None
        edges_before = raw_env.network.edges.copy()

        if mode == "greedy":
            action = deterministic_action(agent, obs)
        else:
            # skrl's act() is not ours to decorate, so the forward is counted here
            cost.tally("forward")
            with torch.no_grad():
                action, _ = agent.act(obs, states=wrapped_env.state(),
                                      timestep=t, timesteps=steps)
        obs, _, terminated, truncated, _ = wrapped_env.step(action)
        after = raw_env.network.edges.tobytes()
        if after != before:          # count only steps that actually changed the graph
            work += 1
            before = after
            if landscape is not None:
                changed = np.argwhere(raw_env.network.edges != edges_before)
                if len(changed) == 1:
                    i, j = (int(x) for x in changed[0])
                    kind = "add" if raw_env.network.edges[i, j] else "remove"
                    rec = decision_record(landscape, (i, j), kind)
                    if rec is not None:
                        rec.update(episode=episode, step=t)
                        decisions.append(rec)
        record(trace, "learned", episode, t + 1, step_stats(raw_env, trace is not None))

        done = terminated.any().item() if torch.is_tensor(terminated) else terminated
        trunc = truncated.any().item() if torch.is_tensor(truncated) else truncated
        if done or trunc:
            break

        # a deterministic policy in a deterministic environment is eventually periodic, so
        # once a state repeats nothing new can be found and the rest of the horizon is waste
        if mode == "greedy":
            key = (after, raw_env.selection.tobytes())
            if key in seen:
                break
            seen.add(key)

    return rollout_result("learned", raw_env, work)


def run_brute_force(env, verbose=True):
    """Fewest-edge rigid graph, then the most rigid one at that edge count.

    Scans m ascending and stops at the first level that admits an IBR graph. Because phi
    rewards rank more than it penalises edges, that level contains the phi optimum. Unlike
    the R^d closed form this makes no homogeneity assumption, it is just slow.
    """
    n = env.network.n
    if n > MAX_BRUTE_FORCE_N:
        return None

    all_edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    saved = env.network.edges.copy()

    best_subset = None
    checked = 0
    for m in range(1, len(all_edges) + 1):
        found_at_this_level = False
        best_eig = -np.inf

        # each level is C(n^2-n, m) rank computations, which grows fast: at n=5 a network
        # needing 9 edges costs ~430k of them. Show progress so it does not look hung.
        subsets = itertools.combinations(all_edges, m)
        if verbose:
            subsets = tqdm(subsets, total=math.comb(len(all_edges), m),
                           desc=f"    brute force m={m}", unit="graph",
                           unit_scale=True, leave=False)

        for subset in subsets:
            env.network.set_edges_list(list(subset))
            checked += 1
            score, is_IBR, is_MBR, rank, _ = score_network(env)
            if not is_IBR:
                continue

            found_at_this_level = True
            eig = rigidity_eigenvalue(env.network, rank_K=env.rank_K)
            if eig > best_eig:
                best_eig = eig
                best_subset = list(subset)

        if found_at_this_level:
            break

    # the inner loop runs hundreds of thousands of times and so skips the is_MBR flag;
    # recompute the winner properly before reporting it
    best = None
    if best_subset is not None:
        env.network.set_edges_list(best_subset)
        st = stats_now(env)
        best = result("optimal", st["score"], st["is_IBR"], st["is_MBR"], st["m"],
                      min_eig=st["min_eig"], shape_err=st.get("shape_err"),
                      edges=env.network.edges)

    env.network.set_edges(saved)
    return best


# --------------------------------------------------------------------------------------
# Every method runs the same way: restore the instance, meter it, keep the row. The order
# here is the order the episode line prints in; the table orders by report.METHOD_ORDER.
ALL_METHODS = ("initial", "random", "degree", "greedy", "spectral", "anneal",
               "constructive", "learned", "optimal")


def run_method(name, ctx):
    """Dispatch one method on the instance currently in ctx['env']. None if unavailable."""
    env, args, traces, ep = ctx["env"], ctx["args"], ctx["traces"], ctx["episode"]
    if name == "initial":
        return run_initial(env, trace=traces, episode=ep)
    if name == "greedy":
        return run_greedy(env, trace=traces, episode=ep)
    if name == "spectral":
        return run_spectral(env, shortlist=args.spectral_shortlist,
                            trace=traces, episode=ep)
    if name == "anneal":
        return run_anneal(env, ctx["anneal_rng"], budget=ctx["anneal_budget"],
                          trace=traces, episode=ep)
    if name == "degree":
        return run_degree(env, ctx["degree_rng"], trace=traces, episode=ep)
    if name == "constructive":
        return run_constructive(env, ctx["construct_rng"], restarts=args.restarts,
                                trace=traces, episode=ep)
    if name == "random":
        return run_random(env, ctx["steps"], trace=traces, episode=ep)
    if name == "learned":
        if ctx["agent"] is None:
            return None
        return run_policy(ctx["agent"], ctx["wrapped"], env, ctx["steps"],
                          mode=args.policy_mode, trace=traces, episode=ep,
                          decisions=ctx["decisions"])
    if name == "optimal":
        return run_brute_force(env)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("environment_name")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model", default=None, help="trained model name from train/")
    parser.add_argument("--brute-force", action="store_true")
    parser.add_argument("--steps", type=int, default=None,
                        help="rollout horizon for random/learned (default: env truncate/max steps)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark", default=None,
                        help="evaluate on a frozen instance set from benchmarks/ instead of "
                             "sampling; makes runs comparable across config regenerations")
    parser.add_argument("--device", default="cpu", help="device for the --model rollout")
    parser.add_argument("--tag", default=None,
                        help="label appended to the run directory name")
    parser.add_argument("--out-dir", default=None,
                        help="write results here instead of runs_evaluation/<generated name>")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip plots; also skips per-step tracing, so rollouts are faster")
    parser.add_argument("--plot-episodes", type=int, default=3,
                        help="how many individual episodes get their own detail figure")
    parser.add_argument("--brief", action="store_true",
                        help="print the table without the explanatory legend")
    parser.add_argument("--methods", default="initial,greedy,constructive,random,learned",
                        help="comma-separated subset of " + ",".join(ALL_METHODS) + ", or "
                             "'all'. greedy and anneal are the expensive ones at large n, and "
                             "optimal is exhaustive search (n <= %d)" % MAX_BRUTE_FORCE_N)
    parser.add_argument("--restarts", type=int, default=20,
                        help="restarts for the constructive baseline; it reports the best of them")
    parser.add_argument("--spectral-shortlist", type=int, default=5,
                        help="how many of the closed form's top candidates the spectral "
                             "baseline rescores before applying one. The ranking is exact "
                             "at stiffness_kappa = 0 and a prior above it, which is what "
                             "the verification is for")
    parser.add_argument("--anneal-budget", type=int, default=None,
                        help="phi evaluations the annealer gets. Default: exactly what greedy "
                             "spent on the same instance, so the two are budget matched; "
                             "4*n*(n-1) when greedy is not being run")
    parser.add_argument("--policy-mode", default="sample", choices=("sample", "greedy"),
                        help="sample (default): sampled actions, i.e. the policy used as a "
                             "sampling search over the horizon, scored on the best state it "
                             "finds. greedy: the single action the policy considers best, "
                             "which is reproducible and terminates on a cycle. A DQN q-network "
                             "has nothing to sample from and is argmax either way.")
    parser.add_argument("--noise-sweep", default=None,
                        help="comma-separated bearing-noise levels in DEGREES; measures "
                             "the shape error each method's graph actually produces "
                             "against the analytic prediction, e.g. 0.5,1,5")
    parser.add_argument("--noise-trials", type=int, default=30,
                        help="noise draws per (method, sigma)")
    parser.add_argument("--replay-env", action="store_true",
                        help="score every method against the environment --model was trained "
                             "on (from its manifest) instead of the current code")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if methods == ["all"]:
        methods = list(ALL_METHODS)
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        print(f"unknown method(s): {unknown}, expected {list(ALL_METHODS)}")
        return 1
    # --brute-force is the older spelling of the same thing
    if args.brute_force and "optimal" not in methods:
        methods.append("optimal")
    methods = [m for m in ALL_METHODS if m in methods]

    filepath = "./environments/" + args.environment_name + ".json"
    if not os.path.exists(filepath):
        print(f"file {filepath} does not exist")
        return 1

    np.random.seed(args.seed)

    with open(filepath, "r") as f:
        env_config_data = json.load(f)

    env = Environment()
    env.load(filepath)
    # gymnasium spaces carry their own RNG, so np.random.seed does not make
    # action_space.sample() reproducible -- the random baseline needs this too
    env.action_space.seed(args.seed)

    steps = args.steps
    if steps is None:
        steps = int(env.truncate_max_steps if env.truncate_enable else env.max_steps)
    n = env.network.n

    agent = wrapped = None
    if args.model:
        from skrl.envs.wrappers.torch import wrap_env
        from agent_loader import load_agent, load_run

        if args.replay_env:
            # every method is then scored through the archived compute_state_score, which
            # is what keeps the table internally consistent with the run being evaluated
            agent, wrapped, env, info = load_run(
                args.model, env_name=args.environment_name, device=args.device
            )
            algorithm = (info or {}).get("algorithm", "?")
            env.action_space.seed(args.seed)
            n = env.network.n
            if args.steps is None:
                steps = int(env.truncate_max_steps if env.truncate_enable else env.max_steps)
        else:
            # the wrapper reads env.device to decide where to put observations; without
            # this it defaults to cuda while the agent is built on cpu
            env.device = args.device
            wrapped = wrap_env(env)
            wrapped.reset()
            agent, algorithm = load_agent(args.model, wrapped, env, device=args.device)
        print(f"loaded {algorithm} model '{args.model}' on {args.device}"
              f"{' (replaying its archived environment)' if args.replay_env else ''}")

    if "optimal" in methods and n > MAX_BRUTE_FORCE_N:
        print(f"brute force refused: n={n} > {MAX_BRUTE_FORCE_N} "
              f"({2 ** (n * n - n):.3g} possible graphs)")
        methods.remove("optimal")

    print(f"env: {args.environment_name} | n={n} | rollout horizon={steps} | episodes={args.episodes}")

    # tracing costs one extra eigendecomposition per step, so it rides with the plots
    tracing = not args.no_plots
    # an archived environment replayed by --replay-env may predate this flag
    if hasattr(env, "trace_min_eig"):
        env.trace_min_eig = tracing
    traces = [] if tracing else None

    # re-seed here so the episodes drawn below do not depend on how much RNG the setup
    # above consumed: loading a model does an extra reset, which would otherwise make a
    # --model run and a plain run evaluate different instances
    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    # a sampling policy draws from torch's global RNG, so this is what makes
    # --policy-mode sample repeatable for a given seed
    torch.manual_seed(args.seed)

    # private to the methods that use randomness: np.random is the stream instances are
    # drawn from, so a method sharing it would change which networks the others are scored
    # on. See the note in _construct_once.
    construct_rng = np.random.default_rng(args.seed)
    anneal_rng = np.random.default_rng(args.seed)
    degree_rng = np.random.default_rng(args.seed)

    instances = []
    # every legal edit is scored at every step, so this is gated on n
    decisions = [] if (agent is not None and not args.no_plots
                       and n <= MAX_DECISION_ANALYSIS_N) else None
    if decisions is None and agent is not None and not args.no_plots:
        print(f"  decision analysis skipped: n={n} > {MAX_DECISION_ANALYSIS_N}")
    # the repair figure enumerates every minimum-size repair, so it is small-n only
    repair_rng = np.random.default_rng(args.seed)
    repair_records = [] if (not args.no_plots and n <= MAX_REPAIR_FIGURE_N) else None
    if repair_records is None and not args.no_plots:
        print(f"  repair-choice figure skipped: n={n} > {MAX_REPAIR_FIGURE_N}")

    # the flag is in degrees because that is what a camera spec is quoted in;
    # everything downstream of here is radians
    sweep_degrees = ([float(x) for x in args.noise_sweep.split(",")]
                     if args.noise_sweep else [])
    sigmas = [float(np.radians(d)) for d in sweep_degrees]
    if sigmas:
        print(f"  noise sweep: {sweep_degrees} degrees of bearing error, "
              f"{args.noise_trials} draws each")

    frozen = None
    if args.benchmark:
        frozen, bench_meta = benchmark.load(args.benchmark)
        if len(frozen) < args.episodes:
            print(f"  benchmark {args.benchmark} has {len(frozen)} instances; "
                  f"running that many instead of {args.episodes}")
            args.episodes = len(frozen)
        print(f"  instances: benchmark {args.benchmark} "
              f"({benchmark.digest(args.benchmark)}), sampling disabled")

    rows = []
    for ep in range(args.episodes):
        env.freeze_network = False
        if frozen is not None:
            env.network = copy.deepcopy(frozen[ep])
            env.freeze_network = True
            env.reset()                          # bookkeeping only
        else:
            env.reset()                          # draw a fresh instance
        instance = copy.deepcopy(env.network)
        env.freeze_network = True                # every reset below keeps it
        if len(instances) < max(args.plot_episodes, 1):
            instances.append(copy.deepcopy(env.network))
        if repair_records is not None and len(repair_records) < REPAIR_FIGURE_INSTANCES:
            rigid = copy.deepcopy(env.network)
            greedy_rigid_construction(rigid, env.rank_K, repair_rng)
            rec = repair_spread(rigid, env.rank_K, repair_rng)
            if rec is not None:
                repair_records.append(rec)

        def restore():
            env.network = copy.deepcopy(instance)
            env.reset()

        ctx = dict(env=env, args=args, traces=traces, episode=ep, steps=steps,
                   agent=agent, wrapped=wrapped, decisions=decisions,
                   construct_rng=construct_rng, anneal_rng=anneal_rng,
                   degree_rng=degree_rng, anneal_budget=args.anneal_budget)

        episode_rows = []
        for name in methods:
            restore()
            # the restore is shared setup, so only the method itself is metered
            with cost.Meter() as meter:
                row = run_method(name, ctx)
            if row is None:                      # unavailable: no model, or n too large
                continue
            row["cost"] = meter.counts
            row["ms"] = meter.ms
            episode_rows.append(row)
            # budget matching, when it was not set on the command line: the annealer gets
            # exactly the phi evaluations greedy spent on this instance
            if name == "greedy" and args.anneal_budget is None:
                ctx["anneal_budget"] = meter.counts.get("score_network", 0) or None

        if sigmas:
            for r in episode_rows:
                measure_noise(env, r, sigmas, args.noise_trials,
                              np.random.default_rng(args.seed + ep))

        for r in episode_rows:
            r["episode"] = ep
        rows.extend(episode_rows)

        line = "  ".join(f"{r['method']}: m={r['m']} phi={r['score']:.1f}" for r in episode_rows)
        print(f"  ep {ep:>3}  {line}")

    # ── report ────────────────────────────────────────────────────────────────────────
    domains = env.domains if isinstance(env.domains, list) else [env.domains]
    domain_str = domains[0] if len(set(domains)) == 1 else f"mixed {sorted(set(domains))}"
    context = {
        "environment": args.environment_name,
        "network": f"{n} agents in {domain_str}, action space {env.action_space_type}",
        "objective": f"{env.state_score_type} state score",
        "instances": (f"{args.episodes} networks from benchmark {args.benchmark} "
                      f"({benchmark.digest(args.benchmark)})" if args.benchmark
                      else f"{args.episodes} random networks, seed {args.seed}"),
    }
    if args.model:
        context["policy"] = (f"{args.model} ({algorithm}, --policy-mode {args.policy_mode}, "
                             f"{steps}-step budget)")

    table = report.format_table(rows, context, brief=args.brief)
    print("\n" + table)

    run_dir = report.make_run_dir("runs_evaluation", args.environment_name,
                                  model_name=args.model, tag=args.tag, out_dir=args.out_dir,
                                  with_plots=bool(traces))
    report.write_summary(run_dir, table)
    report.write_csvs(run_dir, rows, traces)
    wrote_costs = report.write_costs(run_dir, rows)
    if decisions:
        report.write_decisions(run_dir, decisions)
    report.write_meta(run_dir, {
        "args": vars(args),
        "environment_config": env_config_data,
        "n": n, "rollout_steps": steps,
        # the instance set, so two runs are only comparable when these agree
        "benchmark": args.benchmark,
        "benchmark_digest": benchmark.digest(args.benchmark) if args.benchmark else None,
        "provenance": manifest.collect_provenance(seed=args.seed, device=args.device),
    })

    written = ["summary.txt", "results.csv"] + (["trajectories.csv"] if traces else [])
    if wrote_costs:
        written.append("cost.csv and cost.txt")
    if decisions:
        written.append("decisions.csv")
    if traces:
        # the full names go in the figure titles (wrapped): a plot pulled into a slide
        # has to say which model and which environment produced it
        header = {
            "short": report.short_env_name(args.environment_name),
            "env": args.environment_name,
            "model": args.model,
            "network": context["network"],
            "episodes": args.episodes,
            "seed": args.seed,
            "benchmark": args.benchmark,
        }
        # the table itself, so the numbers travel with the figures. The policy line drops
        # the model name -- the header already carries it in full one line above
        policy = (f"{algorithm}, --policy-mode {args.policy_mode}, {steps}-step budget"
                  if args.model else None)
        table_header = dict(header, objective=context["objective"], policy=policy)

        def draw_all():
            report.plot_trajectories(run_dir, traces, rows, header)
            report.plot_outcomes(run_dir, traces, rows, header)
            report.plot_summary(run_dir, rows, header)
            report.plot_noise_sweep(run_dir, rows, header)
            report.plot_prediction_check(run_dir, rows, header)
            report.plot_uncertainty(run_dir, instances, rows, header)
            report.plot_softest_mode(run_dir, instances, rows, header)
            report.plot_sensitivity(run_dir, instances, rows, header)
            report.plot_repair_choice(run_dir, repair_records, header)
            report.plot_cost(run_dir, rows, header)
            report.plot_decisions(run_dir, decisions, header)
            report.plot_table(run_dir, rows, table_header)
            for ep in range(min(args.plot_episodes, args.episodes)):
                sel = [t for t in traces if t["episode"] == ep]
                ep_header = dict(header, episodes=None, subtitle=f"episode {ep}")
                report.plot_trajectories(run_dir, sel,
                                         [r for r in rows if r["episode"] == ep],
                                         ep_header, filename=f"episode_{ep:03d}",
                                         aggregate_over_episodes=False)

        draw_all()
        # and again without the title block or the notes card, for a document that
        # carries its own caption
        with report.plain():
            draw_all()
        # counted rather than predicted: several figures skip themselves when the
        # run does not carry what they draw
        made = len(glob.glob(os.path.join(run_dir, "plots", "png", "*.png")))
        written.append(f"plots/pdf/ and plots/png/ ({made} figures each)")

    print(f"\nwrote {run_dir}/")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
