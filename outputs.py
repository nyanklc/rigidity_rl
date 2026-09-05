"""Every figure and table the thesis reports, in one script.

Every method is scored with Environment.compute_state_score, i.e. the exact state
score phi the agent trains on, so the rows are directly comparable:

  initial  the graph the sampler produced (what the agent starts from)
  random   uniform random actions through env.step()  -> the floor for this action space
  greedy   hill-climbing on phi, one edge toggle at a time
  constructive  builds from the empty graph, keeping any edge that raises rank(B)
  learned  a trained policy, sampling actions (--policy-mode greedy for argmax instead).
           --model takes several, and each becomes its own row
  optimal  exhaustive search for the fewest-edge rigid graph (small n only)

greedy gets stuck exactly where RL should win: states where no single edit improves
phi but a swap of two does.

usage:
  uv run outputs.py <environment_name> [--episodes N] [--model NAME] [--brute-force]
      [--steps K] [--tag NAME] [--device cpu|cuda] [--methods a,b,c] [--restarts K]
      [--policy-mode sample|greedy] [--benchmark NAME] [--noise-sweep 0.5,1,5]
      [--no-plots] [--plot-episodes N] [--brief] [--out-dir PATH] [--replay-env]

Writes one directory per run under runs_outputs/: the table, per-episode results,
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
from datetime import datetime
from types import SimpleNamespace

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
           shape_err=None, edges=None, edges_are="final", m_req=None):
    """One method's outcome on one instance.

    `work` counts graph modifications actually applied and `best_at` is the step the best
    graph was reached at -- the old single `steps` column meant different things per method.

    `m_req` is the instance's own lower bound, recorded per row because it is
    pose-dependent: without it nothing downstream can say whether a graph hit the bound
    without hardcoding a number for one benchmark.
    """
    return {
        "method": method,
        "m_req": None if m_req is None else int(m_req),
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
               decisions=None, label="learned"):
    """Roll out a trained policy, scored on the best state visited.

    `label` is the row this policy occupies. With one model it stays `learned`, which is
    what every run written before --model took a list is keyed on.

    mode="greedy"  the action the policy considers best -- what you would deploy
    mode="sample"  sampled actions, i.e. the policy used as a sampling-based search over the
                   horizon (PPO only; a DQN q-network has nothing to sample from)
    """
    agent.enable_models_training_mode(False)  # eval mode (skrl 2.x naming)
    obs, _ = wrapped_env.reset()  # freeze_network keeps the instance
    seen = set()
    record(trace, label, episode, 0, step_stats(raw_env, trace is not None))
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
                        rec.update(model=label, episode=episode, step=t)
                        decisions.append(rec)
        record(trace, label, episode, t + 1, step_stats(raw_env, trace is not None))

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

    return rollout_result(label, raw_env, work)


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
CLASSICAL_METHODS = ("initial", "random", "degree", "greedy", "spectral", "anneal",
                     "constructive")
# `learned` is the label a single model takes, and the name --methods accepts for "every
# model" whatever they are called. optimal runs last because it is the slowest.
ALL_METHODS = CLASSICAL_METHODS + ("learned", "optimal")


def method_sequence(labels):
    """Run order: the classical methods, then each model, then brute force.

    greedy has to come before anneal, whose budget is the phi-evaluation count greedy
    just spent on the same instance.
    """
    return list(CLASSICAL_METHODS) + list(labels) + ["optimal"]


LABEL_WIDTH = 14


def run_info_blocks(args, env, env_config_data, models, steps, methods, rows):
    """[(section, [(field, value)])] for the run_info figure.

    Everything a reader needs to place this run: what was evaluated, on what, and what
    each model label stands for. The per-model block records the objective a model was
    *trained* on against the one it is *scored* by, which is the one difference between
    models that the table cannot show and that changes how its rows should be read.
    """
    domains = env.domains if isinstance(env.domains, list) else [env.domains]
    dom = domains[0] if len(set(domains)) == 1 else f"mixed {sorted(set(domains))}"
    kappa = env_config_data.get("stiffness_kappa")

    blocks = [("this run", [
        ("environment", args.environment_name),
        ("network", f"{env.network.n} agents in {dom}"),
        ("action space", env.action_space_type),
        ("objective", f"{env.state_score_type}, stiffness_kappa {kappa}"),
        ("instances", (f"{args.episodes} from benchmark {args.benchmark} "
                       f"({benchmark.digest(args.benchmark)})" if args.benchmark
                       else f"{args.episodes} random, seed {args.seed}")),
        ("methods", ", ".join(methods)),
        ("rollout", f"--policy-mode {args.policy_mode}, {steps} steps, every model"),
        ("device", args.device),
        ("written", datetime.now().strftime("%Y-%m-%d %H:%M")),
    ])]

    for label, m in models.items():
        man = m["manifest"] or {}
        cfg = man.get("environment_config_raw") or {}
        prov = man.get("provenance") or {}
        hp = man.get("hyperparameters") or {}
        lr = hp.get("learning_rate")
        trained_kappa = cfg.get("stiffness_kappa")
        objective = f"{cfg.get('state_score_type', '?')}, stiffness_kappa {trained_kappa}"
        if trained_kappa != kappa:
            objective += "   (scored here at %s)" % kappa
        blocks.append((f"model  {label}", [
            ("name", m["name"]),
            ("algorithm", f"{m['algorithm']}, {man.get('backbone', '?')} backbone"),
            ("width", f"gnn {man.get('gnn_hidden_dim', '?')}, "
                      f"head {man.get('head_hidden_dim', '?')}"),
            ("training", f"{man.get('timesteps_completed', '?')} of "
                         f"{man.get('total_timesteps_configured', '?')} steps, "
                         f"{man.get('status', '?')}"),
            ("learning rate", lr if not isinstance(lr, list) else ", ".join(map(str, lr))),
            ("trained on", man.get("environment_config")),
            ("its objective", objective),
            ("seed / commit", f"{prov.get('seed')} / {str(prov.get('git_commit'))[:12]}"
                              f"{' (dirty)' if prov.get('git_dirty') else ''}"),
        ]))
    return blocks


def load_manifest(name):
    """The run manifest for a trained model, or {} when there is none."""
    path = os.path.join("train", f"{name}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# The config keys that decide the observation's width and meaning. A model whose training
# config disagrees with the evaluation one on any of these has a checkpoint of the wrong
# shape, which surfaces deep inside agent_loader as "parameter shapes do not match".
OBS_KEYS = ("n", "obs_type", "action_type", "graph_features",
            "include_candidate_bearings", "rigidity_global", "rigidity_quality",
            "rigidity_flex", "rigidity_edge", "rigidity_stiffness", "rigidity_removal")


def observation_mismatch(trained, current):
    """[(key, trained value, current value)] for the keys that change the observation."""
    return [(k, trained.get(k), current.get(k)) for k in OBS_KEYS
            if trained.get(k) != current.get(k)]


def parse_models(values):
    """--model NAME[=LABEL], repeatable and comma-splittable. -> [(name, label)].

    One model keeps the label `learned`: that is the row name every run and every test
    written before --model took a list is keyed on. Two or more are labelled by the
    `_`-separated tokens that actually differ between them, so
    stiff_dqn_gine / stiff_ppo_gine read as dqn / ppo rather than as two long names.
    """
    names, given = [], {}
    for v in values or []:
        for part in v.split(","):
            part = part.strip()
            if not part:
                continue
            name, _, label = part.partition("=")
            names.append(name)
            if label:
                given[name] = label
    if not names:
        return []
    if len(names) == 1 and not given:
        return [(names[0], "learned")]

    auto = auto_labels(names)
    out, seen = [], {}
    for name in names:
        label = given.get(name, auto[name])
        if label in seen:
            raise SystemExit(
                f"--model: {name} and {seen[label]} both come out as '{label}'. "
                f"Pass an explicit label, e.g. --model {name}=<label>")
        seen[label] = name
        out.append((name, label))
    return out


def auto_labels(names):
    """{name: label} from the `_` tokens that tell `names` apart.

    Tokens every name carries say nothing, so they go. What is left is trimmed from the
    front if it still does not fit, since the distinguishing part of these names is
    usually at the end (stiff_dqn_gine vs stiff_dqn_equi).
    """
    parts = [n.split("_") for n in names]
    shared = set(parts[0]).intersection(*(set(p) for p in parts[1:]))
    out = {}
    for name, p in zip(names, parts):
        keep = [t for t in p if t not in shared] or p
        label = "_".join(keep)
        while len(label) > LABEL_WIDTH and len(keep) > 1:
            keep = keep[1:]
            label = "_".join(keep)
        out[name] = label[:LABEL_WIDTH]
    return out


SECTIONS = ("baselines", "ablation", "training", "generalisation")


def section_generalisation(args, run_dir, models, header):
    """Gather earlier output directories, one per instance set. -> (written, draws).

    Reads what those runs already measured rather than re-running them, so the rows
    can carry the code they were produced by.
    """
    pattern = args.prior or os.path.join("runs_outputs", "*")
    wanted = {m["name"] for m in models.values()} | {"greedy"}
    # the label this run gave each model, so the figure's series match its table's rows
    short = {m["name"]: lab for lab, m in models.items()}

    best = {}                     # (method, benchmark) -> (dir mtime, row)
    for d in sorted(glob.glob(pattern)):
        meta_path = os.path.join(d, "meta.json")
        res_path = os.path.join(d, "results.csv")
        if not (os.path.exists(meta_path) and os.path.exists(res_path)):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        bench = meta.get("benchmark")
        if not bench:
            continue
        # A run written before --model took a list records the model in args and calls
        # its row `learned`. Without this, every such run contributes only its
        # baselines and the figure looks like the models were never evaluated.
        labels = meta.get("models") or {}
        if not labels:
            was = meta.get("args", {}).get("model")
            if isinstance(was, str) and was:
                labels = {"learned": was}
        with open(res_path) as f:
            rows = list(csv.DictReader(f))
        # a run's own label -> the model behind it, so rows from runs that used
        # different labels for the same checkpoint still line up
        for method in {r["method"] for r in rows}:
            name = labels.get(method, method)
            if name not in wanted:
                continue
            sel = [r for r in rows if r["method"] == method]
            row = _generalisation_row(short.get(name, name), bench, sel, d, meta)
            if row is None:
                continue
            key = (name, bench)
            stamp = os.path.getmtime(d)
            if key not in best or stamp > best[key][0]:
                best[key] = (stamp, row)

    gen_rows = [row for _, row in best.values()]
    if not gen_rows:
        print(f"  generalisation: no earlier runs matched {pattern} for "
              f"{', '.join(sorted(wanted))}")
        return [], []

    gen_rows.sort(key=lambda r: (r["n"] or 0, r["benchmark"], r["method"]))
    path = os.path.join(run_dir, "generalisation.csv")
    fields = ["method", "benchmark", "n", "m_req", "episodes", "rigid_pct",
              "at_bound_pct", "edges_over_bound", "phi", "shape_err", "source_dir",
              "git_commit"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in gen_rows:
            wr.writerow(r)
    print(f"  generalisation: {len(gen_rows)} rows over "
          f"{len({r['benchmark'] for r in gen_rows})} instance sets")
    return [os.path.relpath(path, run_dir)], [
        lambda: report.plot_generalisation(run_dir, gen_rows, header)]


def _generalisation_row(name, bench, sel, source_dir, meta):
    """One (method, benchmark) summary from a prior run's rows, or None."""
    def num(r, k):
        v = r.get(k)
        return None if v in (None, "", "nan") else float(v)

    m = [num(r, "m") for r in sel if num(r, "m") is not None]
    if not m:
        return None
    bounds = [num(r, "m_req") for r in sel if num(r, "m_req") is not None]
    rigid = [r.get("is_IBR") == "True" for r in sel]
    at_bound = [r.get("at_bound") == "True" for r in sel
                if r.get("at_bound") not in (None, "")]
    err = [num(r, "shape_err") for r in sel
           if num(r, "shape_err") and num(r, "shape_err") > 0]
    bound = float(np.mean(bounds)) if bounds else None
    return {
        "method": name,
        "benchmark": bench,
        "n": meta.get("n"),
        "m_req": bound,
        "episodes": len(sel),
        "rigid_pct": 100.0 * float(np.mean(rigid)) if rigid else None,
        # None rather than 0 where the prior run predates m_req in results.csv, so a
        # missing bar reads as "not recorded" and not as "never reached the bound"
        "at_bound_pct": 100.0 * float(np.mean(at_bound)) if at_bound else None,
        "edges_over_bound": (float(np.mean(m)) / bound) if bound else None,
        "phi": float(np.mean([num(r, "score") for r in sel])),
        "shape_err": report._gmean(err),
        "source_dir": source_dir,
        "git_commit": str((meta.get("provenance") or {}).get("git_commit", ""))[:12],
    }


def section_training(args, run_dir, models, header):
    """Learning curves from the tensorboard logs under runs/. -> (written, draws)."""
    from tools.compare_runs import load as load_events, tail

    names = ([r.strip() for r in args.runs.split(",") if r.strip()] if args.runs
             else [m["name"] for m in models.values()])
    labels = {}
    for name in names:
        # the label is the model's if this run belongs to one, so a curve and a table
        # row for the same policy carry the same name and the same colour
        label = next((lab for lab, m in models.items() if m["name"] == name), name)
        labels[label] = name

    series, missing = {}, []
    for label, name in labels.items():
        if not os.path.isdir(os.path.join("runs", name)):
            missing.append(name)
            continue
        scalars, _ = load_events(name)
        series[label] = {t: (np.array([x.step for x in s], float),
                             np.array([x.value for x in s], float))
                         for t, s in scalars.items()}
    if missing:
        print(f"  training: no tensorboard directory for {', '.join(missing)}")
    if not series:
        return [], []

    path = os.path.join(run_dir, "training.csv")
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["run", "metric", "steps", "tail_mean"])
        for label, tags in series.items():
            for tag, (st, v) in sorted(tags.items()):
                # the tail mean is what a curve is read for; the raw series stays in
                # the event files rather than being copied here
                wr.writerow([labels[label], tag.strip(), len(v),
                             f"{float(np.mean(v[int(len(v) * 0.85):])):.6g}"])
    return [os.path.relpath(path, run_dir)], [
        lambda: report.plot_training(run_dir, series, header)]


def section_ablation(args, run_dir, env, wrapped, models, header, doing=None):
    """Destroy each observation channel in turn, per model. -> (written, draws).

    Runs against the environment and the agents the baselines section already built, so
    two sections in one directory cannot report numbers measured on different
    environments.
    """
    import ablation
    from skrl.utils.spaces.torch import unflatten_tensorized_space

    out_dir = os.path.join(run_dir, "ablation")
    os.makedirs(out_dir, exist_ok=True)
    space = wrapped.observation_space
    steps = int(min(getattr(env, "max_steps", 100), 100))
    modes = [m.strip() for m in args.ablation_mode.split(",") if m.strip()]
    if modes == ["all"]:
        modes = ["shuffle", "zero", "noise"]
    bad = [m for m in modes if m not in ("shuffle", "zero", "noise")]
    if bad:
        raise SystemExit(f"--ablation-mode: {bad}, expected shuffle, zero, noise or all")

    written, draws = [], []
    for label, m in models.items():
        agent = m["agent"]
        agent.enable_models_training_mode(False)
        env.freeze_network = False
        obs, _ = wrapped.reset()
        layout = ablation.resolve_layout(env, unflatten_tensorized_space(space, obs))
        per_mode = {}
        for mode in modes:
            if doing is not None:
                doing(f"ablation {label}, {mode}, {len(layout)} channels x "
                      f"{args.ablation_episodes} episodes")
            rng = torch.Generator(device="cpu").manual_seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            rows, ref_row, converged = ablation.measure(
                agent, wrapped, env, layout, space, episodes=args.ablation_episodes,
                steps=steps, mode=mode, rng=rng, progress=False)
            per_mode[mode] = (rows, ref_row)
            meta = (f"model {m['name']}  |  env={args.environment_name}"
                    f"  |  n={env.network.n}  |  {len(layout)} channels  |  mode={mode}"
                    f"  |  {args.ablation_episodes} episodes x {steps} steps"
                    f"  |  seed {args.seed}  |  live env"
                    f"  |  stopped at the reference's convergence, "
                    f"median {int(np.median(converged))} of {steps} steps")
            csv_path = os.path.join(out_dir, f"{label}-{mode}.csv")
            ablation.write_csv(csv_path, rows, ref_row,
                               SimpleNamespace(mode=mode, episodes=args.ablation_episodes),
                               meta)
            written.append(os.path.relpath(csv_path, run_dir))

        draws.append(lambda label=label, per_mode=per_mode, m=m:
                     report.plot_ablation(run_dir, per_mode, header, m["name"],
                                          filename=f"ablation-{label}"))
    env.freeze_network = False
    return written, draws


def section_refusal(name, args, models):
    """Why `name` cannot run here, as a sentence, or None."""
    if name in ("ablation", "generalisation") and not models:
        return f"{name} needs at least one --model"
    if name == "training" and not (args.runs or models):
        return "training needs --runs, or a --model whose name is a directory under runs/"
    return None


def run_method(name, ctx):
    """Dispatch one method on the instance currently in ctx['env']. None if unavailable.

    Nothing here draws its own progress bar: the run has one, and a nested bar either
    fights it for the line or scrolls it away.
    """
    env, args, traces, ep = ctx["env"], ctx["args"], ctx["traces"], ctx["episode"]
    if name == "initial":
        return run_initial(env, trace=traces, episode=ep)
    if name == "greedy":
        return run_greedy(env, verbose=False, trace=traces, episode=ep)
    if name == "spectral":
        return run_spectral(env, shortlist=args.spectral_shortlist, verbose=False,
                            trace=traces, episode=ep)
    if name == "anneal":
        return run_anneal(env, ctx["anneal_rng"], budget=ctx["anneal_budget"],
                          verbose=False, trace=traces, episode=ep)
    if name == "degree":
        return run_degree(env, ctx["degree_rng"], verbose=False, trace=traces, episode=ep)
    if name == "constructive":
        return run_constructive(env, ctx["construct_rng"], restarts=args.restarts,
                                verbose=False, trace=traces, episode=ep)
    if name == "random":
        return run_random(env, ctx["steps"], trace=traces, episode=ep)
    if name in ctx["models"]:
        m = ctx["models"][name]
        return run_policy(m["agent"], ctx["wrapped"], env, ctx["steps"],
                          mode=args.policy_mode, trace=traces, episode=ep,
                          decisions=ctx["decisions"], label=name)
    if name == "optimal":
        return run_brute_force(env, verbose=False)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("environment_name")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model", action="append", default=None,
                        help="trained model from train/, repeatable and comma-splittable. "
                             "NAME=LABEL names its row; otherwise a label is derived from "
                             "the parts of the names that differ. One model is labelled "
                             "`learned`")
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
                        help="write results here instead of runs_outputs/<generated name>")
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
    parser.add_argument("--sections", default="baselines",
                        help="comma-separated subset of " + ",".join(SECTIONS) +
                             ", or 'all'. baselines is the method comparison; the rest "
                             "need their own inputs and say so when they are missing")
    parser.add_argument("--runs", default=None,
                        help="tensorboard directories under runs/ for the training "
                             "section; defaults to the model names")
    parser.add_argument("--prior", default=None,
                        help="where the generalisation section looks for the earlier "
                             "runs it aggregates, one per instance set. Default "
                             "runs_outputs/*. It reads what those runs already measured "
                             "rather than re-running them, so evaluate each benchmark "
                             "once with --benchmark and this figure then compares them")
    parser.add_argument("--ablation-mode", default="shuffle",
                        help="shuffle, zero, noise, a comma-separated subset, or 'all'. "
                             "Reading one mode alone is how a channel gets called "
                             "unimportant when it was simply not perturbed")
    parser.add_argument("--ablation-episodes", type=int, default=10)
    parser.add_argument("--replay-env", action="store_true",
                        help="score every method against the environment --model was trained "
                             "on (from its manifest) instead of the current code")
    args = parser.parse_args()

    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    if sections == ["all"]:
        sections = list(SECTIONS)
    unknown = [s for s in sections if s not in SECTIONS]
    if unknown:
        print(f"unknown section(s): {unknown}, expected {list(SECTIONS)}")
        return 1

    model_specs = parse_models(args.model)
    labels = [lab for _, lab in model_specs]
    if args.replay_env and len(model_specs) > 1:
        print("--replay-env replays one model's archived environment, so with several "
              "models the rows would not all be scored by the same phi. Pass one "
              "--model, or drop --replay-env.")
        return 1

    known = method_sequence(labels)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if methods == ["all"]:
        methods = list(known)
    # `learned` names every model whatever they are called, which is what keeps the
    # default --methods string meaning what it always meant
    methods = [m for x in methods for m in (labels if x == "learned" else [x])]
    unknown = [m for m in methods if m not in known]
    if unknown:
        print(f"unknown method(s): {unknown}, expected {known}")
        return 1
    # --brute-force is the older spelling of the same thing
    if args.brute_force and "optimal" not in methods:
        methods.append("optimal")
    methods = [m for m in known if m in methods]

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

    wrapped = None
    models = {}                       # label -> {name, agent, algorithm, manifest}
    if model_specs:
        from skrl.envs.wrappers.torch import wrap_env
        from agent_loader import load_agent, load_run

        if args.replay_env:
            # load_run execs one model's archived rigidity.py and environment.py into
            # sys.modules, so two archives cannot both be live
            (name, label), = model_specs
            agent, wrapped, env, info = load_run(
                name, env_name=args.environment_name, device=args.device
            )
            models[label] = {"name": name, "agent": agent,
                             "algorithm": (info or {}).get("algorithm", "?"),
                             "manifest": info or {}}
            env.action_space.seed(args.seed)
            n = env.network.n
            if args.steps is None:
                steps = int(env.truncate_max_steps if env.truncate_enable else env.max_steps)
        else:
            # the wrapper reads env.device to decide where to put observations; without
            # this it defaults to cuda while the agent is built on cpu
            bad = []
            for name, _ in model_specs:
                diff = observation_mismatch(
                    load_manifest(name).get("environment_config_raw") or {}, env_config_data)
                if diff:
                    bad.append((name, diff))
            if bad:
                print("these models were trained on a different observation, so their "
                      "checkpoints do not fit this environment:")
                for name, diff in bad:
                    for k, was, now in diff:
                        print(f"  {name}: {k} was {was}, this environment has {now}")
                return 1

            env.device = args.device
            wrapped = wrap_env(env)
            wrapped.reset()
            for name, label in model_specs:
                agent, algorithm = load_agent(name, wrapped, env, device=args.device)
                # not `manifest`: that name is the module this file imports
                man = load_manifest(name)
                models[label] = {"name": name, "agent": agent, "algorithm": algorithm,
                                 "manifest": man}
                trained_on = man.get("environment_config")
                # the same environment gets regenerated under different names, so the
                # name is only a warning
                if isinstance(trained_on, str) and trained_on != args.environment_name:
                    print(f"  note: {name} was trained on {trained_on}, "
                          f"scoring it on {args.environment_name}")
        for label, m in models.items():
            print(f"loaded {m['algorithm']} model '{m['name']}' as '{label}' "
                  f"on {args.device}"
                  f"{' (replaying its archived environment)' if args.replay_env else ''}")

    report.configure_methods([(lab, f"trained policy {m['name']} ({m['algorithm']})")
                              for lab, m in models.items()])

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
    decisions = [] if (models and not args.no_plots
                       and n <= MAX_DECISION_ANALYSIS_N) else None
    if decisions is None and models and not args.no_plots:
        print(f"  decision analysis skipped: n={n} > {MAX_DECISION_ANALYSIS_N}")
    elif decisions is not None and len(models) > 1:
        # it scores every legal edit at every step, once per model
        print(f"  decision analysis runs for all {len(models)} models")
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

    # One bar for the run. Every method on every episode is a unit, the noise sweep is
    # one more per episode, and the report is the last; the description says which.
    units = args.episodes * (len(methods) + (1 if sigmas else 0)) + 1
    progress = tqdm(total=units, unit="step", dynamic_ncols=True, leave=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    def doing(what):
        progress.set_description_str(what)

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
                   models=models, wrapped=wrapped, decisions=decisions,
                   construct_rng=construct_rng, anneal_rng=anneal_rng,
                   degree_rng=degree_rng, anneal_budget=args.anneal_budget)

        episode_rows = []
        for name in methods:
            doing(f"episode {ep + 1}/{args.episodes}  {name}")
            restore()
            # the restore is shared setup, so only the method itself is metered
            with cost.Meter() as meter:
                row = run_method(name, ctx)
            progress.update(1)
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
            doing(f"episode {ep + 1}/{args.episodes}  bearing noise")
            for r in episode_rows:
                measure_noise(env, r, sigmas, args.noise_trials,
                              np.random.default_rng(args.seed + ep))
            progress.update(1)

        for r in episode_rows:
            r["episode"] = ep
            # the bound is a property of the poses, so it belongs to the row rather than
            # to the run; stamped here so no run_* can forget it
            r["m_req"] = int(env.m_req)
        rows.extend(episode_rows)


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
    if models:
        context["policy"] = "  |  ".join(
            f"{lab} = {m['name']} ({m['algorithm']})" for lab, m in models.items())
        context["rollout"] = (f"--policy-mode {args.policy_mode}, "
                              f"{steps}-step budget, same for every model")

    doing("writing the table")
    table = report.format_table(rows, context, brief=args.brief)

    run_dir = report.make_run_dir("runs_outputs", args.environment_name,
                                  model_names=[m["name"] for m in models.values()],
                                  tag=args.tag, out_dir=args.out_dir,
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
        # label -> the model behind it, since the labels are derived and the figures
        # only carry the short one
        "models": {lab: m["name"] for lab, m in models.items()},
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
        header = {
            "short": report.short_env_name(args.environment_name),
            "env": args.environment_name,
            "model": ("  ".join(f"{lab} = {m['name']}" for lab, m in models.items())
                      or None),
            "network": context["network"],
            "episodes": args.episodes,
            "seed": args.seed,
            "benchmark": args.benchmark,
        }
        policy = context.get("rollout")
        table_header = dict(header, objective=context["objective"], policy=policy)

        extra_draws = []
        for name in sections:
            if name != "baselines":
                doing(f"section {name}")
            if name == "baselines":
                continue
            why = section_refusal(name, args, models)
            if why:
                print(f"skipping {name}: {why}")
                continue
            if name == "ablation":
                files, draws = section_ablation(args, run_dir, env, wrapped, models,
                                                header, doing=doing)
            elif name == "training":
                files, draws = section_training(args, run_dir, models, header)
            elif name == "generalisation":
                files, draws = section_generalisation(args, run_dir, models, header)
            written.extend(files)
            extra_draws.extend(draws)

        def draw_all():
            for draw in extra_draws:
                draw()
            report.plot_trajectories(run_dir, traces, rows, header)
            report.plot_outcomes(run_dir, traces, rows, header)
            report.plot_summary(run_dir, rows, header)
            report.plot_noise_sweep(run_dir, rows, header)
            report.plot_prediction_check(run_dir, rows, header)
            report.plot_uncertainty(run_dir, instances, rows, header)
            report.plot_softest_mode(run_dir, instances, rows, header)
            report.plot_sensitivity(run_dir, instances, rows, header)
            report.plot_repair_choice(run_dir, repair_records, header)
            report.plot_comparison(run_dir, rows, header)
            report.plot_estimation(run_dir, rows, header)
            report.plot_topology(run_dir, instances, rows, header)
            report.plot_run_info(run_dir, run_info_blocks(
                args, env, env_config_data, models, steps, methods, rows), header)
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

        doing("drawing figures")
        draw_all()
        # and again without the title block or the notes card, for a document that
        # carries its own caption
        doing("drawing figures, plain")
        with report.plain():
            draw_all()
        # counted rather than predicted: several figures skip themselves when the
        # run does not carry what they draw
        made = len(glob.glob(os.path.join(run_dir, "plots", "png", "*.png")))
        written.append(f"plots/pdf/ and plots/png/ ({made} figures each)")

    progress.update(1)
    doing("done")
    progress.close()

    print("\n" + table)
    print(f"\nwrote {run_dir}/")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
