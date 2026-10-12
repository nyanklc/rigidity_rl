"""What each method does to a network, and how it is scored.

The methods every experiment compares, and the measurements they are compared by.
`experiments/` holds the runnable scripts; this is the library they share, so a
method has one implementation and one cost accounting wherever it is called from.

Every method is scored with `Environment.compute_state_score`, the exact phi the
agent trains on, so the rows are directly comparable:

  initial       the graph the sampler produced, i.e. what the agent starts from
  random        uniform actions through env.step() -- the floor for this action space
  degree        adds by lowest degree until rigid, then prunes. Reads nothing but
                degrees and the rigidity test, so it prices the tier-1 row of
                DESIGN_NOTES.md#distributed-feasibility
  greedy        hill-climbs phi one edge toggle at a time, n(n-1) evaluations per edit
  spectral      the same landscape in closed form, verified on a shortlist
  anneal        simulated annealing on phi, budget matched to greedy's measured spend
  constructive  from the empty graph, keeping any edge that raises rank(B)
  optimal       exhaustive search for the fewest-edge rigid graph (n <= 5)

greedy gets stuck exactly where RL should win: states where no single edit improves
phi but a swap of two does.
"""

import copy
import itertools
import math

import numpy as np
import torch
from tqdm import tqdm

from rigidity import (rigidity_eigenvalue, rigidity_decomposition,
                      greedy_rigid_construction, greedy_rigid_repair, repair_edge_count,
                      estimation_error_of, shape_error_of, absent_pairs,
                      extended_bearing_rigidity_matrix, measurement_sensitivity,
                      candidate_gain, characteristic_length, is_IBR_explicit,
                      nullspace_and_softest, nullspace_in_scaled_units, removal_costs)
import cost
import estimation


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
CLOSED_FORM_SCORES = {"WeightedNormalized", "WeightedNormalizedSpectral",
                      "WeightedNormalizedShapeError"}


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

    if stiffness and rank >= env.rank_K and env.stiffness_term_active():
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
                  shape_err=best_shape_error(env),
                  edges=env.best_edges, edges_are="best visited")


def rollout_result_final(method, env, work):
    """The graph the rollout stopped on, which is the answer when stopping is a choice.

    A policy with no stop action has to keep editing, so the best graph it visited is
    what it found and the one it ends on is an accident of the horizon. A policy with a
    stop action decides when to halt, and crediting it with a graph it walked away from
    would measure a search it is not running. The best-visited figures stay on the row
    beside the final ones, since the gap between them is what says whether it stopped in
    the right place.
    """
    st = env.last_stats or {}
    row = result(method, st.get("score", env.last_state_score), st.get("is_IBR", False),
                 st.get("is_MBR", False), st.get("m", int(env.network.edges.sum())),
                 work=work, best_at=env.best_step, min_eig=st.get("min_eig"),
                 shape_err=st.get("shape_err"), edges=env.network.edges,
                 edges_are="final")
    if row["shape_err"] is None and row["is_IBR"] and hasattr(env, "shape_error_now"):
        row["shape_err"] = _final_shape_error(env)
    row["best_m"] = int(env.best_stats["m"]) if env.best_stats else None
    row["best_score"] = float(env.best_state_score)
    # zero exactly when the episode ended on the best graph it saw
    row["best_final_gap"] = float(env.best_state_score - row["score"])
    return row


@cost.measurement
def _final_shape_error(env):
    """Shape error of the graph the rollout stopped on, if the step did not compute it."""
    return env.shape_error_now()


@cost.measurement
def best_shape_error(env):
    """Shape error of the best graph visited, computed here if the step did not.

    `step()` only computes it when the score reads it or tracing is on, so without
    this the rollout methods report a blank shape error under --no-plots while the
    edit-space methods, which measure it at the end, report one. That asymmetry is
    invisible in the table and the column is the one worth reading.
    """
    err = env.best_stats.get("shape_err")
    if err is not None or env.best_edges is None:
        return err
    if not hasattr(env, "shape_error_now"):     # an archived env under --replay-env
        return None
    was = env.network.edges
    env.network.edges = env.best_edges
    try:
        # None on a flexible graph, which has no shape error rather than a huge one
        return env.shape_error_now()
    finally:
        env.network.edges = was


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
    # Setup, not search. Every edit-space method has its restore()'s env.reset() outside
    # the caller's meter, and this reset is the same thing for a rollout; charging it to
    # the policy alone put roughly 60% of the `learned` row on work no other method pays.
    with cost.reporting():
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

    # where stopping is part of the action space the policy chooses when to halt, so the
    # graph it halted on is its answer rather than the best one it passed through
    if getattr(raw_env, "skip_is_stop", False):
        return rollout_result_final(label, raw_env, work)
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
