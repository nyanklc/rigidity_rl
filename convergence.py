"""Driving a formation to a goal shape, and what the edge set does to the trajectory.

One edge set is used twice: the goal bearings are read off the goal formation over it,
and the controller builds `B` from the current formation over the same edges. That is
what `main.py` does, and it is why a method here picks one topology rather than two.

`GradientBasedController` computes `diag(g) B^T p*`, the rigidity matrix transposed
against the goal bearings. That is the gradient of the bearing error, not a missing
subtraction: every 3-row block of `B` carries `P_v = I - v v^T` against its own bearing,
so `B^T p = 0` for the framework's own bearings and `B^T p* == B^T (p* - p)`.

Bearing control cannot observe translation or scale, so the formation converges to the
goal SHAPE and not to the goal poses. Position error is therefore measured with the
gauge projected out (`estimation.shape_error`), which is a small-error metric: read the
early part of a trajectory as approximate.

Two things that would otherwise give a confident wrong answer:

  the gain mask   `control.GradientBasedController` tiles [lin,lin,lin,ang,ang,ang] per
                  agent, which is the order `set_inputs` reads, but `B`'s columns are
                  [3n positions | 3n attitudes]. At lin != ang the gains land on the
                  wrong coordinates. `control.py` is obsolete, so this is not patched
                  there; `gain_mask` is the aligned one and `tiled_mask` the shipped one.
  the start       has to differ from the goal only in directions the agents can move. A
                  planar agent cannot undo a displacement in z, so an unrestricted
                  perturbation is a shape no controller can reach.

The library, not the script: `experiments/control.py` is what runs it.
"""
import copy

import numpy as np

from estimation import gauge_basis, shape_error
from rigidity import (characteristic_length, extended_bearing_rigidity_matrix,
                      node_dof_projectors, rigidity_eigenvalue)

ROW_FIELDS = ("case", "method", "n", "m", "lam", "steps_to_tol", "reached",
              "bearing_rms_0", "bearing_rms_end", "pos_err_0", "pos_err_end",
              "att_err_end", "dt", "tol", "steps")

TRACE_FIELDS = ("case", "method", "step", "t", "bearing_rms", "pos_err", "att_err")

PATH_FIELDS = ("case", "method", "step", "t", "agent", "x", "y", "z")


def gain_mask(n, lin, ang):
    """B's columns are [3n positions | 3n attitudes], so the mask is two blocks."""
    return np.concatenate([np.full(3 * n, float(lin)), np.full(3 * n, float(ang))])


def tiled_mask(n, lin, ang):
    """What control.GradientBasedController builds: per agent, not per block."""
    return np.tile([lin, lin, lin, ang, ang, ang], n).astype(float)


def admissible(net, dp):
    """Project a per-agent displacement onto the DOFs that agent actually has."""
    out = np.array(dp, dtype=float, copy=True)
    for i, agent in enumerate(net.agents):
        S, _ = node_dof_projectors(agent)
        out[i] = S @ out[i]
    return out


def bearing_rms(net, goal_bearings):
    """RMS bearing error per edge, in radians of chord.

    Per edge, not summed: the experiment varies the edge count, and a summed error
    makes a denser graph drive each edge lower to reach the same total.
    """
    d = np.asarray(goal_bearings) - np.asarray(net.get_bearings())
    return float(np.sqrt(d @ d / max(len(d) // 3, 1)))


def perturbed_start(goal_net, rng, sigma):
    """The goal formation displaced, within each agent's own DOFs."""
    start = copy.deepcopy(goal_net)
    dp = admissible(goal_net, rng.normal(0, sigma, (goal_net.n, 3)))
    for agent, d in zip(start.agents, dp):
        agent.pose.position = agent.pose.position + d
    return start


def simulate(start, goal_net, rank_K, mask, dt=1e-3, steps=200_000, tol=1e-3,
             sample_every=200):
    """Run the controller and record the trajectory. -> (rows, summary).

    `tol` is on the RMS bearing error per edge, so it means the same thing whatever
    the edge count. `steps_to_tol` is the first sampled step at or under it, and stays
    None when the run never gets there.
    """
    net = copy.deepcopy(start)
    goal = np.asarray(goal_net.get_bearings())
    length_scale = characteristic_length(goal_net)
    Z_K = gauge_basis(goal_net, rank_K, length_scale)

    def sample(k):
        pos, att = shape_error(goal_net, net, rank_K, Z_K=Z_K, length_scale=length_scale)
        return {"step": k, "t": k * dt, "bearing_rms": bearing_rms(net, goal),
                "pos_err": pos, "att_err": att,
                "positions": np.array([a.pose.position for a in net.agents])}

    rows = [sample(0)]
    steps_to_tol = 0 if rows[0]["bearing_rms"] <= tol else None
    for k in range(1, steps + 1):
        brm = extended_bearing_rigidity_matrix(net)
        net.set_inputs(mask * (brm.T @ goal))
        net.step(dt)
        if k % sample_every == 0 or k == steps:
            rows.append(sample(k))
            if steps_to_tol is None and rows[-1]["bearing_rms"] <= tol:
                steps_to_tol = k
            if not np.isfinite(rows[-1]["bearing_rms"]):
                break
    return rows, steps_to_tol


def edge_sets(env, goal_net, rank_K, methods, models, rng, restarts):
    """{method: edges} on the GOAL poses. -> the topology each method would hold.

    The edge set is chosen for the formation being held, so every method sees the goal
    shape. It is then used for both networks, since the goal bearings and the
    controller's `B` have to be indexed by the same edges.
    """
    import baselines

    out = {}
    if "complete" in methods:
        out["complete"] = goal_net.fully_connected().edges.copy()

    classical = {"greedy": lambda: baselines.run_greedy(env, verbose=False),
                 "spectral": lambda: baselines.run_spectral(env, verbose=False),
                 "constructive": lambda: baselines.run_constructive(
                     env, rng, restarts=restarts, verbose=False)}

    needs_env = [m for m in methods if m in classical or m in (models or {})]
    if not needs_env:
        return out

    env.freeze_network = False
    env.network = copy.deepcopy(goal_net)
    env.freeze_network = True
    env.reset()
    instance = copy.deepcopy(env.network)

    for name in needs_env:
        env.network = copy.deepcopy(instance)
        env.reset()
        if name in classical:
            classical[name]()
            out[name] = env.network.edges.copy()
        else:
            m = models[name]
            horizon = int(env.truncate_max_steps if env.truncate_enable else env.max_steps)
            baselines.run_policy(m["agent"], m["wrapped"], env, horizon,
                                 mode="greedy", label=name)
            # scored on the best state it visited, which is the graph it would hand over
            out[name] = env.best_edges.copy()
    return out


def measure(*, env, goal_networks, methods, models=None, rng=None, restarts=5,
            sigma=0.15, dt=1e-3, steps=200_000, tol=1e-3, sample_every=200,
            lin=1.0, ang=1.0, progress=None):
    """One trajectory per (case, method). -> (rows, traces, topologies).

    Within a case the goal shape and the start formation are shared, so the edge set is
    the only thing that differs between methods.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    models = models or {}
    rows, traces, topologies = [], [], {}

    for case, goal_net in enumerate(goal_networks):
        rank_K = int(np.linalg.matrix_rank(
            extended_bearing_rigidity_matrix(goal_net.fully_connected()), tol=1e-9))
        start = perturbed_start(goal_net, rng, sigma)
        sets = edge_sets(env, goal_net, rank_K, methods, models, rng, restarts)

        for name in methods:
            if name not in sets:
                continue
            if progress:
                progress(f"case {case + 1}/{len(goal_networks)}  {name}")
            goal_with_edges = copy.deepcopy(goal_net)
            goal_with_edges.edges = sets[name].copy()
            start_with_edges = copy.deepcopy(start)
            start_with_edges.edges = sets[name].copy()
            if not np.any(sets[name]):
                continue
            topologies[(case, name)] = sets[name].copy()

            trace, hit = simulate(start_with_edges, goal_with_edges, rank_K,
                                  gain_mask(goal_net.n, lin, ang), dt=dt, steps=steps,
                                  tol=tol, sample_every=sample_every)
            for r in trace:
                traces.append(dict(r, case=case, method=name))
            rows.append({
                "case": case, "method": name, "n": goal_net.n,
                "m": int(sets[name].sum()),
                "lam": float(rigidity_eigenvalue(goal_with_edges, rank_K=rank_K)),
                "steps_to_tol": hit, "reached": hit is not None,
                "bearing_rms_0": trace[0]["bearing_rms"],
                "bearing_rms_end": trace[-1]["bearing_rms"],
                "pos_err_0": trace[0]["pos_err"], "pos_err_end": trace[-1]["pos_err"],
                "att_err_end": trace[-1]["att_err"],
                "dt": dt, "tol": tol, "steps": steps})
    return rows, traces, topologies


def rank_K_of(net):
    """rank(B) of the complete graph on these poses."""
    return int(np.linalg.matrix_rank(
        extended_bearing_rigidity_matrix(net.fully_connected()), tol=1e-9))


def mask_comparison(goal_net, rank_K, edges, rng, sigma=0.15, dt=1e-3,
                    steps=200_000, tol=1e-3, lin=100.0, ang=1.0):
    """{mask name: steps to tolerance} on one topology and one start."""
    goal = copy.deepcopy(goal_net)
    goal.edges = edges.copy()
    start = perturbed_start(goal, rng, sigma)
    out = {}
    for name, mask in (("as written (tiled per agent)", tiled_mask(goal.n, lin, ang)),
                       ("aligned with B's columns", gain_mask(goal.n, lin, ang))):
        _, hit = simulate(start, goal, rank_K, mask, dt=dt, steps=steps, tol=tol)
        out[name] = hit
    return out


def normalised(P):
    """Centred and scaled to unit RMS radius.

    Translation and uniform scale are gauge freedoms of a bearing framework in every
    domain, and the controller's velocity is orthogonal to them, so a raw trajectory
    keeps whatever offset its start had. Removing them is what lets a figure show the
    formation reaching the goal.
    """
    P = np.asarray(P, dtype=float)
    C = P - P.mean(axis=0)
    scale = float(np.sqrt((C ** 2).sum(axis=1).mean())) or 1.0
    return C / scale


def rotation_is_gauge(net):
    """Whether a global rotation is also unobservable, so a figure may still show the
    converged formation rotated away from the goal.

    Rotating the world rotates a frameless agent's bearings, which are in the global
    frame, so one frameless agent makes rotation observable.
    """
    return all(a.domain not in ("R^2", "R^3") for a in net.agents)


def format_table(rows, context, brief=False):
    """The per-run table and its legend, as text."""
    lines = ["=" * 100, "FORMATION CONTROL: DRIVING EACH TOPOLOGY TO THE GOAL SHAPE",
             "=" * 100]
    for k, v in context.items():
        lines.append(f"  {k:<12s}: {v}")
    lines += ["", f"  {'case':>5s}{'method':>14s}{'m':>5s}{'lambda':>11s}"
                  f"{'bearing rms':>13s}{'pos err':>10s}{'steps to tol':>14s}"]
    lines.append("  " + "-" * 72)
    for r in rows:
        hit = "-" if r["steps_to_tol"] is None else str(r["steps_to_tol"])
        lines.append(f"  {r['case']:>5d}{r['method']:>14s}{r['m']:>5d}{r['lam']:>11.3e}"
                     f"{r['bearing_rms_end']:>13.2e}{r['pos_err_end']:>10.3f}"
                     f"{hit:>14s}")
    if not brief:
        lines += ["", "  HOW TO READ IT", "",
                  "  Within a case the goal shape and the start formation are the same for",
                  "  every row, so the edge set is the only thing that differs. The same edge",
                  "  set carries the goal bearings and builds the controller's B.",
                  "",
                  "  m             edges in that topology.",
                  "  lambda        rigidity eigenvalue of the goal formation over it. The",
                  "                linearised error dynamics are xdot = -B'B x, so the slowest",
                  "                mode decays at lambda.",
                  "  bearing rms   RMS bearing error per edge at the end of the run, in radians.",
                  "                Per edge rather than summed, so it means the same thing at",
                  "                any edge count.",
                  "  pos err       RMS position error against the goal SHAPE, in formation",
                  "                radii, with translation and scale projected out: bearing",
                  "                control cannot observe them. It is a small-error metric, so",
                  "                the start of a trajectory is approximate.",
                  "  steps to tol  first sampled step under the tolerance, `-` if it never got",
                  "                there within the step budget."]
    return "\n".join(lines)
