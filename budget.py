"""How good an answer is each method's, given only so many moves?

A rollout here is a search, not a controller. The policies in this experiment carry no
stop action, so an episode is a fixed number of graph modifications and the answer is
the best graph visited along the way. The question is how that answer degrades as the
number of moves is cut.

The measurement is exact rather than sampled, and the reason is worth stating because it
is what makes the experiment cheap. The observation carries no step counter and no notion
of the remaining budget, so a policy's trajectory does not depend on `max_steps` at all.
Truncating one rollout of length T at step t therefore gives exactly the trajectory a
budget-t run would have produced, and the running best over that prefix is exactly the
answer it would have reported. One rollout per (instance, method) settles every budget.

That argument holds only for a search. A policy with a stop action chooses when to halt,
so its trajectory does depend on the horizon it was given, and it cannot be read this
way. Those arms are excluded here for that reason rather than by preference.

greedy is on the same axes with one difference stated in the figure: it applies an edit
at every step of its own trace and stops when no single change helps, so its budget is
counted in applied edits, there being no such thing as a greedy step that does nothing.

The library, not the script: `experiments/budget.py` is what runs it.
"""
import copy
from types import SimpleNamespace

import numpy as np

ROW_FIELDS = ("episode", "method", "budget", "m", "m_req", "at_bound", "is_rigid",
              "score", "shape_err", "best_step", "is_final", "steps_taken")

STEP_FIELDS = ("episode", "method", "step", "score", "m", "is_rigid", "shape_err")

# Dense where the answer is still moving and sparse after it has settled. Every move is
# measured over the range where methods are still improving, because the figure plots the
# budget on a linear axis and a log-spaced grid would put every marker at the left.
def budget_grid(steps, dense_to=24):
    """Budgets from 1 to `steps`: every move early, then thinning out."""
    fine = list(range(1, min(steps, dense_to) + 1))
    coarse = [28, 32, 36, 40, 48, 56, 72, 88, 104, steps]
    return sorted({b for b in fine + coarse if 1 <= b <= steps})


def table_budgets(budgets, count=13):
    """A readable subset of the measured grid. The csv keeps every column; a table with
    forty of them is not a table."""
    if len(budgets) <= count:
        return list(budgets)
    idx = np.unique(np.round(np.linspace(0, len(budgets) - 1, count)).astype(int))
    return [budgets[i] for i in idx]


def curve(trace, m_req, budgets):
    """The best graph visited within each budget. -> [row] on ROW_FIELDS minus the keys.

    `trace` is the per-step record the rollout already writes, sorted by step. The best
    state visited is what every rollout in this project is scored on, so taking the
    running argmax of the score is the same rule applied to a prefix.
    """
    rows = sorted(trace, key=lambda r: r["step"])
    if not rows:
        return []
    last = max(r["step"] for r in rows)
    out = []
    for b in budgets:
        seen = [r for r in rows if r["step"] <= b]
        if not seen:
            continue
        best = max(seen, key=lambda r: r["score"])
        m = int(best["edges"])
        rigid = bool(best["is_IBR"])
        out.append({
            "budget": b, "m": m, "m_req": m_req,
            "at_bound": bool(rigid and m_req and m == m_req),
            "is_rigid": rigid,
            "score": float(best["score"]),
            "shape_err": (float(best["shape_err"])
                          if best.get("shape_err") not in (None, "") else None),
            "best_step": int(best["step"]),
            # the budget was enough: this is already the answer the full run reports
            "is_final": bool(best["step"] <= b and b >= last) or None,
            "steps_taken": min(b, last),
        })
    # `is_final` is really "the best within b equals the best within the whole run",
    # which needs the full-run answer, so it is filled once that is known
    overall = max(rows, key=lambda r: r["score"])["score"]
    for r in out:
        r["is_final"] = bool(r["score"] >= overall - 1e-9)
    return out


def measure(*, env, instances, methods, models=None, wrapped=None, steps=None,
            policy_mode="sample", spectral_shortlist=5, restarts=20, seed=0,
            budgets=None, progress=None):
    """One row per (instance, method, budget), plus the per-step traces. -> (rows, trace)."""
    import baselines
    import cost

    rows, traces = [], []
    models = models or {}
    args = SimpleNamespace(spectral_shortlist=spectral_shortlist, restarts=restarts,
                           policy_mode=policy_mode)
    construct_rng = np.random.default_rng(seed)
    anneal_rng = np.random.default_rng(seed)
    degree_rng = np.random.default_rng(seed)

    for ep, pristine in enumerate(instances):
        if progress:
            progress(f"instance {ep + 1}/{len(instances)}")
        env.freeze_network = False
        env.network = copy.deepcopy(pristine)
        env.freeze_network = True
        env.reset()
        instance = copy.deepcopy(env.network)
        m_req = int(env.m_req)
        horizon = steps or int(env.max_steps)
        grid = budgets or budget_grid(horizon)

        for name in methods:
            env.network = copy.deepcopy(instance)
            env.reset()
            mine = []
            ctx = dict(env=env, args=args, traces=mine, episode=ep,
                       steps=horizon, models=models, wrapped=wrapped, decisions=None,
                       construct_rng=construct_rng, anneal_rng=anneal_rng,
                       degree_rng=degree_rng, anneal_budget=None)
            with cost.reporting():          # the per-step stats are measurement, not search
                out = baselines.run_method(name, ctx)
            if out is None or not mine:
                continue
            for r in curve(mine, m_req, grid):
                rows.append(dict(r, episode=ep, method=name))
            for t in mine:
                traces.append({"episode": ep, "method": name, "step": t["step"],
                               "score": t["score"], "m": t["edges"],
                               "is_rigid": t["is_IBR"], "shape_err": t.get("shape_err")})
    return rows, traces


# ---------------------------------------------------------------- aggregation

def _gmean(values):
    vals = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    return float(np.exp(np.mean(np.log(vals)))) if vals else None


def summarize(rows):
    """One record per (method, budget), over the instances."""
    import report
    methods = [m for m in report.METHOD_ORDER
               if any(r["method"] == m for r in rows)]
    methods += [m for m in dict.fromkeys(r["method"] for r in rows) if m not in methods]

    out = []
    for m in methods:
        for b in sorted({r["budget"] for r in rows if r["method"] == m}):
            sel = [r for r in rows if r["method"] == m and r["budget"] == b]
            if not sel:
                continue
            out.append({
                "method": m, "budget": b, "instances": len(sel),
                "edges_mean": float(np.mean([r["m"] for r in sel])),
                "edges_sd": float(np.std([r["m"] for r in sel])),
                "at_bound_pct": 100.0 * float(np.mean([r["at_bound"] for r in sel])),
                "rigid_pct": 100.0 * float(np.mean([r["is_rigid"] for r in sel])),
                "final_pct": 100.0 * float(np.mean([r["is_final"] for r in sel])),
                "shape_err_gmean": _gmean([r["shape_err"] for r in sel]),
                "score_mean": float(np.mean([r["score"] for r in sel])),
            })
    return out


def best_steps(rows):
    """{method: [step at which the full-budget answer was found, per instance]}."""
    out = {}
    for m in dict.fromkeys(r["method"] for r in rows):
        full = max(r["budget"] for r in rows if r["method"] == m)
        out[m] = [r["best_step"] for r in rows
                  if r["method"] == m and r["budget"] == full]
    return out


def enough(summary, method, target=95.0):
    """Smallest budget at which `method` already holds its full-budget answer on
    `target` percent of instances. None when it never gets there."""
    sel = sorted([s for s in summary if s["method"] == method],
                 key=lambda s: s["budget"])
    for s in sel:
        if s["final_pct"] >= target:
            return s["budget"]
    return None


def format_table(rows, context=None, brief=False):
    summary = summarize(rows)
    if not summary:
        return "no budgets measured"
    methods = list(dict.fromkeys(s["method"] for s in summary))
    measured = sorted({s["budget"] for s in summary})
    budgets = table_budgets(measured)
    by = {(s["method"], s["budget"]): s for s in summary}

    head = f"  {'method':<14}{'quantity':<12}" + "".join(f"{b:>7}" for b in budgets)
    w = max(len(head), 96)
    lines = ["=" * w, "WHAT A BUDGET BUYS", "=" * w]
    for k, v in (context or {}).items():
        lines.append(f"  {k:<12}: {v}")
    lines.append("")
    lines.append(head)
    lines.append(f"  {'':<14}{'budget ->':<12}" + "".join(f"{'moves':>7}" for _ in budgets))
    lines.append("  " + "-" * (len(head) - 2))
    for m in methods:
        for key, label, fmt in (("at_bound_pct", "at bound %", "{:.0f}"),
                                ("edges_mean", "edges", "{:.1f}"),
                                ("shape_err_gmean", "shape err", "{:.1f}"),
                                ("final_pct", "at answer %", "{:.0f}")):
            cells = ""
            for b in budgets:
                s = by.get((m, b))
                v = None if s is None else s[key]
                cells += ("-" if v is None else fmt.format(v)).rjust(7)
            name = m if key == "at_bound_pct" else ""
            lines.append(f"  {name:<14}{label:<12}{cells}")
        lines.append("")

    full = max(measured)
    lines.append("HOW MUCH IS ENOUGH")
    for m in methods:
        b = enough(summary, m)
        lines.append(f"  {m:<14}" + (f"{b} moves already hold the full-budget answer on "
                                     f"95% of instances" if b else
                                     f"never reaches 95% within {full} moves"))
    lines.append("")

    if brief:
        return "\n".join(lines)

    import report
    lines.append("-" * w)
    lines.append("WHAT THE METHODS ARE")
    for m in methods:
        lines.append(f"  {m:<14} {report.METHOD_BLURB.get(m, '')}")
    lines.append("")
    lines.append("WHAT THE ROWS MEAN")
    lines.append(f"  budget      graph modifications the method was allowed, a readable")
    lines.append(f"              subset of the {len(measured)} measured; budget.csv has them all.")
    lines.append("              For a policy a budget is a step of the episode; for greedy")
    lines.append("              it is an applied edit, there being no greedy step that")
    lines.append("              does nothing.")
    lines.append("  at bound %  networks rigid on exactly the fewest edges the poses allow,")
    lines.append("              counting the best graph seen within the budget.")
    lines.append("  edges       edges in that best graph, averaged over the networks.")
    lines.append("  shape err   its RMS state error per radian of bearing noise, as a")
    lines.append("              geometric mean. Lower is better.")
    lines.append("  at answer % networks where the best graph within this budget is already")
    lines.append("              the one the full budget reports. This is the column that")
    lines.append("              says how much budget is actually needed.")
    lines.append("")
    lines.append("HOW TO READ IT")
    lines.append("  The observation carries no step counter, so a policy's trajectory does")
    lines.append("  not depend on the horizon it is given. Truncating one rollout therefore")
    lines.append("  gives exactly what a shorter run would have produced, which is why every")
    lines.append("  column here comes from one rollout per network rather than one per")
    lines.append("  column. That argument needs the method to be a search: a policy with a")
    lines.append("  stop action picks when to halt, so its trajectory does depend on the")
    lines.append("  horizon, and those arms are not in this table.")
    lines.append("=" * w)
    return "\n".join(lines)
