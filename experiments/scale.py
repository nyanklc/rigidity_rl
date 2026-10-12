"""How far does each method hold as the formation grows, and what does it cost?

Aggregates one evaluation run per size rather than re-running them, the same way
`generalisation.py` does, so every row carries the code and the commit it was produced
by. The difference is the axis: `generalisation.py` puts one bar per instance set,
which is the right reading when the sets differ in domain, while here the sets differ
only in `n` and the question is the shape of the curve.

Two quantities make the sizes comparable. `m/m_req` is the edge count over the proven
lower bound for those poses, so it is one at the bound whatever the size or the domain
mix, where a raw edge count is not. Wall time and counted primitive calls say which
methods are still affordable at the far end, which is the other half of the comparison:
a method that would win if it could be run is not a method.

usage:
  uv run experiments/scale.py <environment_name> --model NAME[=LABEL]
      [--methods greedy,spectral,degree] [--prior 'runs_outputs/*__scale']
"""

import os
import sys

sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
import glob
import json

import numpy as np

import report
from experiments import common


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_common_args(p, benchmark=False, episodes=False, brief=False)
    common.add_model_args(p, required=False, policy_mode=False)
    p.add_argument("--methods", default="greedy,spectral,degree,constructive,anneal",
                   help="classical methods to carry through beside the models")
    p.add_argument("--prior", default=None,
                   help="where to look for the earlier runs, one per size "
                        "(default runs_outputs/*)")
    p.add_argument("--lead", default=None,
                   help="the one policy drawn beside the classical methods, the rest "
                        "going on the policy figure")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def _num(r, k):
    v = r.get(k)
    try:
        return None if v in (None, "", "nan") else float(v)
    except ValueError:
        return None


def _summarise(name, n, bench, sel, costs, source_dir, meta):
    """One (method, size) row, or None if the run recorded no edge count for it.

    The cost to read is `calls_to_best`, not `ms_to_best`. A count of calls to the named
    linear algebra primitives is a property of the method; wall time is a property of
    this machine under this load, and is carried beside it rather than instead of it.

    Both `*_to_best` columns are derived, not measured: the cost scaled by the fraction
    of its own run the method spent reaching its best graph. A hill climb stops when
    nothing improves, so its last step is its best and the two columns coincide. A search
    policy with no stop action runs the horizon out whatever it found, so its
    full-horizon cost prices steps it was never meant to skip while this prices the
    answer. The scaling is linear in the step count, which holds while the per-step cost
    is flat: one forward pass and one observation for a policy, one sweep over all
    n(n-1) toggles for a hill climb.
    """
    m = [_num(r, "m") for r in sel if _num(r, "m") is not None]
    if not m:
        return None
    # only over the instances the method actually made rigid. m_req is a lower bound on
    # the edges a *rigid* graph needs, so a flexible graph can sit below it, and averaging
    # those in rewards a method for failing.
    ratios = [_num(r, "m") / _num(r, "m_req") for r in sel
              if _num(r, "m") is not None and _num(r, "m_req")
              and r.get("is_IBR") == "True"]
    rigid = [r.get("is_IBR") == "True" for r in sel]
    at_bound = [r.get("at_bound") == "True" for r in sel
                if r.get("at_bound") not in (None, "")]
    err = [_num(r, "shape_err") for r in sel
           if _num(r, "shape_err") and _num(r, "shape_err") > 0]
    best_at = [_num(r, "best_at") for r in sel if _num(r, "best_at") is not None]
    ms = [_num(c, "ms") for c in costs if _num(c, "ms")]
    calls = [_num(c, "total") for c in costs if _num(c, "total") is not None]
    # the two primitives that dominate every method's count, kept separately so a
    # method that builds many matrices and one that decomposes a few do not read alike
    svd = [_num(c, "rigidity_decomposition") for c in costs
           if _num(c, "rigidity_decomposition") is not None]
    builds = [_num(c, "extended_bearing_rigidity_matrix") for c in costs
              if _num(c, "extended_bearing_rigidity_matrix") is not None]
    # the fraction of its own run each method spent reaching its answer. The span is the
    # method's own step count and not the rollout horizon: a hill climb stops when
    # nothing improves, so its last step is its best one and the fraction is one, while a
    # policy with no stop action runs the horizon out whatever it found.
    steps = [_num(r, "work") for r in sel if _num(r, "work")]
    span = float(np.mean(steps)) if steps else None
    frac = min(1.0, float(np.mean(best_at)) / span) if (best_at and span) else 1.0
    return {
        "method": name,
        "n": n,
        "benchmark": bench,
        "instances": len(sel),
        "m": float(np.mean(m)),
        "m_req": float(np.mean([_num(r, "m_req") for r in sel if _num(r, "m_req")] or [0]))
                 or None,
        "m_over_mreq": float(np.mean(ratios)) if ratios else None,
        "m_over_mreq_sd": float(np.std(ratios)) if ratios else None,
        "rigid_pct": 100.0 * float(np.mean(rigid)) if rigid else None,
        "at_bound_pct": 100.0 * float(np.mean(at_bound)) if at_bound else None,
        "shape_err": report._gmean(err),
        "best_at": float(np.mean(best_at)) if best_at else None,
        # a geometric mean, matching how the comparison's cost block reports it
        "ms": report._gmean(ms),
        "ms_to_best": (report._gmean(ms) * frac) if ms else None,
        "calls": float(np.mean(calls)) if calls else None,
        "calls_to_best": (float(np.mean(calls)) * frac) if calls else None,
        # what one decision costs: greedy scores all n(n-1) toggles before each edit,
        # a policy applies one from one forward pass
        "calls_per_edit": (float(np.mean(calls)) / span) if (calls and span) else None,
        "work": span,
        "svd": float(np.mean(svd)) if svd else None,
        "build_B": float(np.mean(builds)) if builds else None,
        "source_dir": source_dir,
        "git_commit": str((meta.get("provenance") or {}).get("git_commit", ""))[:12],
    }


def measure(args, run_dir, models, header):
    """(written, draws). One row per (method, size), newest run per pair winning."""
    pattern = args.prior or os.path.join("runs_outputs", "*")
    classical = {x.strip() for x in args.methods.split(",") if x.strip()}
    wanted = {m["name"] for m in models.values()} | classical
    short = {m["name"]: lab for lab, m in models.items()}

    best = {}                       # (method, n) -> (dir mtime, row)
    for d in sorted(glob.glob(pattern)):
        meta_path = os.path.join(d, "meta.json")
        res_path = os.path.join(d, "results.csv")
        if not (os.path.exists(meta_path) and os.path.exists(res_path)):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        n = meta.get("n") or (meta.get("environment_config") or {}).get("n")
        if n is None:
            continue
        labels = meta.get("models") or {}
        with open(res_path) as f:
            rows = list(csv.DictReader(f))
        cost_rows = []
        cost_path = os.path.join(d, "cost.csv")
        if os.path.exists(cost_path):
            with open(cost_path) as f:
                cost_rows = list(csv.DictReader(f))

        for method in {r["method"] for r in rows}:
            name = labels.get(method, method)
            if name not in wanted:
                continue
            sel = [r for r in rows if r["method"] == method]
            costs = [c for c in cost_rows if c["method"] == method]
            row = _summarise(short.get(name, name), int(n), meta.get("benchmark"),
                             sel, costs, d, meta)
            if row is None:
                continue
            key = (name, int(n))
            stamp = os.path.getmtime(d)
            if key not in best or stamp > best[key][0]:
                best[key] = (stamp, row)

    scale_rows = [row for _, row in best.values()]
    if not scale_rows:
        print(f"  scale: no earlier runs matched {pattern} for "
              f"{', '.join(sorted(wanted))}")
        return [], []

    scale_rows.sort(key=lambda r: (r["n"], r["method"]))
    path = os.path.join(run_dir, "scale.csv")
    fields = ["method", "n", "benchmark", "instances", "m", "m_req", "m_over_mreq",
              "m_over_mreq_sd", "rigid_pct", "at_bound_pct", "shape_err", "best_at",
              "work", "calls", "calls_to_best", "calls_per_edit", "svd", "build_B",
              "ms", "ms_to_best",
              "source_dir", "git_commit"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in scale_rows:
            wr.writerow(r)

    sizes = sorted({r["n"] for r in scale_rows})
    print(f"  scale: {len(scale_rows)} rows over sizes {sizes}")
    _print_table(scale_rows, sizes)
    # Two figures over subsets rather than one over everything: past about eight lines
    # the palette repeats hues and the panels stop being readable. The classical figure
    # carries one policy so the two can be read against each other, and the policy figure
    # carries greedy for the same reason.
    policies = [m for m in dict.fromkeys(r["method"] for r in scale_rows)
                if m not in report.CLASSICAL_ORDER]
    classical = [m for m in report.CLASSICAL_ORDER
                 if any(r["method"] == m for r in scale_rows)]
    lead = args.lead if args.lead in policies else (policies[0] if policies else None)
    a = classical + ([lead] if lead else [])
    b = (["greedy"] if "greedy" in classical else []) + policies
    def draw(fn, name, only, lead_hue=False):
        """One figure. `lead_hue` moves the single model off a baseline's colour."""
        if lead_hue and lead:
            with report.restyle(lead, color=report.C_ORANGE, ls="-", marker="o", z=9):
                return fn(run_dir, scale_rows, header, filename=name, only=only)
        return fn(run_dir, scale_rows, header, filename=name, only=only)

    draws = []
    for name, only, hue in (("scale", a, True), ("scale_policies", b, False)):
        if len(only) > 1:
            draws.append(lambda f=name, o=only, h=hue:
                         draw(report.plot_scale, f, o, h))
            draws.append(lambda f=name, o=only, h=hue:
                         draw(report.plot_scale_cost, f + "_cost", o, h))
    return [os.path.relpath(path, run_dir)], draws


def _print_table(rows, sizes):
    methods = ([m for m in report.METHOD_ORDER
                if any(r["method"] == m for r in rows)]
               + [m for m in dict.fromkeys(r["method"] for r in rows)
                  if m not in report.METHOD_ORDER])
    print(f"\n{'':>14s} " + " ".join(f"{('n=' + str(n)):>16s}" for n in sizes))
    print(f"{'method':>14s} " + " ".join(f"{'m/m_req   calls':>16s}" for _ in sizes))
    print("  " + "-" * (14 + 17 * len(sizes)))
    for m in methods:
        cells = []
        for n in sizes:
            hit = next((r for r in rows if r["method"] == m and r["n"] == n), None)
            if hit is None or hit.get("m_over_mreq") is None:
                cells.append(f"{'-':>16s}")
            else:
                c = hit.get("calls_to_best")
                cells.append(f"{hit['m_over_mreq']:8.3f}"
                             + (f"{c:8.0f}" if c else f"{'-':>8s}"))
        print(f"{m:>14s} " + " ".join(cells))
    print("\n  m/m_req is the mean over instances of the edge count over the proven lower "
          "bound;\n  calls is the counted primitive calls up to the graph the method "
          "returns. A dash\n  means the method was not run at that size.\n")


def main(argv=None):
    args = parse_args(argv)

    model_specs = common.parse_models(args.model)
    loaded = common.load_environment(args)
    if loaded is None:
        return 1
    env, env_config_data, steps = loaded
    # the rows come from the earlier runs; the checkpoints are loaded so the labels and
    # the colours here are the ones every other figure uses
    loaded = common.load_models(args, env, env_config_data, model_specs, steps)
    if loaded is None:
        return 1
    models, _, env, steps = loaded
    common.announce(models)

    run_dir = common.run_directory(args, models, with_plots=not args.no_plots)
    header = common.header_for(
        args, env, models,
        f"{env.network.n} agents in {common.describe_domains(env)}, "
        f"action space {env.action_space_type}")

    written, draws = measure(args, run_dir, models, header)
    if not written:
        return 1
    report.write_meta(run_dir, common.provenance(args, env_config_data, models))

    if not args.no_plots:
        for draw in draws:
            draw()
        with report.plain():
            for draw in draws:
                draw()

    print(f"\nwrote {run_dir}/")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
