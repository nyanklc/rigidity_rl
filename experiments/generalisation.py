"""Does a policy hold up on instance sets it was not trained on?

Aggregates earlier output directories rather than re-running them, one per instance
set, so each row carries the code and the commit it was actually produced by. Evaluate
each benchmark once with `experiments/comparison.py --benchmark`, then run this.

usage:
  uv run experiments/generalisation.py <environment_name> --model NAME[=LABEL]
      [--prior 'runs_outputs/*']
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
    common.add_model_args(p, required=True, policy_mode=False)
    p.add_argument("--prior", default=None,
                   help="where to look for the earlier runs, one per instance set "
                        "(default runs_outputs/*). It reads what those runs already "
                        "measured rather than re-running them")
    p.add_argument("--methods", default="greedy",
                   help="classical methods to carry through beside the models. At large "
                        "n greedy is not affordable, so the opponent has to be nameable")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def measure(args, run_dir, models, header):
    """(written, draws). One row per (model or greedy, instance set)."""
    pattern = args.prior or os.path.join("runs_outputs", "*")
    classical = {x.strip() for x in args.methods.split(",") if x.strip()}
    wanted = {m["name"] for m in models.values()} | classical
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
        # its row `learned`. Without this, every such run contributes only its baselines
        # and the figure looks like the models were never evaluated.
        labels = meta.get("models") or {}
        if not labels:
            was = meta.get("args", {}).get("model")
            if isinstance(was, str) and was:
                labels = {"learned": was}
        with open(res_path) as f:
            rows = list(csv.DictReader(f))
        # a run's own label -> the model behind it, so rows from runs that used different
        # labels for the same checkpoint still line up
        for method in {r["method"] for r in rows}:
            name = labels.get(method, method)
            if name not in wanted:
                continue
            sel = [r for r in rows if r["method"] == method]
            row = _row(short.get(name, name), bench, sel, d, meta)
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
              "at_bound_pct", "edges_over_bound", "m_over_mreq", "m_over_mreq_sd",
              "phi", "shape_err", "source_dir", "git_commit"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in gen_rows:
            wr.writerow(r)
    print(f"  generalisation: {len(gen_rows)} rows over "
          f"{len({r['benchmark'] for r in gen_rows})} instance sets")
    return [os.path.relpath(path, run_dir)], [
        lambda: report.plot_generalisation(run_dir, gen_rows, header)]


def _row(name, bench, sel, source_dir, meta):
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
    # per-instance m/m_req over the instances the method made rigid, then averaged.
    # m_req bounds the edges a *rigid* graph needs, so a flexible one can sit below it and
    # averaging those in rewards a method for failing. `edges_over_bound` below is a ratio
    # of means over every row, which is not the same number.
    ratios = [num(r, "m") / num(r, "m_req") for r in sel
              if num(r, "m") is not None and num(r, "m_req")
              and r.get("is_IBR") == "True"]
    return {
        "m_over_mreq": float(np.mean(ratios)) if ratios else None,
        "m_over_mreq_sd": float(np.std(ratios)) if ratios else None,
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


def main(argv=None):
    args = parse_args(argv)

    model_specs = common.parse_models(args.model)
    loaded = common.load_environment(args)
    if loaded is None:
        return 1
    env, env_config_data, steps = loaded
    # the rows come from the earlier runs; the checkpoints are loaded so the labels and
    # colours here are the ones every other figure uses
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
