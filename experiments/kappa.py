"""What the conditioning weight buys, with every method optimizing the same weight.

The trade the weight controls is between the number of bearing measurements kept and the
error left in the recovered shape. Reading it off one fixed objective would be unfair to
both ends of the range: a policy trained at a high weight spends edges deliberately, and
scoring it by an objective that charges for edges and barely credits conditioning says
only that it was trained for something else.

So each point here comes from a run in the environment the policy on it was trained for,
with greedy and the annealer hill climbing that same objective. The two axes reported are
the edge count and the shape error, both properties of the graph rather than of any
reward, which is what lets a policy trained at one weight sit beside one trained at
another.

Reads the runs `tools/scale_runs.sh kappa` writes rather than re-running them.

usage:
  uv run experiments/kappa.py <environment_name> [--prior 'runs_outputs/*__front_k*']
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
    p.add_argument("--prior", default=None,
                   help="where to look for the per-weight runs "
                        "(default runs_outputs/*__front_k*)")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def _num(r, k):
    v = r.get(k)
    try:
        return None if v in (None, "", "nan") else float(v)
    except ValueError:
        return None


def measure(args, run_dir, header):
    pattern = args.prior or os.path.join("runs_outputs", "*__front_k*")
    # (weight, method) -> (directory mtime, row). The same environment can have been
    # evaluated more than once, and two rows at one weight would draw the series twice
    best = {}
    for d in sorted(glob.glob(pattern)):
        meta_path = os.path.join(d, "meta.json")
        res_path = os.path.join(d, "results.csv")
        if not (os.path.exists(meta_path) and os.path.exists(res_path)):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        kappa = (meta.get("environment_config") or {}).get("stiffness_kappa")
        if kappa is None:
            continue
        labels = meta.get("models") or {}
        with open(res_path) as f:
            rows = list(csv.DictReader(f))
        # a run carrying several policies has to say which one belongs on this column,
        # and it is the one trained for this environment. Older runs evaluated the whole
        # family in one environment, so without this the row is whichever model sorted
        # first, which is not a comparison of anything
        keep = set(labels)
        if len(labels) > 1:
            want = [lab for lab in labels
                    if lab.lstrip("k").split("_")[0] == ("%g" % kappa)
                    or lab == ("k%g" % kappa)]
            if want:
                keep = set(want)
            else:
                print(f"  {os.path.basename(d)}: {len(labels)} models and none named for "
                      f"kappa={kappa:g}, so no policy row is taken from it")
                keep = set()
        for method in sorted({r["method"] for r in rows}):
            if method in labels and method not in keep:
                continue
            sel = [r for r in rows if r["method"] == method]
            edges = [_num(r, "m") for r in sel if _num(r, "m") is not None]
            if not edges:
                continue
            err = [_num(r, "shape_err") for r in sel
                   if _num(r, "shape_err") and _num(r, "shape_err") > 0]
            # rigid instances only: a flexible graph can sit below the bound, and
            # averaging those in rewards a method for failing to make one rigid
            ratios = [_num(r, "m") / _num(r, "m_req") for r in sel
                      if _num(r, "m") is not None and _num(r, "m_req")
                      and r.get("is_IBR") == "True"]
            at_bound = [r.get("at_bound") == "True" for r in sel
                        if r.get("at_bound") not in (None, "")]
            # every model row is the policy trained for this environment, so it carries
            # one name in the figure whatever label the run gave it
            name = "learned" if method in labels else method
            row = {
                "kappa": float(kappa),
                "method": name,
                "model": labels.get(method, ""),
                "instances": len(sel),
                "edges": float(np.mean(edges)),
                "edges_sd": float(np.std(edges)),
                "m_over_mreq": float(np.mean(ratios)) if ratios else None,
                "at_bound_pct": 100.0 * float(np.mean(at_bound)) if at_bound else None,
                "shape_err": report._gmean(err),
                "phi": float(np.mean([_num(r, "score") for r in sel])),
                "source_dir": d,
            }
            key = (float(kappa), name)
            stamp = os.path.getmtime(d)
            if key not in best or stamp > best[key][0]:
                best[key] = (stamp, row)

    out = [row for _, row in best.values()]
    if not out:
        print(f"  kappa: no runs matched {pattern}")
        return [], []

    out.sort(key=lambda r: (r["kappa"], r["method"]))
    path = os.path.join(run_dir, "kappa.csv")
    fields = ["kappa", "method", "model", "instances", "edges", "edges_sd",
              "m_over_mreq", "at_bound_pct", "shape_err", "phi", "source_dir"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in out:
            wr.writerow(r)

    kappas = sorted({r["kappa"] for r in out})
    print(f"  kappa: {len(out)} rows over weights {kappas}")
    _print_table(out, kappas)
    return [os.path.relpath(path, run_dir)], [
        lambda: report.plot_kappa(run_dir, out, header)]


def _print_table(rows, kappas):
    methods = [m for m in report.KAPPA_SERIES if any(r["method"] == m for r in rows)]
    methods += [m for m in dict.fromkeys(r["method"] for r in rows) if m not in methods]
    print(f"\n{'':>12s} " + " ".join(f"{('kappa=' + ('%g' % k)):>18s}" for k in kappas))
    print(f"{'method':>12s} " + " ".join(f"{'edges   shape err':>18s}" for _ in kappas))
    print("  " + "-" * (12 + 19 * len(kappas)))
    for m in methods:
        cells = []
        for k in kappas:
            hit = next((r for r in rows if r["method"] == m and r["kappa"] == k), None)
            if hit is None:
                cells.append(f"{'-':>18s}")
            else:
                e = hit.get("shape_err")
                cells.append(f"{hit['edges']:9.2f}"
                             + (f"{e:9.2f}" if e else f"{'-':>9s}"))
        print(f"{m:>12s} " + " ".join(cells))
    print("\n  Each column is a separate environment, and every method in it is "
          "optimizing that\n  environment's objective. Edges and shape error are "
          "properties of the graph, so the\n  columns can be read across.\n")


def main(argv=None):
    args = parse_args(argv)

    model_specs = common.parse_models(args.model)
    loaded = common.load_environment(args)
    if loaded is None:
        return 1
    env, env_config_data, steps = loaded
    loaded = common.load_models(args, env, env_config_data, model_specs, steps)
    if loaded is None:
        return 1
    models, _, env, steps = loaded

    run_dir = common.run_directory(args, models, with_plots=not args.no_plots)
    header = common.header_for(
        args, env, models,
        f"{env.network.n} agents in {common.describe_domains(env)}, "
        f"action space {env.action_space_type}")

    written, draws = measure(args, run_dir, header)
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
