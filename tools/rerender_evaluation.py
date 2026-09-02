"""Re-render an evaluation report from the CSVs it already wrote.

Wording in report.py changes more often than the numbers do. This rebuilds
summary.txt and the figures that need only results.csv, trajectories.csv and
meta.json, so a report picks up the new text without a re-run.

The figures that need the Network objects and per-row edge sets are not
regenerable and are left untouched: uncertainty, softest_mode, sensitivity,
repair_choice, decisions, noise, prediction.

    PYTHONPATH=. uv run tools/rerender_evaluation.py [run_dir ...]
"""
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import report

# rebuilt from the CSVs; everything else needs the evaluation to run again
REGENERABLE = ["table", "trajectories", "outcomes", "summary"]
NEEDS_RERUN = ["noise", "prediction", "uncertainty", "softest_mode", "sensitivity",
               "repair_choice", "decisions"]


def _num(v, cast=float):
    return None if v in ("", "nan", None) else cast(v)


def load_rows(run_dir):
    out = []
    with open(os.path.join(run_dir, "results.csv")) as f:
        for r in csv.DictReader(f):
            out.append({
                "episode": int(r["episode"]),
                "method": r["method"],
                "m": int(r["m"]),
                "score": float(r["score"]),
                "is_IBR": r["is_IBR"] == "True",
                "is_MBR": r["is_MBR"] == "True",
                "min_eig": _num(r["min_eig"]),
                "shape_err": _num(r["shape_err"]),
                "work": int(float(r.get("work") or 0)),
                "best_at": int(float(r.get("best_at") or 0)),
                # results.csv does not carry the counters, and a zero here would read
                # as "free" rather than as "not measured". cost.csv holds them.
                "cost": None,
            })
    return out


def load_traces(run_dir):
    path = os.path.join(run_dir, "trajectories.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append({
                "episode": int(r["episode"]),
                "method": r["method"],
                "step": int(r["step"]),
                "score": float(r["score"]),
                "edges": int(r["edges"]),
                "rank": int(float(r["rank"])),
                "rank_K": int(float(r["rank_K"])),
                "is_IBR": r["is_IBR"] == "True",
                "is_MBR": r["is_MBR"] == "True",
                "min_eig": _num(r["min_eig"]),
                "shape_err": _num(r["shape_err"]),
            })
    return out


def noise_block(run_dir):
    """The measured-vs-predicted table, verbatim. Its numbers are not recomputable here."""
    path = os.path.join(run_dir, "summary.txt")
    if not os.path.exists(path):
        return None
    m = re.search(r"(MEASURED SHAPE ERROR UNDER BEARING NOISE.*?)\n\n", open(path).read(),
                  re.S)
    return m.group(1) if m else None


def context_of(meta):
    args, cfg = meta["args"], meta.get("environment_config") or {}
    domains = cfg.get("domains") or []
    domain_str = (domains[0] if len(set(domains)) == 1 else f"mixed {sorted(set(domains))}"
                  ) if domains else "?"
    ctx = {
        "environment": args["environment_name"],
        "network": f"{meta['n']} agents in {domain_str}, "
                   f"action space {cfg.get('action_type', '?')}",
        "objective": f"{cfg.get('state_score_type', '?')} state score",
        "instances": (f"{args['episodes']} networks from benchmark {args['benchmark']} "
                      f"({meta.get('benchmark_digest')})" if args.get("benchmark")
                      else f"{args['episodes']} random networks, seed {args['seed']}"),
    }
    if args.get("model"):
        ctx["policy"] = (f"{args['model']} (?, --policy-mode {args['policy_mode']}, "
                         f"{meta['rollout_steps']}-step budget)")
    return ctx


def rerender(run_dir):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    args = meta["args"]
    rows, traces = load_rows(run_dir), load_traces(run_dir)
    ctx = context_of(meta)

    # the algorithm name is not in meta; keep the one the old summary recorded
    old = open(os.path.join(run_dir, "summary.txt")).read()
    was = re.search(r"policy\s*: \S+ \((\w+),", old)
    if was and "policy" in ctx:
        ctx["policy"] = ctx["policy"].replace("(?,", f"({was.group(1)},")

    table = report.format_table(rows, ctx, brief=args.get("brief", False))
    block = noise_block(run_dir)
    if block and "MEASURED SHAPE ERROR" not in table:
        # format_table drops it because the rebuilt rows carry no per-row noise
        rule = re.search(r"\n-{20,}\n", table)
        if rule:
            table = table.replace(rule.group(0),
                                  "\n" + block + "\n" + rule.group(0), 1)
    report.write_summary(run_dir, table)

    made = ["summary.txt"]
    if traces:
        header = {
            "short": report.short_env_name(args["environment_name"]),
            "env": args["environment_name"],
            "model": args.get("model"),
            "network": ctx["network"],
            "episodes": args["episodes"],
            "seed": args["seed"],
            "benchmark": args.get("benchmark"),
        }
        def draw_all():
            report.plot_trajectories(run_dir, traces, rows, header)
            report.plot_outcomes(run_dir, traces, rows, header)
            report.plot_summary(run_dir, rows, header)
            report.plot_table(run_dir, rows, dict(header, objective=ctx["objective"],
                                                  policy=ctx.get("policy")))
            for ep in range(min(args.get("plot_episodes", 3), args["episodes"])):
                sel = [t for t in traces if t["episode"] == ep]
                if not sel:
                    continue
                report.plot_trajectories(
                    run_dir, sel, [r for r in rows if r["episode"] == ep],
                    dict(header, episodes=None, subtitle=f"episode {ep}"),
                    filename=f"episode_{ep:03d}", aggregate_over_episodes=False)

        draw_all()
        with report.plain():
            draw_all()
        made.append(f"plots for {', '.join(REGENERABLE)} and episode_NNN, "
                    f"each with its -plain twin")
    return made


def main(argv):
    dirs = argv or sorted(glob.glob(os.path.join("runs_evaluation", "*")))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "meta.json"))]
    if not dirs:
        print("no evaluation runs with a meta.json")
        return 1
    for d in dirs:
        try:
            made = rerender(d)
        except Exception as exc:
            print(f"{d}\n  skipped: {type(exc).__name__}: {exc}")
            continue
        print(d)
        for m in made:
            print(f"  {m}")
    print(f"\nnot regenerable without re-running the evaluation: "
          f"{', '.join(NEEDS_RERUN)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
