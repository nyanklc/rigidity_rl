"""How good an answer is each method's, given only so many moves to make?

The policies here carry no stop action, so an episode is a fixed number of graph
modifications and the answer is the best graph visited. This measures how that answer
degrades as the number of moves is cut, which is a question about a search rather than
about a controller. Arms with a stop action are excluded for that reason: they choose
when to halt, so the horizon changes what they do.

usage:
  uv run experiments/budget.py <environment_name> --benchmark NAME
      --model NAME[=LABEL] [--methods greedy,learned] [--steps N]
"""

import os
import sys

# the repository root, in place of experiments/ -- `budget` below is the library at the
# root, which this script shares a name with
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import csv

import numpy as np
from tqdm import tqdm

import baselines
import budget as budget_lib
import report
from experiments import common


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_common_args(p)
    common.add_model_args(p)
    p.add_argument("--methods", default="greedy,learned",
                   help="`learned` names every --model. Only search methods belong here")
    p.add_argument("--steps", type=int, default=None,
                   help="the largest budget, and the horizon every rollout is run at "
                        "(default: the environment's own max_steps)")
    p.add_argument("--budgets", default=None,
                   help="comma-separated budgets; default is log-spaced from 1 to --steps")
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--spectral-shortlist", type=int, default=5)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def resolve_methods(args, labels):
    known = baselines.method_sequence(labels)
    asked = [m.strip() for m in args.methods.split(",") if m.strip()]
    asked = [m for x in asked for m in (list(labels) if x == "learned" else [x])]
    unknown = [m for m in asked if m not in known]
    if unknown:
        print(f"unknown method(s) {unknown}, expected {known} or 'learned'")
        return None
    return [m for m in known if m in asked]


def write(run_dir, rows, traces, summary, table):
    with open(os.path.join(run_dir, "budget.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(budget_lib.ROW_FIELDS),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    written = ["budget.csv"]
    if summary:
        with open(os.path.join(run_dir, "budget_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        written.append("budget_summary.csv")
    # one row per step of every rollout, so a figure redraws without re-running
    with open(os.path.join(run_dir, "budget_steps.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(budget_lib.STEP_FIELDS),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(traces)
    written.append("budget_steps.csv")
    with open(os.path.join(run_dir, "budget.txt"), "w") as fh:
        fh.write(table + "\n")
    written.append("budget.txt")
    return written


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
    models, wrapped, env, steps = loaded
    common.announce(models)

    methods = resolve_methods(args, list(models))
    if methods is None:
        return 1

    # a policy that can stop is not a search: its trajectory depends on the horizon, so
    # truncating one rollout would not give what a shorter run produces
    if env_config_data.get("skip_is_stop"):
        print("this environment has a stop action, so a rollout is a controller rather "
              "than a search and the budget cannot be read off one trajectory")
        return 1

    np.random.seed(args.seed)
    env.action_space.seed(args.seed)

    frozen, instance_note = common.instances_for(args, env)
    if frozen is not None:
        instances = [copy.deepcopy(net) for net in frozen[:args.episodes]]
    else:
        instances = []
        for _ in range(args.episodes):
            env.freeze_network = False
            env.reset()
            instances.append(copy.deepcopy(env.network))

    horizon = args.steps or steps
    grid = ([int(b) for b in args.budgets.split(",")] if args.budgets
            else budget_lib.budget_grid(horizon))
    print(f"env: {args.environment_name} | n={env.network.n} | {instance_note} | "
          f"methods {', '.join(methods)} | budgets {grid[0]}..{grid[-1]} "
          f"({len(grid)} of them)")

    bar = tqdm(total=len(instances), unit="instance", dynamic_ncols=True, leave=False)
    rows, traces = budget_lib.measure(
        env=env, instances=instances, methods=methods, models=models, wrapped=wrapped,
        steps=horizon, policy_mode=args.policy_mode, seed=args.seed, budgets=grid,
        spectral_shortlist=args.spectral_shortlist, restarts=args.restarts,
        progress=lambda s: (bar.set_description_str(s), bar.update(1)))
    bar.close()

    kappa = env_config_data.get("stiffness_kappa")
    context = {
        "environment": args.environment_name,
        "network": f"{env.network.n} agents in {common.describe_domains(env)}, "
                   f"action space {env.action_space_type}",
        "objective": f"{env.state_score_type}, stiffness_kappa {kappa}",
        "instances": instance_note,
        "horizon": f"{horizon} moves, budgets {grid[0]} to {grid[-1]}",
    }
    for lab, m in models.items():
        cfg = (m["manifest"] or {}).get("environment_config_raw") or {}
        context[f"model {lab}"] = (f"{m['name']}, trained at stiffness_kappa "
                                   f"{cfg.get('stiffness_kappa')}")

    summary = budget_lib.summarize(rows)
    table = budget_lib.format_table(rows, context=context, brief=args.brief)

    run_dir = common.run_directory(args, models, with_plots=not args.no_plots)
    written = write(run_dir, rows, traces, summary, table)
    report.write_meta(run_dir, common.provenance(args, env_config_data, models, extra={
        "methods": methods, "budgets": grid, "horizon": horizon, "context": context}))

    header = common.header_for(args, env, models, context["network"])
    if not args.no_plots:
        def draw_all():
            report.plot_budget(run_dir, rows, header)
        draw_all()
        with report.plain():
            draw_all()
        written.append("plots/pdf/ and plots/png/")

    print("\n" + table)
    print(f"\nwrote {run_dir}/")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
