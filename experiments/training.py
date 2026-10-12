"""Learning curves, from the tensorboard event files a training run left under runs/.

Reads `runs/<name>` directly, so it works mid-run. What separates a policy from a
sampling search is in the Decision/ and Probe/ series rather than in the return, which
is why the tail mean of every metric is written out and not just the reward.

usage:
  uv run experiments/training.py <environment_name> [--model NAME[=LABEL]] [--runs a,b]
"""

import os
import sys

sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv

import numpy as np

import report
from experiments import common


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_common_args(p, benchmark=False, episodes=False, brief=False)
    common.add_model_args(p, policy_mode=False)
    p.add_argument("--runs", default=None,
                   help="tensorboard directories under runs/, comma separated; defaults "
                        "to the --model names")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def measure(args, run_dir, models, header):
    """(written, draws) from the event files. Says which runs it could not find."""
    from tools.compare_runs import load as load_events

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
                # the tail mean is what a curve is read for; the raw series stays in the
                # event files rather than being copied here
                wr.writerow([labels[label], tag.strip(), len(v),
                             f"{float(np.mean(v[int(len(v) * 0.85):])):.6g}"])
    return [os.path.relpath(path, run_dir)], [
        lambda: report.plot_training(run_dir, series, header)]


def main(argv=None):
    args = parse_args(argv)
    if not (args.runs or args.model):
        print("training needs --runs, or a --model whose name is a directory under runs/")
        return 1

    model_specs = common.parse_models(args.model)
    loaded = common.load_environment(args)
    if loaded is None:
        return 1
    env, env_config_data, steps = loaded
    # the curves come from the event files; the checkpoints are loaded only so a run
    # named by --model is labelled and coloured the way every other figure labels it
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
