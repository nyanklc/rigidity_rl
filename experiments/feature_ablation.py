"""What is a trained policy actually reading? Destroy one observation channel at a time.

Two readings, because they answer different questions. Sensitivity perturbs the channel
at each state of an unperturbed reference trajectory and reports how often the argmax
action changes; outcome re-runs the whole episode with the channel perturbed and reports
what it cost in phi, edges, rigid% and minimal%. A channel can be sensitive without
mattering, and can matter without being sensitive.

Run more than one --mode before believing a positive: `zero` and `noise` change the
input scale as well as its meaning, so a reaction to them can be out-of-distribution
surprise rather than dependence. The negative results are the ones that survive a mode
change.

usage:
  uv run experiments/feature_ablation.py <environment_name> --model NAME[=LABEL]
      [--mode shuffle,zero,noise|all] [--ablation-episodes N]
"""

import os
import sys

# the repository root, in place of experiments/ -- `ablation` below is the library at
# the root, not this script
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from types import SimpleNamespace

import numpy as np
import torch

import ablation
import report
from experiments import common

MODES = ("shuffle", "zero", "noise")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_common_args(p, benchmark=False, episodes=False, brief=False)
    common.add_model_args(p, required=True, policy_mode=False)
    p.add_argument("--mode", default="shuffle",
                   help="shuffle, zero, noise, a comma-separated subset, or 'all'. "
                        "Reading one mode alone is how a channel gets called unimportant "
                        "when it was simply not perturbed")
    p.add_argument("--ablation-episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=None,
                   help="cap on the episode length (default: min(env.max_steps, 100))")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    # the header names the run by its episode count, and here that is the ablation's
    args.episodes = args.ablation_episodes
    return args


def measure(args, run_dir, env, wrapped, models, header, doing=None):
    """One csv and one txt per (model, mode), plus a figure per model."""
    from skrl.utils.spaces.torch import unflatten_tensorized_space

    out_dir = os.path.join(run_dir, "ablation")
    os.makedirs(out_dir, exist_ok=True)
    space = wrapped.observation_space
    steps = args.steps or int(min(getattr(env, "max_steps", 100), 100))
    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    if modes == ["all"]:
        modes = list(MODES)
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise SystemExit(f"--mode: {bad}, expected {', '.join(MODES)} or all")

    written, draws, raw = [], [], {}
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
            raw.setdefault(label, {})[mode] = {"rows": rows, "reference": ref_row}
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
    # the csvs carry the rendered table, whose column names are not the keys the figure
    # reads; this is the structure plot_ablation takes, so it redraws without re-running
    with open(os.path.join(run_dir, "ablation_raw.json"), "w") as fh:
        json.dump(raw, fh, indent=1, default=float)
    written.append("ablation_raw.json")
    return written, draws


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

    run_dir = common.run_directory(args, models, with_plots=not args.no_plots)
    header = common.header_for(
        args, env, models,
        f"{env.network.n} agents in {common.describe_domains(env)}, "
        f"action space {env.action_space_type}")

    written, draws = measure(args, run_dir, env, wrapped, models, header,
                             doing=lambda s: print(f"  {s}"))
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
