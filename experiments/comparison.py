"""Every method on the same instances, scored by the same phi.

  initial       the graph the sampler produced, i.e. what a policy starts from
  random        uniform random actions through env.step(), the floor for this action space
  degree        add/prune by degree alone: the tier-1 distributed-plausible reference
  greedy        hill climbing on phi, one edge toggle at a time
  spectral      the same landscape in closed form, verified before it commits
  anneal        simulated annealing on phi, budget matched to what greedy spent
  constructive  builds from the empty graph, keeping any edge that raises rank(B)
  learned       a trained policy; --model takes several and each becomes its own row
  optimal       exhaustive search for the fewest-edge rigid graph (small n only)

Every row goes through Environment.compute_state_score, so the comparison is against
the exact objective the agent trains on rather than against a proxy for it.

usage:
  uv run experiments/comparison.py <environment_name> [--episodes N] [--benchmark NAME]
      [--methods a,b,c] [--model NAME[=LABEL]] [--noise-sweep 0.5,1,5] [--no-plots]
"""

import os
import sys

# the repository root, in place of experiments/ -- the drivers import the libraries at
# the root (baselines, report, repair) and some share a name with a script here
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import glob

import numpy as np
import torch
from tqdm import tqdm

import baselines
import benchmark
import cost
import report
from baselines import (ALL_METHODS, MAX_BRUTE_FORCE_N, MAX_DECISION_ANALYSIS_N,
                       measure_noise, method_sequence, run_method)
from experiments import common


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_common_args(p)
    common.add_model_args(p)
    p.add_argument("--methods", default="initial,greedy,constructive,random,learned",
                   help="comma-separated subset of " + ",".join(ALL_METHODS) + ", or "
                        "'all'. greedy and anneal are the expensive ones at large n, and "
                        "optimal is exhaustive search (n <= %d)" % MAX_BRUTE_FORCE_N)
    p.add_argument("--brute-force", action="store_true",
                   help="the older spelling of --methods optimal")
    p.add_argument("--steps", type=int, default=None,
                   help="rollout horizon for random/learned (default: env truncate/max steps)")
    p.add_argument("--restarts", type=int, default=20,
                   help="restarts for the constructive baseline; it reports the best of them")
    p.add_argument("--spectral-shortlist", type=int, default=5,
                   help="how many of the closed form's top candidates the spectral "
                        "baseline rescores before applying one. The ranking is exact at "
                        "stiffness_kappa = 0 and a prior above it, which is what the "
                        "verification is for")
    p.add_argument("--anneal-budget", type=int, default=None,
                   help="phi evaluations the annealer gets. Default: exactly what greedy "
                        "spent on the same instance, so the two are budget matched; "
                        "4*n*(n-1) when greedy is not being run")
    p.add_argument("--noise-sweep", default=None,
                   help="comma-separated bearing-noise levels as the per-axis sigma in "
                        "DEGREES; the RMS angle is sqrt(2) times it. Measures the shape "
                        "error each method's graph actually produces against the analytic "
                        "prediction, e.g. 0.5,1,5")
    p.add_argument("--noise-trials", type=int, default=30,
                   help="noise draws per (method, sigma)")
    p.add_argument("--paired-reference", default="greedy",
                   help="method the paired per-instance block compares against. Empty "
                        "to skip the block")
    p.add_argument("--no-plots", action="store_true",
                   help="skip plots; also skips per-step tracing, so rollouts are faster")
    p.add_argument("--plot-episodes", type=int, default=3,
                   help="how many individual episodes get their own detail figure")
    p.add_argument("--replay-env", action="store_true",
                   help="score every method against the environment --model was trained "
                        "on (from its manifest) instead of the current code")
    return p.parse_args(argv)


def resolve_methods(args, labels):
    """The methods to run, in a fixed order, or None when one is not a method."""
    known = method_sequence(labels)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if methods == ["all"]:
        methods = list(known)
    # `learned` names every model whatever they are called, which is what keeps the
    # default --methods string meaning what it always meant
    methods = [m for x in methods for m in (labels if x == "learned" else [x])]
    unknown = [m for m in methods if m not in known]
    if unknown:
        print(f"unknown method(s): {unknown}, expected {known}")
        return None
    if args.brute_force and "optimal" not in methods:
        methods.append("optimal")
    return [m for m in known if m in methods]


def main(argv=None):
    args = parse_args(argv)

    model_specs = common.parse_models(args.model)
    labels = [lab for _, lab in model_specs]
    if args.replay_env and len(model_specs) > 1:
        print("--replay-env replays one model's archived environment, so with several "
              "models the rows would not all be scored by the same phi. Pass one "
              "--model, or drop --replay-env.")
        return 1

    methods = resolve_methods(args, labels)
    if methods is None:
        return 1

    loaded = common.load_environment(args)
    if loaded is None:
        return 1
    env, env_config_data, steps = loaded

    loaded = common.load_models(args, env, env_config_data, model_specs, steps)
    if loaded is None:
        return 1
    models, wrapped, env, steps = loaded
    common.announce(models)

    n = env.network.n
    if "optimal" in methods and n > MAX_BRUTE_FORCE_N:
        print(f"brute force refused: n={n} > {MAX_BRUTE_FORCE_N} "
              f"({2 ** (n * n - n):.3g} possible graphs)")
        methods.remove("optimal")

    print(f"env: {args.environment_name} | n={n} | rollout horizon={steps} "
          f"| episodes={args.episodes}")

    # tracing costs one extra eigendecomposition per step, so it rides with the plots
    tracing = not args.no_plots
    # an archived environment replayed by --replay-env may predate this flag
    if hasattr(env, "trace_min_eig"):
        env.trace_min_eig = tracing
    traces = [] if tracing else None

    # re-seed here so the episodes drawn below do not depend on how much RNG the setup
    # above consumed: loading a model does an extra reset, which would otherwise make a
    # --model run and a plain run evaluate different instances
    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    # a sampling policy draws from torch's global RNG, so this is what makes
    # --policy-mode sample repeatable for a given seed
    torch.manual_seed(args.seed)

    # private to the methods that use randomness: np.random is the stream instances are
    # drawn from, so a method sharing it would change which networks the others are
    # scored on. See the note in baselines._construct_once.
    construct_rng = np.random.default_rng(args.seed)
    anneal_rng = np.random.default_rng(args.seed)
    degree_rng = np.random.default_rng(args.seed)

    instances = []
    # every legal edit is scored at every step, so this is gated on n
    decisions = [] if (models and tracing and n <= MAX_DECISION_ANALYSIS_N) else None
    if decisions is None and models and tracing:
        print(f"  decision analysis skipped: n={n} > {MAX_DECISION_ANALYSIS_N}")
    elif decisions is not None and len(models) > 1:
        # it scores every legal edit at every step, once per model
        print(f"  decision analysis runs for all {len(models)} models")

    # the flag is in degrees because that is what a camera spec is quoted in;
    # everything downstream of here is radians
    sweep_degrees = ([float(x) for x in args.noise_sweep.split(",")]
                     if args.noise_sweep else [])
    sigmas = [float(np.radians(d)) for d in sweep_degrees]
    if sigmas:
        print(f"  noise sweep: sigma = {sweep_degrees} degrees per axis "
              f"(RMS angle x{np.sqrt(2):.2f}), {args.noise_trials} draws each")

    frozen, instance_note = common.instances_for(args, env)
    if frozen is not None:
        print(f"  instances: {instance_note}, sampling disabled")

    # One bar for the run. Every method on every episode is a unit, the noise sweep is
    # one more per episode, and the report is the last; the description says which.
    units = args.episodes * (len(methods) + (1 if sigmas else 0)) + 1
    progress = tqdm(total=units, unit="step", dynamic_ncols=True, leave=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    def doing(what):
        progress.set_description_str(what)

    rows = []
    for ep in range(args.episodes):
        env.freeze_network = False
        if frozen is not None:
            env.network = copy.deepcopy(frozen[ep])
            env.freeze_network = True
            env.reset()                          # bookkeeping only
        else:
            env.reset()                          # draw a fresh instance
        instance = copy.deepcopy(env.network)
        env.freeze_network = True                # every reset below keeps it
        # every instance, not just the ones a figure draws: the formation figures read
        # instances[0] today and a later redraw may want any of the others, and the
        # poses are the one thing no csv can reconstruct
        instances.append(copy.deepcopy(env.network))

        def restore():
            env.network = copy.deepcopy(instance)
            env.reset()

        ctx = dict(env=env, args=args, traces=traces, episode=ep, steps=steps,
                   models=models, wrapped=wrapped, decisions=decisions,
                   construct_rng=construct_rng, anneal_rng=anneal_rng,
                   degree_rng=degree_rng, anneal_budget=args.anneal_budget)

        episode_rows = []
        for name in methods:
            doing(f"episode {ep + 1}/{args.episodes}  {name}")
            restore()
            # the restore is shared setup, so only the method itself is metered
            with cost.Meter() as meter:
                row = run_method(name, ctx)
            progress.update(1)
            if row is None:                      # unavailable: no model, or n too large
                continue
            row["cost"] = meter.counts
            row["ms"] = meter.ms
            episode_rows.append(row)
            # budget matching, when it was not set on the command line: the annealer gets
            # exactly the phi evaluations greedy spent on this instance
            if name == "greedy" and args.anneal_budget is None:
                ctx["anneal_budget"] = meter.counts.get("score_network", 0) or None

        if sigmas:
            doing(f"episode {ep + 1}/{args.episodes}  bearing noise")
            for r in episode_rows:
                measure_noise(env, r, sigmas, args.noise_trials,
                              np.random.default_rng(args.seed + ep))
            progress.update(1)

        for r in episode_rows:
            r["episode"] = ep
            # the bound is a property of the poses, so it belongs to the row rather than
            # to the run; stamped here so no run_* can forget it
            r["m_req"] = int(env.m_req)
        rows.extend(episode_rows)

    # ── report ────────────────────────────────────────────────────────────────────────
    domain_str = common.describe_domains(env)
    context = {
        "environment": args.environment_name,
        "network": f"{n} agents in {domain_str}, action space {env.action_space_type}",
        "objective": f"{env.state_score_type} state score",
        "instances": (f"{args.episodes} networks from benchmark {args.benchmark} "
                      f"({benchmark.digest(args.benchmark)})" if args.benchmark
                      else f"{args.episodes} random networks, seed {args.seed}"),
    }
    if models:
        context["policy"] = "  |  ".join(
            f"{lab} = {m['name']} ({m['algorithm']})" for lab, m in models.items())
        context["rollout"] = (f"--policy-mode {args.policy_mode}, "
                              f"{steps}-step budget, same for every model")

    doing("writing the table")
    table = report.format_table(rows, context, brief=args.brief)
    # every method ran on the same instances, so the pairing is free; the table's +-
    # is how much the NETWORKS differ, which is far larger than the gap between methods
    table += report.format_paired(rows, reference=args.paired_reference,
                                  brief=args.brief)

    run_dir = common.run_directory(args, models, with_plots=bool(traces))
    report.write_summary(run_dir, table)
    report.write_csvs(run_dir, rows, traces)
    wrote_costs = report.write_costs(run_dir, rows)
    wrote_noise = report.write_noise(run_dir, rows)
    # the poses and domains the formation figures draw, which no csv can hold
    wrote_instances = report.write_instances(run_dir, instances, meta={
        "environment": args.environment_name, "benchmark": args.benchmark,
        "seed": args.seed})
    if decisions:
        report.write_decisions(run_dir, decisions)
    info_blocks = common.run_info_blocks(args, env, env_config_data, models, steps,
                                         methods, rows)
    report.write_meta(run_dir, common.provenance(
        args, env_config_data, models, extra={
            "n": n, "rollout_steps": steps, "methods": methods,
            "context": context,
            # the rendered run_info panel, so it redraws without reloading the models
            "run_info": info_blocks}))

    written = ["summary.txt", "results.csv"] + (["trajectories.csv"] if traces else [])
    if wrote_costs:
        written.append("cost.csv and cost.txt")
    if wrote_noise:
        written.append("noise.csv")
    if wrote_instances:
        written.append("instances.npz")
    if decisions:
        written.append("decisions.csv")

    header = common.header_for(args, env, models, context["network"])
    table_header = dict(header, objective=context["objective"],
                        policy=context.get("rollout"))

    if traces:
        def draw_all():
            report.plot_trajectories(run_dir, traces, rows, header)
            report.plot_outcomes(run_dir, traces, rows, header)
            report.plot_summary(run_dir, rows, header)
            report.plot_noise_sweep(run_dir, rows, header)
            report.plot_prediction_check(run_dir, rows, header)
            report.plot_uncertainty(run_dir, instances, rows, header)
            report.plot_softest_mode(run_dir, instances, rows, header)
            report.plot_sensitivity(run_dir, instances, rows, header)
            report.plot_comparison(run_dir, rows, header)
            report.plot_estimation(run_dir, rows, header)
            report.plot_topology(run_dir, instances, rows, header)
            report.plot_run_info(run_dir, info_blocks, header)
            report.plot_cost(run_dir, rows, header)
            report.plot_decisions(run_dir, decisions, header)
            report.plot_table(run_dir, rows, table_header)
            for ep in range(min(args.plot_episodes, args.episodes)):
                sel = [t for t in traces if t["episode"] == ep]
                ep_header = dict(header, episodes=None, subtitle=f"episode {ep}")
                report.plot_trajectories(run_dir, sel,
                                         [r for r in rows if r["episode"] == ep],
                                         ep_header, filename=f"episode_{ep:03d}",
                                         aggregate_over_episodes=False)

        doing("drawing figures")
        draw_all()
        # and again without the title block or the notes card, for a document that
        # carries its own caption
        doing("drawing figures, plain")
        with report.plain():
            draw_all()
        # counted rather than predicted: several figures skip themselves when the run
        # does not carry what they draw
        made = len(glob.glob(os.path.join(run_dir, "plots", "png", "*.png")))
        written.append(f"plots/pdf/ and plots/png/ ({made} figures each)")

    progress.update(1)
    doing("done")
    progress.close()

    print("\n" + table)
    print(f"\nwrote {run_dir}/")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
