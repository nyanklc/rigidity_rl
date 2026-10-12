"""Drive a formation to a goal shape over the edge set each method chose.

One edge set is used twice, which is what `main.py` does: the goal bearings are read
off the goal formation over it, and the controller builds `B` from the current
formation over the same edges. So a method picks one topology, and it has to work for
both.

Each case is one goal shape and one start formation displaced from it. Every method
sees the same pair, so the edge set is the only thing that can explain a difference in
the trajectory.

Bearing control cannot observe translation or scale, so the formation reaches the goal
SHAPE rather than the goal poses; position error is reported with those projected out.

usage:
  uv run experiments/control.py <environment_name> --model NAME[=LABEL]
  uv run experiments/control.py <environment_name> --methods greedy,complete --cases 5
"""

import os
import sys

# the repository root, in place of experiments/ -- `control` at the root is the
# controller this drives, and this script shares its name
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import csv

import numpy as np
from tqdm import tqdm

import convergence
import report
from experiments import common

CLASSICAL = ("greedy", "spectral", "constructive", "complete")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_common_args(p, benchmark=True, episodes=False)
    common.add_model_args(p, policy_mode=False)
    p.add_argument("--methods", default="greedy,complete,learned",
                   help="comma-separated subset of " + ",".join(CLASSICAL) +
                        ",learned. `learned` names every --model")
    p.add_argument("--cases", type=int, default=3,
                   help="goal shapes. With --benchmark they are its first instances, "
                        "otherwise the environment samples them")
    p.add_argument("--sigma", type=float, default=0.15,
                   help="how far the start formation is displaced from the goal, as a "
                        "fraction of the formation radius. Restricted to each agent's "
                        "own DOFs, since a planar agent cannot undo a shift in z")
    p.add_argument("--dt", type=float, default=1e-3, help="integration step")
    p.add_argument("--steps", type=int, default=200_000, help="step budget per run")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="RMS bearing error per edge that counts as reached")
    p.add_argument("--sample-every", type=int, default=200,
                   help="how often the trajectory is recorded. The position error costs "
                        "an SVD, so this is not free")
    p.add_argument("--lin", type=float, default=1.0, help="linear gain")
    p.add_argument("--ang", type=float, default=1.0, help="angular gain")
    p.add_argument("--check-gain-mask", action="store_true",
                   help="also run the topology twice more, under control.py's own gain "
                        "mask and under the aligned one, and print what the difference "
                        "costs. Off by default because it is two extra simulations and "
                        "a defect report rather than part of the experiment")
    p.add_argument("--mask-lin", type=float, default=100.0,
                   help="linear gain for the gain-mask check; the default is the one "
                        "control.py ships and the one that makes the misalignment bite")
    p.add_argument("--restarts", type=int, default=5,
                   help="restarts for the constructive topology")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def resolve_methods(args, models):
    """The methods to run, or None when one of them is not a name."""
    names = [m.strip() for m in args.methods.split(",") if m.strip()]
    if names == ["all"]:
        names = list(CLASSICAL) + list(models)
    names = [m for x in names for m in (list(models) if x == "learned" else [x])]
    unknown = [m for m in names if m not in CLASSICAL and m not in models]
    if unknown:
        print(f"unknown method(s) {unknown}, expected {list(CLASSICAL) + ['learned']}")
        return None
    return names


def goal_shapes(args, env):
    """The goal formations, and a note for the header."""
    if args.benchmark:
        import benchmark
        nets, _ = benchmark.load(args.benchmark)
        nets = nets[:args.cases]
        return nets, (f"{len(nets)} from benchmark {args.benchmark} "
                      f"({benchmark.digest(args.benchmark)})")
    out = []
    env.freeze_network = False
    for _ in range(args.cases):
        env.reset()
        out.append(copy.deepcopy(env.network))
    return out, f"{len(out)} sampled, seed {args.seed}"


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
    for m in models.values():
        m["wrapped"] = wrapped

    methods = resolve_methods(args, models)
    if methods is None:
        return 1

    np.random.seed(args.seed)
    goals, instance_note = goal_shapes(args, env)
    print(f"env: {args.environment_name} | n={goals[0].n} | {instance_note} "
          f"| {', '.join(methods)} | {args.steps} steps of {args.dt:g}")

    bar = tqdm(total=len(goals) * len(methods), unit="run", dynamic_ncols=True,
               leave=False)
    rows, traces, topologies = convergence.measure(
        env=env, goal_networks=goals, methods=methods, models=models,
        rng=np.random.default_rng(args.seed), restarts=args.restarts,
        sigma=args.sigma, dt=args.dt, steps=args.steps, tol=args.tol,
        sample_every=args.sample_every, lin=args.lin, ang=args.ang,
        progress=lambda s: (bar.set_description_str(s), bar.update(1)))
    bar.close()

    context = {
        "environment": args.environment_name,
        "goal shapes": instance_note,
        "start": f"displaced by sigma {args.sigma:g} of the formation radius",
        "control": f"gains lin {args.lin:g} ang {args.ang:g}, dt {args.dt:g}, "
                   f"tol {args.tol:g} rad per edge",
    }
    if models:
        context["policies"] = "  ".join(f"{lab} = {m['name']}"
                                        for lab, m in models.items())
    text = [convergence.format_table(rows, context, brief=args.brief)]

    # the gain mask control.py ships against the aligned one, on the first topology
    if rows and args.check_gain_mask:
        first = rows[0]
        got = convergence.mask_comparison(
            goals[first["case"]], convergence.rank_K_of(goals[first["case"]]),
            topologies[(first["case"], first["method"])],
            np.random.default_rng(args.seed),
            sigma=args.sigma, dt=args.dt, steps=args.steps, tol=args.tol,
            lin=args.mask_lin, ang=args.ang)
        text += ["", f"WHAT THE GAIN MASK COSTS  (case {first['case']}, "
                     f"{first['method']}, gains lin={args.mask_lin:g} ang={args.ang:g})"]
        for name, hit in got.items():
            text.append(f"  {name:<32s} steps to tol {hit if hit else '-'}")
        text += ["  B's columns are [3n positions | 3n attitudes] and the controller",
                 "  tiles the gains per agent, so at lin != ang they land on the wrong",
                 "  coordinates. control.py is obsolete, so this prices it rather than",
                 "  patching it."]

    run_dir = common.run_directory(args, models, with_plots=not args.no_plots)
    with open(os.path.join(run_dir, "control.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(convergence.ROW_FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in convergence.ROW_FIELDS})
    with open(os.path.join(run_dir, "trajectories.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(convergence.TRACE_FIELDS))
        w.writeheader()
        for r in traces:
            w.writerow({k: r.get(k) for k in convergence.TRACE_FIELDS})
    # one row per (run, sampled step, agent). The goal formation is in here too, under
    # method "goal" at step -1, so a redraw needs no other file
    with open(os.path.join(run_dir, "paths.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(convergence.PATH_FIELDS)
        for case, goal_net in enumerate(goals):
            for i, agent in enumerate(goal_net.agents):
                w.writerow([case, "goal", -1, -1.0, i, *agent.pose.position])
        for r in traces:
            if r.get("positions") is None:
                continue
            for i, xyz in enumerate(r["positions"]):
                w.writerow([r["case"], r["method"], r["step"], r["t"], i, *xyz])
    with open(os.path.join(run_dir, "control.txt"), "w") as fh:
        fh.write("\n".join(text) + "\n")
    report.write_meta(run_dir, common.provenance(args, env_config_data, models,
                                                 extra={"methods": methods}))
    # paths.csv carries the goal positions but not the domains or the goal edge set,
    # which the formation figures need; the npz carries the networks themselves
    report.write_instances(run_dir, goals, meta={
        "environment": args.environment_name, "benchmark": args.benchmark,
        "seed": args.seed, "role": "goal formations"})
    written = ["control.csv", "trajectories.csv", "paths.csv", "instances.npz",
               "control.txt"]

    header = common.header_for(args, env, models,
                               f"{goals[0].n} agents in {common.describe_domains(env)}")
    # the figure draws the edge set beside the path it produced
    for r in rows:
        r["edges"] = topologies.get((r["case"], r["method"]))
    goal_by_case = {i: g for i, g in enumerate(goals)}

    if not args.no_plots:
        def draw_all():
            report.plot_control_trajectories(run_dir, traces, rows, header)
            report.plot_control_formations(run_dir, traces, rows, header,
                                           goals=goal_by_case)
            report.plot_control_summary(run_dir, rows, header)
        draw_all()
        with report.plain():
            draw_all()
        written.append("plots/pdf/ and plots/png/")

    print("\n" + "\n".join(text))
    print(f"\nwrote {run_dir}/")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
