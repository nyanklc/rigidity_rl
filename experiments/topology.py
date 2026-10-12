"""What the graph each method builds looks like, against what the theory predicts.

Five families of property, all read off the topologies the methods produce on one
instance set: how often an edge's reverse is also there, how far each agent's
out-degree sits above the floor its own domain imposes, which ordered domain classes
get the edges, how long the edges are and how degenerate the bearings one agent
measures, and two checks against answers that are already known (the Laman
characterisation in homogeneous R^2, and spectral reproducing greedy at kappa 0).

usage:
  uv run experiments/topology.py <environment_name> [--benchmark NAME]
      [--methods a,b|all] [--model NAME[=LABEL]]
"""

import os
import sys

# the repository root, in place of experiments/ -- `topology` below is the library at the
# root, which this script shares a name with
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import csv

import numpy as np
from tqdm import tqdm

import baselines
import report
import topology as topology_lib
from experiments import common


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_common_args(p)
    common.add_model_args(p)
    p.add_argument("--methods",
                   default="initial,greedy,spectral,constructive,random_rigid,learned",
                   help="`learned` names every --model; `all` is every method plus the "
                        "random rigid null")
    p.add_argument("--steps", type=int, default=None,
                   help="rollout budget for the policies and the random baseline")
    p.add_argument("--restarts", type=int, default=20,
                   help="restarts the constructive baseline reports the best of")
    p.add_argument("--spectral-shortlist", type=int, default=5,
                   help="candidates the spectral baseline rescores before it commits")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def resolve_methods(args, labels):
    """The methods to run, in the shared order, or None when one is not a name."""
    known = baselines.method_sequence(labels) + [topology_lib.NULL_METHOD]
    asked = [m.strip() for m in args.methods.split(",") if m.strip()]
    if asked == ["all"]:
        return known
    asked = [m for x in asked for m in (list(labels) if x == "learned" else [x])]
    unknown = [m for m in asked if m not in known]
    if unknown:
        print(f"unknown method(s) {unknown}, expected {known} or 'learned'")
        return None
    return [m for m in known if m in asked]


def write(run_dir, rows, nodes, pairs, jac, table):
    """The four csvs and the table. -> the relative paths."""
    written = []
    for name, fields, records in (
            ("topology.csv", topology_lib.ROW_FIELDS, rows),
            ("topology_nodes.csv", topology_lib.NODE_FIELDS, nodes),
            ("topology_pairs.csv", topology_lib.PAIR_FIELDS, pairs),
            ("topology_jaccard.csv", topology_lib.JACCARD_FIELDS, jac)):
        if not records:
            continue
        with open(os.path.join(run_dir, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
            w.writeheader()
            w.writerows(records)
        written.append(name)
    with open(os.path.join(run_dir, "topology.txt"), "w") as fh:
        fh.write(table + "\n")
    written.append("topology.txt")
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

    # after the models are loaded, so a --model run and a plain run see the same networks
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

    print(f"env: {args.environment_name} | n={env.network.n} | {instance_note} | "
          f"methods {', '.join(methods)}")
    planar_only = all(set(a.domain for a in net.agents) == {"R^2"} for net in instances)
    if not planar_only:
        print("  the Laman check is skipped: it characterises minimal bearing rigidity "
              "in homogeneous R^2 only, and these instances are not that")

    bar = tqdm(total=len(instances), unit="instance", dynamic_ncols=True, leave=False)
    rows, nodes, pairs, jac = topology_lib.measure(
        env=env, instances=instances, methods=methods, models=models, wrapped=wrapped,
        steps=args.steps or steps, policy_mode=args.policy_mode,
        spectral_shortlist=args.spectral_shortlist, restarts=args.restarts,
        seed=args.seed,
        progress=lambda s: (bar.set_description_str(s), bar.update(1)))
    bar.close()

    kappa = env_config_data.get("stiffness_kappa")
    context = {
        "environment": args.environment_name,
        "network": f"{env.network.n} agents in {common.describe_domains(env)}, "
                   f"action space {env.action_space_type}",
        "objective": f"{env.state_score_type}, stiffness_kappa {kappa}",
        "instances": instance_note,
        "methods": ", ".join(methods),
    }
    # the kappa each policy was TRAINED at, from its own manifest: the length prediction
    # is about that and not about the kappa this run scores by
    for lab, m in models.items():
        cfg = (m["manifest"] or {}).get("environment_config_raw") or {}
        context[f"model {lab}"] = (f"{m['name']}, trained at stiffness_kappa "
                                  f"{cfg.get('stiffness_kappa')}")

    table = topology_lib.format_table(rows, nodes, pairs, jac, context=context,
                                      brief=args.brief, kappa=kappa)

    run_dir = common.run_directory(args, models, with_plots=not args.no_plots)
    written = write(run_dir, rows, nodes, pairs, jac, table)
    report.write_meta(run_dir, common.provenance(args, env_config_data, models, extra={
        "methods": methods, "instances": len(instances)}))

    header = common.header_for(args, env, models, context["network"])
    if not args.no_plots:
        def draw_all():
            report.plot_topology_structure(run_dir, rows, nodes, pairs, header)
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
