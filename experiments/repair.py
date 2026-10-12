"""Break each network, then see how each method puts it right.

Nine events over two classes. The rank-breaking ones (edge drops, a node leaving or
joining) have a well-defined minimum repair, so a method is scored on where its answer
sits among *every* repair of the same size, 100 being the best available. The
non-breaking ones (a redundant edge lost, a spurious edge added, pose drift, and the
`none` control) leave the network rigid, so the count is vacuous and the shape error
relative to the pre-damage graph is the reading.

The damage is derived from the instance index rather than from a shared stream, so
adding a method does not change what any other method was asked to repair. That needs
a frozen instance set, which is why --benchmark is required.

usage:
  uv run experiments/repair.py <environment_name> --benchmark NAME
      [--events a,b|all] [--k 1,2] [--methods a,b|all] [--model NAME[=LABEL]]
"""

import os
import sys

# the repository root, in place of experiments/ -- `repair` below is the library at the
# root, which this script shares a name with
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv

from tqdm import tqdm

import benchmark
import report
import repair as repair_lib
from experiments import common


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_common_args(p)
    common.add_model_args(p)
    p.add_argument("--events",
                   default="none,edge_drop_random,edge_drop_redundant,"
                           "edge_add_spurious,node_leave,node_join",
                   help="comma-separated subset of " + ",".join(repair_lib.EVENTS) +
                        ", or 'all'")
    p.add_argument("--k", default="2",
                   help="how much damage each event does, comma separated")
    p.add_argument("--methods", default="greedy_repair,greedy,optimal_repair,rebuild,learned",
                   help="`learned` names every --model; `all` is every classical repair "
                        "method plus every model")
    p.add_argument("--instances", type=int, default=None,
                   help="benchmark instances to damage (default: --episodes)")
    p.add_argument("--join-domain", default=None,
                   help="fix the joining agent's domain instead of drawing it; it moves "
                        "rank_K and m_req, so it dominates the difficulty of node_join")
    p.add_argument("--drift", type=float, default=0.1,
                   help="pose_drift sigma, as a fraction of the formation radius")
    p.add_argument("--restarts", type=int, default=20,
                   help="restarts for the rebuild method; it reports the best of them")
    p.add_argument("--cap", type=int, default=repair_lib.ENUMERATION_CAP,
                   help="combinations above which the population is sampled rather than "
                        "enumerated. A truncated enumeration is not a percentile, so the "
                        "tier is decided before anything is enumerated")
    p.add_argument("--want", type=int, default=repair_lib.SAMPLE_WANT,
                   help="repairs to rejection-sample when the population is too large "
                        "to enumerate")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def resolve(args, models):
    """(events, ks, methods), or None when one of them is not a name."""
    events = [e.strip() for e in args.events.split(",") if e.strip()]
    if events == ["all"]:
        events = list(repair_lib.EVENTS)
    unknown = [e for e in events if e not in repair_lib.EVENTS]
    if unknown:
        print(f"unknown event(s) {unknown}, expected {list(repair_lib.EVENTS)}")
        return None
    ks = tuple(int(x) for x in args.k.split(","))

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if methods == ["all"]:
        methods = list(repair_lib.CLASSICAL) + list(models)
    methods = [m for x in methods for m in (list(models) if x == "learned" else [x])]
    unknown = [m for m in methods if m not in repair_lib.CLASSICAL and m not in models]
    if unknown:
        print(f"unknown method(s) {unknown}, expected "
              f"{list(repair_lib.CLASSICAL) + ['learned']}")
        return None
    return events, ks, methods


def write(run_dir, rows, spread, summary, table):
    """The three csvs and the table. -> the relative paths."""
    with open(os.path.join(run_dir, "repair.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(repair_lib.ROW_FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in repair_lib.ROW_FIELDS})
    written = ["repair.csv"]
    if summary:
        with open(os.path.join(run_dir, "repair_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        written.append("repair_summary.csv")
    # one row per enumerated repair, so a figure redraws without re-enumerating
    with open(os.path.join(run_dir, "repair_spread.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "event", "event_k", "size", "tier", "shape_err"])
        for rec in spread:
            for err in rec["errors"]:
                w.writerow([rec["instance"], rec["event"], rec["event_k"],
                            rec["size"], rec["tier"], err])
    written.append("repair_spread.csv")
    with open(os.path.join(run_dir, "repair.txt"), "w") as fh:
        fh.write(table + "\n")
    written.append("repair.txt")
    return written


def main(argv=None):
    args = parse_args(argv)
    if not args.benchmark:
        print("repair needs --benchmark: the damage is derived from the instance index")
        return 1

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

    resolved = resolve(args, models)
    if resolved is None:
        return 1
    events, ks, methods = resolved

    nets, _ = benchmark.load(args.benchmark)
    nets = nets[:(args.instances or args.episodes)]
    print(f"env: {args.environment_name} | n={env.network.n} | {len(nets)} instances "
          f"from {args.benchmark} | events {', '.join(events)} | k = "
          f"{', '.join(map(str, ks))}")

    # one unit per (instance, event, k); enumeration dominates and its cost varies by
    # orders of magnitude between events, so the bar is a position rather than an ETA
    cases = len(nets) * sum(len(ks) if e not in ("none", "pose_drift") else 1
                            for e in events)
    bar = tqdm(total=cases, unit="case", dynamic_ncols=True, leave=False)
    rows, spread = repair_lib.measure(
        env=env, instances=nets, events=events, ks=ks, methods=methods,
        models=models, wrapped=wrapped, seed=args.seed,
        policy_mode=args.policy_mode, restarts=args.restarts,
        join_domain=args.join_domain, drift_sigma=args.drift,
        cap=args.cap, want=args.want,
        progress=lambda s: (bar.set_description_str(s), bar.update(1)))
    bar.close()

    context = {
        "environment": args.environment_name,
        "instances": f"{len(nets)} from benchmark {args.benchmark} "
                     f"({benchmark.digest(args.benchmark)})",
        "events": ", ".join(events),
        "damage": f"k = {', '.join(map(str, ks))}",
    }
    table = repair_lib.format_table(rows, context, brief=args.brief)

    run_dir = common.run_directory(args, models, with_plots=not args.no_plots)
    written = write(run_dir, rows, spread, repair_lib.summarise(rows), table)
    # the kappa each policy was TRAINED at, from its own manifest -- not a setting of
    # this run, which scores every row by one phi
    kappas = {lab: ((m["manifest"] or {}).get("environment_config_raw") or {})
                       .get("stiffness_kappa")
              for lab, m in models.items()}

    report.write_meta(run_dir, common.provenance(args, env_config_data, models, extra={
        "events": events, "ks": list(ks), "methods": methods,
        # recorded so repair_kappa redraws without reading train/, which is gitignored
        "trained_kappa": kappas}))

    header = common.header_for(
        args, env, models,
        f"{env.network.n} agents in {common.describe_domains(env)}, "
        f"action space {env.action_space_type}")

    if not args.no_plots:
        def draw_all():
            report.plot_repair_percentile(run_dir, rows, header)
            report.plot_repair_count(run_dir, rows, header)
            report.plot_repair_locality(run_dir, rows, header)
            report.plot_repair_events(run_dir, rows, header)
            report.plot_repair_choice(run_dir, spread, header)
            report.plot_repair_kappa(run_dir, rows, header, kappas)
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
