"""Re-render an evaluation report from what the run already wrote.

Wording and layout in report.py change more often than the numbers do, and a figure
is usually redrawn several times before it reaches the thesis. Every evaluation run
therefore stores enough to rebuild all of its output without evaluating anything
again, and this reads that back:

    results.csv      one row per (episode, method), including the edge set
    trajectories.csv one row per (episode, method, step)
    noise.csv        one row per (episode, method, sigma), measured and predicted
    instances.npz    the Network objects, which no csv can hold
    decisions.csv    one row per edit a policy applied
    cost.csv         every counter
    meta.json        the arguments, the environment config and the provenance

    PYTHONPATH=. uv run tools/rerender_outputs.py [run_dir ...]

A run written before one of those files existed is redrawn as far as it goes, and
the figures whose input is missing are named and skipped rather than half drawn.
"""
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import report

# what each figure needs beyond results.csv, so a missing input is named rather than
# discovered as a blank panel
NEEDS = {
    "table": (), "summary": (), "outcomes": ("traces",), "trajectories": ("traces",),
    "comparison": (), "estimation": (),
    "noise": ("noise",), "prediction": ("noise",),
    "uncertainty": ("instances",), "softest_mode": ("instances",),
    "sensitivity": ("instances",), "topology": ("instances",),
    "cost": ("cost",), "decisions": ("decisions",), "run_info": ("run_info",),
}


def _num(v, cast=float):
    return None if v in ("", "nan", None) else cast(v)


# how each results.csv column comes back. Anything not named here is left as the
# string it was written as, so a column added later still survives the round trip.
ROW_CAST = {"episode": int, "m": int, "m_req": int, "work": int, "best_at": int,
            "score": float, "min_eig": float, "shape_err": float}


def load_rows(run_dir):
    """Every column results.csv carries, cast back to what the figures expect.

    at_bound is derived by report.at_bound from m and m_req rather than read, so a
    rerender agrees with the original run instead of trusting a stored boolean.
    """
    out = []
    with open(os.path.join(run_dir, "results.csv")) as f:
        for r in csv.DictReader(f):
            row = {}
            for key, raw in r.items():
                if key in ("at_bound", "edges"):
                    continue                      # derived, and handled by load_edges
                if key in ROW_CAST:
                    v = _num(raw)
                    row[key] = None if v is None else ROW_CAST[key](v)
                elif raw in ("True", "False"):
                    row[key] = raw == "True"
                else:
                    row[key] = raw
            # results.csv does not carry the counters, and a zero here would read as
            # "free" rather than as "not measured". cost.csv holds them.
            row.setdefault("cost", None)
            out.append(row)
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


def load_noise(run_dir, rows):
    """Put the per-level sweep back on the rows, so format_table and the two noise
    figures see exactly what they saw during the run."""
    path = os.path.join(run_dir, "noise.csv")
    if not os.path.exists(path):
        return False
    by_key = {(r["episode"], r["method"]): r for r in rows}
    with open(path) as f:
        for rec in csv.DictReader(f):
            row = by_key.get((int(rec["episode"]), rec["method"]))
            if row is None:
                continue
            sigma = float(rec["sigma_rad"])
            row.setdefault("noise", {})[sigma] = _num(rec["measured"])
            row.setdefault("noise_failed", {})[sigma] = _num(rec["failed"]) or 0.0
            pred = _num(rec["predicted"])
            if pred is not None and sigma > 0:
                row["pred_err"] = pred / sigma
    return True


def load_instances(run_dir):
    """The Network objects, or [] for a run written before they were stored."""
    path = os.path.join(run_dir, "instances.npz")
    if not os.path.exists(path):
        return []
    import benchmark
    return benchmark.read_networks(path)[0]


def load_edges(rows, run_dir):
    """Put each row's adjacency matrix back, from the results.csv column."""
    n = None
    with open(os.path.join(run_dir, "results.csv")) as f:
        recs = list(csv.DictReader(f))
    if not recs or "edges" not in recs[0]:
        return False
    nets = load_instances(run_dir)
    n = nets[0].n if nets else None
    if n is None:
        return False
    for row, rec in zip(rows, recs):
        row["edges"] = report.decode_edges(rec.get("edges"), n)
        row["edges_are"] = rec.get("edges_are")
    return True


def _cell(v):
    """A csv cell back to the type the figure expects. Empty stays None, since a
    decision with no error ranking carries no rank rather than a zero."""
    if v in ("", None, "nan"):
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return v


def load_decisions(run_dir):
    path = os.path.join(run_dir, "decisions.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [{k: _cell(v) for k, v in rec.items()} for rec in csv.DictReader(f)]


def load_costs(rows, run_dir):
    """Put the counters back on the rows, so the cost block and figure redraw."""
    path = os.path.join(run_dir, "cost.csv")
    if not os.path.exists(path):
        return False
    by_key = {(r["episode"], r["method"]): r for r in rows}
    with open(path) as f:
        for rec in csv.DictReader(f):
            row = by_key.get((int(rec["episode"]), rec["method"]))
            if row is None:
                continue
            row["ms"] = _num(rec.pop("ms")) or 0.0
            rec.pop("episode"), rec.pop("method"), rec.pop("total", None)
            row["cost"] = {k: int(float(v)) for k, v in rec.items() if v not in ("", None)}
    return True


def noise_block(run_dir):
    """The measured-vs-predicted table, verbatim. Its numbers are not recomputable here."""
    path = os.path.join(run_dir, "summary.txt")
    if not os.path.exists(path):
        return None
    m = re.search(r"(MEASURED SHAPE ERROR UNDER BEARING NOISE.*?)\n\n", open(path).read(),
                  re.S)
    return m.group(1) if m else None


def context_of(meta):
    """The table's context block. A run that recorded its own is used verbatim, which
    is the only way the model algorithm and the domain mix come back exactly."""
    if meta.get("context"):
        return dict(meta["context"])

    args, cfg = meta["args"], meta.get("environment_config") or {}
    domains = cfg.get("domains")
    if cfg.get("random_domains"):
        domain_str = "uniformly random domains per agent"
    elif isinstance(domains, str):
        # a homogeneous config stores the domain as a string, and iterating it gave
        # "mixed ['2', 'R', '^']"
        domain_str = domains
    elif domains:
        domain_str = (domains[0] if len(set(domains)) == 1
                      else f"mixed {sorted(set(domains))}")
    else:
        domain_str = "?"
    ctx = {
        "environment": args["environment_name"],
        "network": f"{meta['n']} agents in {domain_str}, "
                   f"action space {cfg.get('action_type', '?')}",
        "objective": f"{cfg.get('state_score_type', '?')} state score",
        "instances": (f"{args['episodes']} networks from benchmark {args['benchmark']} "
                      f"({meta.get('benchmark_digest')})" if args.get("benchmark")
                      else f"{args['episodes']} random networks, seed {args['seed']}"),
    }
    info = meta.get("model_info") or {}
    if info:
        ctx["policy"] = "  |  ".join(
            f"{lab} = {m['name']} ({m.get('algorithm', '?')})" for lab, m in info.items())
        ctx["rollout"] = (f"--policy-mode {args['policy_mode']}, "
                          f"{meta['rollout_steps']}-step budget, same for every model")
    elif args.get("model"):
        ctx["policy"] = (f"{args['model']} (?, --policy-mode {args['policy_mode']}, "
                         f"{meta['rollout_steps']}-step budget)")
    return ctx


def rerender(run_dir):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    args = meta["args"]
    rows, traces = load_rows(run_dir), load_traces(run_dir)
    have = {
        "traces": bool(traces),
        "noise": load_noise(run_dir, rows),
        "cost": load_costs(rows, run_dir),
        "instances": bool(load_instances(run_dir)),
        "decisions": bool(load_decisions(run_dir)),
        "run_info": bool(meta.get("run_info")),
    }
    instances = load_instances(run_dir)
    decisions = load_decisions(run_dir)
    load_edges(rows, run_dir)
    # the same registration the run did, so the colours and the method blurbs match
    info = meta.get("model_info") or {}
    report.configure_methods([
        (lab, f"trained policy {m['name']} ({m.get('algorithm', '?')})")
        for lab, m in info.items()])
    ctx = context_of(meta)

    # the algorithm name is not in meta; keep the one the old summary recorded, where
    # there still is one. A rerender after the summary was deleted just loses the name
    summary_path = os.path.join(run_dir, "summary.txt")
    old = open(summary_path).read() if os.path.exists(summary_path) else ""
    was = re.search(r"policy\s*: \S+ \((\w+),", old)
    if was and "policy" in ctx:
        ctx["policy"] = ctx["policy"].replace("(?,", f"({was.group(1)},")

    table = report.format_table(rows, ctx, brief=args.get("brief", False))
    if not have["noise"]:
        # a run written before noise.csv existed: keep whatever the old summary said,
        # since the numbers cannot be recomputed without repeating the sweep
        block = noise_block(run_dir)
        rule = re.search(r"\n-{20,}\n", table)
        if block and "MEASURED SHAPE ERROR" not in table and rule:
            table = table.replace(rule.group(0),
                                  "\n" + block + "\n" + rule.group(0), 1)
    table += report.format_paired(rows, reference=args.get("paired_reference", "greedy"),
                                  brief=args.get("brief", False))
    report.write_summary(run_dir, table)

    made, skipped = ["summary.txt"], []
    header = {
        "short": report.short_env_name(args["environment_name"]),
        "env": args["environment_name"],
        "model": args.get("model"),
        "network": ctx["network"],
        "episodes": args["episodes"],
        "seed": args["seed"],
        "benchmark": args.get("benchmark"),
    }
    table_header = dict(header, objective=ctx["objective"], policy=ctx.get("policy"))

    drawn = []

    def draw_all():
        drawn.clear()
        for name, needs in NEEDS.items():
            if any(not have[k] for k in needs):
                continue
            if name == "table":
                report.plot_table(run_dir, rows, table_header)
            elif name == "summary":
                report.plot_summary(run_dir, rows, header)
            elif name == "comparison":
                report.plot_comparison(run_dir, rows, header)
            elif name == "estimation":
                report.plot_estimation(run_dir, rows, header)
            elif name == "outcomes":
                report.plot_outcomes(run_dir, traces, rows, header)
            elif name == "trajectories":
                report.plot_trajectories(run_dir, traces, rows, header)
            elif name == "noise":
                report.plot_noise_sweep(run_dir, rows, header)
            elif name == "prediction":
                report.plot_prediction_check(run_dir, rows, header)
            elif name == "uncertainty":
                report.plot_uncertainty(run_dir, instances, rows, header)
            elif name == "softest_mode":
                report.plot_softest_mode(run_dir, instances, rows, header)
            elif name == "sensitivity":
                report.plot_sensitivity(run_dir, instances, rows, header)
            elif name == "topology":
                report.plot_topology(run_dir, instances, rows, header)
            elif name == "cost":
                report.plot_cost(run_dir, rows, header)
            elif name == "decisions":
                report.plot_decisions(run_dir, decisions, header)
            elif name == "run_info":
                # tuples came back from json as lists; plot_run_info unpacks pairs
                report.plot_run_info(run_dir, [(sec, [tuple(kv) for kv in fields])
                                               for sec, fields in meta["run_info"]],
                                     header)
            drawn.append(name)
        if traces:
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

    skipped = sorted({n for n, needs in NEEDS.items()
                      if any(not have[k] for k in needs)})
    if drawn:
        made.append(f"plots for {', '.join(drawn)}"
                    + (" and episode_NNN" if traces else "")
                    + ", each with its -plain twin")
    if skipped:
        missing = sorted({k for n in skipped for k in NEEDS[n] if not have[k]})
        made.append(f"skipped {', '.join(skipped)}: this run stored no "
                    f"{', '.join(missing)}")
    return made


# ---------------------------------------------------------------- the other run kinds

def _rows(path, skip=()):
    """A csv back to row dicts, numbers and booleans cast, blanks as None."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [{k: _cell(v) for k, v in rec.items() if k not in skip}
                for rec in csv.DictReader(f)]


def _header(meta, ctx=None):
    args = meta["args"]
    return {
        "short": report.short_env_name(args["environment_name"]),
        "env": args["environment_name"],
        "model": args.get("model"),
        "network": (ctx or {}).get("network", ""),
        "episodes": args.get("episodes"),
        "seed": args.get("seed"),
        "benchmark": args.get("benchmark"),
    }


def _register(meta):
    info = meta.get("model_info") or {}
    report.configure_methods([
        (lab, f"trained policy {m['name']} ({m.get('algorithm', '?')})")
        for lab, m in info.items()])


def rerender_repair(run_dir):
    """repair.csv plus repair_spread.csv. `picks` is rebuilt from the rows by the same
    rule repair.measure applies: only a method that spent exactly the minimum is in the
    population it is scored against."""
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    _register(meta)
    rows = _rows(os.path.join(run_dir, "repair.csv"))

    spread = {}
    for rec in _rows(os.path.join(run_dir, "repair_spread.csv")):
        key = (rec["instance"], rec["event"], rec["event_k"])
        s = spread.setdefault(key, {"instance": key[0], "event": key[1],
                                    "event_k": key[2], "size": rec["size"],
                                    "tier": rec["tier"], "errors": [], "picks": {}})
        s["errors"].append(rec["shape_err"])
    for r in rows:
        key = (r.get("instance"), r.get("event"), r.get("event_k"))
        if key in spread and r.get("exact_size") and \
                r.get("stop_rule") in (None, "", "final", "first_rigid"):
            spread[key]["picks"][r["method"]] = r.get("shape_err_post")
    spread = list(spread.values())

    header = _header(meta, {"network": f"{meta['args']['environment_name']}"})
    kappas = meta.get("trained_kappa") or {}

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
    return ["plots for repair_percentile, repair_count, repair_locality, "
            "repair_events, repair_choice, repair_kappa, each with its -plain twin"]


def rerender_control(run_dir):
    """control.csv, trajectories.csv and the goal networks in instances.npz."""
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    _register(meta)
    rows = _rows(os.path.join(run_dir, "control.csv"))
    traces = _rows(os.path.join(run_dir, "trajectories.csv"))
    goals = load_instances(run_dir)
    header = _header(meta)

    # the formation figure reads t["positions"], an (n, 3) array that trajectories.csv
    # has no column for; paths.csv carries it as one row per agent, so put it back.
    # The goal formation rides in paths.csv as a pseudo-method and is not a trajectory.
    frames = {}
    for rec in _rows(os.path.join(run_dir, "paths.csv")):
        if rec["method"] == "goal":
            continue
        frames.setdefault((rec["case"], rec["method"], rec["step"]), {})[
            int(rec["agent"])] = (rec["x"], rec["y"], rec["z"])
    import numpy as np
    for t in traces:
        frame = frames.get((t["case"], t["method"], t["step"]))
        t["positions"] = (None if not frame else
                          np.array([frame[i] for i in sorted(frame)], dtype=float))

    def draw_all():
        report.plot_control_trajectories(run_dir, traces, rows, header)
        report.plot_control_formations(run_dir, traces, rows, header,
                                       goals={i: g for i, g in enumerate(goals)})
        report.plot_control_summary(run_dir, rows, header)

    draw_all()
    with report.plain():
        draw_all()
    return ["plots for control_trajectories, control_formations, control_summary, "
            "each with its -plain twin"]


def rerender_ablation(run_dir):
    """ablation_raw.json, which is what plot_ablation consumes."""
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    _register(meta)
    raw = json.load(open(os.path.join(run_dir, "ablation_raw.json")))
    header = _header(meta)
    names = (meta.get("model_info") or {})

    def draw_all():
        for label, modes in raw.items():
            per_mode = {m: (d["rows"], d["reference"]) for m, d in modes.items()}
            report.plot_ablation(run_dir, per_mode, header,
                                 names.get(label, {}).get("name", label),
                                 filename=f"ablation-{label}")

    draw_all()
    with report.plain():
        draw_all()
    return [f"plots for ablation-{lab} ({', '.join(modes)}), each with its -plain twin"
            for lab, modes in raw.items()]


def rerender_budget(run_dir):
    """budget.csv, which carries one row per (episode, method, budget)."""
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    _register(meta)
    rows = _rows(os.path.join(run_dir, "budget.csv"))
    for r in rows:                       # the figure indexes budgets and steps as ints
        for k in ("episode", "budget", "m", "m_req", "best_step", "steps_taken"):
            if r.get(k) is not None:
                r[k] = int(r[k])
    header = _header(meta, meta.get("context"))

    def draw_all():
        report.plot_budget(run_dir, rows, header)

    draw_all()
    with report.plain():
        draw_all()
    return ["plots for budget, with its -plain twin"]


def rerender_generalisation(run_dir):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    _register(meta)
    gen_rows = _rows(os.path.join(run_dir, "generalisation.csv"))
    header = _header(meta)

    def draw_all():
        report.plot_generalisation(run_dir, gen_rows, header)

    draw_all()
    with report.plain():
        draw_all()
    return ["plots for generalisation, with its -plain twin"]


def rerender_topology(run_dir):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    _register(meta)
    rows = _rows(os.path.join(run_dir, "topology.csv"))
    nodes = _rows(os.path.join(run_dir, "topology_nodes.csv"))
    pairs = _rows(os.path.join(run_dir, "topology_pairs.csv"))
    header = _header(meta)

    def draw_all():
        report.plot_topology_structure(run_dir, rows, nodes, pairs, header)

    draw_all()
    with report.plain():
        draw_all()
    return ["plots for topology_structure, with its -plain twin"]


# which file identifies which kind of run
KINDS = [("results.csv", rerender), ("repair.csv", rerender_repair),
         ("control.csv", rerender_control), ("ablation_raw.json", rerender_ablation),
         ("generalisation.csv", rerender_generalisation),
         ("topology.csv", rerender_topology), ("budget.csv", rerender_budget)]


def rerender_any(run_dir):
    for marker, fn in KINDS:
        if os.path.exists(os.path.join(run_dir, marker)):
            return fn(run_dir)
    return ["nothing to rebuild: no recognised output in this directory"]


def main(argv):
    dirs = argv or sorted(glob.glob(os.path.join("runs_outputs", "*")))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "meta.json"))]
    if not dirs:
        print("no evaluation runs with a meta.json")
        return 1
    for d in dirs:
        try:
            made = rerender_any(d)
        except Exception as exc:
            print(f"{d}\n  skipped: {type(exc).__name__}: {exc}")
            continue
        print(d)
        for m in made:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
