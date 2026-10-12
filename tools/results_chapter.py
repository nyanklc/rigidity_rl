"""Reaggregate the selected raw runs and draw the Results chapter figures."""

import csv
import glob
import json
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rigidity import extended_bearing_rigidity_matrix


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_outputs"
FIGS = ROOT / "resources/thesis/tex/img/results"
DATA = RUNS / "results_chapter"
BOOTSTRAPS = 4000

MODELS = {
    "ref_k0_dqn_gine": "k0",
    "ref_k10_s0_dqn_gine": "s0",
    "ref_k10_s1_dqn_gine": "s1",
    "ref_k10_s2_dqn_gine": "s2",
    "estimation_randdom_k50_dqn_gine": "k50",
    "ref_k200_dqn_gine": "k200",
    "estimation_randdom_dqn_gine": "n20k2",
    "estimation_randdom_k50_ppo_gine": "ppo50",
    "ref_n16_k10_dqn_gine": "n16",
    "ref_n10_k10_150k_dqn_gine": "n10_150k",
    "ref_n20_k10_150k_dqn_gine": "n20_150k",
    "ref_select_dqn_gine": "select",
    "ref_stop_dqn_gine": "stop_k10",
    "estimation_stop_dqn_gine": "stop_k2",
    "estimation_stop_k50_dqn_gine": "stop_k50",
    "estimation_stop_k200_dqn_gine": "stop_k200",
}
LABELS = {
    "k0": r"DQN, $\kappa=0$",
    "s0": r"DQN, $\kappa=10$, seed 0",
    "s1": r"DQN, $\kappa=10$, seed 1",
    "s2": r"DQN, $\kappa=10$, seed 2",
    "k50": r"DQN, $\kappa=50$",
    "k10": r"DQN, $\kappa=10$",
    "k200": r"DQN, $\kappa=200$",
    "select": "Sequential node selection",
    "greedy": "Greedy",
    "greedy_repair": "Repair greedy",
    "spectral": "Spectral",
    "degree": "Degree",
    "constructive": "Constructive",
    "anneal": "Annealing",
    "n10_k10": "DQN trained at n=10, kappa=10",
    "n16": "DQN trained at n=16, kappa=10",
    "n20k2": "DQN trained at n=20, kappa=2",
    "stop_k2": "Stop policy, kappa=2",
    "stop_k10": "Stop policy, kappa=10",
    "stop_k50": "Stop policy, kappa=50",
    "stop_k200": "Stop policy, kappa=200",
    "dqn_gine": "DQN, GINE", "dqn_egnn": "DQN, EGNN",
    "ppo_gine": "PPO, GINE", "ppo_egnn": "PPO, EGNN",
}
COLORS = {
    "k0": "#d18b26", "s0": "#6baed6", "s1": "#1f77b4",
    "s2": "#084594", "k50": "#9467bd", "k200": "#2ca02c",
    "greedy": "#333333", "spectral": "#d95f02", "degree": "#7f7f7f",
    "constructive": "#8c6d31", "anneal": "#bc5090",
    "n10_k10": "#1f77b4", "n16": "#009e73", "n20k2": "#cc79a7",
    "n10_150k": "#1f77b4", "n20_150k": "#cc79a7",
    "stop_k2": "#d18b26", "stop_k10": "#1f77b4",
    "stop_k50": "#9467bd", "stop_k200": "#2ca02c",
    "dqn_gine": "#1f77b4", "dqn_egnn": "#17becf",
    "ppo_gine": "#9467bd", "ppo_egnn": "#e377c2",
}
MARKERS = {"k0": "s", "s0": "o", "s1": "o", "s2": "o", "k50": "D",
           "k200": "^", "greedy": "x", "spectral": "P", "degree": "v",
           "constructive": "<", "anneal": ">", "n10_k10": "o", "n16": "D", "n20k2": "^", "n10_150k": "o", "n20_150k": "^",
           "stop_k2": "o", "stop_k10": "s", "stop_k50": "D", "stop_k200": "^", "dqn_gine": "o", "dqn_egnn": "s",
           "ppo_gine": "D", "ppo_egnn": "^"}
SIZE_PREFIX = {
    8: "20260922-033824__addremove__6models__scale",
    10: "20260921-233657__addremove__15models__screen_n10",
    16: "20260922-015029__addremove__6models__scale",
    24: "20260922-032801__addremove__6models__scale",
    32: "20260922-103155__addremove__6models__scale",
    48: "20260922-065806__addremove__3models__scale",
    96: "20260922-183146__addremove__ref_k0_dqn_gine-ref_k10_s1_dqn_gine__scale",
}
HOMOGENEOUS = [
    ("b2_n10_R2", r"$\mathbb{R}^2$"),
    ("b2_n10_R3", r"$\mathbb{R}^3$"),
    ("b2_n10_R2xS1", r"$\mathbb{R}^2\times S^1$"),
    ("b2_n10_R3xS1", r"$\mathbb{R}^3\times S^1$"),
    ("b2_n10_SE3", r"$\mathrm{SE}(3)$"),
]
MIXED = [
    ("b2_n10_mixed_random_R2_SE3", r"$\mathbb{R}^2$/SE(3), random"),
    ("b2_n10_mixed_fixed_R2_SE3", r"$\mathbb{R}^2$/SE(3), 5/5"),
    ("b2_n10_mixed_random_R3_R2xS1", r"$\mathbb{R}^3$/($\mathbb{R}^2\times S^1$), random"),
    ("b2_n10_mixed_fixed_R3_R2xS1", r"$\mathbb{R}^3$/($\mathbb{R}^2\times S^1$), 5/5"),
]


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_run(path):
    path = Path(path)
    with open(path / "meta.json") as fh:
        meta = json.load(fh)
    return path, meta, read_csv(path / "results.csv")


def find_run(prefix):
    matches = [Path(p) for p in glob.glob(str(RUNS / (prefix + "*")))
               if (Path(p) / "meta.json").exists()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one run for {prefix}, found {len(matches)}")
    return matches[0]


def find_tagged_run(benchmark_name, tag, episodes, model_name=None, file="results.csv"):
    matches = []
    for path in RUNS.glob("20*"):
        meta_path = path / "meta.json"
        if not meta_path.exists() or not (path / file).exists():
            continue
        meta = json.loads(meta_path.read_text())
        args = meta.get("args", {})
        if (meta.get("benchmark") != benchmark_name or args.get("tag") != tag
                or args.get("episodes") != episodes):
            continue
        if model_name and model_name not in (meta.get("models") or {}).values():
            continue
        matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {tag} run for {benchmark_name}, found {len(matches)}"
        )
    return matches[0]


def canonical(method, meta):
    model_name = (meta.get("models") or {}).get(method)
    if model_name:
        return MODELS.get(model_name, model_name)
    return method


def truth(value):
    return str(value).lower() in ("true", "1", "yes")


def number(row, key):
    try:
        v = row.get(key)
        return None if v in (None, "", "nan", "inf", "-inf") else float(v)
    except (TypeError, ValueError):
        return None


def rigid_rows(rows):
    return [r for r in rows if truth(r.get("is_IBR"))]


def metric_values(rows, metric):
    if metric == "rigidity":
        return np.asarray([float(truth(r.get("is_IBR"))) for r in rows])
    if metric == "attainment":
        return np.asarray([float(truth(r.get("is_IBR")) and truth(r.get("at_bound")))
                           for r in rows if "at_bound" in r])
    if metric == "final":
        return np.asarray([float(truth(r.get("is_final"))) for r in rows if "is_final" in r])
    rigid = rigid_rows(rows)
    if metric == "bound":
        return np.asarray([float(truth(r.get("at_bound"))) for r in rigid])
    if metric == "ratio":
        return np.asarray([number(r, "m") / number(r, "m_req") for r in rigid
                           if number(r, "m") is not None and number(r, "m_req")])
    if metric == "shape":
        return np.asarray([number(r, "shape_err") for r in rigid
                           if number(r, "shape_err") and number(r, "shape_err") > 0])
    if metric in ("position_component", "attitude_component"):
        return np.asarray([number(r,metric) for r in rigid if number(r,metric) is not None])
    if metric == "edges":
        return np.asarray([number(r, "m") for r in rigid if number(r, "m") is not None])
    if metric in ("calls", "ms"):
        key = "total" if metric == "calls" else "ms"
        return np.asarray([number(r, key) for r in rows if number(r, key) is not None])
    raise ValueError(metric)


def summarize(rows, metric, seed=0):
    values = metric_values(rows, metric)
    if len(values) == 0:
        return np.nan, np.nan, np.nan, 0
    if metric == "attitude_component" and np.any(values == 0):
        if np.all(values == 0):return 0., 0., 0., len(values)
        raise ValueError("Mixed zero/nonzero attitude coefficients need explicit treatment")
    transform = np.log if metric in ("shape", "position_component", "attitude_component") else None
    vals = transform(values) if transform else values
    estimate = float(vals.mean())
    if metric in ("rigidity", "bound", "attainment", "final"):
        z = 1.959963984540054
        denominator = 1 + z * z / len(vals)
        center = (estimate + z * z / (2 * len(vals))) / denominator
        half = z * math.sqrt(estimate * (1 - estimate) / len(vals)
                             + z * z / (4 * len(vals) ** 2)) / denominator
        low, high = min(estimate, max(0.0, center - half)), max(estimate, min(1.0, center + half))
    elif len(vals) < 2:
        low = high = np.nan
    else:
        rng = np.random.default_rng(seed)
        ix = rng.integers(0, len(vals), size=(BOOTSTRAPS, len(vals)))
        boot = vals[ix].mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        low, high = float(low), float(high)
    if transform:
        estimate, low, high = map(float, np.exp([estimate, low, high]))
    if metric in ("rigidity", "bound", "attainment", "final"):
        estimate, low, high = (100 * estimate, 100 * low, 100 * high)
    return estimate, low, high, len(values)


def rows_by_method(path, meta, rows):
    out = {}
    for r in rows:
        out.setdefault(canonical(r["method"], meta), []).append(r)
    return out


def cost_by_method(path, meta):
    cost_path = path / "cost.csv"
    if not cost_path.exists():
        return {}
    out = {}
    for r in read_csv(cost_path):
        out.setdefault(canonical(r["method"], meta), []).append(r)
    return out


def add_source(inventory, path, family, include=True, note=""):
    meta = json.loads((path / "meta.json").read_text())
    args = meta.get("args", {})
    result_path = path / "results.csv"
    result_rows = read_csv(result_path) if result_path.exists() else []
    methods = sorted({canonical(r["method"], meta) for r in result_rows})
    data_files = sorted(file.name for file in path.iterdir()
                        if file.suffix in (".csv", ".npz") or file.name == "ablation_raw.json")
    row_source = next((name for name in ("results.csv", "budget.csv", "repair.csv",
                                         "control.csv", "noise.csv", "decisions.csv",
                                         "training.csv", "generalisation.csv", "scale.csv")
                       if (path / name).exists()), None)
    data_rows = len(read_csv(path / row_source)) if row_source else 0
    if not row_source and (path / "ablation_raw.json").exists():
        raw = json.loads((path / "ablation_raw.json").read_text())
        data_rows = sum(len(item.get("rows", []))
                        for model in raw.values()
                        for perturbation, item in model.items()
                        if isinstance(item, dict))
    inventory.append({
        "directory": str(path.relative_to(ROOT)),
        "family": family,
        "benchmark": meta.get("benchmark") or "",
        "benchmark_digest": meta.get("benchmark_digest") or "",
        "n": meta.get("n") or (meta.get("environment_config") or {}).get("n") or "",
        "episodes_requested": args.get("episodes", args.get("cases", "")),
        "rows": data_rows,
        "methods": ",".join(methods),
        "models": ",".join((meta.get("models") or {}).values()),
        "files": ",".join(data_files),
        "include": str(bool(include)),
        "note": note,
    })

def inventory_sources(selected):
    inventory = []
    raw_files = (
        "results.csv", "cost.csv", "control.csv", "repair.csv", "ablation.csv",
        "noise.csv", "decisions.csv", "topology.csv", "training.csv",
        "trajectories.csv", "paths.csv", "instances.npz",
        "scale.csv", "generalisation.csv", "generalization.csv",
        "ablation_raw.json", "repair_spread.csv", "repair_summary.csv",
        "budget.csv", "budget_steps.csv", "budget_summary.csv",
    )
    for path in sorted(RUNS.glob("20*")):
        meta_path = path / "meta.json"
        if not meta_path.exists():
            continue
        present = [name for name in raw_files if (path / name).exists()]
        if not present:
            continue
        meta = json.loads(meta_path.read_text())
        tag = (meta.get("args") or {}).get("tag") or ""
        family = next((name.removesuffix(".csv") for name in present
                       if name in ("control.csv", "repair.csv", "ablation.csv",
                                   "noise.csv", "decisions.csv", "topology.csv",
                                   "training.csv", "scale.csv", "generalisation.csv",
                                   "generalization.csv")), tag or "comparison")
        if path in selected:
            included = True
            note = "selected raw source"
        else:
            excluded = any(word in (tag + path.name).lower()
                           for word in ("pilot", "smoke", "interrupt", "probe"))
            included = False
            if excluded:
                note = "pilot, smoke, interrupted, or probe run"
            elif "results.csv" not in present:
                note = "summary-only output; instance-level source not selected"
            else:
                note = "raw run not selected for this analysis"
        add_source(inventory, path, family, included, note)
    inventory.sort(key=lambda row: (row["family"], row["directory"]))
    write_csv(DATA / "run_inventory.csv", inventory)

def style(ax):
    ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#333333")


def save_pair(name, draw):
    FIGS.mkdir(parents=True, exist_ok=True)
    for plain in (False, True):
        fig = draw(plain)
        suffix = "-plain" if plain else ""
        for ext in ("pdf", "png"):
            fig.savefig(FIGS / f"{name}{suffix}.{ext}", dpi=220,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)


def metric_panel(ax, data, method_order, metric, xvalues, labels, annotated=False):
    ylab = {"rigidity": "rigid networks, %", "bound": "at lower bound among rigid, %",
            "ratio": "edge ratio m/m_req among rigid", "shape": "shape error",
            "calls": "counted primitive calls per network"}[metric]
    ax.set_title({"rigidity": "Rigidity", "bound": "Lower-bound attainment",
                  "ratio": "Edge ratio", "shape": "Shape error",
                  "calls": "Counted calls"}[metric])
    ax.set_ylabel(ylab)
    ax.set_xticks(xvalues)
    ax.set_xticklabels(labels, fontsize=7)
    for method in method_order:
        ys, lows, highs = [], [], []
        for k, x in enumerate(xvalues):
            rows = data.get((k, method), [])
            est, lo, hi, _ = summarize(rows, metric, seed=100 + 31 * k + len(method))
            ys.append(est)
            lows.append(est - lo)
            highs.append(hi - est)
        valid = np.isfinite(ys)
        if not valid.any():
            continue
        xs = np.asarray(xvalues)[valid]
        vals = np.asarray(ys)[valid]
        err = np.asarray([np.asarray(lows)[valid], np.asarray(highs)[valid]])
        ax.errorbar(xs, vals, yerr=err, color=COLORS.get(method, "#555555"),
                    marker=MARKERS.get(method, "o"), linestyle="-", linewidth=1.0,
                    markersize=3.7, capsize=2, label=LABELS.get(method, method),
                    alpha=0.95)
    if metric in ("rigidity", "bound"):
        ax.set_ylim(-4, 104)
    if metric == "shape":
        ax.set_yscale("log")
    if metric == "calls":
        ax.set_yscale("log")
    style(ax)


def load_size_data():
    data = {}
    costs_data = {}
    sources = {}
    for n, prefix in SIZE_PREFIX.items():
        path, meta, rows = load_run(find_run(prefix))
        sources[n] = path
        methods = rows_by_method(path, meta, rows)
        costs = cost_by_method(path, meta)
        for method, group in methods.items():
            if method in {"k0", "s1", "k50", "k200", "greedy", "spectral", "degree"}:
                data[(n, method)] = group
        for method, group in costs.items():
            if method in {"k0", "s1", "k50", "k200", "greedy", "spectral", "degree"}:
                costs_data[(n, method)] = [dict(total=r.get("total")) for r in group]
    k50_path = find_tagged_run("b2_n48_randdom", "thesis_k50_size", 25,
                               "estimation_randdom_k50_dqn_gine")
    p, m, r = load_run(k50_path)
    data[(48, "k50")] = rows_by_method(p, m, r)["k50"]
    costs = cost_by_method(p, m).get("k50", [])
    costs_data[(48, "k50")] = [dict(total=c.get("total")) for c in costs]

    greedy_path = find_run("20260922-085849__addremove__scale_greedy")
    p, m, r = load_run(greedy_path)
    data[(48, "greedy")] = rows_by_method(p, m, r)["greedy"]
    costs = cost_by_method(p, m).get("greedy", [])
    costs_data[(48, "greedy")] = [dict(total=c.get("total")) for c in costs]
    return data, costs_data, sources


def plot_size(data, costs_data):
    methods = ["k0", "s1", "k50", "k200", "greedy", "spectral", "degree"]
    ns = [8, 10, 16, 24, 32, 48, 96]
    labels = [str(n) for n in ns]

    def draw(plain):
        fig, axs = plt.subplots(3, 2, figsize=(12.2, 8.4), sharex=True)
        for ax, metric in zip(axs.flat, ("rigidity", "bound", "ratio", "shape")):
            local = {(i, method): data.get((n, method), [])
                     for i, n in enumerate(ns) for method in methods}
            metric_panel(ax, local, methods, metric, list(range(len(ns))), labels,
                         annotated=not plain)
            if metric == "rigidity":
                ax.legend(fontsize=6.5, ncol=2, loc="lower left", framealpha=0.95)
        ax = axs.flat[4]
        local = {}
        for i, n in enumerate(ns):
            for method in methods:
                local[(i, method)] = costs_data.get((n, method), [])
        metric_panel(ax, local, methods, "calls", list(range(len(ns))), labels,
                     annotated=not plain)
        axs.flat[5].axis("off")
        if not plain:
            axs.flat[5].text(0.05, 0.85,
                "Policy calls cover the full rollout horizon.\n"
                "Classical methods run to convergence.\n"
                "Sample sizes for n=8, 10, 16, 24, 32, 48, 96.\nN=200, 200, 100, 50, 25, 25, 8.\nGreedy at n=48 uses 8 shared cases.\nn=96 is an eight-case stress sample.\nError bars show 95% bootstrap intervals.",
                fontsize=10, va="top")
        fig.tight_layout(rect=(0, 0.01, 1, 1))
        return fig

    save_pair("network_size", draw)


def find_domain_runs():
    runs = {}
    for bench, _ in HOMOGENEOUS:
        runs[bench] = find_tagged_run(bench, "domain", 100)
    for bench, _ in MIXED:
        runs[bench] = find_tagged_run(bench, "thesis_mix", 100)
    return runs


def load_domain_data(runs):
    data = {}
    sources = {}
    for bench, path in runs.items():
        p, meta, rows = load_run(path)
        sources[bench] = p
        for method, group in rows_by_method(p, meta, rows).items():
            data[(bench, method)] = group
    return data, sources


def plot_domain_figure(name, data, bench_order, labels, methods, note):
    metrics = ("rigidity", "bound", "ratio", "shape")

    def draw(plain):
        fig, axs = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True)
        local = {}
        for i, bench in enumerate(bench_order):
            for method in methods:
                local[(i, method)] = data.get((bench, method), [])
        for ax, metric in zip(axs.flat, metrics):
            metric_panel(ax, local, methods, metric, list(range(len(bench_order))),
                         labels, annotated=not plain)
        axs.flat[0].legend(fontsize=6.8, ncol=2, loc="lower left", framealpha=0.95)
        if not plain:
            fig.text(0.5, 0.01, note + " Error bars show 95% bootstrap intervals over instances.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return fig

    save_pair(name, draw)


def paired_values(rows_a, rows_b, metric):
    a = {int(r["episode"]): r for r in rows_a}
    b = {int(r["episode"]): r for r in rows_b}
    ids = sorted(set(a) & set(b))
    values = []
    for i in ids:
        ra, rb = a[i], b[i]
        if metric == "rigidity":
            values.append(float(truth(ra.get("is_IBR"))) - float(truth(rb.get("is_IBR"))))
        elif metric == "attainment":
            values.append(float(truth(ra.get("is_IBR")) and truth(ra.get("at_bound")))
                          - float(truth(rb.get("is_IBR")) and truth(rb.get("at_bound"))))
        elif metric == "bound":
            if truth(ra.get("is_IBR")) and truth(rb.get("is_IBR")):
                values.append(float(truth(ra.get("at_bound"))) - float(truth(rb.get("at_bound"))))
        elif metric == "ratio":
            if truth(ra.get("is_IBR")) and truth(rb.get("is_IBR")):
                values.append(number(ra, "m") / number(ra, "m_req") -
                              number(rb, "m") / number(rb, "m_req"))
        elif metric == "edges":
            if truth(ra.get("is_IBR")) and truth(rb.get("is_IBR")):
                values.append(number(ra, "m") - number(rb, "m"))
        elif metric == "shape_dex":
            if (truth(ra.get("is_IBR")) and truth(rb.get("is_IBR"))
                    and number(ra, "shape_err") and number(rb, "shape_err")):
                values.append(math.log10(number(ra, "shape_err") /
                                         number(rb, "shape_err")))
    return np.asarray(values), len(ids)


def paired_ci(values, seed=0):
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    est = float(values.mean())
    if len(values) == 1:
        return est, np.nan, np.nan
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    boot = values[ix].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return est, float(lo), float(hi)


def plot_reference(screen_rows, meta):
    by = rows_by_method(None, meta, screen_rows)
    seeds = ["s0", "s1", "s2"]
    metrics = ["rigidity", "bound", "ratio", "shape_dex"]
    titles = ["Rigidity difference", "Lower-bound attainment difference",
              "Edge-ratio difference", "Shape-error difference"]
    units = ["percentage points", "percentage points", "m/m_req", "log10 ratio"]
    summaries = []
    for metric in metrics:
        for j, seed in enumerate(seeds):
            vals, n = paired_values(by[seed], by["greedy"], metric)
            est, lo, hi = paired_ci(vals, seed=200 + j + 31 * metrics.index(metric))
            scale = 100 if metric in ("rigidity", "bound") else 1
            summaries.append({"seed": seed, "metric": metric, "n": len(vals),
                              "paired_ids": n, "estimate": est * scale,
                              "ci_low": lo * scale, "ci_high": hi * scale})
    write_csv(DATA / "reference_seed_effects.csv", summaries)

    def draw(plain):
        fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.4))
        for ax, metric, title, unit in zip(axs.flat, metrics, titles, units):
            recs = [r for r in summaries if r["metric"] == metric]
            y = np.arange(3)
            m = np.asarray([r["estimate"] for r in recs])
            lo = m - np.asarray([r["ci_low"] for r in recs])
            hi = np.asarray([r["ci_high"] for r in recs]) - m
            ax.errorbar(m, y, xerr=[lo, hi], fmt="o", color="#1f77b4",
                        capsize=3, markersize=5)
            ax.axvline(0, color="#333333", linestyle=":", linewidth=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels(["seed 0 (N=200)", "seed 1 (N=200)", "seed 2 (N=200)"])
            ax.set_title(title)
            ax.set_xlabel(f"policy minus greedy ({unit})")

            style(ax)
        if not plain:
            fig.text(0.5, 0.01, "N=200 per seed. Intervals use paired bootstrap resampling of shared cases.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return fig
    save_pair("reference_seed_effects", draw)
    return by


def load_front():
    prefixes = {
        0: "20260922-042026__addremove__ref_k0_dqn_gine__front_k0",
        10: "20260922-042343__addremove__ref_k10_s1_dqn_gine__front_k10",
        50: "20260922-042751__addremove-n10_randdom__estimation_randdom_k50_dqn_g__front_k50",
        200: "20260922-043428__addremove__ref_k200_dqn_gine__front_k200",
    }
    return {k: load_run(find_run(v)) for k, v in prefixes.items()}


def plot_kappa(front):
    policies = {0: "k0", 10: "s1", 50: "k50", 200: "k200"}
    effect_rows = []
    frontier = []
    rates = []
    for k, (path, meta, rows) in front.items():
        by = rows_by_method(path, meta, rows)
        pol = by[policies[k]]
        greedy = by["greedy"]
        for metric in ("edges", "ratio", "shape_dex", "rigidity", "bound"):
            vals, nids = paired_values(pol, greedy, metric)
            est, lo, hi = paired_ci(vals, seed=10 + k + len(metric))
            scale = 100 if metric in ("rigidity", "bound") else 1
            effect_rows.append({"kappa": k, "metric": metric, "n": len(vals),
                                "paired_ids": nids, "estimate": est * scale,
                                "ci_low": lo * scale, "ci_high": hi * scale})
        for method, group in (("policy", pol), ("greedy", greedy)):
            e = metric_values(group, "edges")
            s = metric_values(group, "shape")
            ex = summarize(group, "edges", seed=k + 1)
            sh = summarize(group, "shape", seed=k + 101)
            frontier.append({"kappa": k, "method": method, "n_edges": len(e),
                             "edges": ex[0], "edges_low": ex[1], "edges_high": ex[2],
                             "n_shape": len(s), "shape": sh[0],
                             "shape_low": sh[1], "shape_high": sh[2]})
            rates.append({"kappa": k, "method": method,
                          "rigidity": summarize(group, "rigidity", k)[0],
                          "bound": summarize(group, "bound", k + 10)[0]})
    write_csv(DATA / "kappa_paired_effects.csv", effect_rows)
    write_csv(DATA / "kappa_frontier.csv", frontier)
    write_csv(DATA / "kappa_rates.csv", rates)

    def draw(plain):
        fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.0))
        ax = axs[0, 0]
        for k in (0, 10, 50, 200):
            pts = {r["method"]: r for r in frontier if r["kappa"] == k}
            c = {0: "#d18b26", 10: "#1f77b4", 50: "#9467bd", 200: "#2ca02c"}[k]
            for method, marker, hollow in (("policy", "o", False), ("greedy", "s", True)):
                r = pts[method]
                ax.errorbar(r["edges"], r["shape"],
                    xerr=[[r["edges"]-r["edges_low"]], [r["edges_high"]-r["edges"]]],
                    yerr=[[r["shape"]-r["shape_low"]], [r["shape_high"]-r["shape"]]],
                    fmt=marker, color=c, markerfacecolor="white" if hollow else c,
                    capsize=2.5, markersize=5, linestyle="none")
                if method == "policy":
                    ax.annotate(rf"$\kappa={k}$", (r["edges"], r["shape"]),
                                xytext=(4, 4), textcoords="offset points", fontsize=7)
            ax.plot([pts["policy"]["edges"], pts["greedy"]["edges"]],
                    [pts["policy"]["shape"], pts["greedy"]["shape"]],
                    color=c, alpha=0.5, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("edges among rigid outputs (N=100 per $\\kappa$)")
        ax.set_ylabel("shape error")
        ax.set_title("Edges and shape error")
        ax.plot([], [], "o", color="#444444", label="policy")
        ax.plot([], [], "s", color="#444444", markerfacecolor="white", label="greedy")
        ax.legend(fontsize=7)
        style(ax)

        for ax, metric, title, scale in (
            (axs[0, 1], "edges", "Paired edge-count difference", 1),
            (axs[1, 0], "shape_dex", "Paired shape-error difference", 1),
        ):
            recs = [r for r in effect_rows if r["metric"] == metric]
            y = np.arange(len(recs))
            x = np.asarray([r["estimate"] for r in recs])
            ax.errorbar(x, y, xerr=[x-np.asarray([r["ci_low"] for r in recs]),
                                    np.asarray([r["ci_high"] for r in recs])-x],
                        fmt="o", color="#1f77b4", capsize=3, markersize=4)
            ax.axvline(0, color="#333333", linestyle=":", linewidth=0.9)
            ax.set_yticks(y); ax.set_yticklabels([rf"$\kappa={r['kappa']}$" for r in recs])
            ax.set_title(title)
            ax.set_xlabel("policy minus greedy, paired 95% CI")
            if metric == "edges":
                ax.set_xlabel("policy minus greedy, edges (N=100)")
            if metric == "shape_dex":
                ax.set_xlabel(r"mean $\log_{10}(\varepsilon_{policy}/\varepsilon_{greedy})$ (N=100)")
            style(ax)

        ax = axs[1, 1]
        for method, c, ls in (("policy", "#1f77b4", "-"), ("greedy", "#333333", "--")):
            rr = [r for r in rates if r["method"] == method]
            x = np.arange(4)
            ax.plot(x, [r["rigidity"] for r in rr], marker="o", color=c, ls=ls,
                    label=f"{method}, rigid")
            ax.plot(x, [r["bound"] for r in rr], marker="s", color=c, ls=":",
                    label=f"{method}, at lower bound")
        ax.set_ylim(-4, 104); ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["0\nN=100", "10\nN=100", "50\nN=100", "200\nN=100"])
        ax.set_xlabel(r"evaluation $\kappa$")
        ax.set_ylabel("networks, %")
        ax.set_title("Rigidity and lower-bound attainment")
        ax.legend(fontsize=6.5, ncol=2)
        style(ax)
        if not plain:
            fig.text(0.5, 0.01, "Each policy is paired with greedy under the same evaluation objective. N=100 per group.",
                     ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        return fig
    save_pair("kappa_tradeoff", draw)


def plot_compute(by_reference, costs):
    methods = ["degree", "spectral", "constructive", "greedy", "anneal", "s1"]
    recs = []
    for method in methods:
        rows = by_reference.get(method, [])
        cs = costs.get(method, [])
        calls = np.asarray([number(r, "total") for r in cs if number(r, "total") is not None])
        ms = np.asarray([number(r, "ms") for r in cs if number(r, "ms") is not None])
        recs.append({"method": method, "n": len(rows),
                     "calls": float(calls.mean()) if len(calls) else np.nan,
                     "wall_ms": float(np.exp(np.log(ms).mean())) if len(ms) else np.nan,
                     "bound_pct": summarize(rows, "bound", len(method))[0],
                     "shape_error": summarize(rows, "shape", len(method) + 5)[0]})
    write_csv(DATA / "compute_quality.csv", recs)

    def draw(plain):
        fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.2))
        for ax, ykey, ylabel, title in (
            (axs[0], "bound_pct", "at lower bound among rigid, %", "Lower-bound attainment"),
            (axs[1], "shape_error", "shape error", "Shape error"),
        ):
            for r in recs:
                ax.scatter(r["calls"], r[ykey], color=COLORS[r["method"]],
                           marker=MARKERS[r["method"]], s=48, zorder=3)
                ax.annotate(LABELS.get(r["method"], r["method"]),
                            (r["calls"], r[ykey]), xytext=(5, 2),
                            textcoords="offset points", fontsize=7)
            ax.set_xscale("log")
            if ykey == "shape_error": ax.set_yscale("log")
            ax.set_xlabel("counted primitive calls per network (N=200)")
            ax.set_ylabel(ylabel); ax.set_title(title)
            style(ax)
        if not plain:
            fig.text(0.5, 0.01, "N=200. Policy calls include the full 118-step horizon. Classical searches run to convergence.",
                     ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        return fig
    save_pair("compute_quality", draw)




def graph_structure(screen_path, meta, rows):
    nets, _ = benchmark.read_networks(screen_path / "instances.npz")
    by = rows_by_method(screen_path, meta, rows)
    methods = ["k0", "s0", "s1", "s2", "k50", "k200", "degree", "greedy", "spectral"]
    values = []
    for method in methods:
        for row in by.get(method, []):
            episode = int(row["episode"])
            net = nets[episode]
            A = np.zeros((net.n, net.n), dtype=bool)
            edge_text = row.get("edges", "")
            for edge in edge_text.split():
                i, j = edge.split(">")
                A[int(i), int(j)] = True
            net.edges = A
            edge_list = list(zip(*np.nonzero(A)))
            m = len(edge_list)
            if not m:
                continue
            reverse = sum(bool(A[j, i]) for i, j in edge_list)
            frameless = {i for i, a in enumerate(net.agents)
                         if a.domain in ("R^2", "R^3")}
            free_edges = [(i, j) for i, j in edge_list
                          if i in frameless and j in frameless]
            free_reverse = sum(bool(A[j, i]) for i, j in free_edges)
            B = extended_bearing_rigidity_matrix(net)
            block_ranks = [int(np.linalg.matrix_rank(B[3*k:3*k+3]))
                           for k in range(m)]
            positions = np.asarray([a.pose.position for a in net.agents])
            radius = np.sqrt(np.mean(np.sum((positions - positions.mean(axis=0))**2, axis=1)))
            lengths = [np.linalg.norm(positions[j] - positions[i]) / radius
                       for i, j in edge_list]
            values.append({
                "method": method, "episode": episode, "m": m,
                "reciprocity": reverse / m,
                "frameless_reciprocity": free_reverse / len(free_edges) if free_edges else np.nan,
                "reciprocity_null": (m - 1) / (net.n * (net.n - 1) - 1),
                "rank1_share": block_ranks.count(1) / m,
                "rank2_share": block_ranks.count(2) / m,
                "degree_floor_share": float(np.mean([
                    int(A[i].sum()) >= {"R^2": 0, "R^3": 0, "R^2xS^1": 1,
                           "R^3xS^1": 1, "SE(3)": 2}.get(agent.domain, 0)
                    for i, agent in enumerate(net.agents)])),
                "mean_edge_length_radius": float(np.mean(lengths)),
            })
    write_csv(DATA / "resultant_graph_metrics.csv", values)
    return values


def structural_ci(rows, field, seed):
    vals = np.asarray([number(r, field) for r in rows if number(r, field) is not None])
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    est = float(vals.mean())
    if len(vals) == 1:
        return est, est, est
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(vals), size=(BOOTSTRAPS, len(vals)))
    boot = vals[ix].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return est, float(lo), float(hi)


def plot_structure(values):
    methods = ["degree", "greedy", "spectral", "k0", "s0", "s1", "s2", "k50", "k200"]
    write_csv(DATA / "resultant_graph_metrics.csv", values)

    def draw(plain):
        fig, axs = plt.subplots(2, 2, figsize=(11.8, 7.6))
        y = np.arange(len(methods))
        ax = axs[0, 0]
        for off, field, label, marker, color in (
            (-0.18, "reciprocity", "all edges", "o", "#1f77b4"),
            (0.0, "frameless_reciprocity", "frameless endpoints", "s", "#d95f02"),
            (0.18, "reciprocity_null", "uniform random graph", "D", "#777777"),
        ):
            ests, lows, highs = [], [], []
            for method in methods:
                group = [r for r in values if r["method"] == method]
                est, lo, hi = structural_ci(group, field, seed=len(field)+len(method))
                ests.append(100*est); lows.append(100*(est-lo)); highs.append(100*(hi-est))
            ax.errorbar(ests, y+off, xerr=[lows, highs], fmt=marker, color=color,
                        capsize=2, markersize=4, label=label)
        ax.set_yticks(y); ax.set_yticklabels([LABELS.get(m,m) for m in methods], fontsize=7)
        ax.set_xlabel("directed edges with a reverse, % (N=200 networks)")
        ax.set_title("Reciprocal edges")
        ax.legend(fontsize=6.5)
        style(ax)

        ax = axs[0, 1]
        for off, field, label, marker, color in (
            (-0.12, "rank1_share", "rank 1", "o", "#d95f02"),
            (0.12, "rank2_share", "rank 2", "s", "#1f77b4"),
        ):
            ests, lows, highs = [], [], []
            for method in methods:
                group = [r for r in values if r["method"] == method]
                est, lo, hi = structural_ci(group, field, seed=len(field)+len(method))
                ests.append(100*est); lows.append(100*(est-lo)); highs.append(100*(hi-est))
            ax.errorbar(ests, y+off, xerr=[lows, highs], fmt=marker, color=color,
                        capsize=2, markersize=4, label=label)
        ax.set_yticks(y); ax.set_yticklabels([LABELS.get(m,m) for m in methods], fontsize=7)
        ax.set_xlabel("edge blocks, % (N=200 networks)")
        ax.set_title("Edge-block rank")
        ax.legend(fontsize=7)
        style(ax)

        degree = []
        length = []
        for method in methods:
            group = [r for r in values if r["method"] == method]
            degree.append(structural_ci(group, "degree_floor_share", seed=31+len(method)))
            length.append(structural_ci(group, "mean_edge_length_radius", seed=53+len(method)))
        for ax, records, title, xlabel, scale in (
            (axs[1, 0], degree, "Out-degree floor", "agents meeting domain floor, % (N=200 networks)", 100),
            (axs[1, 1], length, "Edge length", "mean edge length in formation radii (N=200 networks)", 1),
        ):
            means = np.asarray([x[0] * scale for x in records])
            lows = np.asarray([(x[0]-x[1]) * scale for x in records])
            highs = np.asarray([(x[2]-x[0]) * scale for x in records])
            ax.errorbar(means, y, xerr=[lows, highs], fmt="o", color="#1f77b4",
                        capsize=2, markersize=4)
            ax.set_yticks(y); ax.set_yticklabels([LABELS.get(m,m) for m in methods], fontsize=7)
            ax.set_xlabel(xlabel); ax.set_title(title); style(ax)
        if not plain:
            fig.text(0.5, 0.01, "Points are instance means. Bars show 95% bootstrap intervals.",
                     ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        return fig
    save_pair("resultant_graphs", draw)


def load_control():
    matches = []
    required = {"ref_k0_dqn_gine", "ref_k10_s0_dqn_gine",
                "estimation_randdom_k50_dqn_gine", "ref_k200_dqn_gine"}
    for path in RUNS.glob("20*"):
        meta_path = path / "meta.json"
        if not meta_path.exists() or not (path / "control.csv").exists():
            continue
        meta = json.loads(meta_path.read_text())
        args = meta.get("args", {})
        models = set((meta.get("models") or {}).values())
        if args.get("cases") == 5 and required <= models:
            matches.append((path, meta, read_csv(path / "control.csv")))
    if not matches:
        raise RuntimeError("five-case control result not found")
    return sorted(matches, key=lambda z: z[0].name)[-1]


def plot_action_space_effects(effects):
    selected = [r for r in effects if r["method"] == "select"]
    metrics = [("rigidity", "Rigidity", "difference, percentage points"),
               ("bound", "Lower-bound attainment", "difference, percentage points"),
               ("edges", "Edge count", "difference, edges"),
               ("shape_dex", "Shape error", "difference, log10 ratio")]

    def draw(plain):
        fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.4))
        for ax, (metric, title, ylabel) in zip(axs.flat, metrics):
            row = next(r for r in selected if r["metric"] == metric)
            estimate, low, high = (float(row[k]) for k in ("estimate", "ci_low", "ci_high"))
            ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--")
            ax.errorbar([0], [estimate], yerr=[[estimate-low], [high-estimate]],
                        color=COLORS.get("select", COLORS["s1"]), marker="o", capsize=3, linewidth=1.2)
            ax.set_xlim(-0.7, 0.7)
            ax.set_xticks([0])
            ax.set_xticklabels(["pointer minus greedy"], fontsize=8)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            style(ax)
        if not plain:
            fig.text(0.5, 0.01,
                     "Paired bootstrap estimates on 20 shared instances. Shape error is shown as a log10 ratio.",
                     ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return fig
    save_pair("action_space", draw)
def plot_control(control):
    path, meta, rows = control
    selected = [{**r, "method": canonical(r["method"], meta)} for r in rows
                if canonical(r["method"], meta) in {"complete", "greedy", "k0", "s0", "k50", "k200"}]
    write_csv(DATA / "formation_control_cases.csv", selected)
    methods = ["complete", "greedy", "k0", "s0", "k50", "k200"]
    cases = sorted({r["case"] for r in selected})

    def draw(plain):
        fig, ax = plt.subplots(figsize=(10.0, 4.8))
        offsets = np.linspace(-0.24, 0.24, len(methods))
        for offset, method in zip(offsets, methods):
            group = [r for r in selected if r["method"] == method]
            for r in group:
                case_index = cases.index(r["case"])
                hit = number(r, "steps_to_tol")
                y = hit if hit is not None else number(r, "steps")
                ax.scatter(case_index + offset, y, s=38,
                           color=COLORS.get(method, "#555555"),
                           marker="o" if truth(r.get("reached")) else "x",
                           label=LABELS.get(method, method) if r is group[0] else None,
                           zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(cases)))
        ax.set_xticklabels([str(case) for case in cases])
        ax.set_xlabel("goal/start case (N=5)")
        ax.set_ylabel("steps to tolerance (failure plotted at step limit)")
        ax.set_title("Steps to tolerance")
        ax.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
        style(ax)
        if not plain:
            fig.text(0.5, 0.01, "Five shared goal/start cases. Cross marks runs that did not reach tolerance.",
                     ha="center", fontsize=8)
        fig.subplots_adjust(right=0.78, bottom=0.17)
        return fig
    save_pair("formation_control_cases", draw)


def summary_records(path, meta, rows, family, methods_filter=None):
    env = meta.get("environment_config") or {}
    benchmark_name = meta.get("benchmark") or ""
    n = meta.get("n") or env.get("n") or ""
    eval_kappa = env.get("stiffness_kappa", "")
    output = []
    for method, group in rows_by_method(path, meta, rows).items():
        if methods_filter and method not in methods_filter:
            continue
        output.append({
            "family": family, "benchmark": benchmark_name, "n": n,
            "evaluation_kappa": eval_kappa, "method": method,
            "instances": len(group),
            "rigidity_pct": summarize(group, "rigidity")[0],
            "bound_pct_rigid": summarize(group, "bound")[0],
            "edge_ratio_rigid": summarize(group, "ratio")[0],
            "edge_count_rigid": summarize(group, "edges")[0],
            "shape_error_rigid": summarize(group, "shape")[0],
        })
    return output


def paired_effect_records(by, family, benchmark_name, references,
                          metrics=("rigidity", "bound", "ratio", "edges", "shape_dex")):
    output = []
    for reference in references:
        if reference not in by:
            continue
        for method, rows in by.items():
            if method == reference:
                continue
            for metric in metrics:
                values, paired_ids = paired_values(rows, by[reference], metric)
                estimate, low, high = paired_ci(
                    values, seed=100 + len(family) + len(method) + len(metric))
                scale = 100 if metric in ("rigidity", "bound") else 1
                output.append({
                    "family": family, "benchmark": benchmark_name,
                    "method": method, "reference": reference,
                    "metric": metric, "n": len(values),
                    "paired_ids": paired_ids,
                    "estimate": estimate * scale,
                    "ci_low": low * scale, "ci_high": high * scale,
                })
    return output


def row_metric(row, metric):
    if metric == "rigidity":
        return float(truth(row.get("is_IBR")))
    if not truth(row.get("is_IBR")):
        return None
    if metric == "bound":
        return float(truth(row.get("at_bound")))
    if metric == "ratio":
        m, m_req = number(row, "m"), number(row, "m_req")
        return m / m_req if m is not None and m_req else None
    if metric == "shape":
        shape = number(row, "shape_err")
        return math.log(shape) if shape is not None and shape > 0 else None
    raise ValueError(metric)


def paired_eval_summary(rows10, rows50, methods, seed=0):
    result = []
    for method in methods:
        a = {int(r["episode"]): r for r in rows10.get(method, [])}
        b = {int(r["episode"]): r for r in rows50.get(method, [])}
        ids = sorted(set(a) & set(b))
        for metric in ("rigidity", "bound", "ratio", "shape"):
            pairs = []
            for i in ids:
                value10, value50 = row_metric(a[i], metric), row_metric(b[i], metric)
                if value10 is None or value50 is None:
                    continue
                if not (math.isfinite(value10) and math.isfinite(value50)):
                    continue
                pairs.append((value10, value50))
            pair_array = np.asarray(pairs, dtype=float).reshape((-1, 2))
            if len(pair_array):
                means = pair_array.mean(axis=0)
                rng = np.random.default_rng(seed + len(method) + 17 * len(metric))
                draws = rng.integers(0, len(pair_array), size=(BOOTSTRAPS, len(pair_array)))
                boot_means = pair_array[draws].mean(axis=1)
                low, high = np.percentile(boot_means, [2.5, 97.5], axis=0)
                diffs = pair_array[:, 1] - pair_array[:, 0]
                diff = float(diffs.mean())
                diff_low, diff_high = np.percentile(
                    np.mean(diffs[draws], axis=1), [2.5, 97.5])
            else:
                means = np.asarray([np.nan, np.nan])
                low = high = np.asarray([np.nan, np.nan])
                diff = diff_low = diff_high = np.nan
            if metric == "shape":
                estimates = np.exp(means)
                lows, highs = np.exp(low), np.exp(high)
                diff_scale = 1 / math.log(10)
                scale = 1
            else:
                scale = 100 if metric in ("rigidity", "bound") else 1
                estimates, lows, highs = means * scale, low * scale, high * scale
                diff_scale = scale
            result.append({
                "method": method, "metric": metric,
                "shared_instances": len(ids), "paired_instances": len(pair_array),
                "k10": float(estimates[0]), "k10_low": float(lows[0]),
                "k10_high": float(highs[0]), "k50": float(estimates[1]),
                "k50_low": float(lows[1]), "k50_high": float(highs[1]),
                "k50_minus_k10": float(diff * diff_scale),
                "difference_low": float(diff_low * diff_scale),
                "difference_high": float(diff_high * diff_scale),
            })
    return result

def plot_kappa_cross(rows10, rows50, meta10, meta50):
    methods = ["k0", "s1", "k50", "k200", "greedy", "spectral"]
    effects = paired_eval_summary(rows10, rows50, methods, seed=246)
    write_csv(DATA / "n24_kappa_cross_effects.csv", effects)
    titles = {"rigidity": "Rigidity", "bound": "Lower-bound attainment",
              "ratio": "Edge ratio", "shape": "Shape error"}
    ylabels = {"rigidity": "rigid networks, %",
               "bound": "at lower bound among rigid, %",
               "ratio": "edge ratio m/m_req among rigid",
               "shape": "shape error"}

    def draw(plain):
        fig, axs = plt.subplots(2, 2, figsize=(10.4, 7.0))
        for ax, metric in zip(axs.flat, ("rigidity", "bound", "ratio", "shape")):
            records = {r["method"]: r for r in effects if r["metric"] == metric}
            for method in methods:
                r = records[method]
                x = np.array([0, 1])
                mean = np.array([r["k10"], r["k50"]])
                low = np.array([r["k10_low"], r["k50_low"]])
                high = np.array([r["k10_high"], r["k50_high"]])
                ax.errorbar(x, mean, yerr=[mean-low, high-mean],
                            color=COLORS.get(method, "#555555"),
                            marker=MARKERS.get(method, "o"), linewidth=1.0,
                            markersize=4, capsize=2, label=LABELS.get(method, method))
            ax.set_title(titles[metric])
            ax.set_xticks([0, 1]); ax.set_xticklabels(["10\nN=50", "50\nN=50"])
            ax.set_xlabel(r"evaluation $\kappa$")
            ax.set_ylabel(ylabels[metric])
            if metric in ("rigidity", "bound"):
                ax.set_ylim(-4, 104)
            if metric == "shape":
                ax.set_yscale("log")
            style(ax)
        axs[0, 0].legend(fontsize=6.5, ncol=2, loc="lower left")
        if not plain:
            fig.text(0.5, 0.01,
                     "Whiskers use paired bootstrap resampling of the shared benchmark instances.",
                     ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        return fig
    save_pair("n24_kappa_cross", draw)
    return effects

def plot_training_size(size_sources, screen_path):
    ns = [8, 10, 16, 24, 32, 48]
    methods = ["n10_k10", "n16", "n20k2", "greedy"]
    data = {}
    summary = []
    for n in ns:
        p, meta, rows = load_run(screen_path if n == 10 else size_sources[n])
        by = rows_by_method(p, meta, rows)
        data[(n, "n10_k10")] = by.get("s1", [])
        data[(n, "n20k2")] = by.get("n20k2", [])
        data[(n, "greedy")] = by.get("greedy", [])
        arm = find_tagged_run(f"b2_n{n}_randdom", "scale_arm",
                              {8: -1, 10: 200, 16: 100, 24: 50, 32: -1, 48: 25}[n],
                              "ref_n16_k10_dqn_gine") if n in (10, 16, 24, 48) else None
        if arm:
            ap, am, ar = load_run(arm)
            arm_by = rows_by_method(ap, am, ar)
            data[(n, "n16")] = arm_by.get("n16", [])
            summary.extend(summary_records(ap, am, ar, "training_size", {"n16"}))
    # The n=48 greedy evaluation covers only the first eight shared instances.
    greedy_path = find_run("20260922-085849__addremove__scale_greedy")
    gp, gm, gr = load_run(greedy_path)
    data[(48, "greedy")] = rows_by_method(gp, gm, gr)["greedy"]
    for i, n in enumerate(ns):
        for method in methods:
            group = data.get((n, method), [])
            if not group:
                continue
            summary.append({
                "family": "training_size_transfer", "benchmark": f"b2_n{n}_randdom",
                "n": n, "evaluation_kappa": 10, "method": method,
                "instances": len(group),
                "rigidity_pct": summarize(group, "rigidity")[0],
                "bound_pct_rigid": summarize(group, "bound")[0],
                "edge_ratio_rigid": summarize(group, "ratio")[0],
                "edge_count_rigid": summarize(group, "edges")[0],
                "shape_error_rigid": summarize(group, "shape")[0],
            })
    # Use benchmark/run sample labels, with the smaller greedy subset visible at n=48.
    labels = [str(n) for n in ns]
    local = {(i, method): data.get((n, method), [])
             for i, n in enumerate(ns) for method in methods}
    def draw(plain):
        fig, axs = plt.subplots(2, 2, figsize=(11.8, 7.2), sharex=True)
        for ax, metric in zip(axs.flat, ("rigidity", "bound", "ratio", "shape")):
            metric_panel(ax, local, methods, metric, list(range(len(ns))), labels,
                         annotated=not plain)
        axs[0, 0].legend(fontsize=6.8, ncol=2, loc="lower left")
        if not plain:
            fig.text(0.5, 0.01,
                     "N=200, 200, 100, 50, 25, 25 for n=8, 10, 16, 24, 32, 48. Greedy at n=48 uses 8 shared cases.\n"
                     "The n=16 checkpoint used changed reward shaping. All cases are scored at evaluation kappa 10.\n"
                     "Error bars show 95% bootstrap intervals over instances.",
                     ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        return fig
    save_pair("training_size_transfer", draw)
    write_csv(DATA / "training_size_transfer.csv", summary)
    return summary


def summarize_values(values, seed=0, scale=1):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan, np.nan
    estimate = float(values.mean())
    if len(values) == 1:
        return estimate * scale, estimate * scale, estimate * scale
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return estimate * scale, float(low * scale), float(high * scale)


def plot_algorithm_backbone(path, meta, rows):
    aliases = {
        "k50dg": "dqn_gine", "k50de": "dqn_egnn",
        "k50pg": "ppo_gine", "k50pe": "ppo_egnn",
    }
    raw = {}
    for row in rows:
        raw.setdefault(row["method"], []).append(row)
    selected = {label: raw.get(method, []) for method, label in aliases.items()}
    selected["greedy"] = raw.get("greedy", [])
    names = ["dqn_gine", "dqn_egnn", "ppo_gine", "ppo_egnn", "greedy"]
    recs = []
    for method in names:
        group = selected[method]
        edge = summarize(group, "edges", seed=201+len(method))
        shape = summarize(group, "shape", seed=231+len(method))
        rigid = summarize(group, "rigidity", seed=241+len(method))
        bound = summarize(group, "bound", seed=251+len(method))
        cost_rows = cost_by_method(path, meta).get(
            {"dqn_gine": "k50", "dqn_egnn": "estimation_randdom_k50_dqn_equi",
             "ppo_gine": "ppo50", "ppo_egnn": "estimation_randdom_k50_ppo_equi",
             "greedy": "greedy"}[method], [])
        calls = [number(r, "total") for r in cost_rows if number(r, "total") is not None]
        recs.append({"method": method, "instances": len(group),
                     "rigidity_pct": rigid[0], "bound_pct_rigid": bound[0],
                     "edge_count_rigid": edge[0], "shape_error_rigid": shape[0],
                     "calls_per_network": float(np.mean(calls)) if calls else np.nan})
    write_csv(DATA / "algorithm_backbone_metrics.csv", recs)

    def draw(plain):
        fig, axs = plt.subplots(2, 2, figsize=(10.6, 7.0))
        ax = axs[0, 0]
        for method in names:
            group = selected[method]
            edge = summarize(group, "edges", seed=301+len(method))
            shape = summarize(group, "shape", seed=331+len(method))
            ax.errorbar(edge[0], shape[0],
                        xerr=[[edge[0]-edge[1]], [edge[2]-edge[0]]],
                        yerr=[[shape[0]-shape[1]], [shape[2]-shape[0]]],
                        fmt=MARKERS.get(method, "o"), color=COLORS.get(method, "#555555"),
                        capsize=2.5, markersize=5, label=LABELS.get(method, method))
        ax.set_xlabel("edges among rigid outputs (N=200)")
        ax.set_ylabel("shape error")
        ax.set_title("Edges and shape error")
        ax.set_yscale("log")
        style(ax)
        for ax, key, title, ylabel, scale in (
            (axs[0, 1], "rigidity_pct", "Rigidity", "rigid networks, %", 1),
            (axs[1, 0], "bound_pct_rigid", "Lower-bound attainment",
             "at lower bound among rigid, %", 1),
            (axs[1, 1], "calls_per_network", "Counted calls",
             "primitive calls per network (N=200)", 1),
        ):
            for i, r in enumerate(recs):
                ax.errorbar(i, r[key] * scale, fmt=MARKERS.get(r["method"], "o"),
                            color=COLORS.get(r["method"], "#555555"), markersize=5,
                            label=LABELS.get(r["method"], r["method"]))
            ax.set_xticks(np.arange(len(names)))
            ax.set_xticklabels([LABELS.get(m,m) for m in names], rotation=25, ha="right", fontsize=7)
            ax.set_title(title); ax.set_ylabel(ylabel)
            if key == "rigidity_pct":
                ax.set_ylim(75, 102)
            if key == "bound_pct_rigid":
                ax.set_ylim(-4, 104)
            if key == "calls_per_network":
                ax.set_yscale("log")
            style(ax)
        axs[0, 0].legend(fontsize=7)
        if not plain:
            fig.text(0.5, 0.01,
                     "All policies were trained at kappa 50 and scored on the same n=10 instances at evaluation kappa 10.",
                     ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        return fig
    save_pair("algorithm_backbone", draw)
    return recs


def plot_stopping_policies():
    prefixes = {
        10: "20260922-124632__addremove__4models__scale_stop",
        16: "20260922-125417__addremove__4models__scale_stop",
        24: "20260922-130947__addremove__4models__scale_stop",
    }
    methods = ["stop_k2", "stop_k10", "stop_k50", "stop_k200"]
    labels = {10: "10\nN=200", 16: "16\nN=100", 24: "24\nN=50"}
    rows_out = []
    values_by = {}
    for n, prefix in prefixes.items():
        path, meta, rows = load_run(find_run(prefix))
        horizon = int(meta.get("rollout_steps") or (meta.get("environment_config") or {}).get("max_steps"))
        by = rows_by_method(path, meta, rows)
        for method in methods:
            group = by.get(method, [])
            if not group:
                continue
            stopped = np.asarray([float(number(r, "work") < horizon) for r in group])
            rigidity = np.asarray([float(truth(r.get("is_IBR"))) for r in group])
            bound = np.asarray([float(truth(r.get("at_bound"))) for r in rigid_rows(group)])
            ratios = metric_values(group, "ratio")
            stop_est = summarize_values(stopped, 40 + n + len(method), 100)
            rigid_est = summarize_values(rigidity, 70 + n + len(method), 100)
            bound_est = summarize_values(bound, 100 + n + len(method), 100)
            ratio_est = summarize_values(ratios, 130 + n + len(method))
            values_by[(n, method)] = {
                "stop": stop_est, "rigidity": rigid_est, "bound": bound_est,
                "ratio": ratio_est,
            }
            rows_out.append({
                "n": n, "method": method, "instances": len(group),
                "horizon": horizon, "stopped": int(stopped.sum()),
                "stopped_pct": stop_est[0], "stopped_low": stop_est[1],
                "stopped_high": stop_est[2], "rigidity_pct": rigid_est[0],
                "bound_pct_rigid": bound_est[0], "edge_ratio_rigid": ratio_est[0],
                "rigid_instances": int(rigidity.sum()),
                "at_bound_instances": int(bound.sum()),
            })
    write_csv(DATA / "stopping_policy_metrics.csv", rows_out)
    panel_specs = [
        ("stop", "Early stopping", "episodes stopping before horizon, %"),
        ("rigidity", "Rigidity", "rigid networks, %"),
        ("bound", "Lower-bound attainment", "at lower bound among rigid, %"),
        ("ratio", "Edge ratio", "edge ratio m/m_req among rigid"),
    ]
    ns = [10, 16, 24]
    def draw(plain):
        fig, axs = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex=True)
        for ax, (metric, title, ylabel) in zip(axs.flat, panel_specs):
            for method in methods:
                est = [values_by.get((n, method), {}).get(metric, (np.nan, np.nan, np.nan)) for n in ns]
                mean = np.asarray([x[0] for x in est]); low = np.asarray([x[1] for x in est]); high = np.asarray([x[2] for x in est])
                valid = np.isfinite(mean)
                ax.errorbar(np.asarray(ns)[valid], mean[valid],
                            yerr=[mean[valid]-low[valid], high[valid]-mean[valid]],
                            color=COLORS.get(method, "#555555"), marker=MARKERS.get(method, "o"),
                            linewidth=1.0, markersize=4, capsize=2,
                            label=LABELS.get(method, method))
            ax.set_title(title); ax.set_ylabel(ylabel)
            ax.set_xticks(ns); ax.set_xticklabels([labels[n] for n in ns])
            if metric in ("stop", "rigidity", "bound"):
                ax.set_ylim(-4, 104)
            style(ax)
        axs[0, 0].legend(fontsize=6.8, ncol=2, loc="lower left")
        if not plain:
            fig.text(0.5, 0.01,
                     "A policy stops when it selects the terminal skip action; N differs by network size.",
                     ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        return fig
    save_pair("stopping_policies", draw)
    return rows_out

def write_training_configs(selected):
    names = set()
    for path in selected:
        meta_path = path / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            names.update((meta.get("models") or {}).values())
    rows = []
    for model_name in sorted(names):
        manifest_path = ROOT / "train" / f"{model_name}.json"
        if not manifest_path.exists():
            rows.append({"model": model_name, "manifest": str(manifest_path.relative_to(ROOT)),
                         "status": "manifest missing"})
            continue
        manifest = json.loads(manifest_path.read_text())
        env = manifest.get("environment_config_raw") or {}
        rows.append({
            "model": model_name, "manifest": str(manifest_path.relative_to(ROOT)),
            "algorithm": manifest.get("algorithm", ""),
            "training_environment": manifest.get("environment_config", ""),
            "training_n": env.get("n", ""), "training_domains": env.get("domains", ""),
            "random_domains": env.get("random_domains", ""),
            "training_kappa": env.get("stiffness_kappa", ""),
            "training_score": env.get("state_score_type", ""),
            "action_type": env.get("action_type", ""),
            "observation_type": env.get("obs_type", ""),
            "backbone": manifest.get("backbone", ""),
            "configured_steps": manifest.get("total_timesteps_configured", ""),
            "completed_steps": manifest.get("timesteps_completed", ""),
            "status": manifest.get("status", ""),
            "started": manifest.get("timestamp_started", ""),
        })
    write_csv(DATA / "training_configurations.csv", rows)

def copy_report_pair(plots_dir, source_name, target_name):
    for suffix in ("", "-plain"):
        for extension in ("pdf", "png"):
            source = plots_dir / extension / f"{source_name}{suffix}.{extension}"
            target = FIGS / f"{target_name}{suffix}.{extension}"
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, target)

def collect_appendix_metrics():
    ablation_rows = []
    ablation_runs = {
        0: find_run("20260912-183108__addremove__ref_k0_dqn_gine"),
        200: find_run("20260912-183357__addremove__ref_k200_dqn_gine"),
    }
    for train_kappa, path in ablation_runs.items():
        raw = json.loads((path / "ablation_raw.json").read_text())
        for perturbation, record in raw["learned"].items():
            for row in record["rows"]:
                ablation_rows.append({"training_kappa": train_kappa,
                                      "perturbation": perturbation, **row})
    write_csv(DATA / "ablation_channel_metrics.csv", ablation_rows)

    action_path = find_run("20260912-164003__selectseq__ref_select_dqn_gine__arm_action")
    action_p, action_meta, action_rows = load_run(action_path)
    action_by = rows_by_method(action_p, action_meta, action_rows)
    write_csv(DATA / "action_space_metrics.csv",
              summary_records(action_p, action_meta, action_rows, "action_space",
                              {"select", "greedy", "spectral", "degree"}))
    action_effects = paired_effect_records(
        action_by, "action_space", action_meta.get("benchmark", ""), ["greedy"])
    write_csv(DATA / "action_space_paired_effects.csv", action_effects)
    plot_action_space_effects(action_effects)

    decisions_path = find_run("20260912-162935__addremove__3models__main")
    decisions = read_csv(decisions_path / "decisions.csv")
    decision_rows = []
    for model in ("s0", "s1", "s2"):
        group = [row for row in decisions if row["model"] == model]
        phi = np.asarray([number(row, "phi_pct") for row in group
                          if number(row, "phi_pct") is not None])
        err = np.asarray([number(row, "err_pct") for row in group
                          if number(row, "err_pct") is not None])
        decision_rows.append({"model": model, "decisions": len(group),
                              "phi_decisions": len(phi), "shape_error_decisions": len(err),
                              "phi_percentile_mean": float(phi.mean()) if len(phi) else np.nan,
                              "phi_percentile_median": float(np.median(phi)) if len(phi) else np.nan,
                              "shape_error_percentile_mean": float(err.mean()) if len(err) else np.nan,
                              "shape_error_percentile_median": float(np.median(err)) if len(err) else np.nan,
                              "phi_best_pct": float(np.mean([truth(row.get("phi_best")) for row in group])) if group else np.nan,
                              "shape_error_best_pct": float(np.mean([truth(row.get("err_best")) for row in group])) if group else np.nan})
    write_csv(DATA / "decision_metrics.csv", decision_rows)

    noise = read_csv(decisions_path / "noise.csv")
    noise_rows = []
    groups = {}
    for row in noise:
        groups.setdefault((row["method"], number(row, "sigma_deg")), []).append(row)
    for (method, sigma_deg), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        valid = [row for row in group if not truth(row.get("failed"))
                 and number(row, "measured") and number(row, "predicted")]
        log_ratios = np.asarray([math.log(number(row, "measured") / number(row, "predicted"))
                                 for row in valid])
        if len(log_ratios):
            est, lo, hi = summarize_values(log_ratios, seed=83 + len(method) + int(sigma_deg * 100), scale=1)
            estimate, low, high = math.exp(est), math.exp(lo), math.exp(hi)
        else:
            estimate = low = high = np.nan
        noise_rows.append({"method": method, "sigma_deg": sigma_deg,
                           "instances": len(group), "valid": len(valid),
                           "failures": sum(truth(row.get("failed")) for row in group),
                           "geomean_measured_over_predicted": estimate,
                           "ratio_low": low, "ratio_high": high})
    write_csv(DATA / "noise_validation_metrics.csv", noise_rows)

    repair_path = find_run("20260912-185118__addremove__4models")
    repair_rows = read_csv(repair_path / "repair.csv")
    from repair import _headline
    repair_rows = _headline(repair_rows)
    repair_summary = []
    repair_groups = {}
    for row in repair_rows:
        if truth(row.get("skipped")) or not row.get("method"):
            continue
        repair_groups.setdefault((row["event"], row["method"]), []).append(row)
    for (event, method), group in sorted(repair_groups.items()):
        values = [number(row, "pctile") for row in group if number(row, "pctile") is not None]
        percentiles = summarize_values(np.asarray(values), seed=101 + len(event) + len(method), scale=1)
        edge_ratios = [number(row, "m_over_m_req") for row in group
                       if number(row, "m_over_m_req") is not None]
        err_ratios = [number(row, "err_ratio_to_best") for row in group
                      if number(row, "err_ratio_to_best") is not None]
        repair_summary.append({"event": event, "method": method, "rows": len(group),
                               "n_percentile": len(values), "percentile_mean": percentiles[0],
                               "percentile_low": percentiles[1], "percentile_high": percentiles[2],
                               "n_edge_ratio": len(edge_ratios),
                               "edge_ratio_mean": float(np.mean(edge_ratios)) if edge_ratios else np.nan,
                               "n_error_ratio": len(err_ratios),
                               "error_ratio_geomean": float(np.exp(np.mean(np.log(err_ratios)))) if err_ratios else np.nan})
    write_csv(DATA / "repair_metrics.csv", repair_summary)

    budget_path = find_run("20260913-043413__addremove__4models__budget")
    budget_rows = read_csv(budget_path / "budget.csv")
    budget_metrics = []
    budget_groups = {}
    for row in budget_rows:
        budget_groups.setdefault((row["method"], int(row["budget"])), []).append(row)
    for (method, budget), group in sorted(budget_groups.items()):
        rigid = [row for row in group if truth(row.get("is_rigid"))]
        errors = [number(row, "shape_err") for row in rigid
                  if number(row, "shape_err") and number(row, "shape_err") > 0]
        budget_metrics.append({"method": method, "budget": budget, "instances": len(group),
                              "rigid_pct": 100 * np.mean([truth(row.get("is_rigid")) for row in group]),
                              "bound_pct_rigid": 100 * np.mean([truth(row.get("at_bound")) for row in rigid]) if rigid else np.nan,
                              "edge_count_rigid": float(np.mean([number(row, "m") for row in rigid])) if rigid else np.nan,
                              "shape_error_rigid": float(np.exp(np.mean(np.log(errors)))) if errors else np.nan,
                              "final_row_pct": 100 * np.mean([truth(row.get("is_final")) for row in group])})
    write_csv(DATA / "budget_metrics.csv", budget_metrics)

    training_path = find_run("20260912-155001__addremove")
    copy_report_pair(budget_path / "plots", "budget", "budget")
    copy_report_pair(training_path / "plots", "training", "training")
    copy_report_pair(decisions_path / "plots", "decisions", "decisions")
    copy_report_pair(decisions_path / "plots", "noise", "noise")
    copy_report_pair(decisions_path / "plots", "prediction", "prediction")
    copy_report_pair(decisions_path / "plots", "uncertainty", "uncertainty")
    for kappa, path in ablation_runs.items():
        copy_report_pair(path / "plots", "ablation-learned", f"ablation-k{kappa}")
    copy_report_pair(repair_path / "plots", "repair_percentile", "repair_percentile")
    copy_report_pair(repair_path / "plots", "repair_choice", "repair_choice")


def _chapter_csv(name):
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))


def plot_budget_progress():
    rows = _chapter_csv("budget_metrics.csv")
    methods = ["greedy", "k0", "k10", "k50", "k200"]
    panels = (
        ("rigid_pct", "Rigidity", "percent"),
        ("bound_pct_rigid", "Lower-bound attainment", "percent"),
        ("edge_count_rigid", "Edge count", "edges"),
        ("shape_error_rigid", "Shape error", "error"),
    )

    def draw(plain):
        fig, axs = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex=True)
        for ax, (field, title, ylabel) in zip(axs.flat, panels):
            for method in methods:
                group = sorted((r for r in rows if r["method"] == method),
                               key=lambda r: int(r["budget"]))
                xs = [int(r["budget"]) for r in group]
                ys = [float(r[field]) if r[field] not in ("", "nan") else np.nan
                      for r in group]
                ax.plot(xs, ys, marker=MARKERS.get(method, "o"),
                        color=COLORS.get(method, "#333333"), markersize=3,
                        linewidth=1, label=LABELS.get(method, method))
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            if field == "shape_error_rigid":
                ax.set_yscale("log")
            style(ax)
        for ax in axs[-1, :]:
            ax.set_xlabel("Moves allowed")
        axs[0, 0].legend(fontsize=7, ncol=2)
        if not plain:
            fig.text(0.5, 0.01,
                     "Each point uses the best state found within that move budget. "
                     "N=20 instances per method.",
                     ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return fig

    save_pair("budget_progress", draw)


def plot_decision_percentiles():
    rows = _chapter_csv("decision_metrics.csv")
    seeds = ["s0", "s1", "s2"]
    panels = (
        ("phi_percentile_mean", "phi_percentile_median", "Objective percentile"),
        ("shape_error_percentile_mean", "shape_error_percentile_median",
         "Shape-error percentile"),
    )

    def draw(plain):
        fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.1), sharey=True)
        by_seed = {r["model"]: r for r in rows}
        for ax, (mean_key, median_key, title) in zip(axs, panels):
            for key, label, marker, color in (
                (mean_key, "Mean", "o", "#1f77b4"),
                (median_key, "Median", "s", "#d95f02"),
            ):
                values = [float(by_seed[s][key]) for s in seeds]
                ax.plot(np.arange(3), values, marker=marker, linewidth=1,
                        color=color, label=label)
            ax.axhline(50, color="#555555", linestyle=":", linewidth=0.9)
            ax.set_xticks(np.arange(3), ["Seed 0", "Seed 1", "Seed 2"])
            ax.set_title(title)
            ax.set_ylabel("Percentile")
            ax.set_ylim(0, 100)
            ax.legend(fontsize=8)
            style(ax)
        if not plain:
            fig.text(0.5, 0.01,
                     "Percentiles summarize recorded decisions. "
                     "The dotted line marks the median of the reference set.",
                     ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        return fig

    save_pair("decision_percentiles", draw)


def plot_repair_percentiles():
    rows = _chapter_csv("repair_metrics.csv")
    events = [
        ("edge_drop_random", "Edge drop"),
        ("node_join", "Node join"),
        ("node_leave", "Node leave"),
    ]
    methods = ["greedy", "greedy_repair", "k0", "k10", "k50", "k200"]

    def draw(plain):
        fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.6), sharey=True)
        groups = {(r["event"], r["method"]): r for r in rows}
        for ax, (event, title) in zip(axs, events):
            for i, method in enumerate(methods):
                r = groups.get((event, method))
                if not r or r["percentile_mean"] in ("", "nan"):
                    continue
                est = float(r["percentile_mean"])
                lo = est - float(r["percentile_low"])
                hi = float(r["percentile_high"]) - est
                ax.errorbar(i, est, yerr=[[lo], [hi]], fmt="o",
                            color=COLORS.get(method, "#333333"),
                            capsize=2.5, markersize=4)
            ax.set_xticks(np.arange(len(methods)),
                          [LABELS.get(m, m) for m in methods],
                          rotation=45, ha="right", fontsize=7)
            ax.set_title(title)
            ax.set_ylim(0, 105)
            ax.set_ylabel("Percentile among equal-size repairs")
            ax.axhline(50, color="#555555", linestyle=":", linewidth=0.9)
            style(ax)
        if not plain:
            fig.text(0.5, 0.01,
                     "Percentiles are defined for rank-breaking repairs. "
                     "Points are instance means with 95% bootstrap intervals.",
                     ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        return fig

    save_pair("repair_percentiles", draw)


def plot_noise_prediction_ratio():
    rows = _chapter_csv("noise_validation_metrics.csv")
    methods = ["degree", "greedy", "spectral", "s0", "s1", "s2",
               "anneal", "constructive"]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]

    def draw(plain):
        fig, ax = plt.subplots(figsize=(8.8, 5.1))
        for method, marker in zip(methods, markers):
            group = sorted((r for r in rows if r["method"] == method),
                           key=lambda r: float(r["sigma_deg"]))
            if not group:
                continue
            xs = np.asarray([float(r["sigma_deg"]) for r in group])
            ys = np.asarray([float(r["geomean_measured_over_predicted"])
                             for r in group])
            lo = np.asarray([float(r["ratio_low"]) for r in group])
            hi = np.asarray([float(r["ratio_high"]) for r in group])
            ax.errorbar(xs, ys, yerr=[ys-lo, hi-ys], marker=marker,
                        color=COLORS.get(method, "#555555"), capsize=2.5,
                        markersize=4, linewidth=1,
                        label=LABELS.get(method, method))
        ax.axhline(1, color="#333333", linestyle=":", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Bearing noise per measurement (degrees)")
        ax.set_ylabel("Measured / predicted shape error")
        ax.set_title("Measured-to-predicted shape error")
        ax.legend(fontsize=7, ncol=2)
        style(ax)
        if not plain:
            fig.text(0.5, 0.01,
                     "Each point is the geometric mean ratio over 20 instances. "
                     "Bars show 95% bootstrap intervals.",
                     ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return fig

    save_pair("noise_prediction_ratio", draw)



def plot_ablation_summary():
    rows = _chapter_csv("ablation_channel_metrics.csv")
    mode_colors = {"shuffle": "#f28e2b", "zero": "#cc79a7", "noise": "#d62728"}

    for kappa in (0, 200):
        selected = [r for r in rows if int(r["training_kappa"]) == kappa]
        channels = list(dict.fromkeys(r["channel"] for r in selected))
        worst = {channel: max((float(r["d_phi"]) for r in selected
                               if r["channel"] == channel and r["d_phi"] not in ("", "nan")),
                              default=0.0) for channel in channels}
        channels.sort(key=lambda channel: worst[channel], reverse=True)

        def draw(plain):
            fig, axs = plt.subplots(1, 2, figsize=(11.4, 7.4), sharey=True)
            for ax, field, title in (
                (axs[0], "d_phi", "Objective-score difference"),
                (axs[1], "flip", "Changes in best actions"),
            ):
                for offset, mode in enumerate(("shuffle", "zero", "noise")):
                    group = {(r["channel"]): r for r in selected
                             if r["perturbation"] == mode}
                    ys = np.arange(len(channels)) + (offset - 1) * 0.22
                    values = []
                    positions = []
                    for y, channel in zip(ys, channels):
                        row = group.get(channel)
                        value = float(row[field]) if row and row[field] not in ("", "nan") else np.nan
                        if field == "flip" and np.isfinite(value):
                            value *= 100
                        if np.isfinite(value):
                            positions.append(y)
                            values.append(value)
                    ax.barh(positions, values, height=0.2,
                            color=mode_colors[mode], label=mode)
                ax.set_yticks(np.arange(len(channels)), channels, fontsize=7)
                ax.set_title(title)
                ax.set_xlabel("score difference" if field == "d_phi" else "percent of steps")
                ax.legend(fontsize=7)
                style(ax)
            if not plain:
                fig.text(0.5, 0.01,
                         "Each channel is perturbed separately. Shuffle preserves its "
                         "distribution; zero and noise also change its scale.",
                         ha="center", fontsize=8)
            fig.tight_layout(rect=(0, 0.035, 1, 1))
            return fig

        save_pair(f"ablation-k{kappa}", draw)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    size_data, size_costs, size_sources = load_size_data()
    plot_size(size_data, size_costs)

    domain_runs = find_domain_runs()
    domain_data, domain_sources = load_domain_data(domain_runs)
    homogeneous_order = [b for b, _ in HOMOGENEOUS]
    homogeneous_labels = [label for _, label in HOMOGENEOUS]
    plot_domain_figure(
        "domain_reference", domain_data, homogeneous_order, homogeneous_labels,
        ["k0", "s1", "k50", "k200", "greedy", "spectral", "degree"],
        "Homogeneous-domain results. N=100 per domain.",
    )
    mix_order = [b for b, _ in MIXED]
    mix_labels = ["R2/SE3 random" + chr(10) + "N=100", "R2/SE3 5/5" + chr(10) + "N=100", "R3/(R2xS1) random" + chr(10) + "N=100", "R3/(R2xS1) 5/5" + chr(10) + "N=100"]
    plot_domain_figure(
        "domain_training_kappa", domain_data, mix_order, mix_labels,
        ["k0", "s0", "s1", "s2", "k50", "k200", "greedy", "spectral", "degree"],
        "All methods are scored at evaluation $\\kappa=10$. N=100 per composition and assignment design.",
    )

    screen_path = find_run("20260921-233657__addremove__15models__screen_n10")
    sp, smeta, srows = load_run(screen_path)
    ref_by = plot_reference(srows, smeta)
    training_size_rows = plot_training_size(size_sources, screen_path)
    stop_policy_rows = plot_stopping_policies()
    front = load_front()
    plot_kappa(front)
    plot_compute(ref_by, cost_by_method(sp, smeta))
    algorithm_backbone_rows = plot_algorithm_backbone(sp, smeta, srows)
    control = load_control()
    plot_control(control)
    structure = graph_structure(screen_path, smeta, srows)
    plot_structure(structure)
    collect_appendix_metrics()
    plot_budget_progress()
    plot_decision_percentiles()
    plot_repair_percentiles()
    plot_noise_prediction_ratio()
    plot_ablation_summary()

    n24_k50_path = find_tagged_run("b2_n24_randdom", "thesis_k50_eval", 50,
                                   "estimation_randdom_k50_dqn_gine")
    n24_k50_p, n24_k50_meta, n24_k50_rows = load_run(n24_k50_path)
    n24_k10_p, n24_k10_meta, n24_k10_rows = load_run(size_sources[24])
    n24_kappa_effects = plot_kappa_cross(
        rows_by_method(n24_k10_p, n24_k10_meta, n24_k10_rows),
        rows_by_method(n24_k50_p, n24_k50_meta, n24_k50_rows),
        n24_k10_meta, n24_k50_meta)

    k50_n48_path = find_tagged_run("b2_n48_randdom", "thesis_k50_size", 25,
                                   "estimation_randdom_k50_dqn_gine")
    old48_p, old48_meta, old48_rows = load_run(size_sources[48])
    k50_48_p, k50_48_meta, k50_48_rows = load_run(k50_n48_path)
    greedy_path = find_run("20260922-085849__addremove__scale_greedy")
    greedy_p, greedy_meta, greedy_rows = load_run(greedy_path)
    n48_by = rows_by_method(old48_p, old48_meta, old48_rows)
    n48_by.update(rows_by_method(k50_48_p, k50_48_meta, k50_48_rows))
    n48_by["greedy"] = rows_by_method(greedy_p, greedy_meta, greedy_rows)["greedy"]

    selected = {path: f"size_n{n}" for n, path in size_sources.items()}
    selected[screen_path] = "reference_200"
    selected.update({path: f"domain_{bench}" for bench, path in domain_sources.items()})
    selected.update({v[0]: f"kappa_{k}" for k, v in front.items()})
    selected[control[0]] = "formation_control"
    selected[k50_n48_path] = "size_n48_k50"
    selected[greedy_path] = "size_n48_greedy"
    selected[n24_k50_path] = "kappa_eval50_n24"
    for n, episodes in ((10, 200), (16, 100), (24, 50), (48, 25)):
        path = find_tagged_run(f"b2_n{n}_randdom", "scale_arm", episodes,
                               "ref_n16_k10_dqn_gine")
        selected[path] = f"training_size_arm_n{n}"
    for n, prefix in ((10, "20260922-124632__addremove__4models__scale_stop"),
                      (16, "20260922-125417__addremove__4models__scale_stop"),
                      (24, "20260922-130947__addremove__4models__scale_stop")):
        selected[find_run(prefix)] = f"stopping_policy_n{n}"
    selected[find_run("20260913-043413__addremove__4models__budget")] = "step_budget"
    selected[find_run("20260912-155001__addremove")] = "training_curves"
    selected[find_run("20260912-183108__addremove__ref_k0_dqn_gine")] = "ablation_k0"
    selected[find_run("20260912-183357__addremove__ref_k200_dqn_gine")] = "ablation_k200"
    selected[find_run("20260912-162935__addremove__3models__main")] = "noise_validation"
    selected[find_run("20260912-164003__selectseq__ref_select_dqn_gine__arm_action")] = "action_space"
    selected[find_run("20260912-163456__addremove__ref_norig_dqn_gine__arm_obs")] = "observation_ablation"
    selected[find_run("20260912-185118__addremove__4models")] = "repair"
    inventory_sources(selected)
    write_training_configs(selected)

    summary_rows = []
    for n, path in size_sources.items():
        p, meta, rows = load_run(path)
        summary_rows.extend(summary_records(
            p, meta, rows, "size", {"k0", "s1", "k50", "k200", "greedy", "spectral", "degree"}))
    for bench, path in domain_sources.items():
        p, meta, rows = load_run(path)
        summary_rows.extend(summary_records(
            p, meta, rows, "domain", {"k0", "s0", "s1", "s2", "k50", "k200", "greedy", "spectral", "degree"}))
    summary_rows.extend(summary_records(
        sp, smeta, srows, "reference", {"k0", "s0", "s1", "s2", "k50", "k200", "greedy", "spectral", "degree"}))
    for k, (path, meta, rows) in front.items():
        summary_rows.extend(summary_records(
            path, meta, rows, "kappa_front", {"k0", "s1", "k50", "k200", "greedy", "spectral", "degree"}))
    summary_rows.extend(summary_records(
        n24_k50_p, n24_k50_meta, n24_k50_rows, "n24_eval_k50",
        {"k0", "s1", "k50", "k200", "greedy", "spectral"}))
    summary_rows.extend(summary_records(
        k50_48_p, k50_48_meta, k50_48_rows, "size",
        {"k50"}))
    summary_rows.extend(summary_records(
        greedy_p, greedy_meta, greedy_rows, "size",
        {"greedy"}))
    write_csv(DATA / "summary_metrics.csv", summary_rows)

    paired_rows = []
    for bench, path in domain_sources.items():
        if bench not in {b for b, _ in MIXED}:
            continue
        p, meta, rows = load_run(path)
        by = rows_by_method(p, meta, rows)
        paired_rows.extend(paired_effect_records(
            by, "mixed_domain", bench, ["greedy", "spectral"]))
    paired_rows.extend(paired_effect_records(
        n48_by, "size_n48", "b2_n48_randdom", ["greedy", "spectral"]))
    paired_rows.extend(paired_effect_records(
        ref_by, "reference_n10", smeta.get("benchmark", ""), ["greedy", "spectral"]))
    write_csv(DATA / "paired_effects.csv", paired_rows)
    print(f"wrote figures to {FIGS}")
    print(f"wrote derived CSVs to {DATA}")


# Reproducible tables and figures for the approved chapter revision.
CHAPTER_FIGURES = []
CHAPTER_USED = set()
TABLES = ROOT / 'resources/thesis/tex/tables/results'
SHORT = {'s0': 'DQN, seed 0', 's1': 'DQN, seed 1', 's2': 'DQN, seed 2',
         'k0': r'DQN, $\kappa_t=0$', 'k50': r'DQN, $\kappa_t=50$',
         'k200': r'DQN, $\kappa_t=200$', 'norig': 'Without rigidity features',
         'select': 'Sequential selection', 'greedy': 'Greedy', 'spectral': 'Spectral',
         'degree': 'Degree', 'constructive': 'Constructive', 'anneal': 'Annealing',
         'complete': 'Complete', 'k10': r'DQN, $\kappa_t=10$'}


def chapter_run(prefix):
    p, meta, rows = load_run(find_run(prefix))
    CHAPTER_USED.add(p)
    return p, meta, rows_by_method(p, meta, rows)


def require_shared(meta_a, meta_b):
    a, b = meta_a.get('benchmark_digest'), meta_b.get('benchmark_digest')
    if not a or a != b:
        raise ValueError('Paired sources must have the same saved benchmark digest')


def chapter_save(name, draw, purpose, sources):
    save_pair(name, draw)
    CHAPTER_FIGURES.append({'figure': name, 'purpose': purpose,
                            'sources': '|'.join(str(p.relative_to(ROOT)) for p in sources)})


def chapter_layout(fig, axes, plain, note='', legend=True, top=.88):
    if legend:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=min(3, len(labels)),
                   frameon=False, fontsize=9, bbox_to_anchor=(.5, 1.01))
    if not plain:
        fig.text(.5, .006, note, ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, .04 if not plain else 0, 1, top if legend else 1),
                     w_pad=1.7, h_pad=1.8)


def chapter_grid(name, data, keys, labels, methods, metrics, purpose, sources,
                 note='', numeric=False):
    titles = {'rigidity': 'Rigid outputs', 'attainment': 'Rigid outputs at the bound',
              'ratio': 'Edge ratio', 'shape': 'Error coefficient', 'calls': 'Primitive calls', 'final': 'Final best score reached',
              'edges': 'Edge count', 'ms': 'Recorded wall time',
              'position_component':'Position error coefficient','attitude_component':'Attitude error coefficient'}
    units = {'rigidity': '% of all instances', 'attainment': '% of all instances', 'final': '% of all instances',
             'ratio': r'$m/m_{\rm req}$, rigid outputs', 'shape': r'$\varepsilon$, rigid outputs',
             'calls': 'Calls per instance', 'edges': r'$m$, rigid outputs', 'ms': 'Milliseconds per instance',
             'position_component':r'$\varepsilon_p$, rigid outputs','attitude_component':r'$\varepsilon_a$, rigid outputs'}
    def draw(plain):
        nr = math.ceil(len(metrics) / 2)
        fig, axs = plt.subplots(nr, 2, figsize=(7.0, 3.25 * nr + .45), squeeze=False)
        x = np.asarray(keys, dtype=float) if 'budget' in name else np.arange(len(keys))
        for ax, metric in zip(axs.flat, metrics):
            for j, method in enumerate(methods):
                vals = [summarize(data.get((key, method), []), metric,
                                  seed=50 + i * 19 + j) for i, key in enumerate(keys)]
                y = np.array([v[0] for v in vals]); lo = np.array([v[1] for v in vals]); hi = np.array([v[2] for v in vals])
                offset = 0 if 'budget' in name else (j - (len(methods)-1)/2) * (.025 if numeric else .10)
                ax.errorbar(x + offset, y, yerr=[np.maximum(0,y-lo), np.maximum(0,hi-y)],
                            color=COLORS.get(method, '#444444'), marker=MARKERS.get(method, 'o'),
                            linestyle='-' if numeric else 'none', lw=1.2, markersize=4,
                            errorevery=[i for i,k in enumerate(keys) if k in [1,2,5,10,20,50,118]] if 'budget' in name else 1,
                            capsize=2, label=SHORT.get(method, LABELS.get(method, method)))
            ax.set_title(titles[metric]); ax.set_ylabel(units[metric])
            ax.set_xticks(x, labels, fontsize=9)
            if 'budget' in name:
                ax.set_xscale('log')
                ax.set_xticks([1,2,5,10,20,50,118], ['1','2','5','10','20','50','118'])
                ax.minorticks_off()
                ax.set_xlabel('Edge-edit budget')
            elif numeric:
                ax.set_xlabel('Agents $n$')
            if metric in ('shape','calls','ms','position_component','attitude_component'): ax.set_yscale('log')
            if metric=='attitude_component':
                absent=[]
                for i,key in enumerate(keys):
                    vals=metric_values(data.get((key,methods[0]),[]),metric)
                    if len(vals) and np.all(vals==0):
                        absent.append(x[i]);ax.axvspan(x[i]-.5,x[i]+.5,color='#eeeeee',zorder=0)
                if absent:ax.text(np.mean(absent),.5,'No attitude\nDOFs',transform=ax.get_xaxis_transform(),ha='center',va='center',fontsize=8)
            if metric=='shape' and 'position_component' in metrics:ax.set_title('Combined error coefficient')
            if metric in ('rigidity','attainment'): ax.set_ylim(-3,103)
            if metric == 'ratio': ax.axhline(1, color='#888888', ls=':', lw=.8)
            style(ax)
        for ax in list(axs.flat)[len(metrics):]: ax.set_visible(False)
        chapter_layout(fig, axs, plain, note)
        return fig
    chapter_save(name, draw, purpose, sources)


def tex_number(x, digits=2):
    return '--' if not np.isfinite(x) else f'${x:.{digits}f}$'


def chapter_table(name, header, body, align):
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [r'\begin{tabular}{'+align+'}', r'\toprule', ' & '.join(header)+r' \\', r'\midrule']
    lines += [' & '.join(row)+r' \\' for row in body]
    lines += [r'\bottomrule',r'\end{tabular}']
    (TABLES / (name+'.tex')).write_text('\n'.join(lines)+'\n')


def chapter_outcome_row(label, rows):
    rigid = rigid_rows(rows)
    bound = sum(truth(r.get('is_IBR')) and truth(r.get('at_bound')) for r in rows)
    return [label, f'${len(rigid)}/{len(rows)}$', f'${bound}/{len(rows)}$' if all('at_bound' in r for r in rows) else '--',
            tex_number(summarize(rows,'edges')[0]), tex_number(summarize(rows,'shape')[0])]


def chapter_model_map(meta):
    models = dict(meta.get('models') or {})
    if not models:
        arg = meta.get('args',{}).get('model')
        if isinstance(arg,str) and ',' not in arg:
            models['learned'] = arg.split('=')[0]
    return models


def chapter_audit():
    audit, summaries, configs, protocols = [], [], [], []
    manifests = {}
    for p in sorted((ROOT/'train').glob('*.json')):
        if not p.stem.startswith(('ref_','estimation_','stiff_')): continue
        m=json.loads(p.read_text()); e=m.get('environment_config_raw') or {};manifests[p.stem]=m
        configs.append({'model':p.stem,'algorithm':m.get('algorithm'),'backbone':m.get('backbone'),
                        'n':e.get('n'),'random_domains':e.get('random_domains'),
                        'domains':str(e.get('domains')),'score':e.get('state_score_type'),
                        'kappa':e.get('stiffness_kappa'),'action':e.get('action_type'),
                        'steps_configured':m.get('total_timesteps_configured'),
                        'steps_completed_index':m.get('timesteps_completed'),'status':m.get('status'),
                        'seed':(m.get('provenance') or {}).get('seed'),
                        'rigidity_features':','.join(k for k,v in e.items() if k.startswith('rigidity_') and v is True)})
    for p in sorted(RUNS.glob('20*')):
        if not (p/'meta.json').exists():continue
        meta=json.loads((p/'meta.json').read_text());models=chapter_model_map(meta);args=meta.get('args',{})
        allowed={k:v for k,v in models.items() if v in manifests}
        run_names=args.get('runs') or ''
        if not allowed and not any(n in run_names for n in manifests) and p not in CHAPTER_USED:continue
        files=[f.name for f in p.iterdir() if f.suffix in ('.csv','.npz','.json')]
        env=meta.get('environment_config') or {}
        if p in CHAPTER_USED: status='included';reason='Source for a chapter or appendix figure/table'
        elif any(w in p.name.lower() for w in ('smoke','pilot','probe','timing')):
            status='excluded';reason='Diagnostic or pilot subset; full evaluation retained separately'
        elif 'results.csv' in files:
            status='supporting';reason='Per-instance results retained in chapter_all_runs.csv; not pooled across protocols'
        elif 'control.csv' in files:
            status='excluded';reason='Earlier control protocol; corrected shared-case control source used'
        elif any(f in files for f in ('generalisation.csv','scale.csv')):
            status='excluded';reason='Aggregation of evaluations, not independent instances; use underlying raw runs'
        elif 'training.csv' in files:
            status='supporting';reason='Training summary export; original event logs used for curves'
        elif 'topology.csv' in files:
            status='supporting';reason='Topology diagnostic; common-instance saved graphs used for chapter comparison'
        elif 'ablation_raw.json' in files:
            status='supporting';reason='Older inference perturbation protocol; reference-family ablations shown'
        elif 'repair.csv' in files:
            status='supporting';reason='Older score-family repair evaluation; retained separately from reference-family repair'
        else:status='excluded';reason='No additional instance-level comparison for the chapter'
        audit.append({'directory':str(p.relative_to(ROOT)),'status':status,'reason':reason,
                      'models':json.dumps(allowed,sort_keys=True),'benchmark':meta.get('benchmark'),
                      'digest':meta.get('benchmark_digest'),'score':env.get('state_score_type'),
                      'evaluation_kappa':env.get('stiffness_kappa'),'horizon':meta.get('rollout_steps',meta.get('horizon')),
                      'policy_mode':args.get('policy_mode'),'files':','.join(sorted(files))})
        if 'results.csv' not in files:continue
        by={}
        for r in read_csv(p/'results.csv'):by.setdefault(r['method'],[]).append(r)
        for method,rs in by.items():
            name=models.get(method,method); train=manifests.get(name,{}).get('environment_config_raw') or {}
            k=env.get('stiffness_kappa');tk=train.get('stiffness_kappa')
            score_match=bool(train) and all(train.get(key)==env.get(key) for key in ('state_score_type','stiffness_kappa','spectral_functional'))
            rec={'directory':str(p.relative_to(ROOT)),'method':method,'model':name,'benchmark':meta.get('benchmark'),
                 'digest':meta.get('benchmark_digest'),'n':meta.get('n',env.get('n')),'N':len(rs),
                 'rigid':len(rigid_rows(rs)),'at_bound':sum(truth(r.get('is_IBR')) and truth(r.get('at_bound')) for r in rs) if all('at_bound' in r for r in rs) else '',
                 'mean_edges':summarize(rs,'edges')[0],'geomean_error':summarize(rs,'shape')[0],
                 'mean_ratio':summarize(rs,'ratio')[0],'training_kappa':tk,'evaluation_kappa':k,
                 'training_score':train.get('state_score_type'),'evaluation_score':env.get('state_score_type'),
                 'same_score_definition':score_match,'output_rules':','.join(sorted({r.get('edges_are','unrecorded') for r in rs}))}
            summaries.append(rec)
        if p in CHAPTER_USED:
            protocols.append({'directory':p.name,'benchmark':meta.get('benchmark'), 'digest':meta.get('benchmark_digest'),
                              'horizon':meta.get('rollout_steps'), 'args':json.dumps(args,sort_keys=True),
                              'environment':json.dumps(env,sort_keys=True),'model_info':json.dumps(meta.get('model_info',{}),sort_keys=True)})
    import hashlib
    fingerprints = {}
    for entry in audit:
        path = ROOT / entry['directory']
        if not (path/'results.csv').exists(): continue
        meta = json.loads((path/'meta.json').read_text())
        mapping = chapter_model_map(meta)
        normalized = sorted((r['episode'], mapping.get(r['method'],r['method']),
                             r.get('m'), r.get('is_IBR'),
                             round(float(r['score']),7) if number(r,'score') is not None else None)
                            for r in read_csv(path/'results.csv'))
        signature = hashlib.sha256(json.dumps([entry['digest'],entry['score'],entry['evaluation_kappa'],
                                               entry['horizon'],normalized],sort_keys=True).encode()).hexdigest()
        previous = fingerprints.get(signature)
        if previous and entry['status'] != 'included':
            entry['status'] = 'duplicate outcomes'
            entry['reason'] = 'Same benchmark, objective, horizon, and recorded scores/counts as '+previous+'; not independent evidence'
        else: fingerprints[signature] = entry['directory']
    # Make nondisplayed sources traceable without treating repeated protocols as replications.
    included = [r for r in audit if r['status'] == 'included']
    for entry in audit:
        if entry['status'] != 'supporting' or not entry['reason'].startswith('Per-instance'):
            continue
        names = set(json.loads(entry['models']).values())
        matches = [r['directory'] for r in included
                   if entry['benchmark'] and entry['benchmark'] == r['benchmark']
                   and entry['score'] == r['score'] and entry['evaluation_kappa'] == r['evaluation_kappa']
                   and names.intersection(json.loads(r['models']).values())]
        if matches:
            reason = 'This benchmark/objective/model combination is represented by ' + matches[0] + '; exports are not pooled'
        elif '__gen' in entry['directory'] and any(n.startswith('ref_') for n in names):
            reason = 'Small reference-seed-0 transfer set; Chapter 6 uses the larger frozen size/domain sets and keeps seeds separate'
        elif '__front_' in entry['directory']:
            reason = 'Small cross-weight diagnostic; matched-weight evidence uses the 100-case comparisons, and selection-weight sensitivity is reported separately'
        elif any('select' in n for n in names):
            reason = 'Legacy sequential-action configurations combine action-space and objective changes; the appendix uses the matched reference action-space comparison and separate stopping results'
        elif any('stop' in n for n in names):
            reason = 'Small stopping diagnostic; the appendix uses the larger stopping evaluations across sizes'
        elif not entry['benchmark']:
            reason = 'Unfrozen configuration comparison; the appendix uses the shared frozen matched-weight architecture experiment'
        elif any(n.startswith('ref_') for n in names):
            reason = 'Additional reference export; the chapter uses the shared reference screening and size sources rather than pooling separate exports'
        elif 'three-policies' in entry['directory'] or 'n20_randdom' in entry['directory']:
            reason = 'Legacy comparison mixes training sizes or objective families; retained summaries do not supply an isolated effect and are not added to the reference comparisons'
        else:
            reason = 'Additional legacy size/domain evaluation; the appendix retains representative matched-objective transfers, while this full outcome summary remains available for review'
        entry['reason'] = reason + '. Per-method outcomes remain in chapter_all_runs.csv.'
    write_csv(DATA/'chapter_run_audit.csv',audit)
    write_csv(DATA/'chapter_all_runs.csv',summaries)
    config_rows=[]
    for r in configs:
        label=str(r['algorithm'])+'/'+('EGNN' if r['backbone']=='Equivariant' else str(r['backbone']))
        if r['model'].startswith('ref_k10_s'): label+=' (seed '+r['model'].split('_')[2][1:]+')'
        if 'norig' in r['model']: label+=' (reduced)'
        action='S' if 'SelectNodes' in str(r['action']) else 'E'
        env=manifests[r['model']].get('environment_config_raw') or {}
        if env.get('skip_is_stop'): action+='+stop'
        objective=r'$\varepsilon$' if r['score']=='WeightedNormalizedShapeError' else (r'Trace' if env.get('spectral_functional')=='trace' else r'$\lambda$')
        steps=r['steps_completed_index']
        config_rows.append([label, str(r['n'])+(' random' if r['random_domains'] else ' fixed'),
                            tex_number(float(r['kappa']),0),objective,action,
                            tex_number(round((steps+1)/1000),0) if isinstance(steps,(int,float)) else '--'])
    chapter_table('policy_configurations',['Configuration','Formation',r'$\kappa_t$','Criterion','Actions','Steps (k)'],config_rows,'llrlrr')
    write_csv(DATA/'chapter_policy_configurations.csv',configs)
    write_csv(DATA/'chapter_protocols.csv',protocols)


def chapter_primary():
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                         'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
                         'pdf.fonttype':42})
    size={}; costs={};sources=[];metas={}
    methods=['s1','greedy','spectral','degree','constructive']
    for n,pre in SIZE_PREFIX.items():
        p,m,by=chapter_run(pre);sources.append(p);metas[n]=m
        for method,rs in by.items():size[n,method]=rs
        for method,rs in cost_by_method(p,m).items():costs[n,method]=rs
    p48,m48,b48=chapter_run('20260922-085849');require_shared(metas[48],m48)
    p50,m50,b50=chapter_run('20260923-004100');require_shared(metas[48],m50)
    size[48,'k50']=b50['k50']
    ns=list(SIZE_PREFIX);labs=[str(n) for n in ns]
    for name,metrics in [('chapter_size_success',['rigidity','attainment']),('chapter_size_quality',['ratio','shape'])]:
        chapter_grid(name,size,ns,labs,methods,metrics,'Size transfer at matched training/evaluation weight 10',sources,
                     'All rates use all instances. Error and edge ratio use rigid outputs.',numeric=True)
    shared_ids={r['episode'] for r in b48['greedy']};body=[]
    for method in methods:
        rs=b48['greedy'] if method=='greedy' else [r for r in size[48,method] if r['episode'] in shared_ids]
        body.append(chapter_outcome_row(SHORT[method],rs))
    chapter_table('shared48',['Method','Rigid','At bound',r'$\bar m$',r'$\varepsilon_{\rm geom}$'],body,'lrrrr')
    body=[]
    for method in methods:
        if (96,method) in size:body.append(chapter_outcome_row(SHORT[method],size[96,method]))
    chapter_table('stress96',['Method','Rigid','At bound',r'$\bar m$',r'$\varepsilon_{\rm geom}$'],body,'lrrrr')
    cp,cm,ref=chapter_run(SIZE_PREFIX[10])
    order=['greedy','spectral','degree','constructive','anneal','s0','s1','s2']
    chapter_table('reference',['Method','Rigid','At bound',r'$\bar m$',r'$\varepsilon_{\rm geom}$'],
                  [chapter_outcome_row(SHORT[x],ref[x]) for x in order],'lrrrr')
    def draw_ref(plain):
        fig,axs=plt.subplots(1,2,figsize=(7,3.0),squeeze=False)
        for ax,metric,title in zip(axs.flat,['edges','shape_dex'],['Edge-count difference','Error ratio']):
            for j,s in enumerate(['s0','s1','s2']):
                v,_=paired_values(ref[s],ref['greedy'],metric);e,l,h=paired_ci(v,123+j)
                if metric=='shape_dex':e,l,h=10**e,10**l,10**h
                ax.errorbar(e,j,xerr=[[e-l],[h-e]],fmt='o',color=COLORS[s],capsize=3)
            ax.set_yticks(range(3),['Seed 0','Seed 1','Seed 2']);ax.set_title(title)
            ax.set_xlabel('Policy minus greedy' if metric=='edges' else 'Policy / greedy')
            ax.axvline(0 if metric=='edges' else 1,c='#888888',ls=':');style(ax)
        chapter_layout(fig,axs,plain,'Paired 95% bootstrap intervals over 200 shared instances.',legend=False)
        return fig
    chapter_save('chapter_reference',draw_ref,'Paired edge and error differences for the three reference seeds',[cp])
    # Domain summaries use each frozen set independently.
    domain,ds=load_domain_data(find_domain_runs());CHAPTER_USED.update(ds.values())
    for group,name in [(HOMOGENEOUS,'homogeneous'),(MIXED,'mixed')]:
        keys=[x[0] for x in group];labels=[x[1] for x in group]
        if name=='mixed':labels=[r'$\mathbb{R}^2$ / SE(3)'+'\nrandom',r'$\mathbb{R}^2$ / SE(3)'+'\nbalanced',r'$\mathbb{R}^3$ /'+'\n'+r'$\mathbb{R}^2\times S^1$'+'\nrandom',r'$\mathbb{R}^3$ /'+'\n'+r'$\mathbb{R}^2\times S^1$'+'\nbalanced']
        for suffix,metrics in [('success',['rigidity','attainment']),('quality',['ratio','shape'])]:
            chapter_grid('chapter_'+name+'_'+suffix,domain,keys,labels,
                         ['s0','s1','s2','greedy'] if name=='mixed' else ['s1','greedy','spectral','degree'],
                         metrics,'Matched-objective domain transfer',list(ds.values()),
                         'Training and evaluation weight 10. N=100 per composition.')
    # Full-cost measurements on the same reference instances.
    cby=cost_by_method(cp,cm); body=[]
    for method in ['greedy','spectral','degree','constructive','anneal','s1']:
        rs=cby[method]
        body.append([SHORT[method]]+[tex_number(np.mean([number(r,k) or 0 for r in rs]),1)
                    for k in ['total','extended_bearing_rigidity_matrix','rigidity_decomposition','forward','ms']])
    chapter_table('cost',['Method','Calls',r'$B$ builds','SVDs','Forwards','ms'],body,'lrrrrr')
    chapter_grid('chapter_cost_scaling',costs,ns,labs,methods,['calls','ms'],
                 'Measured primitive-call scaling',sources,'Full search, without trajectory plotting.',numeric=True)
    # Compare size generalization across training weights under common selection.
    chapter_grid('chapter_weight_transfer',size,ns,labs,['k0','s1','k50','k200'],
                 ['rigidity','attainment','ratio','shape'],'Size transfer with different training weights',sources+[p50],
                 'Shared size benchmarks. Rigidity and attainment use all instances; quality uses rigid outputs.',numeric=True)
    return size,costs,ref,(cp,cm),domain,ds


def chapter_tradeoff():
    front=load_front();sources=[v[0] for v in front.values()];CHAPTER_USED.update(sources)
    body=[];matched={};effects=[]
    for k,(p,m,rs) in front.items():
        by=rows_by_method(p,m,rs);method={0:'k0',10:'s1',50:'k50',200:'k200'}[k]
        matched[k]=by,method
        for who in ['greedy',method]:body.append([f'${k}$', 'Greedy' if who=='greedy' else 'DQN']+chapter_outcome_row('',by[who])[1:])
        for metric in ['edges','shape_dex','attainment']:
            values,_=paired_values(by[method],by['greedy'],metric);e,l,h=paired_ci(values,47+k)
            effects.append({'kappa':k,'metric':metric,'N':len(values),'estimate':e,'low':l,'high':h})
    write_csv(DATA/'chapter_matched_effects.csv',effects)
    chapter_table('matched_weights',[r'$\kappa_t=\kappa_e$','Method','Rigid','At bound',r'$\bar m$',r'$\varepsilon_{\rm geom}$'],body,'rlrrrr')
    def draw(plain):
        fig,axs=plt.subplots(1,2,figsize=(7,3.7),squeeze=False)
        for ax,metric,title in zip(axs.flat,['edges','shape_dex'],['Edge-count difference','Error ratio']):
            for j,r in enumerate(x for x in effects if x['metric']==metric):
                e,l,h=r['estimate'],r['low'],r['high']
                if metric=='shape_dex':e,l,h=10**e,10**l,10**h
                ax.errorbar(e,j,xerr=[[e-l],[h-e]],fmt='o',color='#1f77b4',capsize=3)
            ax.set_yticks(range(4),[r'$\kappa=0$',r'$\kappa=10$',r'$\kappa=50$',r'$\kappa=200$'])
            ax.axvline(1 if metric=='shape_dex' else 0,c='#888888',ls=':');ax.set_title(title)
            ax.set_xlabel('Policy / greedy' if metric=='shape_dex' else 'Policy minus greedy');style(ax)
        chapter_layout(fig,axs,plain,'Matched objectives. Paired intervals over 100 shared instances.',legend=False)
        return fig
    chapter_save('chapter_matched_effects',draw,'Effect of learned versus greedy search at each matched objective',sources)
    p10,m10,b10=chapter_run(SIZE_PREFIX[24]);p50,m50,b50=chapter_run('20260923-005611');require_shared(m10,m50)
    body=[]
    for method in ['greedy','spectral','k0','s1','k50','k200']:
        for k,b in [(10,b10),(50,b50)]:body.append([SHORT[method],f'${k}$']+chapter_outcome_row('',b[method])[1:])
    chapter_table('evaluation_weights',['Method',r'$\kappa_e$','Rigid','At bound',r'$\bar m$',r'$\varepsilon_{\rm geom}$'],body,'lrrrrr')


def chapter_training():
    # A training curve uses actual event steps and a fixed physical-outcome metric.
    from tools.compare_runs import load as load_events
    seeds=['s0','s1','s2']
    binned=['Episode/ Best is rigid','Episode/ Best nr edges','Episode/ Best state score',
            'Episode/ Best shape err','Episode/ Best-final score gap','Episode/ Best step']
    train_series={};training_rows=[]
    for seed in seeds:
        name=next(k for k,v in MODELS.items() if v==seed);scalars,_=load_events(name,root=str(ROOT/'runs'))
        for tag in binned:
            if not scalars.get(tag):raise ValueError('Missing training series '+tag)
            # Average disjoint 10,000-transition bins; do not merge different training processes.
            groups={}
            for event in scalars[tag]:
                if np.isfinite(event.value):groups.setdefault(int(event.step)//10000,[]).append(event)
            # The error coefficient spans decades, so its bins use the geometric mean.
            average=(lambda v:float(np.exp(np.mean(np.log(v))))) if tag=='Episode/ Best shape err' else np.mean
            points=[(np.mean([e.step for e in es]),average([e.value for e in es])) for _,es in sorted(groups.items())]
            train_series[seed,tag]=points
            training_rows += [{'seed':seed,'metric':tag,'step':x,'mean':y} for x,y in points]
    write_csv(DATA/'chapter_training_curves.csv',training_rows)
    panels=[('Episode/ Best is rigid','Rigid best graphs','% of episodes',100),
            ('Episode/ Best nr edges','Best-graph edge count','Mean edges',1),
            ('Episode/ Best state score','Best-graph objective score',r'Mean $\varphi$',1),
            ('Episode/ Best shape err','Best-graph error coefficient',r'Geometric mean $\varepsilon$',1),
            ('Episode/ Best-final score gap','Best-final score gap',r'Mean $\varphi_{\rm best}-\varphi_{\rm final}$',1),
            ('Episode/ Best step','Step of the best graph','Mean step within episode',1)]
    def draw_training(plain):
        fig,axs=plt.subplots(3,2,figsize=(7,9.5),squeeze=False)
        for ax,(tag,title,ylabel,scale) in zip(axs.flat,panels):
            for seed in seeds:
                xy=np.array(train_series[seed,tag]);ax.plot(xy[:,0]/1000,xy[:,1]*scale,label=SHORT[seed],color=COLORS[seed])
            ax.set_xlabel('Training transitions (thousands)');ax.set_title(title);ax.set_ylabel(ylabel);style(ax)
        chapter_layout(fig,axs,plain,'Disjoint 10,000-transition bins; exploratory training episodes.',top=.95)
        return fig
    chapter_save('chapter_training',draw_training,'Learning progress for the three matched reference seeds',[ROOT/'runs'/('ref_k10_'+s+'_dqn_gine') for s in seeds])


def chapter_exploration():
    # One curve per epsilon schedule; every run in a group must have logged the same values.
    from tools.compare_runs import load as load_events
    groups=[('400k steps, final 0.02',['ref_k0_dqn_gine','ref_k10_s0_dqn_gine','ref_k10_s1_dqn_gine','ref_k10_s2_dqn_gine',
                                        'ref_k200_dqn_gine','ref_select_dqn_gine','ref_stop_dqn_gine','ref_norig_dqn_gine']),
            ('400k steps, final 0.05',['estimation_randdom_k50_dqn_gine','estimation_randdom_k50_dqn_equi']),
            ('250k steps, final 0.05',['estimation_randdom_dqn_gine','estimation_stop_dqn_gine','estimation_stop_k50_dqn_gine',
                                        'estimation_stop_k200_dqn_gine','stiff_dqn_gine','estimation_dqn_gine','estimation_dqn_equi',
                                        'estimation_trace_dqn_gine','estimation_k10_dqn_gine']),
            ('150k steps, final 0.02',['ref_n16_k10_dqn_gine'])]
    curves={};rows=[]
    for label,names in groups:
        logged=[]
        for name in names:
            scalars,_=load_events(name,root=str(ROOT/'runs'))
            logged.append(np.array([(e.step,e.value) for e in scalars['Exploration / Exploration epsilon']]))
        for other,name in zip(logged[1:],names[1:]):
            if other.shape!=logged[0].shape or not np.allclose(other,logged[0]):raise ValueError('Epsilon schedule differs: '+name)
        # skrl takes uniformly random actions before random_timesteps and logs epsilon only afterwards.
        random_steps=json.loads((ROOT/'train'/(names[0]+'.json')).read_text())['hyperparameters']['random_timesteps']
        curves[label]=np.vstack([[0,1],[random_steps,1],logged[0]])
        rows += [{'schedule':label,'runs':'|'.join(names),'step':x,'epsilon':y} for x,y in curves[label]]
    write_csv(DATA/'chapter_exploration.csv',rows)
    def draw(plain):
        fig,axs=plt.subplots(1,1,figsize=(7,3.3),squeeze=False);ax=axs[0,0]
        for label,_ in groups:
            ax.plot(curves[label][:,0]/1000,curves[label][:,1],label=label)
        ax.set_xlabel('Training transitions (thousands)');ax.set_ylabel(r'$\varepsilon$');ax.set_title('Probability of a random action')
        ax.set_ylim(0,1.03);ax.legend(fontsize=8);style(ax)
        chapter_layout(fig,axs,plain,'Uniformly random actions for the first 10,000 transitions, then logged epsilon per write interval.',legend=False)
        return fig
    chapter_save('chapter_exploration',draw,'Epsilon-greedy exploration schedules of the DQN policies',
                 [ROOT/'runs'/n for _,names in groups for n in names])


def chapter_ablation_figures():
    # Perturbation results are already paired episode aggregates, without episode-level intervals.
    abr=[]
    for k,pre in [(0,'20260912-183108'),(200,'20260912-183357')]:
        path=find_run(pre);CHAPTER_USED.add(path);raw=json.loads((path/'ablation_raw.json').read_text())['learned']
        for mode,rec in raw.items():
            for r in rec['rows']:abr.append({'kappa':k,'mode':mode,**r})
    write_csv(DATA/'chapter_ablation.csv',abr)
    for k in [0,200]:
        def draw(plain,k=k):
            channels=[r['channel'] for r in abr if r['kappa']==k and r['mode']=='shuffle']
            fig,axs=plt.subplots(1,2,figsize=(7,6.7),squeeze=False)
            for ax,field,title in zip(axs.flat,['d_phi','flip'],['Score loss','Changed actions']):
                for j,mode in enumerate(['shuffle','zero','noise']):
                    rs={r['channel']:r for r in abr if r['kappa']==k and r['mode']==mode}
                    ax.plot([rs[c][field]*(100 if field=='flip' else 1) if rs[c].get('perturbed',0)>=.01 else np.nan for c in channels],np.arange(len(channels))+(j-1)*.20,
                            marker=['o','s','^'][j],ms=4,ls='none',label=mode.capitalize())
                ax.set_yticks(range(len(channels)),[{'rigidity_glob':'Global rigidity', 'quality':'Global quality', 'node_freedom':'Node freedom', 'node_slack':'Node slack', 'common_nbrs':'Common neighbors', 'pair_max_rank':'Pair block rank', 'add_rank':'Addition rank gain', 'remove_rank':'Removal rank loss', 'add_stiffness':'Addition conditioning', 'remove_stiffness':'Removal conditioning', 'add_independence':'Addition independence', 'coord_features':'Coordinates', 'adj':'Adjacency', 'edge_exists':'Edge presence'}.get(c,c.capitalize()) for c in channels],fontsize=8.5)
                # Alternate row bands and a line per row tie the three offset markers to their channel.
                for i in range(len(channels)):
                    if i%2==0:ax.axhspan(i-.5,i+.5,color='#f0f0f0',zorder=0,lw=0)
                    ax.axhline(i,color='#d8d8d8',lw=.6,zorder=0)
                ax.set_ylim(len(channels)-.5,-.5);ax.axvline(0,c='#888888',ls=':');ax.set_title(title)
                if field == 'flip': ax.set_yticklabels([])
                else:
                    ax.set_xscale('symlog',linthresh=1)
                    ax.margins(x=.10)
                ax.set_xlabel(r'$\phi_{\rm original}-\phi_{\rm perturbed}$' if field=='d_phi' else '% of reference decisions');style(ax)
            chapter_layout(fig,axs,plain,f'Training and evaluation weight {k}; 20 episodes. No-op perturbations are omitted.',top=.95)
            return fig
        chapter_save('chapter_ablation_'+str(k),draw,'Within-policy feature sensitivity under matched objective',[find_run('20260912-183108' if k==0 else '20260912-183357')])


def chapter_behavior():
    p,m,b=chapter_run('20260912-162935');pn,mn,bn=chapter_run('20260912-163456');ps,ms,bs=chapter_run('20260912-164003')
    require_shared(m,mn);require_shared(m,ms)
    norig=next(rs for k,rs in bn.items() if k not in ['greedy','spectral','degree','initial','constructive','random','anneal'])
    body=[chapter_outcome_row('Greedy',b['greedy'])]
    body += [chapter_outcome_row(SHORT[k],b[k]) for k in ['s0','s1','s2']]
    body += [chapter_outcome_row('Without rigidity features',norig)]
    chapter_table('observation_removal',['Method','Rigid','At bound',r'$\bar m$',r'$\varepsilon_{\rm geom}$'],body,'lrrrr')
    chapter_table('action_space',['Method','Rigid','At bound',r'$\bar m$',r'$\varepsilon_{\rm geom}$'],
                  [chapter_outcome_row('Greedy',b['greedy']),chapter_outcome_row('Direct edges, seed 0',b['s0']),chapter_outcome_row('Sequential nodes',bs['select'])],'lrrrr')
    effects=paired_effect_records({'norig':norig,'s0':b['s0'],'select':bs['select'],'greedy':b['greedy']},'features_actions',m['benchmark'],['s0','greedy'])
    write_csv(DATA/'chapter_feature_action_effects.csv',effects)
    chapter_ablation_figures()
    chapter_training()
    chapter_exploration()
    # Architecture has an objective-matched experiment as well as a larger transfer set.
    pa,ma,ba=chapter_run('20260912-170134');pc,mc,bc=chapter_run(SIZE_PREFIX[10])
    model_names=['estimation_randdom_k50_dqn_gine','estimation_randdom_k50_dqn_equi','estimation_randdom_k50_ppo_gine','estimation_randdom_k50_ppo_equi']
    names=['DQN/GINE','DQN/EGNN','PPO/GINE','PPO/EGNN'];body=[]
    for k,bx in [(50,ba),(10,bc)]:
        body.append([f'${k}$','Greedy']+chapter_outcome_row('',bx['greedy'])[1:])
        for name,lab in zip(model_names,names):body.append([f'${k}$',lab]+chapter_outcome_row('',bx[MODELS.get(name,name)])[1:])
    chapter_table('architecture',[r'$\kappa_e$','Configuration','Rigid','At bound',r'$\bar m$',r'$\varepsilon_{\rm geom}$'],body,'rlrrrr')
    # Per-episode decision summaries avoid treating correlated edits as replicates.
    dec=read_csv(p/'decisions.csv');body=[];out=[]
    for seed in ['s0','s1','s2']:
        rs=[r for r in dec if r['model']==seed];valid=[r for r in rs if number(r,'err_pct') is not None]
        for field in ['phi_pct','err_pct']:
            ep={}
            for r in rs:
                value=number(r,field)
                if value is not None:ep.setdefault(r['episode'],[]).append(value)
            vals=np.array([np.mean(v) for v in ep.values()]);e,l,h=paired_ci(vals,98)
            out.append({'seed':seed,'metric':field,'eligible_decisions':sum(map(len,ep.values())), 'episodes':len(ep),'mean_episode_percentile':e,'low':l,'high':h})
        body.append([seed.replace('s','Seed '),f'${len(rs)}$',f'${len(valid)}$',tex_number(np.median([number(r,'phi_pct') for r in rs]),1),tex_number(np.median([number(r,'err_pct') for r in valid]),1)])
    write_csv(DATA/'chapter_decision_metrics.csv',out)
    chapter_table('decisions',['Policy','Score edits','Error edits','Score median','Error median'],body,'lrrrr')


def chapter_graphs(ref,cp):
    import networkx as nx
    nets,_=benchmark.read_networks(cp/'instances.npz');episode=0;n=nets[episode].n
    net=nets[episode];domains=[a.domain for a in net.agents]
    domain_order=['R^2','R^3','R^2xS^1','R^3xS^1','SE(3)']
    colors=['#56b4e9','#0072b2','#e69f00','#cc79a7','#009e73']
    methods=['greedy','spectral','degree','s1'];pos=nx.circular_layout(range(n))
    def draw(plain):
        fig,axs=plt.subplots(2,2,figsize=(7,7.0),squeeze=False)
        for ax,method in zip(axs.flat,methods):
            row=next(r for r in ref[method] if int(r['episode'])==episode);G=nx.DiGraph();G.add_nodes_from(range(n))
            G.add_edges_from(tuple(map(int,x.split('>'))) for x in row['edges'].split())
            nx.draw_networkx_edges(G,pos,ax=ax,arrows=True,arrowsize=11,width=.9,alpha=.65,
                                   connectionstyle='arc3,rad=0.07',node_size=200)
            for d,c in zip(domain_order,colors):
                ids=[i for i in range(n) if domains[i]==d]
                if ids:nx.draw_networkx_nodes(G,pos,nodelist=ids,node_color=c,node_size=200,edgecolors='white',ax=ax)
            nx.draw_networkx_labels(G,pos,font_size=9,font_color='black',ax=ax)
            ax.set_title(SHORT[method]);ax.set_aspect('equal');ax.set_axis_off();ax.margins(.12)
        from matplotlib.lines import Line2D
        fig.legend([Line2D([],[],marker='o',ls='none',color=c) for c in colors],[r'$\mathbb{R}^2$', r'$\mathbb{R}^3$', r'$\mathbb{R}^2\times S^1$', r'$\mathbb{R}^3\times S^1$', r'$\mathrm{SE}(3)$'],ncol=5,loc='upper center',frameon=False)
        fig.tight_layout(rect=(0,.025 if not plain else 0,1,.95))
        if not plain:fig.text(.5,.003,'First saved benchmark instance. Circular layout preserves node identities, not spatial positions.',ha='center',fontsize=8)
        return fig
    chapter_save('chapter_topologies',draw,'Directed output graphs on a prespecified shared instance',[cp])
    body=[chapter_outcome_row(SHORT[m],[next(r for r in ref[m] if int(r['episode'])==episode)]) for m in methods]
    chapter_table('topology_case',['Method','Rigid','At bound',r'$m$',r'$\varepsilon$'],body,'lrrrr')
    structure=read_csv(DATA/'resultant_graph_metrics.csv');body=[]
    for method in ['greedy','spectral','degree','s0','s1','s2']:
        rs=[r for r in structure if r['method']==method]
        vals=[np.mean([number(r,k) for r in rs]) for k in ['reciprocity','reciprocity_null','rank2_share','degree_floor_share','mean_edge_length_radius']]
        body.append([SHORT[method]]+[tex_number(v*(100 if i<4 else 1),1 if i<4 else 2) for i,v in enumerate(vals)])
    chapter_table('structure',['Method','Recip.','Uniform','Rank two','Degree floor','Length'],body,'lrrrrr')


def chapter_control():
    p=find_run('20260912-194452');CHAPTER_USED.add(p);rows=read_csv(p/'control.csv');traj=read_csv(p/'trajectories.csv')
    methods=['greedy','complete','k0','k10','k50','k200'];COLORS.update({'complete':'#777777','k10':COLORS['s0']})
    body=[]
    for method in methods:
        rs=[r for r in rows if r['method']==method];body.append([SHORT[method],f'${sum(truth(r["reached"]) for r in rs)}/{len(rs)}$']+
                     [('$'+str(int(float(r['steps_to_tol'])))+'$') if number(r,'steps_to_tol') is not None else '--' for r in rs])
    chapter_table('control',['Method','Reached']+[f'Case {i+1}' for i in range(5)],body,'lrrrrrr')
    for name,which in [('chapter_control',[0]),('chapter_control_all',list(range(5))),('chapter_control_attitude_all',list(range(5)))]:
        def draw(plain,which=which,name=name):
            if len(which)==1:
                fig,axs=plt.subplots(3,1,figsize=(7,7.2),squeeze=False);panels=[(0,'bearing_rms'),(0,'pos_err'),(0,'att_err')]
            else:
                fig,axs=plt.subplots(3,2,figsize=(7,8),squeeze=False);panels=[(i,'att_err' if name=='chapter_control_attitude_all' else 'bearing_rms') for i in which];axs[-1,-1].set_visible(False)
            for ax,(case,field) in zip(axs.flat,panels):
                for method in methods:
                    rs=[r for r in traj if int(r['case'])==case and r['method']==method]
                    ax.plot([float(r['t']) for r in rs],[float(r[field]) for r in rs],color=COLORS.get(method),label=SHORT[method],lw=1.1)
                if field!='att_err':ax.set_yscale('log')
                else:ax.set_ylim(bottom=0)
                ax.set_xlabel('Simulation time');ax.set_title({'bearing_rms':'Bearing error','pos_err':'Position error','att_err':'Attitude error'}[field]+f', case {case+1}')
                ax.set_ylabel({'bearing_rms':'RMS per edge','pos_err':'Position RMS / radius','att_err':'Attitude RMS (rad)'}[field])
                if len(which)==1 or field=='att_err':
                    ax.set_xscale('symlog',linthresh=1);ax.set_xticks([0,1,10,100],['0','1','10','100']);ax.set_xlim(0,200)
                if field=='bearing_rms':ax.axhline(.001,c='#888888',ls=':',lw=.8)
                style(ax)
            chapter_layout(fig,axs,plain,'Same goal and start within each case. Attitude is in radians.' if name=='chapter_control_attitude_all' else 'Same goal and start within each case. Dotted line is the bearing tolerance.')
            return fig
        chapter_save(name,draw,'Control trajectories with shared initial and goal formations',[p])


def chapter_stopping_details():
    outcomes=[];timing=[];summaries=[];episodes=[]
    for n,pre in [(10,'20260922-124632'),(16,'20260922-125417'),(24,'20260922-130947')]:
        path,meta,by=chapter_run(pre);horizon=int(meta['rollout_steps'])
        for method,rows in by.items():
            N=len(rows);late=[];redundant=[];final_best=lost=early=0;late_horizon=0
            for row in rows:
                work=int(row['work']);best_at=int(row['best_at']);gap=float(row['best_final_gap'])
                assert 0<=best_at<=work<=horizon
                assert np.isclose(float(row['best_score'])-float(row['score']),gap,atol=1e-8,rtol=0)
                assert gap>=-1e-8
                matched=abs(gap)<=1e-8;delay=work-best_at;redundant.append(delay)
                final_best+=matched;lost+=not matched;early+=work<horizon
                if matched and delay>0:
                    late.append(delay);late_horizon+=work==horizon
                episodes.append({'source':str(path.relative_to(ROOT)),'n':n,'method':method,'episode':row['episode'],
                                 'work':work,'best_at':best_at,'best_final_gap':gap,'final_best_score':matched,
                                 'edits_after_first_best':delay,'later_matching_final':matched and delay>0,
                                 'at_horizon':work==horizon})
            policy=method.replace('stop_k',r'$\kappa_t=')+'$'
            outcomes.append([f'${n}$',policy,f'${early}/{N}$',f'${len(rigid_rows(rows))}/{N}$',
                             tex_number(summarize(rows,'edges')[0]),tex_number(summarize(rows,'shape')[0]),
                             tex_number(np.mean([int(r['work']) for r in rows]),1)])
            # Edits after the best score was first reached, over every episode.
            timing.append([f'${n}$',policy,f'${lost}/{N}$',f'${final_best}/{N}$',f'${len(late)}/{N}$',tex_number(np.mean(redundant),1)])
            summaries.append({'n':n,'method':method,'N':N,'early':early,'rigid':len(rigid_rows(rows)),
                              'lost_best':lost,'final_best_score':final_best,'later_matching_final':len(late),
                              'later_at_horizon':late_horizon,'delay_median':np.median(late) if late else None,
                              'delay_min':min(late) if late else None,'delay_max':max(late) if late else None,
                              'horizon':horizon,'mean_steps':np.mean([int(r['work']) for r in rows]),
                              'redundant_mean':np.mean(redundant)})
    chapter_table('stopping',[r'$n$','Policy','Early','Rigid',r'$\bar m$',r'$\varepsilon$','Edits'],outcomes,'rlrrrrr')
    top=(TABLES/'stopping.tex').read_text()
    chapter_table('stopping',[r'$n$','Policy','Lost best','Final best','Later best','Redundant edits'],timing,'rlrrrr')
    bottom=(TABLES/'stopping.tex').read_text()
    (TABLES/'stopping.tex').write_text(r'\textbf{Final outcomes}\par\smallskip'+'\n'+top+
                                    r'\par\medskip\textbf{Best-score timing}\par\smallskip'+'\n'+bottom)
    write_csv(DATA/'chapter_stopping.csv',summaries)
    write_csv(DATA/'chapter_stopping_episodes.csv',episodes)
    print('Later best-score endings:',sum(r['later_matching_final'] for r in summaries),
          'at horizon:',sum(r['later_at_horizon'] for r in summaries),flush=True)


def chapter_budget_figures():
    # Direct budget observations, not a proportional estimate of time to best.
    p=find_run('20260913-043413');CHAPTER_USED.add(p);raw=read_csv(p/'budget.csv');data={}
    budgets=sorted({int(r['budget']) for r in raw})
    for r in raw:
        row=dict(r,is_IBR=r['is_rigid']);data.setdefault((int(r['budget']),r['method']),[]).append(row)
    chapter_grid('chapter_budget',data,budgets,[str(x) for x in budgets],['greedy','k10'],
                 ['rigidity','shape'],'Observed best graph at each edit budget, matched weight 10',[p],
                 '20 shared instances. A greedy edit evaluates all toggles; a policy edit does not.',numeric=True)
    chapter_grid('chapter_budget_transfer',data,budgets,[str(x) for x in budgets],['k0','k10','k50','k200'],
                 ['rigidity','edges','shape','final'],'Search budget under a common evaluation weight 10',[p],
                 'Training weights differ. Rigidity and final-score fractions use all instances.',numeric=True)


def chapter_training_size():
    # Three policies that differ only in training size (150000 steps, kappa_t=10), on every size benchmark.
    runs=['20260930-034849','20260930-035236','20260930-035441','20260930-040035','20260930-040618',
          '20260930-041637','20260930-041734','20260930-042017']
    methods=['n10_150k','n16','n20_150k'];td={};sources=[]
    for pre in runs:
        p,m,b=chapter_run(pre);n=int(m['n']);sources.append(p)
        for method in methods:
            if method in b:
                if (n,method) in td:raise ValueError(f'two evaluations of {method} at n={n}')
                td[n,method]=b[method]
    SHORT.update({'n10_150k':r'Trained at $n=10$','n16':r'Trained at $n=16$','n20_150k':r'Trained at $n=20$'})
    sizes=[8,10,16,24,32,48]
    chapter_grid('chapter_training_size',td,sizes,[str(x) for x in sizes],methods,['rigidity','shape'],
                 'Transfer across training sizes at equal training length',sources,
                 'Evaluation weight 10. All policies trained with weight 10 for 150000 steps.',numeric=True)


def chapter_supplement(size,costs):
    chapter_budget_figures()
    chapter_stopping_details()
    chapter_training_size()
    # Repair success and count are reported alongside selection-conditioned percentiles.
    p=find_run('20260912-185118');CHAPTER_USED.add(p)
    from repair import _headline
    rs=_headline(read_csv(p/'repair.csv'));body=[];rep=[]
    methods=['greedy_repair','greedy','k0','k10','k50','k200']
    names={'greedy_repair':'Rank repair','greedy':'Greedy','k0':r'DQN $0$','k10':r'DQN $10$','k50':r'DQN $50$','k200':r'DQN $200$'}
    for event,lab in [('edge_drop_random','Edge drop'),('node_leave','Node leave'),('node_join','Node join')]:
        for method in methods:
            group=[r for r in rs if r.get('event')==event and r.get('method')==method and not truth(r.get('skipped'))]
            vals=[number(r,'pctile') for r in group if number(r,'pctile') is not None]
            body.append([lab,names[method],f'${sum(truth(r["repaired"]) for r in group)}/{len(group)}$',
                         tex_number(np.mean([number(r,'added') for r in group]),1),tex_number(np.mean([number(r,'removed') for r in group]),1),f'${len(vals)}/{len(group)}$',tex_number(np.mean(vals) if vals else np.nan,1)])
            rep.append({'event':event,'method':method,'cases':len(group),'repaired':sum(truth(r['repaired']) for r in group),
                        'eligible':len(vals),'percentile':np.mean(vals) if vals else np.nan})
    chapter_table('repair',['Event','Method','Rigid','Added','Removed','Scored','Percentile'],body,'llrrrrr')
    body=[]
    for event,lab in [('none','None'),('edge_drop_redundant','Redundant drop'),('edge_add_spurious','Spurious edges')]:
        for method in ['greedy','k0','k10','k50','k200']:
            group=[r for r in rs if r.get('event')==event and r.get('method')==method and not truth(r.get('skipped'))]
            if not group:continue
            vals=[number(r,'err_ratio_to_pre') for r in group if number(r,'err_ratio_to_pre') and number(r,'err_ratio_to_pre')>0]
            body.append([lab,names[method],f'${len(group)}$',tex_number(np.mean([number(r,'churn') for r in group]),2),
                         tex_number(np.exp(np.mean(np.log(vals))) if vals else np.nan)])
    chapter_table('repair_nonbreaking',['Event','Method','Cases','Changed edges', 'Error / before'],body,'llrrr')
    write_csv(DATA/'chapter_repair.csv',rep)
    p=find_run('20260912-162935');noise=read_csv(p/'noise.csv');out=[]
    for method in ['greedy','spectral','degree','s0','s1','s2']:
        for sigma in [.1,.5,1,2,5]:
            group=[r for r in noise if r['method']==method and float(r['sigma_deg'])==sigma]
            v=np.array([math.log(float(r['measured'])/float(r['predicted'])) for r in group if number(r,'measured') and number(r,'predicted') and number(r,'measured')>0])
            e,l,h=paired_ci(v,93);out.append({'method':method,'sigma_deg':sigma,'instances':len(group),'valid':len(v),
                                            'failed_trials':sum(round(100 * (number(r,'failed') or 0)) for r in group),'ratio':math.exp(e),'low':math.exp(l),'high':math.exp(h)})
    write_csv(DATA/'chapter_noise.csv',out)
    def draw_noise(plain):
        fig,axs=plt.subplots(1,2,figsize=(7,3.7),squeeze=False)
        for ax,methods in zip(axs.flat,[['greedy','spectral','degree'],['s0','s1','s2']]):
            for j,method in enumerate(methods):
                rr=[r for r in out if r['method']==method];x=np.arange(len(rr))+(j-1)*.07;y=np.array([r['ratio'] for r in rr])
                ax.errorbar(x,y,yerr=[y-np.array([r['low'] for r in rr]),np.array([r['high'] for r in rr])-y],marker=MARKERS.get(method,'o'),label=SHORT[method],color=COLORS[method],capsize=2,ms=4)
            ax.set_xticks(range(5),['0.1','0.5','1','2','5']);ax.set_xlabel('Bearing noise (degrees)');ax.set_ylabel('Measured / predicted position RMS')
            ax.set_yscale('log');ax.axhline(1,c='#888888',ls=':');ax.legend(fontsize=8);style(ax)
        chapter_layout(fig,axs,plain,'Geometric mean and 95% bootstrap intervals over 20 formations.',legend=False)
        return fig
    chapter_save('chapter_noise',draw_noise,'Empirical position-error prediction under bearing noise',[p])
    # Older objectives remain separate from the reference-family evaluations.
    older=[('20260831-160651','DQN/GINE','Eigenvalue',2),('20260831-161827','PPO/GINE, 200k','Eigenvalue',2),
           ('20260831-163049','PPO/GINE, 225k','Eigenvalue',2)]
    body=[[r'\multicolumn{6}{l}{\emph{Earlier eigenvalue objective}}']]
    for pre,lab,obj,k in older:
        p,m,b=chapter_run(pre)
        for method in (['greedy','learned'] if pre=='20260831-160651' else ['learned']):
            body.append([lab if method=='learned' else 'Greedy',obj,f'${k}$']+chapter_outcome_row('',b[method])[1:2]+chapter_outcome_row('',b[method])[3:])
    body.append([r'\multicolumn{6}{l}{\emph{Matched spectral objectives}}'])
    for pre,lab,obj,k in [('20260830-083704','DQN/GINE','Eigenvalue',2),
                          ('20260830-212646','DQN/GINE','Trace',2),
                          ('20260831-025344','DQN/GINE','Eigenvalue',10)]:
        po,mo,bo=chapter_run(pre)
        for method in ['greedy','learned']:
            body.append([lab if method=='learned' else 'Greedy',obj,f'${k}$']+chapter_outcome_row('',bo[method])[1:2]+chapter_outcome_row('',bo[method])[3:])
    body.append([r'\multicolumn{6}{l}{\emph{Common eigenvalue evaluation}}'])
    p,m,b=chapter_run('20260903-153205')
    for name,lab in [('estimation_dqn_gine','DQN/GINE'),('estimation_dqn_equi','DQN/EGNN'),('estimation_trace_dqn_gine','DQN/GINE, trace'),('estimation_k10_dqn_gine','DQN/GINE, weight 10')]:
        method=MODELS.get(name,name);body.append([lab,'Eigenvalue',r'$2$']+chapter_outcome_row('',b[method])[1:2]+chapter_outcome_row('',b[method])[3:])
    chapter_table('older_objectives',['Policy','Evaluation',r'$\kappa_e$','Rigid',r'$\bar m$',r'$\varepsilon$'],body,'llrrrr')
    # Selected objective-matched legacy transfer sets include their own greedy baseline.
    body=[]
    for pre,lab in [('20260831-225011','Mixed, 5'),('20260831-225421','Random, 6'),('20260831-230434',r'$\mathbb R^2\times S^1$, 8'),
                    ('20260831-230906',r'$\mathbb R^3$, rotated, 8'),('20260831-231812',r'$\mathbb R^3\times S^1$, 8'),
                    ('20260831-233045',r'$\mathrm{SE}(3)$, 8'),('20260831-234540','Random, 12'),('20260831-235911','Random, 16'),
                    ('20260901-000800',r'$\mathbb R^3$, 16'),('20260831-205345',r'$\mathbb R^3$, 50')]:
        p,m,b=chapter_run(pre)
        for method in ['greedy','learned']:
            if method not in b:continue
            body.append([lab,'Greedy' if method=='greedy' else 'DQN']+chapter_outcome_row('',b[method])[1:2]+chapter_outcome_row('',b[method])[3:])
    chapter_table('older_transfer',['Formation','Method','Rigid',r'$\bar m$',r'$\varepsilon$'],body,'llrrr')


def chapter_visual_additions():
    """Geometric views and complementary physical-outcome plots from saved results."""
    import copy
    from matplotlib.lines import Line2D
    from rigidity import scaled_rigidity_matrix, error_covariance, characteristic_length
    from report import _ellipsoid
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':9,'legend.fontsize':9})
    cp,cm,ref=chapter_run(SIZE_PREFIX[10]);nets,_=benchmark.read_networks(cp/'instances.npz');net=nets[0]
    P=np.array([a.pose.position for a in net.agents]);L=characteristic_length(net);P=(P-P.mean(0))/L
    domains=['R^2','R^3','R^2xS^1','R^3xS^1','SE(3)'];colors=['#56b4e9','#0072b2','#e69f00','#cc79a7','#009e73']
    labels=[r'$\mathbb{R}^2$',r'$\mathbb{R}^3$',r'$\mathbb{R}^2\times S^1$',r'$\mathbb{R}^3\times S^1$',r'$\mathrm{SE}(3)$']
    nodecolors=[colors[domains.index(a.domain)] for a in net.agents]
    methods=['greedy','spectral','degree','s1'];geometry={};out=[]
    rank_K=np.linalg.matrix_rank(extended_bearing_rigidity_matrix(net.fully_connected()))
    for method in methods+['s0','s2']:
        row=next(r for r in ref[method] if int(r['episode'])==0)
        edges=[tuple(map(int,x.split('>'))) for x in row['edges'].split()]
        work=copy.deepcopy(net);work.edges[:]=False
        for i,j in edges:work.edges[i,j]=True
        B=scaled_rigidity_matrix(work);C=error_covariance(B,rank_K);w,V=np.linalg.eigh(B.T@B)
        mode=V[:,len(w)-rank_K][:3*net.n].reshape(net.n,3)
        cov=[C[3*i:3*i+3,3*i:3*i+3] for i in range(net.n)]
        geometry[method]=(edges,cov,mode)
        out.append({'method':method,'episode':0,'radius':L,'smallest_observable_eigenvalue':w[len(w)-rank_K],
                    'position_trace':np.trace(C[:3*net.n,:3*net.n])})
    maxsd=max(np.sqrt(max(np.linalg.eigvalsh(c).max(),0)) for _,cs,_ in geometry.values() for c in cs)
    shell_scale=.7/maxsd
    covariance_bounds=np.vstack([P+sign*shell_scale*np.sqrt(np.maximum(np.array([np.diag(C) for C in cs]),0))
                                for _,cs,_ in geometry.values() for sign in [-1,1]])
    for r in out:r['ellipsoid_display_factor']=shell_scale
    write_csv(DATA/'chapter_geometry.csv',out)
    def axes3(ax,points):
        lo=points.min(0);hi=points.max(0);center=(lo+hi)/2;span=max(hi-lo)*.64
        ax.set(xlim=(center[0]-span,center[0]+span),ylim=(center[1]-span,center[1]+span),zlim=(center[2]-span,center[2]+span))
        ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-20)
        ax.set_xlabel('x',labelpad=-2);ax.set_ylabel('y',labelpad=-2);ax.set_zlabel('z',labelpad=-2)
        ax.tick_params(labelsize=7,pad=0)
        from matplotlib.ticker import MaxNLocator
        for a in [ax.xaxis,ax.yaxis,ax.zaxis]:a.set_major_locator(MaxNLocator(3))
    def tight_geometry_axes(ax,points):
        lo=points.min(0);hi=points.max(0);width=hi-lo
        padding=np.maximum(.10*width,.08)
        for k,setter in enumerate([ax.set_xlim,ax.set_ylim,ax.set_zlim]):
            setter(lo[k]-padding[k],hi[k]+padding[k])
        ax.set_box_aspect(width+2*padding,zoom=1.08)
        ax.view_init(elev=25,azim=40)

    def shaded_edges(ax,edges):
        for i,j in edges:
            ax.quiver(*P[i],*((P[j]-P[i])*.94),color='#8c969e',alpha=.35,
                      linewidth=.7,arrow_length_ratio=.10,zorder=1)
    def draw_formation(plain):
        fig=plt.figure(figsize=(7,8.2))
        grid=fig.add_gridspec(3,4)
        axiscolors=['#d62728','#228833','#3366cc']
        for j,method in enumerate(methods+['frames']):
            slot=grid[j//2,2*(j%2):2*(j%2)+2] if j<4 else grid[2,1:3]
            ax=fig.add_subplot(slot,projection='3d');axes3(ax,P)
            frame_points=np.vstack([P]+[P[i]+.23*agent.pose.rotation_mat()[:,k]
                                      for i,agent in enumerate(net.agents)
                                      if agent.domain not in ['R^2','R^3'] for k in range(3)])
            tight_geometry_axes(ax,frame_points)
            ax.set_title(SHORT[method] if method!='frames' else 'Agent orientations',pad=8)
            ax.scatter(*P.T,c=nodecolors,s=52,depthshade=False,edgecolors='white',linewidths=.6)
            from mpl_toolkits.mplot3d import proj3d
            offsets=[(9,0),(8,6),(0,10),(-10,-5),(-9,8),(0,-11),(9,8),(-9,-8),(-10,8),(-10,-7)]
            for i,point in enumerate(P):
                x,y,_=proj3d.proj_transform(*point,ax.get_proj())
                if method=='frames' and i==2:offsets[i]=(10,10)
                ax.annotate(str(i),(x,y),xytext=offsets[i],textcoords='offset points',
                            ha='center',va='center',fontsize=9,annotation_clip=False)
            if method!='frames':
                for i,k in geometry[method][0]:
                    delta=P[k]-P[i]
                    ax.quiver(*(P[i]+.04*delta),*(.89*delta),color='#444444',alpha=.8,
                              linewidth=1.0,arrow_length_ratio=.13)
            else:
                for point,agent in zip(P,net.agents):
                    if agent.domain in ['R^2','R^3']:continue
                    R=agent.pose.rotation_mat()
                    for k,color in enumerate(axiscolors):
                        ax.quiver(*point,*(.23*R[:,k]),color=color,linewidth=1.8,arrow_length_ratio=.25)
            for artist in ax.collections:artist.set_clip_on(False)
        fig.legend([Line2D([],[],marker='o',ls='none',color=c) for c in colors],labels,
                   ncol=5,loc='upper center',frameon=False)
        fig.legend([Line2D([],[],color=c,lw=2) for c in axiscolors],
                   [r'Local $x_i$',r'Local $y_i$',r'Local $z_i$'],ncol=1,
                   loc='center',bbox_to_anchor=(.86,.16),frameon=False)
        fig.subplots_adjust(left=.035,right=.95,bottom=.05,top=.925,wspace=.20,hspace=.25)
        if not plain:fig.text(.5,.003,'First reference instance; matched weight 10. Shared poses in formation radii.',ha='center',fontsize=8)
        return fig
    chapter_save('chapter_formation_3d',draw_formation,'Directed measurements and shared agent orientations',[cp])
    # Each covariance figure shares a display factor across its panels.
    for suffix, pair in [('', ['greedy','s1']), ('_classical', ['spectral','degree'])]:
        covariance_methods=['greedy','s0','s1','s2'] if not suffix else pair
        scale=.65/max(np.sqrt(max(np.linalg.eigvalsh(C).max(),0))
                      for method in covariance_methods for C in geometry[method][1])
        bounds=np.vstack([P+sign*scale*np.sqrt(np.maximum(np.array([np.diag(C) for C in geometry[m][1]]),0))
                          for m in pair for sign in [-1,1]])
        mode_bounds=np.vstack([P]+[P+geometry[m][2]*(.65/(np.linalg.norm(geometry[m][2],axis=1).max() or 1)) for m in pair])
        for row in out:
            if row['method'] in covariance_methods:row['ellipsoid_display_factor']=scale
        for kind in ['uncertainty','mode']:
            shown=['greedy','s0','s1','s2'] if not suffix else pair
            shown_mode_bounds=np.vstack([P]+[P+geometry[m][2]*(.65/(np.linalg.norm(geometry[m][2],axis=1).max() or 1)) for m in shown])
            shown_bounds=np.vstack([P+sign*scale*np.sqrt(np.maximum(np.array([np.diag(C) for C in geometry[m][1]]),0))
                                   for m in shown for sign in [-1,1]])
            def draw_detail(plain,kind=kind,pair=shown,scale=scale,bounds=shown_bounds,mode_bounds=shown_mode_bounds):
                fig=plt.figure(figsize=(7,7.2 if len(pair)==4 else 8.0))
                for j,method in enumerate(pair):
                    ax=fig.add_subplot(2,2 if len(pair)==4 else 1,j+1,projection='3d')
                    edges,cov,mode=geometry[method]
                    displacement=mode*(.65/(np.linalg.norm(mode,axis=1).max() or 1))
                    points=bounds if kind=='uncertainty' else mode_bounds
                    axes3(ax,points)
                    tight_geometry_axes(ax,points)
                    ax.set_title(SHORT[method],fontsize=13,pad=8)
                    shaded_edges(ax,edges)
                    ax.scatter(*P.T,c=nodecolors,s=42 if kind=='uncertainty' else 90,depthshade=False,edgecolors='white',linewidths=.6)
                    if kind=='uncertainty':
                        for i,C in enumerate(cov):_ellipsoid(ax,C,P[i],scale,nodecolors[i],res=20)
                    else:
                        ax.quiver(*P.T,*displacement.T,color='#222222',linewidth=2.5,arrow_length_ratio=.28)
                    for artist in ax.collections:artist.set_clip_on(False)
                fig.legend([Line2D([],[],marker='o',ls='none',color=c) for c in colors],labels,ncol=5,loc='upper center',frameon=False)
                fig.subplots_adjust(left=.04,right=.93,bottom=.055,top=.90,hspace=.23,wspace=.20)
                if not plain:
                    note='Covariance scale is shared within each figure.' if kind=='uncertainty' else 'The longest mode arrow has the same display length in each panel.'
                    fig.text(.5,.008,'First reference instance. '+note,ha='center',fontsize=8)
                return fig
            chapter_save('chapter_'+kind+'_3d'+suffix,draw_detail,
                         'Positional covariance' if kind=='uncertainty' else 'Weakest observable positional deformation',[cp])
    write_csv(DATA/'chapter_geometry.csv',out)
    def draw_frames(plain):
        fig=plt.figure(figsize=(7,6.2));ax=fig.add_subplot(111,projection='3d');axes3(ax,P);ax.view_init(elev=25,azim=40)
        ax.set_box_aspect((1,1,1),zoom=1.10);ax.tick_params(labelsize=10)
        shaded_edges(ax,geometry['s1'][0])
        ax.scatter(*P.T,c=nodecolors,s=65,depthshade=False,edgecolors='white',linewidths=.8)
        axiscolors=['#d62728','#228833','#3366cc']
        for i,(point,agent) in enumerate(zip(P,net.agents)):
            ax.text(*(point+np.array([-.06,.10,.06])),str(i),fontsize=11)
            if agent.domain in ['R^2','R^3']:continue
            R=agent.pose.rotation_mat()
            for k,color in enumerate(axiscolors):
                ax.quiver(*point,*(.23*R[:,k]),color=color,linewidth=1.7,arrow_length_ratio=.22)
        fig.legend([Line2D([],[],color=c,lw=2) for c in axiscolors],
                   [r'Local $x_i$',r'Local $y_i$',r'Local $z_i$'],ncol=3,loc='upper center',frameon=False)
        fig.subplots_adjust(left=.03,right=.91,bottom=.08,top=.92)
        if not plain:fig.text(.5,.01,'Shared poses for every topology. Position-only agents use the global frame.',ha='center',fontsize=9)
        return fig
    chapter_save('chapter_agent_frames',draw_frames,'Local measurement frames of the shared formation',[cp])
    # Recorded controller paths stay in the fixed sampled coordinate frame.
    p=find_run('20260912-194452');paths=read_csv(p/'paths.csv');case=[r for r in paths if int(r['case'])==0]
    goal=np.array([[float(r[k]) for k in ['x','y','z']] for r in sorted([r for r in case if r['method']=='goal'],key=lambda r:int(r['agent']))])
    selected=['greedy','complete','k10','k50'];series={}
    for method in selected:
        rows=[r for r in case if r['method']==method];steps=sorted({int(r['step']) for r in rows})
        by={}
        for r in rows:by.setdefault(int(r['step']),[]).append(r)
        series[method]=np.array([[[float(r[k]) for k in ['x','y','z']] for r in sorted(by[s],key=lambda r:int(r['agent']))] for s in steps])
    allpoints=np.vstack([goal]+[a.reshape(-1,3) for a in series.values()])
    agent_colors=plt.get_cmap('tab10')(np.arange(len(goal)))
    def motion_path(ax,points,color):
        ax.plot(*points.T,c=color,lw=2.2)
        lengths=np.linalg.norm(np.diff(points,axis=0),axis=1);arc=np.cumsum(lengths)
        if len(arc) and arc[-1]>1e-12:
            k=min(np.searchsorted(arc,.5*arc[-1]),len(points)-2)
            direction=points[k+1]-points[k]
            if np.linalg.norm(direction)>0:
                direction=direction/np.linalg.norm(direction)*min(.06,.15*arc[-1])
                ax.quiver(*points[k],*direction,color=color,linewidth=1.6,arrow_length_ratio=.4)
    def draw_paths(plain):
        fig=plt.figure(figsize=(7,8.5))
        for j,method in enumerate(['greedy','k10']):
            ax=fig.add_subplot(2,1,j+1,projection='3d');axes3(ax,allpoints);a=series[method]
            ax.view_init(elev=25,azim=40);ax.set_box_aspect((1,1,1),zoom=1.12);ax.set_title(SHORT[method],pad=8)
            for i,color in enumerate(agent_colors):
                motion_path(ax,a[:,i,:],color)
                ax.scatter(*goal[i],facecolors='none',edgecolors=color,s=42,depthshade=False)
                ax.scatter(*a[0,i],c=[color],marker='x',s=30,depthshade=False)
                ax.scatter(*a[-1,i],c=[color],s=20,depthshade=False)
        fig.legend([Line2D([],[],color=c,lw=2) for c in agent_colors],[str(i) for i in range(len(goal))],
                   title='Agent',ncol=10,loc='upper center',frameon=False)
        fig.subplots_adjust(left=.05,right=.9,bottom=.06,top=.88,hspace=.32)
        if not plain:fig.text(.5,.005,'Hollow: goal; cross: start; filled: final. Arrows indicate motion.',ha='center',fontsize=8)
        return fig
    chapter_save('chapter_control_paths_3d',draw_paths,'Physical control paths on the first shared case',[p])
    control_nets,_=benchmark.read_networks(p/'instances.npz');control_radius=characteristic_length(control_nets[0])
    assert np.allclose(goal,np.array([a.pose.position for a in control_nets[0].agents]))
    assert all(np.allclose(a[0],series['greedy'][0]) for a in series.values())
    displacements={m:(a-a[0])/control_radius for m,a in series.items()}
    displacement_goals={m:(goal-a[0])/control_radius for m,a in series.items()}
    bounds=np.vstack([a.reshape(-1,3) for a in displacements.values()]+list(displacement_goals.values()))
    def draw_displacements(plain):
        fig=plt.figure(figsize=(7,7.4))
        for j,method in enumerate(selected):
            ax=fig.add_subplot(2,2,j+1,projection='3d');axes3(ax,bounds);ax.view_init(elev=25,azim=40)
            ax.set_title(SHORT[method],pad=8);a=displacements[method]
            for i,color in enumerate(agent_colors):
                motion_path(ax,a[:,i,:],color)
                ax.scatter(*a[-1,i],c=[color],s=24,depthshade=False)
                ax.scatter(*displacement_goals[method][i],facecolors='none',edgecolors=color,s=32,depthshade=False)
            ax.scatter(0,0,0,c='black',marker='x',s=35,depthshade=False)
            ax.set_xlabel(r'$\Delta x/L$',labelpad=-2);ax.set_ylabel(r'$\Delta y/L$',labelpad=-2);ax.set_zlabel(r'$\Delta z/L$',labelpad=-2)
        fig.legend([Line2D([],[],color=c,lw=2) for c in agent_colors],[str(i) for i in range(len(goal))],
                   title='Agent',ncol=10,loc='upper center',frameon=False)
        fig.subplots_adjust(left=.01,right=.95,bottom=.055,top=.88,wspace=.03,hspace=.30)
        if not plain:fig.text(.5,.005,'Each path is translated by its own initial position and divided by the goal radius.',ha='center',fontsize=8)
        return fig
    chapter_save('chapter_control_displacements_3d',draw_displacements,'Per-agent displacement paths without the formation-wide position offset',[p])
    def draw_coordinates(plain):
        fig,axs=plt.subplots(3,2,figsize=(7,7.4),sharex=True)
        for col,method in enumerate(['greedy','k10']):
            times=sorted({float(r['t']) for r in case if r['method']==method});a=displacements[method]
            assert len(times)==len(a)
            for coord in range(3):
                ax=axs[coord,col]
                for i,color in enumerate(agent_colors):ax.plot(times,a[:,i,coord],color=color,lw=1.2,label=str(i))
                ax.set_xscale('symlog',linthresh=1);ax.set_xticks([0,1,10,100],['0','1','10','100']);ax.set_xlim(0,200)
                ax.set_ylabel([r'$\Delta x/L$',r'$\Delta y/L$',r'$\Delta z/L$'][coord]);style(ax)
                if coord==0:ax.set_title(SHORT[method])
                if coord==2:ax.set_xlabel('Simulation time')
        chapter_layout(fig,axs,plain,'First shared case. Time is linear up to 1, logarithmic thereafter.',legend=False)
        handles,labels=axs[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,title='Agent',ncol=10,loc='upper center',frameon=False)
        fig.subplots_adjust(top=.86)
        return fig
    chapter_save('chapter_control_coordinates',draw_coordinates,'Per-agent position coordinates over recorded time',[p])
    # Absolute trade-offs complement paired policy-minus-greedy differences.
    front=load_front();summaries=[]
    for k,(p,m,rs) in front.items():
        by=rows_by_method(p,m,rs)
        for method in ['greedy',{0:'k0',10:'s1',50:'k50',200:'k200'}[k]]:
            e=summarize(by[method],'edges',seed=k+3);q=summarize(by[method],'shape',seed=k+13)
            summaries.append({'kappa':k,'method':'Greedy' if method=='greedy' else 'DQN','edges':e,'error':q})
    def draw_trade(plain):
        fig,ax=plt.subplots(figsize=(7,4.5))
        for method,color,marker in [('Greedy','#333333','s'),('DQN','#2878aa','o')]:
            for j,r in enumerate(x for x in summaries if x['method']==method):
                x,l,h=r['edges'][:3];y,lo,hi=r['error'][:3]
                ax.errorbar(x,y,xerr=[[x-l],[h-x]],yerr=[[y-lo],[hi-y]],fmt=marker,color=color,capsize=3,label=method if j==0 else None)
                left=(method=='DQN') if r['kappa'] in [0,10] else (method=='Greedy')
                offset=(-12,12 if r['kappa']==0 else -20) if left else (12,10 if r['kappa']!=0 else -16)
                ax.annotate(r'$\kappa='+str(r['kappa'])+'$',(x,y),xytext=offset,
                            ha='right' if left else 'left',textcoords='offset points',fontsize=9,color=color)
        ax.set(xlim=(16,25.5),ylim=(1.3,130),yscale='log',xlabel='Mean directed edge count',ylabel=r'Geometric mean $\varepsilon$')
        style(ax);ax.legend(loc='upper right',frameon=False);fig.tight_layout(rect=(0,.045 if not plain else 0,1,1))
        if not plain:fig.text(.5,.005,'100 shared formations; matched training and evaluation weights.',ha='center',fontsize=8)
        return fig
    chapter_save('chapter_absolute_tradeoff',draw_trade,'Absolute measurement/error balance under matched objectives',[x[0] for x in front.values()])

    p=find_run('20260912-162935');noise=read_csv(p/'noise.csv');decisions=read_csv(p/'decisions.csv')
    chapter_prediction_scatter()
    def draw_decisions(plain):
        fig,axs=plt.subplots(1,3,figsize=(7,3.6),sharey=True)
        for ax,seed in zip(axs,['s0','s1','s2']):
            for field,label,color in [('phi_pct','Objective','#2878aa'),('err_pct','Error','#bc6500')]:
                for kind,ls in [('add','-'),('remove','--')]:
                    v=sorted(number(r,field) for r in decisions if r['model']==seed and r['kind']==kind and number(r,field) is not None)
                    ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=color,ls=ls,label=label+', '+kind)
            ax.set(title=seed.replace('s','Seed '),xlabel='Edit percentile',xlim=(0,100),ylim=(0,1));style(ax)
        axs[0].set_ylabel('Fraction of eligible decisions')
        chapter_layout(fig,axs,plain,'Pooled recorded decisions; correlated within episodes. Error eligibility differs from score eligibility.')
        return fig
    chapter_save('chapter_decision_distributions',draw_decisions,'Addition/removal distributions distinguish high objective rank from error rank',[p])


def chapter_decision_phases():
    # Edits before the best graph of the episode (search) and after it, ranked among all legal toggles.
    p=find_run('20260912-162935');CHAPTER_USED.add(p);decisions=read_csv(p/'decisions.csv')
    best_at={(r['method'],r['episode']):int(float(r['best_at'])) for r in read_csv(p/'results.csv')}
    seeds=['s0','s1','s2'];kinds=[('add','Addition','o'),('remove','Removal','s')]
    phases=[('search','Before the best graph'),('after','After the best graph')]
    phase=lambda r:'search' if int(r['step'])<best_at[(r['model'],r['episode'])] else 'after'
    groups={(s,k,ph):[r for r in decisions if r['model']==s and r['kind']==k and phase(r)==ph]
            for s in seeds for k,_,_ in kinds for ph,_ in phases}
    def agreement(rows,seed):
        # Fraction of edits that are the objective-optimal toggle, bootstrapped over episodes.
        ep={}
        for r in rows:
            if r['phi_best'] not in ('','None'):ep.setdefault(r['episode'],[]).append(r['phi_best']=='True')
        if not ep:return np.nan,np.nan,np.nan,0
        hits=np.array([sum(v) for v in ep.values()]);counts=np.array([len(v) for v in ep.values()])
        rng=np.random.default_rng(seed);ix=rng.integers(0,len(hits),size=(BOOTSTRAPS,len(hits)))
        boot=hits[ix].sum(axis=1)/counts[ix].sum(axis=1)
        return 100*hits.sum()/counts.sum(),*(100*np.percentile(boot,[2.5,97.5])),int(counts.sum())
    out=[]
    for (s,k,ph),rows in groups.items():
        phi=[number(r,'phi_pct') for r in rows if number(r,'phi_pct') is not None]
        err=[number(r,'err_pct') for r in rows if number(r,'err_pct') is not None]
        a,lo,hi,na=agreement(rows,11)
        out.append({'seed':s,'kind':k,'phase':ph,'edits':len(rows),'phi_median':np.median(phi) if phi else None,
                    'err_edits':len(err),'err_median':np.median(err) if err else None,
                    'phi_best_pct':a,'phi_best_low':lo,'phi_best_high':hi})
    write_csv(DATA/'chapter_decision_phases.csv',out)
    # After the best graph, how often the score equals the score two edits earlier (an edit and its undo).
    trajectories=read_csv(p/'trajectories.csv');cycle=[]
    for s in seeds:
        returns=steps=0;best=[]
        for (m,ep),b in best_at.items():
            if m!=s:continue
            scores=[round(float(r['score']),9) for r in sorted((r for r in trajectories if r['method']==s and r['episode']==ep),key=lambda r:int(r['step']))][b:]
            returns+=sum(scores[t]==scores[t-2] for t in range(2,len(scores)));steps+=max(0,len(scores)-2);best.append(b)
        cycle.append({'seed':s,'episodes':len(best),'best_at_median':np.median(best),'best_at_max':max(best),
                      'post_best_steps':steps,'two_step_return_pct':100*returns/steps})
    write_csv(DATA/'chapter_decision_cycle.csv',cycle)
    def draw(plain):
        fig,axs=plt.subplots(3,2,figsize=(7,8.6),sharey='row',squeeze=False)
        x=lambda i,j:i*2.6+j
        rng=np.random.default_rng(3)
        for c,(ph,title) in enumerate(phases):
            for r_,(field,ylabel) in enumerate([('phi_pct','Objective percentile'),('err_pct','Error percentile')]):
                ax=axs[r_,c]
                for i,s in enumerate(seeds):
                    for j,(k,lab,mk) in enumerate(kinds):
                        v=np.array([number(r,field) for r in groups[s,k,ph] if number(r,field) is not None])
                        if len(v)==0:continue
                        ax.scatter(x(i,j)+rng.uniform(-.3,.3,len(v)),v,s=5,alpha=.35,color=COLORS[s],marker=mk,lw=0)
                        ax.plot([x(i,j)-.38,x(i,j)+.38],[np.median(v)]*2,color='black',lw=1.4)
                ax.axhline(50,c='#888888',ls=':',lw=1)
                ax.set(ylim=(-3,103),ylabel=ylabel if c==0 else None)
                if r_==0:ax.set_title(title)
            ax=axs[2,c]
            for i,s in enumerate(seeds):
                for j,(k,lab,mk) in enumerate(kinds):
                    row=next(o for o in out if o['seed']==s and o['kind']==k and o['phase']==ph)
                    y=row['phi_best_pct']
                    ax.errorbar(x(i,j),y,yerr=[[y-row['phi_best_low']],[row['phi_best_high']-y]],color=COLORS[s],marker=mk,
                                capsize=2,ms=5,ls='none')
            ax.set(ylim=(-3,103),ylabel='Objective-optimal edits (%)' if c==0 else None)
            for a in axs[:,c]:
                a.set_xticks([x(i,j) for i in range(3) for j in range(2)],['add','remove']*3,fontsize=7.5)
                a.set_xlim(-.7,x(2,1)+.7);style(a)
                for i,s in enumerate(seeds):
                    a.text(x(i,.5),-.13,SHORT[s].replace('DQN, ',''),transform=a.get_xaxis_transform(),ha='center',va='top',fontsize=8)
        chapter_layout(fig,axs,plain,'Points are edits; black lines are medians. Agreement intervals bootstrap episodes.',legend=False)
        return fig
    chapter_save('chapter_decision_phases',draw,'Decision ranks before and after the best graph of the episode',[p])


def chapter_prediction_scatter():
    # One point per returned graph at 1 degree; axes capped at one formation radius.
    p=find_run('20260912-162935');noise=read_csv(p/'noise.csv')
    def draw_prediction(plain):
        fig,axs=plt.subplots(1,2,figsize=(7,3.9))
        chosen=[r for r in noise if float(r['sigma_deg'])==1 and r['method'] in ['greedy','spectral','degree','s0','s1','s2']]
        values=[float(r[k]) for r in chosen for k in ['predicted','measured'] if float(r[k])>0];lo=min(values)*.7;hi=1.0
        for ax,methods in zip(axs,[['greedy','spectral','degree'],['s0','s1','s2']]):
            for method in methods:
                rs=[r for r in chosen if r['method']==method]
                ax.scatter([float(r['predicted']) for r in rs],[float(r['measured']) for r in rs],s=24,alpha=.75,c=COLORS[method],marker=MARKERS.get(method,'o'),label=SHORT[method])
            ax.plot([lo,hi],[lo,hi],':',c='#555555')
            ax.set(xscale='log',yscale='log',xlim=(lo,hi),ylim=(lo,hi),aspect='equal',
                   xlabel='Predicted position RMS (formation radii)',ylabel='Measured position RMS (formation radii)')
            ax.legend(fontsize=8,loc='upper left');style(ax)
        chapter_layout(fig,axs,plain,'One point per graph at 1 degree noise. Points above one formation radius are outside the axes.',legend=False)
        return fig
    chapter_save('chapter_prediction_scatter',draw_prediction,'Formation-level agreement and deviations hidden by average error ratios',[p])


def chapter_error_components():
    """Split saved-output covariance into position and attitude contributions."""
    import copy
    from rigidity import scaled_rigidity_matrix
    domain, sources = load_domain_data(find_domain_runs())
    front = load_front()
    output=[]
    cache_path=DATA/'chapter_error_components.csv'
    cached={(r['source'],r['method'],int(r['episode'])):r for r in read_csv(cache_path)} if cache_path.exists() else {}
    def augment(path, groups, methods):
        nets,_=benchmark.read_networks(path/'instances.npz')
        ranks={}
        result={}
        for method in methods:
            result[method]=[]
            for original in groups[method]:
                row=dict(original);episode=int(row['episode'])
                if truth(row['is_IBR']):
                    previous=cached.get((str(path.relative_to(ROOT)),method,episode))
                    if previous and previous.get('edges')==row['edges'] and float(previous['stored_combined'])==float(row['shape_err']):
                        row['position_component']=float(previous['position_component']);row['attitude_component']=float(previous['attitude_component'])
                        output.append(previous);result[method].append(row);continue
                    net=copy.deepcopy(nets[episode]);n=net.n
                    if episode not in ranks:
                        ranks[episode]=np.linalg.matrix_rank(extended_bearing_rigidity_matrix(net.fully_connected()))
                    net.edges[:]=False
                    for edge in row['edges'].split():
                        i,j=map(int,edge.split('>'));net.edges[i,j]=True
                    assert int(net.edges.sum())==int(float(row['m']))
                    _,singular,Vt=np.linalg.svd(scaled_rigidity_matrix(net),full_matrices=False)
                    rank=ranks[episode];Vt=Vt[:rank];inv=1/singular[:rank]**2
                    ep=float(np.sqrt(np.sum(inv*np.sum(Vt[:,:3*n]**2,axis=1))/n))
                    attitude_dofs=any(a.domain not in ['R^2','R^3'] for a in net.agents)
                    ea=float(np.sqrt(np.sum(inv*np.sum(Vt[:,3*n:]**2,axis=1))/n)) if attitude_dofs else 0.
                    combined=np.hypot(ep,ea)
                    assert np.isclose(combined,float(row['shape_err']),rtol=2e-5), (path,method,episode,combined,row['shape_err'])
                    row['position_component']=ep;row['attitude_component']=ea
                    output.append({'source':str(path.relative_to(ROOT)),'method':method,'episode':episode,'n':n,
                                   'position_component':ep,'attitude_component':ea,'combined':combined,
                                   'stored_combined':row['shape_err'],'attitude_dofs':attitude_dofs,'edges':row['edges']})
                result[method].append(row)
        write_csv(cache_path,output)
        print('Components saved:',path.name,len(output),flush=True)
        return result
    enriched={}
    for group,name in [(HOMOGENEOUS,'homogeneous'),(MIXED,'mixed')]:
        methods=['s1','greedy','spectral','degree'] if name=='homogeneous' else ['s0','s1','s2','greedy']
        for bench,label in group:
            groups={method:domain[bench,method] for method in methods}
            by=augment(sources[bench],groups,methods)
            enriched.update({(bench,m):rs for m,rs in by.items()})
        keys=[x[0] for x in group];labels=[x[1] for x in group]
        if name=='mixed':labels=[r'$\mathbb{R}^2$ / SE(3)'+'\nrandom',r'$\mathbb{R}^2$ / SE(3)'+'\nbalanced',r'$\mathbb{R}^3$ /'+'\n'+r'$\mathbb{R}^2\times S^1$'+'\nrandom',r'$\mathbb{R}^3$ /'+'\n'+r'$\mathbb{R}^2\times S^1$'+'\nbalanced']
        chapter_grid('chapter_'+name+'_quality',enriched,keys,labels,methods,
                     ['ratio','shape','position_component','attitude_component'],
                     'Sparsity and separate position/attitude conditioning across domains',list(sources.values()),
                     'Rigid outputs. Components use the same per-agent normalization as the combined coefficient.')
    size={}; size_sources=[]; size_metas={}
    methods=['s1','greedy','spectral','degree','constructive']
    reference_methods=['greedy','spectral','degree','constructive','anneal','s0','s1','s2']
    for n,prefix in SIZE_PREFIX.items():
        path,meta,groups=chapter_run(prefix);size_sources.append(path);size_metas[n]=meta
        selected=reference_methods if n==10 else [m for m in methods if m in groups]
        by=augment(path,groups,selected)
        size.update({(n,m):rs for m,rs in by.items()})
    path,meta,groups=chapter_run('20260922-085849');require_shared(size_metas[48],meta)
    shared=augment(path,groups,['greedy'])['greedy'];shared_ids={int(r['episode']) for r in shared}
    def component_row(method,rows):
        return chapter_outcome_row(SHORT[method],rows)+[tex_number(summarize(rows,c)[0],2) for c in ['position_component','attitude_component']]
    header=['Method','Rigid','At bound',r'$\bar m$',r'$\varepsilon_{\rm geom}$',r'$\varepsilon_{p,\rm geom}$',r'$\varepsilon_{a,\rm geom}$']
    chapter_table('reference',header,[component_row(m,size[10,m]) for m in reference_methods],'lrrrrrr')
    chapter_table('shared48',header,[component_row(m,shared if m=='greedy' else [r for r in size[48,m] if int(r['episode']) in shared_ids]) for m in methods],'lrrrrrr')
    chapter_table('stress96',header,[component_row(m,size[96,m]) for m in methods if (96,m) in size],'lrrrrrr')
    chapter_grid('chapter_size_quality',size,list(SIZE_PREFIX),[str(n) for n in SIZE_PREFIX],methods,
                 ['ratio','shape','position_component','attitude_component'],
                 'Size transfer with separate position and attitude conditioning',size_sources,
                 'Rigid outputs only. No interval is estimated from a single surviving output.',numeric=True)
    reference_effects=[]
    def draw_reference(plain):
        fig,axs=plt.subplots(2,2,figsize=(7,5.7))
        for ax,metric,title in zip(axs.flat,['edges','shape_dex','position_component','attitude_component'],
                                  ['Edge-count difference','Combined error ratio','Position error ratio','Attitude error ratio']):
            for j,seed in enumerate(['s0','s1','s2']):
                if metric in ['edges','shape_dex']:
                    values,_=paired_values(size[10,seed],size[10,'greedy'],metric)
                else:
                    greedy={int(r['episode']):r for r in size[10,'greedy']}
                    values=np.array([np.log10(float(r[metric])/float(greedy[int(r['episode'])][metric]))
                                     for r in size[10,seed] if truth(r['is_IBR']) and truth(greedy[int(r['episode'])]['is_IBR'])])
                e,l,h=paired_ci(values,123+j)
                if metric!='edges':e,l,h=10**e,10**l,10**h
                ax.errorbar(e,j,xerr=[[e-l],[h-e]],fmt='o',color=COLORS[seed],capsize=3)
                if plain:reference_effects.append({'seed':seed,'metric':metric,'n':len(values),'estimate':e,'low':l,'high':h})
            ax.set_yticks(range(3),['Seed 0','Seed 1','Seed 2']);ax.set_title(title)
            ax.set_xlabel('Policy minus greedy' if metric=='edges' else 'Policy / greedy')
            ax.axvline(0 if metric=='edges' else 1,color='#888888',ls=':');style(ax)
        chapter_layout(fig,axs,plain,'Paired 95% bootstrap intervals over 200 shared formations.',legend=False)
        return fig
    chapter_save('chapter_reference',draw_reference,'Paired sparsity and combined/position/attitude error differences',[size_sources[list(SIZE_PREFIX).index(10)]])
    write_csv(DATA/'chapter_reference_component_effects.csv',reference_effects)
    trade={};policies={0:'k0',10:'s1',50:'k50',200:'k200'}
    for k,(path,meta,rows) in front.items():
        by=augment(path,rows_by_method(path,meta,rows),['greedy',policies[k]])
        trade[k,'greedy']=by['greedy'];trade[k,'policy']=by[policies[k]]
    def draw(plain):
        fig,ax=plt.subplots(figsize=(7,4.5))
        for method,label,color,marker in [('greedy','Greedy','#333333','s'),('policy','DQN','#2878aa','o')]:
            for j,k in enumerate([0,10,50,200]):
                x,xl,xh,_=summarize(trade[k,method],'position_component',seed=k+800)
                y,yl,yh,_=summarize(trade[k,method],'attitude_component',seed=k+900)
                ax.errorbar(x,y,xerr=[[x-xl],[xh-x]],yerr=[[y-yl],[yh-y]],fmt=marker,color=color,capsize=3,label=label if j==0 else None)
                ax.annotate(r'$\kappa='+str(k)+'$',(x,y),xytext=(-12,12) if method=='greedy' else (12,-17),
                            ha='right' if method=='greedy' else 'left',textcoords='offset points',fontsize=9,color=color)
        ax.set(xscale='log',yscale='log',xlim=(1,80),ylim=(.85,70),xlabel=r'Position coefficient $\varepsilon_p$',ylabel=r'Attitude coefficient $\varepsilon_a$')
        from matplotlib.ticker import NullFormatter
        ax.set_xticks([1,3,10,30],['1','3','10','30']);ax.set_yticks([1,3,10,30],['1','3','10','30'])
        ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_minor_formatter(NullFormatter());style(ax)
        ax.legend(loc='upper left',frameon=False)
        fig.tight_layout(rect=(0,.045 if not plain else 0,1,1))
        if not plain:fig.text(.5,.005,'Matched weights; 100 shared formations. Geometric means and 95% bootstrap intervals.',ha='center',fontsize=8)
        return fig
    chapter_save('chapter_conditioning_components',draw,'Position versus attitude conditioning under matched weights',[v[0] for v in front.values()])
    write_csv(DATA/'chapter_error_components.csv',output)
    print('Verified component decompositions:',len(output),'zero attitude:',sum(float(r['attitude_component'])==0 for r in output),flush=True)


def chapter_revision():
    CHAPTER_USED.clear();CHAPTER_FIGURES.clear()
    size,costs,ref,(cp,cm),domain,ds=chapter_primary()
    chapter_tradeoff();chapter_behavior();chapter_graphs(ref,cp);chapter_control();chapter_supplement(size,costs)
    chapter_visual_additions()
    chapter_decision_phases()
    chapter_error_components()
    chapter_audit()
    CHAPTER_FIGURES[:]=list({r['figure']:r for r in CHAPTER_FIGURES}.values())
    write_csv(DATA/'chapter_figure_inventory.csv',CHAPTER_FIGURES)
    print('Revised figures:',len(CHAPTER_FIGURES),'sources:',len(CHAPTER_USED),flush=True)


if __name__ == "__main__":
    if "--revision-only" not in sys.argv:
        main()
    chapter_revision()

