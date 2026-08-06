"""Rendering for baselines.py: the comparison table, the CSVs and the plots.

The table is written for someone who does not know the topic: every column says which
direction is better, jargon is spelled out in a legend, and the two different meanings the
old `steps` column carried are split into `work` and `best@`.
"""

import csv
import json
import os
import re
from datetime import datetime

import numpy as np

# ── palette ───────────────────────────────────────────────────────────────────────────
# From the data-viz reference palette, used unchanged and in its documented order.
# greedy/learned/random take categorical slots 1-3, which are certified for the
# all-pairs case (overlapping lines) in both modes. initial and optimal are *reference
# points* rather than methods under comparison, so they take neutral inks and dashed
# strokes instead of a categorical hue -- that also keeps the categorical count at 3.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

METHOD_STYLE = {
    "greedy":  {"color": "#2a78d6", "ls": "-",  "z": 3},   # categorical slot 1
    "learned": {"color": "#eb6834", "ls": "-",  "z": 4},   # slot 2
    "random":  {"color": "#1baf7a", "ls": "-",  "z": 2},   # slot 3
    "initial": {"color": MUTED,     "ls": ":",  "z": 1},   # reference
    "optimal": {"color": INK_2,     "ls": "--", "z": 5},   # reference
}
# slot 3 (aqua) sits below 3:1 on the light surface, so the relief rule applies: every
# series carries a direct label at its line end, and the table view always ships.

METHOD_ORDER = ["initial", "random", "greedy", "learned", "optimal"]

METHOD_BLURB = {
    "initial": "the random graph each method starts from",
    "random":  "uniform random actions - the floor any method should beat",
    "greedy":  "repeatedly applies the single best edge change until none helps",
    "learned": "the trained policy",
    "optimal": "exhaustive search over every graph (small networks only)",
}

ACTION_SHORT = {
    "SelectNodesSequentially": "selectseq",
    "AddRemoveEdgeDiscreteNoSelfLoops": "addremove",
    "AddEdgeDiscreteNoSkipNoSelfLoops": "addonly",
}


# ── run directory ─────────────────────────────────────────────────────────────────────
def short_env_name(env_name):
    action = re.search(r"_action([A-Za-z0-9]+?)_obs", env_name)
    action = ACTION_SHORT.get(action.group(1), action.group(1)) if action else "env"
    tail = re.search(r"_(n\d+_[A-Za-z0-9]+|hetero[A-Za-z0-9]*)$", env_name)
    return f"{action}-{tail.group(1)}" if tail else action


def short_model_name(model_name):
    if not model_name:
        return None
    return re.split(r"_action", model_name)[0][:28]


def make_run_dir(root, env_name, model_name=None, tag=None, out_dir=None, with_plots=True):
    if out_dir:
        path = out_dir
    else:
        parts = [datetime.now().strftime("%Y%m%d-%H%M%S"), short_env_name(env_name)]
        if model_name:
            parts.append(short_model_name(model_name))
        if tag:
            parts.append(tag)
        path = os.path.join(root, "__".join(parts))
    os.makedirs(os.path.join(path, "plots") if with_plots else path, exist_ok=True)
    return path


# ── aggregation ───────────────────────────────────────────────────────────────────────
def _fmt(mean, sd=None, width=None):
    if sd is None:
        return f"{mean:.2f}"
    return f"{mean:.2f}+-{sd:.2f}"


def aggregate(rows):
    """{method: {...}} means/sds over episodes, in a stable display order."""
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    methods += [m for m in dict.fromkeys(r["method"] for r in rows) if m not in methods]

    opt = {r["episode"]: r["score"] for r in rows if r["method"] == "optimal"}
    out = {}
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        eig = [r["min_eig"] for r in sel if r.get("min_eig") is not None]
        matched = [r for r in sel if r["episode"] in opt]
        out[m] = {
            "episodes": len(sel),
            "edges_mean": float(np.mean([r["m"] for r in sel])),
            "edges_sd": float(np.std([r["m"] for r in sel])),
            "score_mean": float(np.mean([r["score"] for r in sel])),
            "score_sd": float(np.std([r["score"] for r in sel])),
            "rigid_pct": 100.0 * float(np.mean([r["is_IBR"] for r in sel])),
            "minimal_pct": 100.0 * float(np.mean([r["is_MBR"] for r in sel])),
            "min_eig_mean": float(np.mean(eig)) if eig else None,
            "work_mean": float(np.mean([r.get("work", 0) for r in sel])),
            "best_at_mean": float(np.mean([r.get("best_at", 0) for r in sel])),
            "matches_opt_pct": (100.0 * sum(1 for r in matched
                                            if r["score"] >= opt[r["episode"]] - 1e-9)
                                / len(matched)) if matched else None,
        }
    return out


# ── the table ─────────────────────────────────────────────────────────────────────────
def format_table(rows, context, brief=False):
    agg = aggregate(rows)
    has_opt = any(v["matches_opt_pct"] is not None for v in agg.values())
    lines = []
    w = 78

    lines.append("=" * w)
    lines.append("BASELINE COMPARISON")
    lines.append("=" * w)
    for k, v in context.items():
        lines.append(f"  {k:<12}: {v}")
    lines.append("")

    head1 = f"  {'method':<9}{'edges':>12}{'score':>14}{'rigid':>8}{'minimal':>9}{'rigidity':>11}{'work':>7}{'best@':>7}"
    head2 = f"  {'':<9}{'(fewer)':>12}{'(higher)':>14}{'%':>8}{'%':>9}{'(higher)':>11}{'edits':>7}{'step':>7}"
    if has_opt:
        head1 += f"{'=best':>7}"
        head2 += f"{'%':>7}"
    lines.append(head1)
    lines.append(head2)
    lines.append("  " + "-" * (len(head1) - 2))

    for m, v in agg.items():
        eig = f"{v['min_eig_mean']:.1e}" if v["min_eig_mean"] is not None else "-"
        work = "-" if m in ("initial", "optimal") else f"{v['work_mean']:.0f}"
        best = "-" if m in ("initial", "optimal") else f"{v['best_at_mean']:.0f}"
        row = (f"  {m:<9}"
               f"{_fmt(v['edges_mean'], v['edges_sd']):>12}"
               f"{_fmt(v['score_mean'], v['score_sd']):>14}"
               f"{v['rigid_pct']:>8.0f}{v['minimal_pct']:>9.0f}{eig:>11}{work:>7}{best:>7}")
        if has_opt:
            row += ("-" if v["matches_opt_pct"] is None
                    else f"{v['matches_opt_pct']:.0f}").rjust(7)
        lines.append(row)
    lines.append("")

    if brief:
        return "\n".join(lines)

    lines.append("-" * w)
    lines.append("WHAT THE METHODS ARE")
    for m in agg:
        lines.append(f"  {m:<9} {METHOD_BLURB.get(m, '')}")
    lines.append("")
    lines.append("WHAT THE COLUMNS MEAN")
    lines.append("  edges     how many directed bearing measurements the final network needs.")
    lines.append("            Each edge is a sensor/communication link, so fewer is better.")
    lines.append("  score     the objective every method is scored with (phi). Higher is better;")
    lines.append("            it rewards rigidity and penalises each extra edge.")
    lines.append("  rigid     % of networks whose shape is fully determined by its bearing")
    lines.append("            measurements. This is the property being solved for.")
    lines.append("  minimal   % that are rigid AND use the fewest possible edges.")
    lines.append("            (heuristic on mixed-domain networks - may under-report)")
    lines.append("  rigidity  how much margin the network has before it stops being rigid.")
    lines.append("            Larger is more robust; ~1e-5 is normal at this scale.")
    lines.append("  work      how many changes to the network the method actually made.")
    lines.append("  best@     the step at which its best network was found. Lower means it")
    lines.append("            converged faster; the rest of the budget added nothing.")
    if has_opt:
        lines.append("  =best     % of networks where the method tied the exhaustive optimum.")
    lines.append("")
    lines.append("HOW TO READ IT")
    lines.append("  'initial' and 'optimal' are reference points, not competing methods:")
    lines.append("  every method starts from 'initial', and 'optimal' is the best achievable.")
    lines.append("  A method is doing well when it approaches 'optimal' with low 'work'.")
    lines.append("  All methods are run on the same networks, so rows compare directly.")
    lines.append("=" * w)
    return "\n".join(lines)


# ── output files ──────────────────────────────────────────────────────────────────────
RESULT_FIELDS = ["episode", "method", "m", "score", "is_IBR", "is_MBR",
                 "min_eig", "work", "best_at"]
TRACE_FIELDS = ["episode", "method", "step", "score", "edges", "rank", "rank_K",
                "is_IBR", "is_MBR", "min_eig"]


def write_csvs(run_dir, rows, traces):
    with open(os.path.join(run_dir, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    if traces:
        with open(os.path.join(run_dir, "trajectories.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=TRACE_FIELDS, extrasaction="ignore")
            wr.writeheader()
            for t in traces:
                wr.writerow(t)


def write_meta(run_dir, meta):
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def write_summary(run_dir, text):
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")


# ── plots ─────────────────────────────────────────────────────────────────────────────
# Static output for a thesis, so light mode only and no hover layer. Identity is never
# colour alone: every series carries a legend entry and a direct label at its line end.
PANELS = [
    ("score", "objective score\n(higher is better)", False),
    ("edges", "edges used\n(fewer is better)", False),
    ("rank", "rigidity matrix rank\n(dashed = fully rigid)", False),
    ("min_eig", "rigidity margin\n(higher is more robust)", True),
]


def _style_axes(ax, log=False):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)
    if log:
        ax.set_yscale("log")


def _series(traces, method, field):
    """{episode: [value per step]} for one method/field."""
    out = {}
    for t in traces:
        if t["method"] != method or t.get(field) is None:
            continue
        out.setdefault(t["episode"], []).append((t["step"], t[field]))
    return {ep: [v for _s, v in sorted(pts)] for ep, pts in out.items()}


def _padded(series, length):
    """Forward-fill each episode to a common length: a converged method holds its graph."""
    rows = []
    for vals in series.values():
        if not vals:
            continue
        rows.append(list(vals) + [vals[-1]] * (length - len(vals)))
    return np.array(rows, dtype=float) if rows else None


def _label_box():
    # a surface-coloured pad so an end label stays legible where it crosses a line
    return dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.9)


def _place_end_labels(ax, entries, x_at, min_gap=0.06):
    """Direct labels at the line ends, nudged apart so they never overlap."""
    if not entries:
        return
    lo, hi = ax.get_ylim()
    is_log = ax.get_yscale() == "log"

    def frac(v):
        if is_log:
            v, lo_, hi_ = max(v, 1e-300), max(lo, 1e-300), max(hi, 1e-299)
            return (np.log10(v) - np.log10(lo_)) / (np.log10(hi_) - np.log10(lo_))
        return (v - lo) / (hi - lo) if hi > lo else 0.5

    placed = []
    for f, text in sorted(((frac(v), t) for v, t in entries), key=lambda p: p[0]):
        if placed and f - placed[-1][0] < min_gap:
            f = placed[-1][0] + min_gap
        placed.append((f, text))
    for f, text in placed:
        ax.annotate(text, xy=(x_at, float(np.clip(f, 0.02, 0.98))),
                    xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points", color=INK_2, fontsize=7.5,
                    va="center", zorder=25, bbox=_label_box())


def _draw_panel(ax, traces, field, log, methods, ref_lines, aggregate_over_episodes):
    max_len = max((len(v) for m in methods for v in _series(traces, m, field).values()),
                  default=1)
    max_len = max(max_len, 1)

    positives, end_labels = [], []
    for method in methods:
        series = _series(traces, method, field)
        if not series:
            continue
        style = METHOD_STYLE.get(method, {"color": INK_2, "ls": "-", "z": 2})
        data = _padded(series, max_len)
        if data is None:
            continue
        x = np.arange(max_len)
        mean = data.mean(axis=0)
        positives.extend(v for v in data.ravel() if v > 0)

        if aggregate_over_episodes and data.shape[0] > 1:
            band_lo, band_hi = np.percentile(data, [25, 75], axis=0)
            ax.fill_between(x, band_lo, band_hi, color=style["color"], alpha=0.13,
                            linewidth=0, zorder=style["z"])
        ax.plot(x, mean, color=style["color"], linestyle=style["ls"], linewidth=2.0,
                zorder=style["z"] + 5, label=method, solid_capstyle="round")
        end_labels.append((mean[-1], method))

    for label, value in ref_lines.get(field, []):
        ax.axhline(value, color=INK_2, linestyle="--", linewidth=1.2, zorder=6)
        ax.annotate(label, xy=(0.015, value), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points", color=INK_2,
                    fontsize=7.5, zorder=20, bbox=_label_box())

    _style_axes(ax, log=log)
    # a non-rigid graph has a zero eigenvalue, which drags a log axis to 1e-300;
    # floor it just under the smallest value that is actually meaningful
    if log and positives:
        ax.set_ylim(bottom=10 ** np.floor(np.log10(np.percentile(positives, 1))))
    ax.set_xlabel("step", color=INK_2, fontsize=8.5)
    ax.margins(x=0.14)
    _place_end_labels(ax, end_labels, x_at=max_len - 1)


def plot_trajectories(run_dir, traces, rows, title, filename="trajectories",
                      aggregate_over_episodes=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [m for m in METHOD_ORDER
               if m not in ("initial", "optimal") and any(t["method"] == m for t in traces)]
    if not methods:
        return None

    # references: what fully rigid looks like, and what the exhaustive optimum achieved
    ref = {}
    rank_K = next((t["rank_K"] for t in traces if t.get("rank_K")), None)
    if rank_K:
        ref["rank"] = [("fully rigid", rank_K)]
    opt = [r for r in rows if r["method"] == "optimal"]
    if opt:
        ref["score"] = [("optimal", float(np.mean([r["score"] for r in opt])))]
        ref["edges"] = [("optimal", float(np.mean([r["m"] for r in opt])))]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor=SURFACE)
    for ax, (field, label, log) in zip(axes.ravel(), PANELS):
        _draw_panel(ax, traces, field, log, methods, ref, aggregate_over_episodes)
        ax.set_title(label, color=INK, fontsize=9.5, loc="left", pad=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title, color=INK, fontsize=11.5, x=0.012, ha="left", y=0.995)
    sub = ("mean across episodes, shaded band = middle 50%"
           if aggregate_over_episodes else "single episode")
    fig.text(0.012, 0.955, sub, color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    return _save(fig, run_dir, filename)


def plot_summary(run_dir, rows, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    colors = [METHOD_STYLE.get(m, {}).get("color", INK_2) for m in methods]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor=SURFACE)

    def box(ax, field, label):
        data = [[r[field] for r in rows if r["method"] == m] for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color=INK),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED,
                                        markeredgecolor="none"))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
            patch.set_edgecolor(c)
        for part in ("whiskers", "caps"):
            for artist in bp[part]:
                artist.set_color(AXIS)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=0)
        _style_axes(ax)
        ax.set_title(label, color=INK, fontsize=9.5, loc="left", pad=8)

    box(axes[0][0], "m", "edges used per network\n(fewer is better)")
    box(axes[0][1], "score", "objective score per network\n(higher is better)")

    ax = axes[1][0]
    rigid = [100 * np.mean([r["is_IBR"] for r in rows if r["method"] == m]) for m in methods]
    minimal = [100 * np.mean([r["is_MBR"] for r in rows if r["method"] == m]) for m in methods]
    x = np.arange(len(methods))
    ax.bar(x - 0.19, rigid, 0.36, color=colors, alpha=0.9, label="rigid")
    ax.bar(x + 0.19, minimal, 0.36, color=colors, alpha=0.4, label="also minimal")
    for xi, (a, b) in enumerate(zip(rigid, minimal)):
        ax.annotate(f"{a:.0f}", xy=(xi - 0.19, a), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK_2)
        ax.annotate(f"{b:.0f}", xy=(xi + 0.19, b), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylim(0, 112)
    _style_axes(ax)
    ax.set_title("% of networks solved\n(solid = rigid, faded = also minimal)",
                 color=INK, fontsize=9.5, loc="left", pad=8)

    ax = axes[1][1]
    roll = [m for m in methods if m not in ("initial", "optimal")]
    if roll:
        data = [[r.get("best_at", 0) for r in rows if r["method"] == m] for m in roll]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color=INK),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED,
                                        markeredgecolor="none"))
        for patch, m in zip(bp["boxes"], roll):
            c = METHOD_STYLE.get(m, {}).get("color", INK_2)
            patch.set_facecolor(c); patch.set_alpha(0.65); patch.set_edgecolor(c)
        for part in ("whiskers", "caps"):
            for artist in bp[part]:
                artist.set_color(AXIS)
        ax.set_xticks(range(1, len(roll) + 1)); ax.set_xticklabels(roll)
    _style_axes(ax)
    ax.set_title("steps taken to find the best network\n(lower converges faster)",
                 color=INK, fontsize=9.5, loc="left", pad=8)

    fig.suptitle(title, color=INK, fontsize=11.5, x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, run_dir, "summary")


def _save(fig, run_dir, name):
    import matplotlib.pyplot as plt
    out = []
    for ext in ("pdf", "png"):
        path = os.path.join(run_dir, "plots", f"{name}.{ext}")
        fig.savefig(path, facecolor=SURFACE, dpi=200, bbox_inches="tight")
        out.append(path)
    plt.close(fig)
    return out
