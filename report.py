"""Rendering for baselines.py: the comparison table, the CSVs and the plots.

The table is written for someone who does not know the topic: every column says which
direction is better, jargon is spelled out in a legend, and the two different meanings the
old `steps` column carried are split into `work` and `best@`.
"""

import csv
import json
import os
import re
import textwrap
from datetime import datetime

import numpy as np

# ── palette ───────────────────────────────────────────────────────────────────────────
# Data-viz reference palette, unchanged.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CARD = "#f4f3ee"   # one step off the surface, so the notes card reads as a panel

METHOD_STYLE = {
    "greedy":  {"color": "#2a78d6", "ls": "-",  "z": 3},   # categorical slot 1
    "learned": {"color": "#eb6834", "ls": "-",  "z": 4},   # slot 2
    "random":  {"color": "#1baf7a", "ls": "-",  "z": 2},   # slot 3
    "constructive": {"color": "#7d4bb5", "ls": "-", "z": 3},   # slot 4
    "initial": {"color": MUTED,     "ls": ":",  "z": 1},   # reference
    "optimal": {"color": INK_2,     "ls": "--", "z": 5},   # reference
}
# slot 3 (aqua) sits below 3:1 on the light surface, so the relief rule applies: every
# series carries a direct label at its line end, and the table view always ships.

METHOD_ORDER = ["initial", "random", "greedy", "constructive", "learned", "optimal"]

METHOD_BLURB = {
    "initial": "the random graph each method starts from",
    "random":  "uniform random actions - the floor any method should beat",
    "greedy":  "repeatedly applies the single best edge change until none helps",
    "constructive": "builds from the empty graph, keeping any edge that raises rank(B)",
    "learned": "the trained policy",
    "optimal": "exhaustive search over every graph (small networks only)",
}

# both formats, one directory each under plots/
PLOT_FORMATS = ("pdf", "png")

ACTION_SHORT = {
    "SelectNodesSequentially": "selectseq",
    "AddRemoveEdgeDiscreteNoSelfLoops": "addremove",
    "AddEdgeDiscreteNoSkipNoSelfLoops": "addonly",
}


# ── run directory ─────────────────────────────────────────────────────────────────────
def short_env_name(env_name):
    action = re.search(r"_action([A-Za-z0-9]+?)_(?:obs|reward)", env_name)
    action = ACTION_SHORT.get(action.group(1), action.group(1)) if action else "env"
    tail = re.search(r"_(n\d+_[A-Za-z0-9]+|mixed\d*|hetero[A-Za-z0-9]*)$", env_name)
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
    if with_plots:
        for ext in PLOT_FORMATS:
            os.makedirs(os.path.join(path, "plots", ext), exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)
    return path


# ── aggregation ───────────────────────────────────────────────────────────────────────
def _gmean(values):
    """exp(mean(log x)). Positive values only -- the caller filters."""
    return float(np.exp(np.mean(np.log(values)))) if values else None


def _gsd(values):
    """Multiplicative standard deviation: a dimensionless factor >= 1, used as
    `gmean x/ gsd` rather than `mean +- sd`."""
    return float(np.exp(np.std(np.log(values)))) if values else None


def _fmt_geo(v, times=" x/", mark="*"):
    """`gmean x/ gsd`, flagged when zero-margin networks had to be left out."""
    if v["min_eig_gmean"] is None:
        return "-"
    partial = mark if v["min_eig_n_pos"] < v["min_eig_n"] else ""
    # one surviving network has no spread; printing "x/1.0" would read as "no spread"
    # rather than "nothing to spread over"
    if v["min_eig_n_pos"] < 2:
        return f"{v['min_eig_gmean']:.1e}{partial}"
    return f"{v['min_eig_gmean']:.1e}{times}{v['min_eig_gsd']:.1f}{partial}"


def _fmt(mean, sd=None, spec=".2f", pm="+-"):
    """mean +- sd in a given number format. Every value in the table is a mean over the
    networks, so every one of them carries its spread."""
    if mean is None:
        return "-"
    if sd is None:
        return f"{mean:{spec}}"
    return f"{mean:{spec}}{pm}{sd:{spec}}"


def aggregate(rows):
    """{method: {...}} means/sds over episodes, in a stable display order."""
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    methods += [m for m in dict.fromkeys(r["method"] for r in rows) if m not in methods]

    opt = {r["episode"]: r["score"] for r in rows if r["method"] == "optimal"}
    out = {}
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        eig = [r["min_eig"] for r in sel if r.get("min_eig") is not None]
        pos = [e for e in eig if e > 0]
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
            "min_eig_sd": float(np.std(eig)) if eig else None,
            # the margin spans decades, so an arithmetic sd implies a range crossing
            # zero. The geometric pair is the honest spread for it: a multiplicative
            # factor you divide and multiply the geometric mean by.
            "min_eig_gmean": _gmean(pos),
            "min_eig_gsd": _gsd(pos),
            # a non-rigid network has margin exactly 0, which no geometric mean can
            # take -- record how many networks the geometric pair actually saw
            "min_eig_n": len(eig),
            "min_eig_n_pos": len(pos),
            "work_mean": float(np.mean([r.get("work", 0) for r in sel])),
            "work_sd": float(np.std([r.get("work", 0) for r in sel])),
            "best_at_mean": float(np.mean([r.get("best_at", 0) for r in sel])),
            "best_at_sd": float(np.std([r.get("best_at", 0) for r in sel])),
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

    head1 = (f"  {'method':<9}{'edges':>12}{'score':>14}{'rigid':>8}{'minimal':>9}"
             f"{'rigidity':>19}{'rigidity(geo)':>17}{'work':>11}{'best@':>12}")
    head2 = (f"  {'':<9}{'(fewer)':>12}{'(higher)':>14}{'%':>8}{'%':>9}"
             f"{'mean+-sd':>19}{'gmean x/gsd':>17}{'edits':>11}{'step':>12}")
    if has_opt:
        head1 += f"{'=best':>7}"
        head2 += f"{'%':>7}"
    # the rules follow the table, not the other way round
    w = max(len(head1), 100)

    lines.append("=" * w)
    lines.append("BASELINE COMPARISON")
    lines.append("=" * w)
    for k, v in context.items():
        lines.append(f"  {k:<12}: {v}")
    lines.append("")

    lines.append(head1)
    lines.append(head2)
    lines.append("  " + "-" * (len(head1) - 2))

    for m, v in agg.items():
        ref = m in ("initial", "optimal")
        eig = _fmt(v["min_eig_mean"], v["min_eig_sd"], ".1e")
        work = "-" if ref else _fmt(v["work_mean"], v["work_sd"], ".1f")
        best = "-" if ref else _fmt(v["best_at_mean"], v["best_at_sd"], ".1f")
        row = (f"  {m:<9}"
               f"{_fmt(v['edges_mean'], v['edges_sd']):>12}"
               f"{_fmt(v['score_mean'], v['score_sd']):>14}"
               f"{v['rigid_pct']:>8.0f}{v['minimal_pct']:>9.0f}"
               f"{eig:>19}{_fmt_geo(v):>17}{work:>11}{best:>12}")
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
    lines.append("  rigidity  how strongly the bearings react to a change in shape. Every")
    lines.append("            rigid network recovers its shape from exact bearings; larger")
    lines.append("            means it still does so under measurement noise, since shape")
    lines.append("            error scales as 1/sqrt(this). Its absolute size depends on how")
    lines.append("            far apart the agents are, so compare rows, not the number.")
    lines.append("  rigidity(geo)")
    lines.append("            the same margin as a geometric mean and spread, because it")
    lines.append("            ranges over orders of magnitude: 'a x/b' means the typical")
    lines.append("            network sits between a/b and a*b. A '*' marks rows where")
    lines.append("            non-rigid networks (margin exactly 0) had to be left out --")
    lines.append("            a zero cannot enter a geometric mean, so those rows describe")
    lines.append("            only the networks that came out rigid.")
    lines.append("  work      how many changes to the network the method actually made.")
    lines.append("  best@     the step at which its best network was found. Lower means it")
    lines.append("            converged faster; the rest of the budget added nothing.")
    if has_opt:
        lines.append("  =best     % of networks where the method tied the exhaustive optimum.")
    lines.append("")
    lines.append("HOW TO READ IT")
    lines.append("  Every value is a mean over the networks; '+-' is the standard deviation")
    lines.append("  across them, i.e. how much the method varies from one network to the next.")
    lines.append("  The percentage columns carry no '+-': they already are means of a yes/no")
    lines.append("  outcome, whose spread is fixed by the percentage itself.")
    lines.append("  'initial' and 'optimal' are reference points, not competing methods:")
    lines.append("  every method starts from 'initial', and 'optimal' is the best achievable.")
    lines.append("  A method is doing well when it approaches 'optimal' with low 'work'.")
    lines.append("  All methods are run on the same networks, so rows compare directly.")
    if any(r["method"] == "constructive" for r in rows):
        lines.append("  'constructive' is the one exception: it throws the initial edges away")
        lines.append("  and builds from nothing, because it is a construction, not an edit.")
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
# Light mode only, no hover layer, identity never by colour alone.
PANELS = [
    dict(field="score", title="Objective score  φ",
         note="rewards rigidity, charges for every edge - higher is better", log=False),
    dict(field="edges", title="Network size",
         note="directed bearing measurements in use - fewer is better", log=False),
    dict(field="rank", title="Rigidity matrix rank",
         note="the shape is fully determined once it reaches the dashed line", log=False),
    dict(field="min_eig", title="Rigidity margin",
         note="smallest nonzero eigenvalue of BᵀB - higher survives more noise", log=True),
]

# final / best / mean, the same three views the environment logs per episode
STAT_ORDER = ["final", "best", "mean"]
STAT_ALPHA = {"final": 0.95, "best": 0.55, "mean": 0.22}
STAT_BLURB = {
    "final": "the network the run ended on",
    "best":  "the best-scoring network the run passed through",
    "mean":  "averaged over every step of the run",
}


# ── headers and cards ─────────────────────────────────────────────────────────────────
def _wrap(text, width_in, fontsize):
    """Wrap to a physical width, since the strings here are long generated identifiers.

    0.60 em per character is measured, not nominal: it has to over-estimate slightly or
    the card is sized for fewer lines than it ends up drawing and the text runs out of it.
    """
    chars = max(24, int(width_in * 72.0 / (fontsize * 0.60)))
    return textwrap.wrap(text, chars) or [""]


def _header_height(header, width_in):
    """Inches the title block needs -- the caller sizes the figure around it."""
    return 0.30 + 0.16 * len(_header_detail_lines(header, width_in))


HEADER_FIELDS = [("model", "model"), ("env", "environment"), ("network", "network"),
                 ("objective", "objective"), ("policy", "policy")]


def _header_detail_lines(header, width_in):
    lines = []
    for key, label in HEADER_FIELDS:
        if header.get(key):
            lines += _wrap(f"{label:<11} : {header[key]}", width_in - 0.2, 8.0)
    bits = []
    if header.get("episodes"):
        # a frozen instance set is not "random networks, seed N"; saying so in a
        # figure that outlives the run is how a stale provenance line happens
        if header.get("benchmark"):
            bits.append(f"{header['episodes']} networks from benchmark "
                        f"{header['benchmark']}")
        else:
            bits.append(f"{header['episodes']} random networks")
            if header.get("seed") is not None:
                bits.append(f"seed {header['seed']}")
    elif header.get("seed") is not None:
        bits.append(f"seed {header['seed']}")
    if header.get("subtitle"):
        bits.append(header["subtitle"])
    if bits:
        lines.append(f"{'measured on':<11} : " + "  ·  ".join(bits))
    return lines


def _draw_header(fig, header, kind):
    """Title block: what the figure is, then the full env/model names, wrapped."""
    width_in, height_in = fig.get_size_inches()
    title = kind + (f"  -  {header['short']}" if header.get("short") else "")
    y = 1.0 - 0.26 / height_in
    fig.text(0.008, y, title, color=INK, fontsize=12.5, ha="left", va="top")
    y -= 0.30 / height_in
    for line in _header_detail_lines(header, width_in):
        fig.text(0.008, y, line, color=MUTED, fontsize=8.0, ha="left", va="top",
                 family="monospace")
        y -= 0.16 / height_in
    return y - 0.10 / height_in   # top of the plotting area, as a figure fraction


def _card_rows(methods, notes, width_in):
    """Wrapped card content plus the row count, so the figure can be sized for it.

    Two columns: the method key, then how the figure is built. When the notes are much
    longer than the method list they flow newspaper-style -- the rest of the left column
    first, continuing at the top of the right -- rather than leaving half the card empty.
    Splits happen between notes, never mid-sentence.
    """
    left = [("METHODS", None, None)]
    for m in methods:
        style = METHOD_STYLE.get(m, {"color": INK_2, "ls": "-"})
        left.append((m, METHOD_BLURB.get(m, ""), style))

    blocks = [_wrap(note, width_in * 0.42, 7.5) for note in notes]
    n_notes = sum(len(b) for b in blocks)
    heading = ("HOW IS THIS FIGURE BUILT?", True)

    if 1 + n_notes <= len(left) + 2:
        right = [heading] + [(line, False) for b in blocks for line in b]
        return left, right, max(len(left), len(right))

    # room to fill under the method key before the right column has to start
    target = (len(left) + 2 + 1 + n_notes) // 2
    used, first, second = len(left) + 2, [], []
    for block in blocks:
        # once the right column has started, everything follows it -- letting a later
        # short note slip back into the left column would break the reading order
        if not second and (not first or used + len(block) <= target):
            first += block
            used += len(block)
        else:
            second += block
    left = left + [("", None, None), ("HOW IS THIS FIGURE BUILT?", None, None)]
    left += [(line, None, "note") for line in first]
    return left, [(line, False) for line in second], max(len(left), len(second))


def _card_height(methods, notes, width_in):
    return 0.16 * _card_rows(methods, notes, width_in)[2] + 0.24


def _draw_card(fig, methods, notes, width_in, card_h):
    """The 'what am I looking at' card: method key on the left, how-it-works on the right.

    Drawn straight onto the figure in a reserved band rather than as a gridspec row --
    a row's height is scaled down by whatever the panels spend on their decorations, so
    the card came out shorter than the text it had to hold.
    """
    from matplotlib.patches import Rectangle

    left, right, rows = _card_rows(methods, notes, width_in)
    fig_h = fig.get_size_inches()[1]
    y0, y1 = 0.09 / fig_h, (card_h - 0.14) / fig_h
    x0, x1 = 0.035, 0.965

    fig.patches.append(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                                 facecolor=CARD, edgecolor=GRID, linewidth=1.0, zorder=0))

    step = (y1 - y0) / (rows + 0.8)
    top = y1 - step * 0.9
    col_l, col_r = x0 + 0.012, x0 + 0.47

    for i, (name, blurb, style) in enumerate(left):
        y = top - i * step
        if style == "note":                     # notes that spilled over from the right
            fig.text(col_l, y, name, color=INK_2, fontsize=7.5, va="center")
            continue
        if style is None:                       # column heading (or a blank spacer)
            fig.text(col_l, y, name, color=INK_2, fontsize=7.0, va="center",
                     family="monospace")
            continue
        fig.add_artist(_fig_line(fig, [col_l + 0.002, col_l + 0.030], [y, y],
                                 style["color"], style["ls"]))
        fig.text(col_l + 0.040, y, name, color=INK, fontsize=7.8, va="center")
        if blurb:
            fig.text(col_l + 0.108, y, blurb, color=INK_2, fontsize=7.5, va="center")

    for i, (line, heading) in enumerate(right):
        y = top - i * step
        fig.text(col_r, y, line, color=INK_2, fontsize=7.0 if heading else 7.5,
                 va="center", family="monospace" if heading else None)


def _fig_line(fig, xs, ys, color, ls):
    from matplotlib.lines import Line2D
    return Line2D(xs, ys, transform=fig.transFigure, color=color, linestyle=ls,
                  linewidth=2.4, solid_capstyle="round", zorder=3)


def _panel_title(ax, title, note):
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=16)
    ax.annotate(note, xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 5),
                textcoords="offset points", color=MUTED, fontsize=7.8, ha="left",
                va="bottom", annotation_clip=False)


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
        from matplotlib.ticker import NullFormatter
        ax.set_yscale("log")
        # decade labels only; the default adds 2x/3x/4x minor labels on a short range
        ax.yaxis.set_minor_formatter(NullFormatter())


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
    # methods that end on the same value all get pushed up; without this they would then
    # be clipped back onto each other at the top of the axis
    overflow = placed[-1][0] - 0.98
    if overflow > 0:
        placed = [(f - overflow, t) for f, t in placed]
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


def _figure(header, kind, methods, notes, panel_h=3.6, width=11.0):
    """A figure laid out as: title block, 2x2 panels, notes card. Sized to fit all three."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    head_h = _header_height(header, width)
    card_h = _card_height(methods, notes, width)
    fig_h = 2 * panel_h + head_h + card_h
    fig = plt.figure(figsize=(width, fig_h), facecolor=SURFACE)
    axes = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
    top = _draw_header(fig, header, kind)
    return fig, axes, (card_h, card_h / fig_h), top, width


def _finish(fig, card, methods, notes, width, top, run_dir, name):
    card_h, bottom = card
    fig.tight_layout(rect=(0, bottom, 1, top), h_pad=1.6, w_pad=2.0)
    _draw_card(fig, methods, notes, width, card_h)
    return _save(fig, run_dir, name)


def plot_trajectories(run_dir, traces, rows, header, filename="trajectories",
                      aggregate_over_episodes=True):
    """How each method's network evolves over the run."""
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

    notes = [
        "x axis: one step of the run. greedy has no step budget - it contributes one "
        "point per edge change it applies, and stops when no single change helps.",
        ("Each line is the mean over the networks; the shaded band is the middle 50% of them "
         "(25th-75th percentile)." if aggregate_over_episodes else
         "Each line is a single network - this is one episode, not an average."),
        "A method that finishes early is held at its last network for the rest of the axis, "
        "so the curves stay comparable.",
        "Every method is run on the same networks from the same starting graph, so the "
        "curves can be read against each other directly.",
    ]
    if ref:
        notes.append("Dashed reference lines mark full rigidity and, where exhaustive "
                     "search ran, the optimum it found.")

    fig, axes, card, top, width = _figure(header, "Run trajectories", methods, notes)
    for ax, panel in zip(axes, PANELS):
        _draw_panel(ax, traces, panel["field"], panel["log"], methods, ref,
                    aggregate_over_episodes)
        _panel_title(ax, panel["title"], panel["note"])
    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


# ── final / best / mean ───────────────────────────────────────────────────────────────
def outcome_stats(traces):
    """Per method, the three views of each run: {method: {stat: [per-episode dict]}}.

    Mirrors what the environment logs per episode, so a baselines figure and a
    tensorboard curve mean the same thing by "final", "best" and "mean".
    """
    per = {}
    for t in traces:
        per.setdefault((t["method"], t["episode"]), []).append(t)

    def snapshot(p):
        return {"score": float(p["score"]), "edges": float(p["edges"]),
                "rigid": float(bool(p["is_IBR"])), "minimal": float(bool(p["is_MBR"])),
                "min_eig": None if p.get("min_eig") is None else float(p["min_eig"])}

    out = {}
    for (method, _ep), pts in per.items():
        pts = sorted(pts, key=lambda p: p["step"])
        eig = [p["min_eig"] for p in pts if p.get("min_eig") is not None]
        rec = out.setdefault(method, {s: [] for s in STAT_ORDER})
        rec["final"].append(snapshot(pts[-1]))
        rec["best"].append(snapshot(max(pts, key=lambda p: p["score"])))
        rec["mean"].append({
            "score": float(np.mean([p["score"] for p in pts])),
            "edges": float(np.mean([p["edges"] for p in pts])),
            # a fraction of the run rather than a fraction of the runs -- said so on the card
            "rigid": float(np.mean([bool(p["is_IBR"]) for p in pts])),
            "minimal": float(np.mean([bool(p["is_MBR"]) for p in pts])),
            "min_eig": float(np.mean(eig)) if eig else None,
        })
    return out


OUTCOME_PANELS = [
    dict(field="score", title="Objective score  φ",
         note="rewards rigidity, charges for every edge - higher is better",
         log=False, scale=1.0, fmt="{:.0f}"),
    dict(field="edges", title="Network size",
         note="directed bearing measurements in use - fewer is better",
         log=False, scale=1.0, fmt="{:.1f}"),
    dict(field="rigid", title="Rigidity achieved",
         note="% of networks (final, best) or % of the run spent rigid (mean)",
         log=False, scale=100.0, fmt="{:.0f}"),
    dict(field="min_eig", title="Rigidity margin",
         note="smallest nonzero eigenvalue of BᵀB - higher survives more noise",
         log=True, scale=1.0, fmt="{:.1e}"),
]


def plot_outcomes(run_dir, traces, rows, header, filename="outcomes"):
    """final / best / mean side by side, for every method that was run."""
    stats = outcome_stats(traces)
    methods = [m for m in METHOD_ORDER if m in stats]
    methods += [m for m in stats if m not in methods]
    if not methods:
        return None

    notes = [f"{s}: {STAT_BLURB[s]}" for s in STAT_ORDER]
    notes += [
        "Bars are the mean over the networks; the whisker is ±1 standard deviation "
        "across them.",
        "For a method that never moves (initial) or only improves (greedy), final and "
        "best are the same bar by construction.",
        "The rigidity margin is plotted on a log axis; a non-rigid network has margin 0 "
        "and cannot be drawn there.",
    ]

    fig, axes, card, top, width = _figure(header, "Final / best / mean outcome",
                                             methods, notes)
    x = np.arange(len(methods))
    w = 0.26

    for ax, panel in zip(axes, OUTCOME_PANELS):
        field, scale = panel["field"], panel["scale"]
        colors = [METHOD_STYLE.get(m, {}).get("color", INK_2) for m in methods]
        # a percentage of runs has a Bernoulli spread that says nothing useful, so the
        # rigidity panel gets bars only
        show_err = field != "rigid"
        drawn = []
        for k, stat in enumerate(STAT_ORDER):
            vals, errs = [], []
            for m in methods:
                got = [d[field] for d in stats[m][stat] if d.get(field) is not None]
                vals.append(float(np.mean(got)) * scale if got else np.nan)
                errs.append((float(np.std(got)) * scale if got else 0.0) if show_err else 0.0)
            offset = (k - 1) * w
            ax.bar(x + offset, np.nan_to_num(vals), w * 0.92,
                   color=colors, alpha=STAT_ALPHA[stat], zorder=3,
                   edgecolor=colors, linewidth=0.8)
            if show_err:
                ax.errorbar(x + offset, np.nan_to_num(vals), yerr=errs, fmt="none",
                            ecolor=AXIS, elinewidth=1.0, capsize=2.5, zorder=4)
            drawn.append((vals, errs, stat))

        _style_axes(ax, log=panel["log"])
        if panel["log"]:
            pos = [v for vals, _e, _s in drawn for v in vals
                   if v is not None and np.isfinite(v) and v > 0]
            if pos:
                # room above the tallest bar for its (rotated) value label
                ax.set_ylim(10 ** np.floor(np.log10(min(pos))) / 2, max(pos) * 8)
        elif field == "rigid":
            ax.set_ylim(0, 118)
        else:
            hi = max((v + e for vals, errs, _s in drawn
                      for v, e in zip(vals, errs) if np.isfinite(v)), default=1.0)
            ax.set_ylim(0, hi * 1.20 if hi > 0 else 1.0)

        # value labels, so the three alphas never have to be told apart by eye alone
        for vals, errs, stat in drawn:
            k = STAT_ORDER.index(stat)
            for xi, (v, e) in enumerate(zip(vals, errs)):
                if not np.isfinite(v):
                    continue
                ax.annotate(panel["fmt"].format(v), xy=(xi + (k - 1) * w, v + e),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=6.5, color=INK_2, rotation=90 if panel["log"] else 0,
                            va="bottom", zorder=6)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8.5)
        # which bar is which -- one line under the axis beats a legend or a rotated
        # label per bar, and the shading order is the same in every panel
        ax.set_xlabel("bars in each group, left to right:   final  ·  best  ·  mean",
                      color=MUTED, fontsize=7.5, labelpad=6)
        _panel_title(ax, panel["title"], panel["note"])

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


def plot_summary(run_dir, rows, header):
    """Spread across networks of the outcome each method is scored on."""
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    colors = [METHOD_STYLE.get(m, {}).get("color", INK_2) for m in methods]

    notes = [
        "One value per network per method. The line is the median, the box the middle "
        "50%, the whiskers reach 1.5x that range and the dots are networks outside it.",
        "Scored on the best network each run visited - that is what the comparison table "
        "reports, and it separates 'found a good topology' from 'stopped on it'.",
        "'rigid' means the network's shape is fully determined by its bearing measurements; "
        "'also minimal' means it does that with the fewest possible edges.",
        "'steps to best' counts steps for the rollout methods and applied edge changes for "
        "greedy, so compare it within a method rather than across them.",
    ]

    fig, axes, card, top, width = _figure(header, "Outcome across networks",
                                             methods, notes)

    def box(ax, data, labels, cols, title, note):
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color=INK),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED,
                                        markeredgecolor="none"))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
            patch.set_edgecolor(c)
        for part in ("whiskers", "caps"):
            for artist in bp[part]:
                artist.set_color(AXIS)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8.5)
        _style_axes(ax)
        _panel_title(ax, title, note)

    box(axes[0], [[r["m"] for r in rows if r["method"] == m] for m in methods],
        methods, colors, "Network size",
        "edges in the best network found - fewer is better")
    box(axes[1], [[r["score"] for r in rows if r["method"] == m] for m in methods],
        methods, colors, "Objective score  φ",
        "score of the best network found - higher is better")

    ax = axes[2]
    rigid = [100 * np.mean([r["is_IBR"] for r in rows if r["method"] == m]) for m in methods]
    minimal = [100 * np.mean([r["is_MBR"] for r in rows if r["method"] == m]) for m in methods]
    x = np.arange(len(methods))
    ax.bar(x - 0.19, rigid, 0.36, color=colors, alpha=0.9, zorder=3)
    ax.bar(x + 0.19, minimal, 0.36, color=colors, alpha=0.4, zorder=3)
    for xi, (a, b) in enumerate(zip(rigid, minimal)):
        ax.annotate(f"{a:.0f}", xy=(xi - 0.19, a), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK_2)
        ax.annotate(f"{b:.0f}", xy=(xi + 0.19, b), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8.5)
    ax.set_ylim(0, 112)
    _style_axes(ax)
    _panel_title(ax, "Networks solved",
                 "% rigid (solid) and % also using the fewest edges (faded)")

    ax = axes[3]
    roll = [m for m in methods if m not in ("initial", "optimal")]
    if roll:
        box(ax, [[r.get("best_at", 0) for r in rows if r["method"] == m] for m in roll],
            roll, [METHOD_STYLE.get(m, {}).get("color", INK_2) for m in roll],
            "Steps to the best network",
            "how long each method took to reach its best - lower converges sooner")
    else:
        _style_axes(ax)
        _panel_title(ax, "Steps to the best network", "no rollout method was run")

    return _finish(fig, card, methods, notes, width, top, run_dir, "summary")


# ── the table, as a figure ────────────────────────────────────────────────────────────
# Same numbers as summary.txt, laid out rather than printed: the direction of each column
# is a subtitle instead of a "(fewer)" tag, reference rows are drawn as reference rows,
# and it drops into a slide next to the other figures without a monospace dump.
TABLE_COLUMNS = [
    dict(key="method",  title="method",   unit="",                 w=1.05, align="left"),
    dict(key="edges",   title="edges",    unit="fewer is better",  w=1.15, align="right"),
    dict(key="score",   title="score  φ", unit="higher is better", w=1.20, align="right"),
    dict(key="rigid",   title="rigid",    unit="% of networks",    w=0.85, align="right"),
    dict(key="minimal", title="minimal",  unit="% of networks",    w=0.85, align="right"),
    dict(key="margin",  title="margin",   unit="mean ± sd, higher is better",
         w=1.50, align="right"),
    dict(key="margin_geo", title="margin (geo)", unit="gmean ×/÷ gsd",
         w=1.25, align="right"),
    dict(key="work",    title="work",     unit="edits applied",    w=1.05, align="right"),
    dict(key="best_at", title="best at",  unit="step reached",     w=1.10, align="right"),
    dict(key="opt",     title="= best",   unit="% matched",        w=0.80, align="right"),
]

TABLE_NOTES = [
    "edges: directed bearing measurements the network needs - each one is a sensing or "
    "communication link, so fewer is better.",
    "score φ: the objective every method is scored with. It rewards rigidity and charges "
    "for each extra edge.",
    "rigid: the network's shape is fully determined by its bearing measurements - the "
    "property being solved for. minimal: rigid with the fewest possible edges "
    "(a heuristic on mixed-domain networks, so it can under-report).",
    "margin: how strongly the bearings react to a change in shape. Every rigid network "
    "recovers its shape from exact bearings; larger means it still does so under "
    "measurement noise, since shape error scales as 1/sqrt(margin). Its absolute size "
    "depends on how far apart the agents are, so compare rows rather than the number.",
    "margin (geo): the same quantity as a geometric mean and spread, because it ranges "
    "over orders of magnitude - 'a ×/÷ b' means the typical network sits between a/b and "
    "a·b. A '*' marks rows where non-rigid networks had to be left out: their margin is "
    "exactly 0, which no geometric mean can take, so those rows describe only the "
    "networks that came out rigid.",
    "work: changes to the network the method actually applied. best at: the step its best "
    "network was reached - lower means it converged sooner and the rest of the budget "
    "added nothing.",
    "= best: share of networks where the method tied the exhaustive optimum.",
    "initial and optimal are reference rows, not competing methods: every method starts "
    "from initial, and optimal is the best achievable. All methods see the same networks. "
    "constructive is the exception: it discards the initial edges and builds from empty, "
    "because it is a construction algorithm rather than an edit one.",
    "Every value is a mean over the networks and ± is the standard deviation across them. "
    "The percentage columns carry no ±: they are already means of a yes/no outcome, whose "
    "spread is fixed by the percentage itself.",
]


def plot_table(run_dir, rows, header, filename="table", width=12.0):
    """The comparison table as a figure, so it can ship next to the plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    agg = aggregate(rows)
    if not agg:
        return None
    methods = list(agg)
    has_opt = any(v["matches_opt_pct"] is not None for v in agg.values())
    cols = [c for c in TABLE_COLUMNS if c["key"] != "opt" or has_opt]
    notes = [n for n in TABLE_NOTES if has_opt or not n.startswith("= best")]

    head_h = _header_height(header, width)
    card_h = _card_height(methods, notes, width)
    row_h, head_row_h = 0.30, 0.52
    table_h = head_row_h + row_h * len(methods) + 0.30
    fig_h = head_h + table_h + card_h
    fig = plt.figure(figsize=(width, fig_h), facecolor=SURFACE)
    top = _draw_header(fig, header, "Baseline comparison")

    def cell(method, key, v):
        ref = method in ("initial", "optimal")
        if key == "edges":
            return f"{v['edges_mean']:.2f} ±{v['edges_sd']:.2f}"
        if key == "score":
            return f"{v['score_mean']:.2f} ±{v['score_sd']:.2f}"
        if key == "rigid":
            return f"{v['rigid_pct']:.0f}"
        if key == "minimal":
            return f"{v['minimal_pct']:.0f}"
        if key == "margin":
            return ("-" if v["min_eig_mean"] is None
                    else _fmt(v["min_eig_mean"], v["min_eig_sd"], ".1e", " ±"))
        if key == "margin_geo":
            return "-" if v["min_eig_gmean"] is None else _fmt_geo(v, times=" ×/÷")
        if key == "work":
            return "-" if ref else _fmt(v["work_mean"], v["work_sd"], ".1f", " ±")
        if key == "best_at":
            return "-" if ref else _fmt(v["best_at_mean"], v["best_at_sd"], ".1f", " ±")
        if key == "opt":
            return "-" if v["matches_opt_pct"] is None else f"{v['matches_opt_pct']:.0f}"
        return ""

    # column geometry, in figure fractions
    x0, x1 = 0.035, 0.965
    span = x1 - x0
    total = sum(c["w"] for c in cols)
    edges, acc = [], x0
    for c in cols:
        edges.append((acc, acc + span * c["w"] / total))
        acc += span * c["w"] / total

    y_top = top - 0.10 / fig_h
    y_head = y_top - head_row_h / fig_h

    for c, (cx0, cx1) in zip(cols, edges):
        x = cx0 + 0.006 if c["align"] == "left" else cx1 - 0.006
        ha = "left" if c["align"] == "left" else "right"
        fig.text(x, y_top - 0.16 / fig_h, c["title"], color=INK, fontsize=9.5,
                 ha=ha, va="center")
        if c["unit"]:
            fig.text(x, y_top - 0.34 / fig_h, c["unit"], color=MUTED, fontsize=7.0,
                     ha=ha, va="center")

    fig.add_artist(_fig_line(fig, [x0, x1], [y_head, y_head], AXIS, "-"))

    for i, m in enumerate(methods):
        v = agg[m]
        ref = m in ("initial", "optimal")
        yc = y_head - (i + 0.5) * row_h / fig_h
        if i % 2 == 1:
            fig.patches.append(Rectangle((x0, yc - 0.5 * row_h / fig_h), span,
                                         row_h / fig_h, transform=fig.transFigure,
                                         facecolor=CARD, edgecolor="none", zorder=0))
        style = METHOD_STYLE.get(m, {"color": INK_2, "ls": "-"})
        mx0 = edges[0][0]
        fig.add_artist(_fig_line(fig, [mx0 + 0.006, mx0 + 0.030], [yc, yc],
                                 style["color"], style["ls"]))
        fig.text(mx0 + 0.038, yc, m, color=INK_2 if ref else INK, fontsize=9, va="center")
        for c, (cx0, cx1) in list(zip(cols, edges))[1:]:
            fig.text(cx1 - 0.006, yc, cell(m, c["key"], v), color=INK_2 if ref else INK,
                     fontsize=9, ha="right", va="center", family="monospace")

    _draw_card(fig, methods, notes, width, card_h)
    return _save(fig, run_dir, filename)


def _save(fig, run_dir, name):
    """Every figure ships in both formats, filed by format: plots/pdf/ and plots/png/.
    PDF is what goes in the thesis, PNG is what you actually look at, and mixing them in
    one directory meant scrolling past each figure twice."""
    import matplotlib.pyplot as plt
    out = []
    for ext in PLOT_FORMATS:
        directory = os.path.join(run_dir, "plots", ext)
        os.makedirs(directory, exist_ok=True)   # callers may pass a bare --out-dir
        path = os.path.join(directory, f"{name}.{ext}")
        fig.savefig(path, facecolor=SURFACE, dpi=200, bbox_inches="tight")
        out.append(path)
    plt.close(fig)
    return out
