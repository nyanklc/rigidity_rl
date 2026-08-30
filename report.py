"""Rendering for evaluation.py: the comparison table, the CSVs and the plots.

The table is written for someone who does not know the topic: every column says which
direction is better, jargon is spelled out in a legend, and the two different meanings the
old `steps` column carried are split into `work` and `best@`.
"""

import copy
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


def _fmt_geo(v, key="min_eig", times=" x/", mark="*"):
    """`gmean x/ gsd`, flagged when non-rigid networks had to be left out.

    Serves both spectral columns; neither 0 nor inf can enter a geometric mean.
    """
    gmean, gsd = v[f"{key}_gmean"], v[f"{key}_gsd"]
    if gmean is None:
        return "-"
    partial = mark if v[f"{key}_n_pos"] < v[f"{key}_n"] else ""
    # one surviving network has no spread; printing "x/1.0" would read as "no spread"
    # rather than "nothing to spread over"
    if v[f"{key}_n_pos"] < 2:
        return f"{gmean:.1e}{partial}"
    return f"{gmean:.1e}{times}{gsd:.1f}{partial}"


def _fmt(mean, sd=None, spec=".2f", pm="+-"):
    """mean +- sd in a given number format. Every value in the table is a mean over the
    networks, so every one of them carries its spread."""
    if mean is None:
        return "-"
    if sd is None:
        return f"{mean:{spec}}"
    return f"{mean:{spec}}{pm}{sd:{spec}}"


def _noise_summary(sel):
    """{sigma: (measured gmean, predicted gmean, failed fraction)} per noise level.

    `failed` is the share of recoveries that blew up rather than landing near the
    truth. Those carry no error to average, so they are counted, not averaged.
    """
    def usable(r, sigma):
        v = r.get("noise", {}).get(sigma)
        return (v is not None and np.isfinite(v) and v > 0
                and r.get("pred_err") and np.isfinite(r["pred_err"]))

    out = {}
    for sigma in sorted({s for r in sel for s in r.get("noise", {})}):
        rows_here = [r for r in sel if usable(r, sigma)]
        failed = [r.get("noise_failed", {}).get(sigma, 0.0) for r in sel
                  if r.get("noise_failed") is not None]
        if rows_here:
            out[sigma] = (_gmean([r["noise"][sigma] for r in rows_here]),
                          _gmean([sigma * r["pred_err"] for r in rows_here]),
                          float(np.mean(failed)) if failed else 0.0)
    return out


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
        # None exactly where the network came out flexible; the count is kept so
        # the table can mark the row
        err = [r["shape_err"] for r in sel
               if r.get("shape_err") is not None and r["shape_err"] > 0]
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
            # stiffness spans decades, so an arithmetic sd implies a range crossing
            # zero. The geometric pair is the honest spread for it: a multiplicative
            # factor you divide and multiply the geometric mean by.
            "min_eig_gmean": _gmean(pos),
            "min_eig_gsd": _gsd(pos),
            # a non-rigid network has stiffness exactly 0, which no geometric mean can
            # take -- record how many networks the geometric pair actually saw
            "min_eig_n": len(eig),
            "min_eig_n_pos": len(pos),
            "noise": _noise_summary(sel),
            "shape_err_gmean": _gmean(err),
            "shape_err_gsd": _gsd(err),
            "shape_err_n": len(sel),
            "shape_err_n_pos": len(err),
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
             f"{'stiffness(geo)':>17}{'shape err':>16}{'work':>11}{'best@':>12}")
    head2 = (f"  {'':<9}{'(fewer)':>12}{'(higher)':>14}{'%':>8}{'%':>9}"
             f"{'gmean x/gsd':>17}{'gmean x/gsd':>16}{'edits':>11}{'step':>12}")
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
        work = "-" if ref else _fmt(v["work_mean"], v["work_sd"], ".1f")
        best = "-" if ref else _fmt(v["best_at_mean"], v["best_at_sd"], ".1f")
        row = (f"  {m:<9}"
               f"{_fmt(v['edges_mean'], v['edges_sd']):>12}"
               f"{_fmt(v['score_mean'], v['score_sd']):>14}"
               f"{v['rigid_pct']:>8.0f}{v['minimal_pct']:>9.0f}"
               f"{_fmt_geo(v):>17}{_fmt_geo(v, key='shape_err'):>16}"
               f"{work:>11}{best:>12}")
        if has_opt:
            row += ("-" if v["matches_opt_pct"] is None
                    else f"{v['matches_opt_pct']:.0f}").rjust(7)
        lines.append(row)
    lines.append("")

    sweep = {m: v["noise"] for m, v in agg.items() if v["noise"]}
    if sweep:
        sigmas = sorted({s for v in sweep.values() for s in v})
        lines.append("MEASURED SHAPE ERROR UNDER BEARING NOISE  (predicted in brackets)")
        lines.append("  " + "method".ljust(13)
                     + "".join(f"{np.degrees(s):.2f} deg".rjust(22) for s in sigmas))
        marked = False
        for m, v in sweep.items():
            cells = ""
            for s in sigmas:
                if s not in v:
                    cells += "-".rjust(22)
                    continue
                got, pred, failed = v[s]
                mark = "*" if failed > 0.005 else ""
                cells += f"{got:.3f} [{pred:.3f}]{mark}".rjust(22)
                marked = marked or bool(mark)
            lines.append(f"  {m:<13}{cells}")
        if marked:
            lines.append("  * some recoveries blew up at that noise level and are not "
                         "in the average")
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
    lines.append("  stiffness how strongly the bearings react to a change in shape, as a")
    lines.append("            geometric mean and spread since it ranges over orders of")
    lines.append("            magnitude: 'a x/b' means the typical network sits between a/b")
    lines.append("            and a*b. Higher is better. Its absolute size depends on how far")
    lines.append("            apart the agents are, so compare rows, not the number.")
    lines.append("  shape err how far the recovered formation is from the true one, per")
    lines.append("            radian of error in the bearing measurements. Position is")
    lines.append("            counted in formation radii and attitude in radians, so the")
    lines.append("            number is a fraction: 8.0 means one degree of bearing error")
    lines.append("            (0.017 rad) displaces the shape by about 14% of its own size.")
    lines.append("            LOWER is better, and unlike stiffness it is comparable across")
    lines.append("            network sizes, domains and pose ranges.")
    lines.append("            A '*' on either column marks rows where non-rigid networks had")
    lines.append("            to be left out -- their stiffness is 0 and their shape error is")
    lines.append("            infinite, and neither can enter a geometric mean, so those rows")
    lines.append("            describe only the networks that came out rigid.")
    lines.append("  work      how many changes to the network the method actually made.")
    lines.append("  best@     the step at which its best network was found. Lower means it")
    lines.append("            converged faster; the rest of the budget added nothing.")
    if has_opt:
        lines.append("  =best     % of networks where the method tied the exhaustive optimum.")
    lines.append("")
    if sweep:
        lines.append("  The noise block is what actually happens when every bearing is")
        lines.append("  perturbed by that many degrees and the formation is recovered from")
        lines.append("  the noisy measurements: RMS position error in formation radii. The")
        lines.append("  bracketed number is what the rigidity matrix predicts. They agree")
        lines.append("  while the error stays small; a measured value far below the")
        lines.append("  prediction means the noise is too large for the prediction to hold.")
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
                 "min_eig", "shape_err", "work", "best_at"]
TRACE_FIELDS = ["episode", "method", "step", "score", "edges", "rank", "rank_K",
                "is_IBR", "is_MBR", "min_eig", "shape_err"]


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
    dict(field="min_eig", title="Stiffness",
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


# two lines of title, in inches, reserved above every 3-D panel
PANEL_TITLE_BAND = 0.52


def _title_band(fig):
    return PANEL_TITLE_BAND / fig.get_figheight()


def _panel_title_3d(ax, title, note):
    """Titles for a 3-D panel, in the band _formation_axes leaves free above it."""
    fig = ax.figure
    x0, y0, w, h = ax._panel_rect
    band = _title_band(fig)
    fig.text(x0 + 0.02 * w, y0 + h, title, color=INK, fontsize=11,
             ha="left", va="top")
    fig.text(x0 + 0.02 * w, y0 + h - band * 0.52, note, color=MUTED, fontsize=8.2,
             ha="left", va="top")


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


def _figure(header, kind, methods, notes, panel_h=3.6, width=11.0, panels=(2, 2),
            axes3d=False):
    """A figure laid out as: title block, panels, notes card. Sized to fit all three."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    head_h = _header_height(header, width)
    card_h = _card_height(methods, notes, width)
    nrows, ncols = panels
    fig_h = nrows * panel_h + head_h + card_h
    fig = plt.figure(figsize=(width, fig_h), facecolor=SURFACE)
    # a 3-D caller adds its own projection="3d" panels
    axes = ([] if axes3d
            else [fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)])
    top = _draw_header(fig, header, kind)
    return fig, axes, (card_h, card_h / fig_h), top, width


def _finish(fig, card, methods, notes, width, top, run_dir, name, tight=True):
    card_h, bottom = card
    # a figure that placed its own panels (the 3-D ones) must not be re-laid out
    if tight:
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

    Mirrors what the environment logs per episode, so an evaluation figure and a
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
    dict(field="min_eig", title="Stiffness",
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
        "Stiffness is plotted on a log axis; a non-rigid network has stiffness 0 "
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


def plot_noise_sweep(run_dir, rows, header, filename="noise"):
    """Measured shape error against bearing noise, per method, over the prediction."""
    agg = aggregate(rows)
    sweep = {m: v["noise"] for m, v in agg.items() if v["noise"]}
    if not sweep:
        return None

    methods = list(sweep)
    notes = [
        "Each line is one method: the formation it produced, with every bearing "
        "perturbed by the noise on the x axis, recovered from those noisy "
        "measurements and compared against the truth.",
        "y is RMS position error in formation radii, so 0.1 means the recovered "
        "shape is off by a tenth of the formation's own size. Lower is better.",
        "The dashed line is what the rigidity matrix predicts for the same "
        "topology. Measurement following prediction means the topology's "
        "conditioning explains the error; measurement falling below it means the "
        "noise is past the point where the prediction applies.",
    ]
    fig, axes, card, top, width = _figure(
        header, "Shape error under bearing noise", methods, notes,
        panel_h=4.6, width=11.0, panels=(1, 1))
    ax = axes[0]

    for m, v in sweep.items():
        style = METHOD_STYLE.get(m, {"color": INK_2, "ls": "-", "z": 2})
        sig = np.array(sorted(v))
        got = np.array([v[s][0] for s in sig])
        pred = np.array([v[s][1] for s in sig])
        ax.plot(np.degrees(sig), got, "-o", color=style["color"], linewidth=2.0,
                markersize=4, label=m, zorder=style["z"] + 2)
        ax.plot(np.degrees(sig), pred, "--", color=style["color"], alpha=0.45,
                linewidth=1.5, zorder=style["z"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bearing noise per measurement (degrees)")
    ax.set_ylabel("RMS position error (formation radii)")
    _style_axes(ax, log=True)
    _panel_title(ax, "Measured against predicted",
                 "solid: measured   dashed: predicted from the rigidity matrix")
    ax.legend(frameon=False, fontsize=8)

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


# ── formation drawings ────────────────────────────────────────────────────────────────
# These draw the network itself rather than a statistic of it, in real 3-D: a bearing
# formation is a spatial object and a flat projection hides the axis it is worst
# determined along. Marker shape carries the agent's domain, which is the other thing
# a plain scatter throws away on a mixed formation.
DOMAIN_MARKER = {
    "R^2":     ("s", "planar, no heading"),
    "R^2xS^1": ("^", "planar with heading"),
    "R^3":     ("o", "spatial, no heading"),
    "R^3xS^1": ("D", "spatial, one rotation axis"),
    "SE(3)":   ("P", "full pose"),
}
DOMAIN_ORDER = ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"]


def _domains_present(net):
    seen = {a.domain for a in net.agents}
    return [d for d in DOMAIN_ORDER if d in seen]


def _domain_note(net):
    """The marker key, as a card line, only when there is more than one domain."""
    present = _domains_present(net)
    if len(present) < 2:
        return None
    return "Marker shape is the agent's domain: " + ", ".join(
        f"{DOMAIN_MARKER[d][0]} {d} ({DOMAIN_MARKER[d][1]})" for d in present) + "."


def _formation_cell(nrows, ncols, index, band):
    """The rect of panel `index` (1-based) inside the band left by header and card."""
    top, bottom = band
    row, col = divmod(index - 1, ncols)
    cell_w, cell_h = 1.0 / ncols, (top - bottom) / nrows
    return col * cell_w, top - (row + 1) * cell_h, cell_w, cell_h


def _formation_axes(fig, nrows, ncols, index, positions, band, pad=0.14):
    """A 3-D panel with equal scale on every axis and recessive furniture.

    Placed explicitly rather than through tight_layout, which does not know about
    the title band a 3-D panel needs above it.
    """
    x0, y0, w, h = _formation_cell(nrows, ncols, index, band)
    ax = fig.add_axes([x0 + 0.012 * w, y0, w * 0.98, h], projection="3d")
    ax._panel_rect = (x0, y0, w, h)
    centre = positions.mean(axis=0)
    span = float(np.abs(positions - centre).max()) * (1.0 + pad) or 1.0
    for setter, c in ((ax.set_xlim, centre[0]), (ax.set_ylim, centre[1]),
                      (ax.set_zlim, centre[2])):
        setter(c - span, c + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(SURFACE)
        axis.pane.set_edgecolor(AXIS)
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.6
    ax.tick_params(colors=MUTED, labelsize=6.5, pad=1)   # negative pad clips minus signs
    # matplotlib leaves a wide margin round a 3-D box; the drawing is the point
    title_h = _title_band(fig)
    ax.set_position([x0 + 0.005 * w, y0 + 0.010, w * 0.99, h - title_h - 0.010])
    for lab, setter in (("x", ax.set_xlabel), ("y", ax.set_ylabel), ("z", ax.set_zlabel)):
        setter(lab, color=MUTED, fontsize=7, labelpad=-6)
    return ax


def _draw_agents(ax, net, P, size=52, extra=None):
    """One scatter per domain so the marker shape reads as the domain."""
    doms = [a.domain for a in net.agents]
    for d in _domains_present(net):
        idx = [i for i, x in enumerate(doms) if x == d]
        sizes = size if extra is None else size + extra[idx]
        ax.scatter(P[idx, 0], P[idx, 1], P[idx, 2], s=sizes,
                   marker=DOMAIN_MARKER[d][0], facecolors=SURFACE, edgecolors=INK,
                   linewidths=1.1, depthshade=False, zorder=6)


def _draw_edges_3d(ax, P, edges, color=INK_2, alpha=0.28, width=0.8, widths=None):
    ii, jj = np.nonzero(edges)
    for k, (i, j) in enumerate(zip(ii, jj)):
        a, b = P[i], P[j]
        d = b - a
        lw = width if widths is None else widths[k]
        al = alpha if widths is None else min(0.95, 0.25 + 0.65 * lw / max(widths.max(), 1e-9))
        # stop short of the marker so the arrowhead is not buried in it
        ax.quiver(a[0], a[1], a[2], *(d * 0.88), color=color, alpha=al,
                  linewidth=lw, arrow_length_ratio=0.16, zorder=3)


def _ellipsoid(ax, cov3, centre, scale, color, res=18):
    """The 1-sigma ellipsoid of a 3x3 covariance, as a translucent shell."""
    w, V = np.linalg.eigh(cov3)
    w = np.maximum(w, 0.0)
    u = np.linspace(0, 2 * np.pi, 2 * res)
    v = np.linspace(0, np.pi, res)
    unit = np.stack([np.outer(np.cos(u), np.sin(v)),
                     np.outer(np.sin(u), np.sin(v)),
                     np.outer(np.ones_like(u), np.cos(v))])
    pts = (V @ (np.sqrt(w)[:, None] * unit.reshape(3, -1))) * scale
    x, y, z = (pts.reshape(3, *unit.shape[1:]) + centre[:, None, None])
    ax.plot_surface(x, y, z, color=color, alpha=0.38, linewidth=0, shade=False,
                    zorder=8)


def _nice_factor(x):
    """Round to something a caption can state: 1, 2, 5, 10, 20, 50, ..."""
    if x <= 1.5:
        return 1
    mag = 10 ** int(np.floor(np.log10(x)))
    return int(min((c * mag for c in (1, 2, 5, 10)), key=lambda c: abs(c - x)))


def _grid_for(count):
    """Panel grid that does not leave a hole: 1x1, 1x2, 2x2, then rows of three."""
    if count <= 2:
        return 1, count
    if count <= 4:
        return 2, 2
    return int(np.ceil(count / 3)), 3


def _panel_rows(rows, episode=0):
    """The rows one formation figure draws, in display order."""
    sel = [r for r in rows if r.get("episode") == episode
           and r.get("edges") is not None and r.get("is_IBR")]
    return sorted(sel, key=lambda r: METHOD_ORDER.index(r["method"])
                  if r["method"] in METHOD_ORDER else 99)


def _which_graph(row):
    return f"{row['m']} edges, {row.get('edges_are', 'final')}"


def plot_uncertainty(run_dir, instances, rows, header, sigma=0.0175,
                     filename="uncertainty"):
    """Where each agent could actually be, given noisy bearings, per method."""
    from rigidity import error_covariance, extended_bearing_rigidity_matrix
    if not instances:
        return None
    net = instances[0]
    ep_rows = _panel_rows(rows)
    if not ep_rows:
        return None

    P = np.array([a.pose.position for a in net.agents], dtype=float)
    radius = float(np.sqrt(np.mean(((P - P.mean(axis=0)) ** 2).sum(axis=1)))) or 1.0

    rank_K = int(np.linalg.matrix_rank(
        extended_bearing_rigidity_matrix(net.fully_connected())))
    blocks, worsts = [], []
    for row in ep_rows:
        work = copy.deepcopy(net)
        work.edges = row["edges"].copy()
        cov = error_covariance(extended_bearing_rigidity_matrix(work), rank_K)
        per_agent = [cov[3 * i:3 * i + 3, 3 * i:3 * i + 3] for i in range(work.n)]
        blocks.append(per_agent)
        worsts.append(max(float(np.sqrt(max(np.linalg.eigvalsh(c).max(), 0.0)))
                          for c in per_agent))

    # one factor for every panel, or the panels stop being comparable. It is set from
    # the MEDIAN panel: setting it from the worst made every other panel invisible.
    typical = float(np.median([w * sigma for w in worsts])) or 1.0
    exaggeration = _nice_factor(0.30 * radius / typical)

    methods = [r["method"] for r in ep_rows]
    notes = [
        f"One formation, one panel per method: identical agents and poses, only the "
        f"measured bearings differ. Drawn on the first evaluated network.",
        f"Each shell is where that agent ends up when every bearing carries "
        f"{np.degrees(sigma):.1f} degrees of error. Bigger means the topology pins that "
        f"agent down less well; the percentage above each panel is the worst agent, at "
        f"true scale.",
        f"Shells are drawn {exaggeration}x larger than life so they are visible at all, "
        f"and the same factor is used in every panel so the panels compare.",
        "Arrows run from the measuring agent to the one it measures.",
    ]
    dom = _domain_note(net)
    if dom:
        notes.append(dom)

    nrows, ncols = _grid_for(len(ep_rows))
    fig, _, card, top, width = _figure(
        header, "Where the noise puts each agent", methods, notes,
        panel_h=5.4, width=6.8 * ncols, panels=(nrows, ncols), axes3d=True)
    band = (top, card[1])

    for k, (row, per_agent, worst) in enumerate(zip(ep_rows, blocks, worsts)):
        ax = _formation_axes(fig, nrows, ncols, k + 1, P, band)
        work = copy.deepcopy(net)
        work.edges = row["edges"].copy()
        style = METHOD_STYLE.get(row["method"], {"color": INK_2})
        _draw_edges_3d(ax, P, work.edges)
        _draw_agents(ax, work, P)
        for i, cov3 in enumerate(per_agent):
            _ellipsoid(ax, cov3, P[i], sigma * exaggeration, style["color"])
        _panel_title_3d(ax, row["method"],
                        f"{_which_graph(row)}   worst agent off by "
                        f"{worst * sigma / radius:.1%} of the formation")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename, tight=False)


def plot_softest_mode(run_dir, instances, rows, header, filename="softest_mode"):
    """The deformation the bearings can barely see, per method."""
    from rigidity import (extended_bearing_rigidity_matrix, nullspace_and_softest,
                          rigidity_decomposition)
    if not instances:
        return None
    net = instances[0]
    ep_rows = _panel_rows(rows)
    if not ep_rows:
        return None

    P = np.array([a.pose.position for a in net.agents], dtype=float)
    span = float(np.abs(P - P.mean(axis=0)).max()) or 1.0
    rank_K = int(np.linalg.matrix_rank(
        extended_bearing_rigidity_matrix(net.fully_connected())))

    modes, lams = [], []
    for row in ep_rows:
        work = copy.deepcopy(net)
        work.edges = row["edges"].copy()
        brm = extended_bearing_rigidity_matrix(work)
        rank, _, lam = rigidity_decomposition(brm, rank_K)
        _, v, _, _ = nullspace_and_softest(brm, rank)
        modes.append(v[:3 * work.n, 0].reshape(-1, 3) if v.shape[1] else np.zeros_like(P))
        lams.append(lam)
    scales = [0.30 * span / (float(np.abs(d).max()) or 1.0) for d in modes]

    methods = [r["method"] for r in ep_rows]
    notes = [
        "The softest mode: the way of deforming the formation that changes the bearings "
        "least, and so the direction an estimator confuses most easily. Drawn on the "
        "first evaluated network.",
        "The mode is normalised and each panel is scaled to its own largest arrow, so "
        "the arrows show the SHAPE of that deformation - which agents move, together or "
        "against each other. Arrow lengths do not compare between panels.",
        "How soft it is, is the rigidity eigenvalue above each panel: smaller means the "
        "bearings resist that deformation less and the shape is pinned down worse. "
        "Larger is better.",
    ]
    dom = _domain_note(net)
    if dom:
        notes.append(dom)

    nrows, ncols = _grid_for(len(ep_rows))
    fig, _, card, top, width = _figure(
        header, "The deformation the bearings barely see", methods, notes,
        panel_h=5.4, width=6.8 * ncols, panels=(nrows, ncols), axes3d=True)
    band = (top, card[1])

    for k, (row, disp, lam, scale) in enumerate(zip(ep_rows, modes, lams, scales)):
        ax = _formation_axes(fig, nrows, ncols, k + 1, P, band)
        work = copy.deepcopy(net)
        work.edges = row["edges"].copy()
        style = METHOD_STYLE.get(row["method"], {"color": INK_2})
        _draw_edges_3d(ax, P, work.edges)
        _draw_agents(ax, work, P)
        d = disp * scale
        ax.quiver(P[:, 0], P[:, 1], P[:, 2], d[:, 0], d[:, 1], d[:, 2],
                  color=style["color"], linewidth=2.0, arrow_length_ratio=0.22, zorder=9)
        _panel_title_3d(ax, row["method"],
                        f"{_which_graph(row)}   rigidity eigenvalue {lam:.2e}")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename, tight=False)


def plot_sensitivity(run_dir, instances, rows, header, filename="sensitivity"):
    """Which measurements the shape error actually comes from, per method."""
    from rigidity import extended_bearing_rigidity_matrix, measurement_sensitivity
    if not instances:
        return None
    net = instances[0]
    ep_rows = _panel_rows(rows)
    if not ep_rows:
        return None

    P = np.array([a.pose.position for a in net.agents], dtype=float)
    methods = [r["method"] for r in ep_rows]
    notes = [
        "Noise on one bearing propagates into the recovered shape by a known amount, and "
        "those amounts add up to the total error exactly. So every measurement has a "
        "share of the error and the shares sum to 100%. Drawn on the first evaluated "
        "network.",
        "Arrow thickness is that single bearing's share. Marker size is the share "
        "contributed by every bearing that agent takes - a large marker is an agent "
        "whose own sensing the formation leans on.",
        "A formation that leans hard on one measurement is fragile in a way the edge "
        "count does not show.",
    ]
    dom = _domain_note(net)
    if dom:
        notes.append(dom)

    nrows, ncols = _grid_for(len(ep_rows))
    fig, _, card, top, width = _figure(
        header, "Where the error comes from", methods, notes,
        panel_h=5.4, width=6.8 * ncols, panels=(nrows, ncols), axes3d=True)
    band = (top, card[1])

    rank_K = int(np.linalg.matrix_rank(
        extended_bearing_rigidity_matrix(net.fully_connected())))

    for k, row in enumerate(ep_rows):
        ax = _formation_axes(fig, nrows, ncols, k + 1, P, band)
        work = copy.deepcopy(net)
        work.edges = row["edges"].copy()
        per_edge, per_node = measurement_sensitivity(work, rank_K)
        total = per_edge.sum()
        if not np.isfinite(total) or total <= 0:
            continue
        e_share, n_share = per_edge / total, per_node / total
        style = METHOD_STYLE.get(row["method"], {"color": INK_2})

        _draw_edges_3d(ax, P, work.edges, color=style["color"],
                       widths=0.7 + 6.0 * e_share)
        _draw_agents(ax, work, P, size=44, extra=1400 * n_share)
        _panel_title_3d(ax, row["method"],
                        f"{_which_graph(row)}   worst bearing {e_share.max():.0%}, "
                        f"worst agent {n_share.max():.0%} of the error")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename, tight=False)


def plot_prediction_check(run_dir, rows, header, filename="prediction"):
    """Does the rigidity matrix predict the error the noise actually causes?

    One point per (method, instance, noise level). On the diagonal means the
    topology's conditioning explains the error; below it means the noise is past
    the point where the analytic metric applies.
    """
    pts = [(r["method"], sigma * r["pred_err"], measured)
           for r in rows
           if r.get("noise") and r.get("pred_err") and np.isfinite(r["pred_err"])
           for sigma, measured in r["noise"].items()
           if measured is not None and np.isfinite(measured) and measured > 0]
    if len(pts) < 3:
        return None

    methods = [m for m in METHOD_ORDER if any(p[0] == m for p in pts)]
    notes = [
        "x is the error the rigidity matrix predicts for that network at that noise "
        "level; y is the error measured by actually perturbing every bearing and "
        "recovering the formation.",
        "On the dashed diagonal, the prediction is right and the analytic metric can "
        "be trusted. Points falling below it are where the noise is large enough "
        "that the linear theory stops applying.",
        "Both axes are RMS position error in formation radii. One point per method, "
        "network and noise level.",
    ]
    fig, axes, card, top, width = _figure(
        header, "Predicted against measured error", methods, notes,
        panel_h=4.6, width=11.0, panels=(1, 1))
    ax = axes[0]

    lo = min(min(p[1] for p in pts), min(p[2] for p in pts)) * 0.7
    hi = max(max(p[1] for p in pts), max(p[2] for p in pts)) * 1.4
    ax.plot([lo, hi], [lo, hi], "--", color=MUTED, linewidth=1.2, zorder=1)

    for m in methods:
        sel = [p for p in pts if p[0] == m]
        style = METHOD_STYLE.get(m, {"color": INK_2, "z": 2})
        ax.scatter([p[1] for p in sel], [p[2] for p in sel], s=26,
                   color=style["color"], alpha=0.75, edgecolors=SURFACE,
                   linewidths=0.6, label=m, zorder=style.get("z", 2) + 2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted RMS position error (formation radii)")
    ax.set_ylabel("measured RMS position error")
    _style_axes(ax, log=True)
    _panel_title(ax, "Prediction against measurement",
                 "on the diagonal: the rigidity matrix explains the error")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


def plot_repair_choice(run_dir, spread, header, filename="repair_choice"):
    """Among repairs of the same size, how much does the choice matter?

    `spread` is what evaluation.py collected: one record per broken instance with
    every minimum-size repair's shape error and where greedy's pick landed.
    """
    if not spread:
        return None

    notes = [
        "After a formation breaks, several different edge sets of the same minimum "
        "size restore rigidity. Each grey dot is one of them, on one broken network.",
        "Height is the shape error that repair leaves behind, relative to the best "
        "repair available on that network - so 1 is the best possible and 10 is ten "
        "times worse, for the same number of edges.",
        "The marked point is what marginal-gain greedy picked. A wide column means "
        "the choice matters; greedy sitting high in it means the count-optimal "
        "criterion does not make that choice well.",
    ]
    methods = ["greedy"]
    fig, axes, card, top, width = _figure(
        header, "Does it matter which repair you pick?", methods, notes,
        panel_h=4.6, width=11.0, panels=(1, 1))
    ax = axes[0]

    rng = np.random.default_rng(0)
    for k, rec in enumerate(spread):
        errs = np.asarray(rec["errors"], dtype=float)
        errs = errs[np.isfinite(errs) & (errs > 0)]
        if len(errs) < 2:
            continue
        rel = errs / errs.min()
        jitter = k + (rng.random(len(rel)) - 0.5) * 0.55
        ax.scatter(jitter, rel, s=16, color=MUTED, alpha=0.45, linewidths=0, zorder=2)
        if rec.get("greedy") and rec["greedy"] > 0:
            ax.scatter([k], [rec["greedy"] / errs.min()], s=70, marker="D",
                       color=METHOD_STYLE["greedy"]["color"], edgecolors=SURFACE,
                       linewidths=1.0, zorder=5,
                       label="greedy's pick" if k == 0 else None)

    import matplotlib.ticker as mticker
    ax.axhline(1.0, color=INK_2, linestyle="--", linewidth=1.1, zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("broken network")
    ax.set_ylabel("shape error, relative to the best repair of the same size")
    ax.set_xticks([])
    _style_axes(ax, log=True)
    # a log axis over one decade labels only the decade, which reads as "no scale"
    ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:g}x" if v >= 1 else ""))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    _panel_title(ax, "Every minimum-size repair, on each broken network",
                 "1x is the best repair available; higher is worse for the same edge count")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


DECISION_FIELDS = ["episode", "step", "kind", "phi_pct", "err_pct", "share_pct",
                   "phi_best", "err_best", "dphi", "derr"]


def write_decisions(run_dir, decisions):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "decisions.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=DECISION_FIELDS, extrasaction="ignore")
        wr.writeheader()
        for d in decisions:
            wr.writerow(d)


def plot_decisions(run_dir, decisions, header, filename="decisions"):
    """Where the policy's chosen edit ranked among all the edits it could have made.

    Scored two ways on purpose. phi is what it was trained on; shape error is what
    we care about but never asked for. The gap between the two panels says whether
    a shortfall belongs to the policy or to the objective.
    """
    if not decisions:
        return None

    panels = [
        ("phi_pct", "Ranked by phi", "the objective it was trained on"),
        ("err_pct", "Ranked by shape error", "the objective it was NOT trained on"),
        ("share_pct", "Ranked by how sensitive the agent is",
         "does it act where the error already is?"),
    ]
    kinds = ["add", "remove"]
    notes = [
        "At every step, every legal single-edge change is scored, and the one the "
        "policy actually made is ranked among them. 100% would be the best available "
        "edit; the dashed line at 50% is what picking at random scores.",
        "The policy is trained on phi, which rewards rank and charges for edges - not "
        "on shape error. A high phi panel with a middling error panel therefore says "
        "the objective is what leaves error behind, not the policy.",
        "The third panel asks whether the edit touches an agent already carrying much "
        "of the error. Adds and removes are separated because they mean opposite "
        "things: acting on a loaded agent is what you want when adding, and what you "
        "want to avoid when removing.",
    ]
    methods = ["learned"]
    fig, axes, card, top, width = _figure(
        header, "Were the policy's edits good ones?", methods, notes,
        panel_h=4.0, width=13.0, panels=(1, 3))

    for ax, (key, title, sub) in zip(axes, panels):
        data = [[d[key] for d in decisions if d["kind"] == k and d.get(key) is not None]
                for k in kinds]
        if not any(data):
            _style_axes(ax)
            _panel_title(ax, title, "no edits to rank")
            continue

        ax.axhline(50.0, color=INK_2, linestyle="--", linewidth=1.1, zorder=1)
        rng = np.random.default_rng(0)
        for pos, (k, vals) in enumerate(zip(kinds, data)):
            if not vals:
                continue
            colour = METHOD_STYLE["learned"]["color"] if k == "add" else INK_2
            ax.scatter(pos + (rng.random(len(vals)) - 0.5) * 0.5, vals, s=16,
                       color=colour, alpha=0.40, linewidths=0, zorder=2)
            ax.scatter([pos], [np.mean(vals)], s=110, marker="D", color=colour,
                       edgecolors=SURFACE, linewidths=1.2, zorder=5)
            ax.annotate(f"{np.mean(vals):.0f}%", (pos, np.mean(vals)), fontsize=8.5,
                        color=INK, ha="center", va="bottom",
                        xytext=(0, 11), textcoords="offset points", zorder=6)

        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels([f"{k}\n({len(v)})" for k, v in zip(kinds, data)])
        ax.set_ylim(-4, 108)
        ax.set_ylabel("percentile among all legal edits")
        _style_axes(ax)
        _panel_title(ax, title, sub)

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
    dict(key="margin_geo", title="stiffness", unit="gmean ×/÷ gsd, higher is better",
         w=1.30, align="right"),
    dict(key="shape_err", title="shape error", unit="gmean ×/÷ gsd, lower is better",
         w=1.35, align="right"),
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
    "stiffness: how strongly the bearings react to a change in shape, higher is better. "
    "Shown as a geometric mean and spread because it ranges over orders of magnitude - "
    "'a ×/÷ b' means the typical network sits between a/b and a·b. Its absolute size "
    "depends on how far apart the agents are, so compare rows rather than the number.",
    "shape error: how far the recovered formation lands from the true one, per radian of "
    "error in the bearing measurements - position counted in formation radii, attitude in "
    "radians. LOWER is better. 8.0 means one degree of bearing error (0.017 rad) displaces "
    "the shape by about 14% of its own size. Unlike stiffness it is comparable across "
    "network sizes, domains and pose ranges. A '*' on either column marks rows where "
    "non-rigid networks had to be left out: their stiffness is exactly 0 and their shape "
    "error infinite, and neither can enter a geometric mean.",
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
        if key == "margin_geo":
            return "-" if v["min_eig_gmean"] is None else _fmt_geo(v, times=" ×/÷")
        if key == "shape_err":
            return ("-" if v["shape_err_gmean"] is None
                    else _fmt_geo(v, key="shape_err", times=" ×/÷"))
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
