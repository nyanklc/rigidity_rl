"""Rendering for outputs.py: the comparison table, the CSVs and the plots.

The table is written for someone who does not know the topic: every column says which
direction is better, jargon is spelled out in a legend, and the two different meanings the
old `steps` column carried are split into `work` and `best@`.
"""

import contextlib
import copy
import csv
import json
import os
import re
import textwrap
from datetime import datetime

import numpy as np

import cost

# ── palette ───────────────────────────────────────────────────────────────────────────
# Data-viz reference palette, unchanged.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CARD = "#f4f3ee"   # one step off the surface, so the notes card reads as a panel

# The categorical hues, in the reference palette's documented order. They belong to the
# trained models: a run compares models against a fixed set of classical opponents, so
# colour answers "which model" and the baselines carry no hue at all.
MODEL_HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#7d4bb5", "#e34948"]

# The baselines separate by ink level and dash. SURFACE supports three usable line tones
# and no more (INK ~19:1, INK_2 ~6.5:1, MUTED ~2.9:1; AXIS and GRID cannot carry a line),
# so the dash does the rest. `marker` is read only by the two figures that carry a legend
# instead of end labels. greedy takes the darkest ink as the reference opponent.
BASELINE_STYLE = {
    "greedy":       {"color": INK,   "ls": "-",                    "marker": "s", "z": 4},
    "spectral":     {"color": INK,   "ls": (0, (6, 1.6, 1, 1.6)),  "marker": "D", "z": 3},
    "constructive": {"color": INK_2, "ls": "-",                    "marker": "^", "z": 3},
    "anneal":       {"color": INK_2, "ls": (0, (4, 1.6, 1, 1.6)),  "marker": "v", "z": 2},
    "optimal":      {"color": INK_2, "ls": "--",                   "marker": "*", "z": 5},
    "degree":       {"color": MUTED, "ls": "-",                    "marker": "P", "z": 2},
    "random":       {"color": MUTED, "ls": (0, (5, 2)),            "marker": "X", "z": 2},
    "initial":      {"color": MUTED, "ls": ":",                    "marker": ".", "z": 1},
}

# With one policy there is a hue to spare for every method, which is what the figures
# looked like before --model took a list. `learned` is renamed to the run's label.
SINGLE_MODEL_STYLE = {
    "greedy":  {"color": "#2a78d6", "ls": "-", "marker": "s", "z": 3},
    "learned": {"color": "#eb6834", "ls": "-", "marker": "o", "z": 6},
    "random":  {"color": "#1baf7a", "ls": "-", "marker": "X", "z": 2},
    "constructive": {"color": "#7d4bb5", "ls": "-", "marker": "^", "z": 3},
    "degree":  {"color": "#eda100", "ls": "-", "marker": "P", "z": 2},
    "spectral": {"color": "#008300", "ls": "-", "marker": "D", "z": 3},
    "anneal":  {"color": "#e87ba4", "ls": "-", "marker": "v", "z": 2},
    "initial": {"color": MUTED,     "ls": ":", "marker": ".", "z": 1},
    "optimal": {"color": INK_2,     "ls": "--", "marker": "*", "z": 5},
}

CLASSICAL_ORDER = ["initial", "random", "degree", "greedy", "spectral", "anneal",
                   "constructive"]

METHOD_STYLE = {}
METHOD_ORDER = []

# The formation figures draw one 3-D panel per method. Every panel is a fixed width, so
# `_grid_for` widens the figure rather than shrinking them and this cap bounds height.
MAX_FORMATION_PANELS = 9

# Which panels survive that cap, which is NOT the table's order. Filled by
# configure_methods, which is where the rule is written down.
FORMATION_PRIORITY = []

CLASSICAL_BLURB = {
    "initial": "the random graph each method starts from",
    "random":  "uniform random actions, the floor any method should beat",
    "degree":  "connects the least-connected pair until rigid, then prunes",
    "greedy":  "repeatedly applies the single best edge change until none helps",
    "spectral": "greedy's hill climb, read off the rigidity algebra directly",
    "anneal":  "random changes, accepting worse ones ever less often",
    "constructive": "builds from the empty graph, keeping any edge that raises rank(B)",
    "optimal": "exhaustive search over every graph (small networks only)",
}

METHOD_BLURB = {}


def configure_methods(models=()):
    """Set the run's method identities. `models` is [(label, blurb)] in command order.

    Rebuilt in place so the tables stay module-level names that everything else reads
    without being handed a registry. Called with no models it restores the single-policy
    tables, where the one policy is called `learned`, which is what every run written
    before multi-model support is keyed on.
    """
    labels = [lab for lab, _ in models] or ["learned"]
    blurbs = dict(models) or {"learned": "the trained policy"}

    METHOD_STYLE.clear()
    if len(labels) > 1:
        # colour has to answer "which model", so the baselines give up their hues and
        # separate by ink and dash instead
        METHOD_STYLE.update({m: dict(BASELINE_STYLE[m]) for m in BASELINE_STYLE})
        for k, lab in enumerate(labels):
            METHOD_STYLE[lab] = {"color": MODEL_HUES[k % len(MODEL_HUES)], "ls": "-",
                                 "marker": "o", "z": 6 + k}
    else:
        # one policy needs one hue, so every method can keep a categorical slot
        METHOD_STYLE.update({m: dict(s) for m, s in SINGLE_MODEL_STYLE.items()})
        METHOD_STYLE[labels[0]] = dict(SINGLE_MODEL_STYLE["learned"])

    METHOD_ORDER[:] = CLASSICAL_ORDER + labels + ["optimal"]

    METHOD_BLURB.clear()
    METHOD_BLURB.update(CLASSICAL_BLURB)
    METHOD_BLURB.update(blurbs)

    # These figures exist to compare the policies against their opponents, so a policy is
    # never the panel the cap drops, and `initial` is the reference the rest are read
    # against. Selection uses this; drawing order stays METHOD_ORDER.
    FORMATION_PRIORITY[:] = (labels + ["initial", "greedy", "optimal", "constructive",
                                       "spectral", "anneal", "degree", "random"])


def method_style(name):
    """The style for `name`, taking the next free hue if it was never configured.

    The old `.get(name, {...})` fallback handed every unregistered method the same ink,
    which draws two policies as one line.
    """
    if name not in METHOD_STYLE:
        used = {s["color"] for s in METHOD_STYLE.values()}
        free = [c for c in MODEL_HUES if c not in used]
        METHOD_STYLE[name] = {"color": free[0] if free else INK_2, "ls": "-",
                              "marker": "o", "z": 6 + len(METHOD_STYLE)}
    return METHOD_STYLE[name]

configure_methods()          # single-policy tables until a run says otherwise


# both formats, one directory each under plots/
PLOT_FORMATS = ("pdf", "png")

# Every figure is written twice under the same name, the second with `-plain` appended:
# one carries the title block and the notes card, one is the panels alone. Plain is not
# raw -- everything inside the axes stays, since without it the plot cannot be read.
_PLAIN = False


@contextlib.contextmanager
def plain():
    """Draw the panels only, and file them under `<name>-plain`."""
    global _PLAIN
    was, _PLAIN = _PLAIN, True
    try:
        yield
    finally:
        _PLAIN = was

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


def make_run_dir(root, env_name, model_name=None, model_names=(), tag=None, out_dir=None,
                 with_plots=True):
    """One directory per run. `model_names` names it after the models it compared.

    Past two the names stop fitting, so the count stands in for them and the run's own
    meta.json says which they were. `model_name=` is the older single-model spelling.
    """
    names = list(model_names) or ([model_name] if model_name else [])
    if out_dir:
        path = out_dir
    else:
        parts = [datetime.now().strftime("%Y%m%d-%H%M%S"), short_env_name(env_name)]
        if len(names) > 2:
            parts.append(f"{len(names)}models")
        elif names:
            parts.append("-".join(short_model_name(x) for x in names))
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


def _cost_summary(sel):
    """Mean calls per primitive, and the wall time, over the episodes carrying them.

    None when the rows predate the counters or came from an archived environment,
    which is not the same as a method that cost nothing.
    """
    metered = [r for r in sel if r.get("cost") is not None]
    if not metered:
        return None
    keys = sorted({k for r in metered for k in r["cost"]})
    out = {"episodes": len(metered),
           "calls": {k: float(np.mean([r["cost"].get(k, 0) for r in metered])) for k in keys}}
    out["total"] = sum(out["calls"].get(k, 0.0) for k in cost.LEAVES)
    ms = [r["ms"] for r in metered if r.get("ms")]
    out["ms_gmean"], out["ms_gsd"] = _gmean(ms), _gsd(ms)
    return out


# The columns of the cost block, in the order a method reaches them. `score_network` and
# the forward pass are what a method decides with; the rest is what those decisions cost.
COST_COLUMNS = [
    ("score_network", "phi"),
    ("deterministic_action", "fwd"),
    ("forward", "fwd*"),
    ("extended_bearing_rigidity_matrix", "build B"),
    ("rigidity_decomposition", "svd"),
    ("nullspace", "null"),
    ("nullspace_and_softest", "null+v"),
    ("candidate_gain", "cand"),
    ("removal_costs", "remove"),
    ("edge_block_ranks", "blocks"),
    ("is_IBR_explicit", "rigid?"),
    ("eigenvalues", "eig"),
]


def _cost_block(agg):
    """What each method spent, as counted calls and wall time. [] when nothing was metered."""
    metered = {m: v["cost"] for m, v in agg.items() if v.get("cost")}
    if not metered:
        return []
    used = [(k, h) for k, h in COST_COLUMNS
            if any(c["calls"].get(k, 0) for c in metered.values())]
    other = sorted({k for c in metered.values() for k in c["calls"]}
                   - {k for k, _ in COST_COLUMNS} - {"_reporting"})

    lines = ["COMPUTATIONAL COST PER NETWORK",
             "  " + "method".ljust(13) + "".join(h.rjust(10) for _, h in used)
             + "total".rjust(10) + "ms".rjust(14)]
    for m, c in metered.items():
        cells = "".join(f"{c['calls'].get(k, 0):.0f}".rjust(10) for k, _ in used)
        ms = "-" if c["ms_gmean"] is None else f"{c['ms_gmean']:.1f} x/{c['ms_gsd']:.1f}"
        lines.append(f"  {m:<13}{cells}{c['total']:>10.0f}{ms:>14}")
    if other:
        lines.append(f"  not shown: {', '.join(other)}")
    # --replay-env execs an archived rigidity.py, which carries no counters. A method
    # that evaluated phi without any matrix work did not do it for free, it was not seen.
    blind = [m for m, c in metered.items()
             if c["calls"].get("score_network", 0) > 0 and c["total"] == 0]
    if blind:
        lines.append(f"  UNMEASURED for {', '.join(blind)}: the rigidity primitives were "
                     f"not counted, which an archived environment does not carry")
    lines.append("")
    return lines


def cost_legend():
    """What one call of each counted primitive actually does."""
    out = ["  These are counts of CALLS, not of work. One `null` is an eigendecomposition",
           "  of a 6n x 6n matrix and one `blocks` is m rank computations on 3 x 6n slices,",
           "  and both count as one, which is what the ms column is there to weigh. ms is a",
           "  geometric mean and spread over the networks, and it depends on the machine and",
           "  its load in a way the counts do not; measure it with BLAS pinned to one thread.",
           "  total sums only the primitives that call no other, so nothing is counted twice.",
           ""]
    for key, head in COST_COLUMNS:
        op = cost.OPERATION.get(key)
        if op:
            out.append(f"  {head:<9} {key} - {op}")
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
        bound = [r for r in sel if r.get("m_req") not in (None, "")]
        out[m] = {
            "episodes": len(sel),
            "edges_mean": float(np.mean([r["m"] for r in sel])),
            "edges_sd": float(np.std([r["m"] for r in sel])),
            "score_mean": float(np.mean([r["score"] for r in sel])),
            "score_sd": float(np.std([r["score"] for r in sel])),
            "rigid_pct": 100.0 * float(np.mean([r["is_IBR"] for r in sel])),
            "minimal_pct": 100.0 * float(np.mean([r["is_MBR"] for r in sel])),
            # rigid at the proven lower bound, which is exact where is_MBR is a
            # heuristic. None when no row carried a bound to compare against.
            "at_bound_pct": (100.0 * float(np.mean([bool(at_bound(r)) for r in bound]))
                             if bound else None),
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
            "cost": _cost_summary(sel),
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

    head1 = (f"  {'method':<13}{'edges':>12}{'score':>14}{'rigid':>8}{'at bound':>10}"
             f"{'minimal':>9}"
             f"{'stiffness(geo)':>17}{'shape err':>16}{'work':>13}{'best@':>12}")
    head2 = (f"  {'':<13}{'(fewer)':>12}{'(higher)':>14}{'%':>8}{'%':>10}{'%':>9}"
             f"{'gmean x/gsd':>17}{'gmean x/gsd':>16}{'edits':>13}{'step':>12}")
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
        bound = "-" if v["at_bound_pct"] is None else f"{v['at_bound_pct']:.0f}"
        row = (f"  {m:<13}"
               f"{_fmt(v['edges_mean'], v['edges_sd']):>12}"
               f"{_fmt(v['score_mean'], v['score_sd']):>14}"
               f"{v['rigid_pct']:>8.0f}{bound:>10}{v['minimal_pct']:>9.0f}"
               f"{_fmt_geo(v):>17}{_fmt_geo(v, key='shape_err'):>16}"
               f"{work:>13}{best:>12}")
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

    lines.extend(_cost_block(agg))

    if brief:
        return "\n".join(lines)

    lines.append("-" * w)
    lines.append("WHAT THE METHODS ARE")
    for m in agg:
        lines.append(f"  {m:<13} {METHOD_BLURB.get(m, '')}")
    lines.append("")
    lines.append("WHAT THE COLUMNS MEAN")
    lines.append("  edges     how many directed bearing measurements the final network")
    lines.append("            needs. Each edge is a sensor or communication link, so")
    lines.append("            fewer is better.")
    lines.append("  score     the objective every method is scored with (phi). It rewards")
    lines.append("            rigidity and charges for each extra edge. Higher is better.")
    lines.append("  rigid     percent of networks whose shape is fully determined by their")
    lines.append("            bearing measurements.")
    lines.append("  at bound  percent that are rigid using exactly the fewest edges these")
    lines.append("            poses allow. The bound is proven, so this is exact.")
    lines.append("  minimal   percent the minimality heuristic calls rigid with the fewest")
    lines.append("            possible edges. It agrees with at bound on homogeneous")
    lines.append("            networks and can disagree either way on mixed-domain ones,")
    lines.append("            which is why both columns are here.")
    lines.append("  stiffness how strongly the bearings react to a change in shape, as a")
    lines.append("            geometric mean and spread since it ranges over orders of")
    lines.append("            magnitude, where 'a x/b' means the typical network sits")
    lines.append("            between a/b and a*b. Higher is better. Its absolute size")
    lines.append("            depends on how far apart the agents are, so compare rows")
    lines.append("            rather than the number itself.")
    lines.append("  shape err how far the recovered formation is from the true one, per")
    lines.append("            radian of error in the bearing measurements. Position is")
    lines.append("            counted in formation radii and attitude in radians, so 8.0")
    lines.append("            means one degree of bearing error (0.017 rad) displaces the")
    lines.append("            shape by about 14 percent of its own size. Lower is better,")
    lines.append("            and unlike stiffness it is comparable across network sizes,")
    lines.append("            domains and pose ranges. A '*' on either column marks rows")
    lines.append("            where non-rigid networks had to be left out, since their")
    lines.append("            stiffness is 0 and their shape error infinite and neither")
    lines.append("            can enter a geometric mean.")
    lines.append("  work      how many changes to the network the method applied.")
    lines.append("  best@     the step at which its best network was found. A lower number")
    lines.append("            means the rest of the budget added nothing.")
    if has_opt:
        lines.append("  =best     percent of networks where the method tied the exhaustive")
        lines.append("            optimum.")
    lines.append("")
    if any(v.get("cost") for v in agg.values()):
        lines.append("WHAT THE COST BLOCK MEANS")
        lines.extend(cost_legend())
    if sweep:
        lines.append("  The noise block perturbs every bearing by that many degrees,")
        lines.append("  recovers the formation from the noisy measurements and reports the")
        lines.append("  RMS position error in formation radii. The bracketed number is what")
        lines.append("  the rigidity matrix predicts. The two agree while the error stays")
        lines.append("  small and separate once the noise is past the range the prediction")
        lines.append("  covers.")
        lines.append("")
    lines.append("HOW TO READ IT")
    lines.append("  Every value is a mean over the networks and '+-' is the standard")
    lines.append("  deviation across them, which is how much the method varies from one")
    lines.append("  network to the next. The percentage columns carry no '+-' because they")
    lines.append("  are already means of a yes/no outcome, whose spread is fixed by the")
    lines.append("  percentage itself.")
    lines.append("  initial and optimal are reference rows rather than competing methods.")
    lines.append("  Every method starts from initial and optimal is the best achievable.")
    lines.append("  All methods are run on the same networks, so rows compare directly.")
    if any(r["method"] == "constructive" for r in rows):
        lines.append("  constructive is the exception. It discards the initial edges and")
        lines.append("  builds from nothing, being a construction rather than an edit.")
    lines.append("=" * w)
    return "\n".join(lines)


# ── output files ──────────────────────────────────────────────────────────────────────
RESULT_FIELDS = ["episode", "method", "m", "m_req", "score", "is_IBR", "is_MBR",
                 "at_bound", "min_eig", "shape_err", "work", "best_at"]
TRACE_FIELDS = ["episode", "method", "step", "score", "edges", "rank", "rank_K",
                "is_IBR", "is_MBR", "min_eig", "shape_err"]


def at_bound(row):
    """Rigid using exactly the fewest edges the poses allow. None when unknowable.

    m_req is a proven lower bound, so this is exact where `is_MBR` is a heuristic that
    can over- and under-report on heterogeneous networks. Both are reported.
    """
    if not row.get("is_IBR") or row.get("m_req") in (None, ""):
        return False if row.get("m_req") not in (None, "") else None
    return int(row["m"]) == int(row["m_req"])


def write_csvs(run_dir, rows, traces):
    with open(os.path.join(run_dir, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(dict(r, at_bound=at_bound(r)))
    if traces:
        with open(os.path.join(run_dir, "trajectories.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=TRACE_FIELDS, extrasaction="ignore")
            wr.writeheader()
            for t in traces:
                wr.writerow(t)


def write_costs(run_dir, rows):
    """One row per (episode, method) with every counter. The legend goes beside it."""
    metered = [r for r in rows if r.get("cost") is not None]
    if not metered:
        return False
    keys = sorted({k for r in metered for k in r["cost"]})
    with open(os.path.join(run_dir, "cost.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["episode", "method", "ms", "total"] + keys)
        wr.writeheader()
        for r in metered:
            row = {"episode": r.get("episode"), "method": r["method"],
                   "ms": round(r.get("ms", 0.0), 3),
                   "total": sum(r["cost"].get(k, 0) for k in cost.LEAVES)}
            row.update({k: r["cost"].get(k, 0) for k in keys})
            wr.writerow(row)
    with open(os.path.join(run_dir, "cost.txt"), "w") as f:
        f.write("\n".join(cost_legend()) + "\n")
    return True


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
def _clip(text, width_in, fontsize):
    """Trim to a physical width. The card's two columns must not run into each other.

    0.50 em per character where _wrap uses 0.60: wrapping has to over-estimate or the
    card is sized for fewer lines than it draws, and clipping has to under-estimate or
    it eats the end of a line that would have fitted.
    """
    chars = max(16, int(width_in * 72.0 / (fontsize * 0.50)))
    return text if len(text) <= chars else text[:chars - 1].rstrip() + "\u2026"


def _wrap(text, width_in, fontsize):
    """Wrap to a physical width, since the strings here are long generated identifiers.

    0.60 em per character is measured, not nominal: it has to over-estimate slightly or
    the card is sized for fewer lines than it ends up drawing and the text runs out of it.
    """
    chars = max(24, int(width_in * 72.0 / (fontsize * 0.60)))
    return textwrap.wrap(text, chars) or [""]


def _header_height(header, width_in):
    """Inches the title block needs -- the caller sizes the figure around it.

    Only the title line. What the run was (environment, models, network, instances) used
    to sit under it on every figure and now lives once, in the run_info figure, which
    gives the panels back four lines of height each.
    """
    return 0.34


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
    return y - 0.10 / height_in   # top of the plotting area, as a figure fraction


def _card_rows(methods, notes, width_in):
    """Wrapped card content plus the row count, so the figure can be sized for it.

    Two columns: the method key, then how the figure is built. When the notes are much
    longer than the method list they flow newspaper-style -- the rest of the left column
    first, continuing at the top of the right -- rather than leaving half the card empty.
    Splits happen between notes, never mid-sentence.
    """
    # a figure whose series are not methods passes none, and an empty heading on the
    # card is worse than no heading
    left = [("METHODS", None, None)] if methods else []
    for m in methods:
        style = method_style(m)
        left.append((m, METHOD_BLURB.get(m, ""), style))

    blocks = [_wrap(note, width_in * 0.42, 7.5) for note in notes]
    n_notes = sum(len(b) for b in blocks)
    heading = ("WHAT THE COLUMNS MEAN", True)

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
    left = left + [("", None, None), ("WHAT THE COLUMNS MEAN", None, None)]
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
            fig.text(col_l + 0.108, y, _clip(blurb, width_in * (col_r - col_l - 0.12), 7.5),
                     color=INK_2, fontsize=7.5, va="center")

    for i, (line, heading) in enumerate(right):
        y = top - i * step
        fig.text(col_r, y, line, color=INK_2, fontsize=7.0 if heading else 7.5,
                 va="center", family="monospace" if heading else None)


def _fig_line(fig, xs, ys, color, ls):
    from matplotlib.lines import Line2D
    return Line2D(xs, ys, transform=fig.transFigure, color=color, linestyle=ls,
                  linewidth=2.4, solid_capstyle="round", zorder=3)


def _method_ticklabels(ax, methods, fontsize=8.5):
    """Method names under an axis, tilted once there are too many to sit flat.

    `constructive` is twelve characters and eight of them do not fit across a half-width
    panel, so past six the labels tilt rather than run into each other.
    """
    if len(methods) > 6:
        ax.set_xticklabels(methods, fontsize=fontsize, rotation=30, ha="right",
                           rotation_mode="anchor")
    else:
        ax.set_xticklabels(methods, fontsize=fontsize)


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


def _style_axes(ax, log=False, logx=False):
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
    if logx:
        from matplotlib.ticker import NullFormatter
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())


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
        style = method_style(method)
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

    head_h = 0.0 if _PLAIN else _header_height(header, width)
    card_h = 0.0 if _PLAIN else _card_height(methods, notes, width)
    nrows, ncols = panels
    fig_h = nrows * panel_h + head_h + card_h
    fig = plt.figure(figsize=(width, fig_h), facecolor=SURFACE)
    # a 3-D caller adds its own projection="3d" panels
    axes = ([] if axes3d
            else [fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)])
    top = 1.0 if _PLAIN else _draw_header(fig, header, kind)
    return fig, axes, (card_h, card_h / fig_h), top, width


def _finish(fig, card, methods, notes, width, top, run_dir, name, tight=True):
    card_h, bottom = card
    # a figure that placed its own panels (the 3-D ones) must not be re-laid out
    if tight:
        fig.tight_layout(rect=(0, bottom, 1, top), h_pad=1.6, w_pad=2.0)
    if not _PLAIN:
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
        "The x axis is one step of the run. greedy has no step budget. It contributes "
        "one point per edge change it applies and stops when no single change helps.",
        ("Each line is the mean over the networks; the shaded band is the middle 50% of them "
         "(25th-75th percentile)." if aggregate_over_episodes else
         "Each line is a single network. This is one episode, not an average."),
        "A method that finishes early is held at its last network for the rest of the axis, "
        "so the curves stay comparable.",
        "Every method is run on the same networks from the same starting graph.",
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

    notes = [f"{s} is {STAT_BLURB[s]}" for s in STAT_ORDER]
    notes += [
        "Bars are the mean over the networks; the whisker is ±1 standard deviation "
        "across them.",
        "For a method that never moves (initial) or only improves (greedy), final and "
        "best are the same bar.",
        "Stiffness is plotted on a log axis. A non-rigid network has stiffness 0 and "
        "cannot be drawn there.",
    ]

    fig, axes, card, top, width = _figure(header, "Final / best / mean outcome",
                                             methods, notes)
    x = np.arange(len(methods))
    w = 0.26

    for ax, panel in zip(axes, OUTCOME_PANELS):
        field, scale = panel["field"], panel["scale"]
        colors = [method_style(m)["color"] for m in methods]
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
        _method_ticklabels(ax, methods)
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
        "Each line is one method. Its formation is taken with every bearing perturbed "
        "by the noise on the x axis, recovered from those noisy measurements and "
        "compared against the truth.",
        "The y axis is RMS position error in formation radii, so 0.1 means the "
        "recovered shape is off by a tenth of the formation's own size. Lower is better.",
        "The dashed line is what the rigidity matrix predicts for the same topology. "
        "The two agree while the error stays small and separate once the noise is past "
        "the range the linear prediction covers.",
    ]
    fig, axes, card, top, width = _figure(
        header, "Shape error under bearing noise", methods, notes,
        panel_h=4.6, width=11.0, panels=(1, 1))
    ax = axes[0]

    for m, v in sweep.items():
        style = method_style(m)
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
                 "solid is measured, dashed is predicted from the rigidity matrix")
    ax.legend(frameon=False, fontsize=8)

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


# ── formation drawings ────────────────────────────────────────────────────────────────
# These draw the network itself rather than a statistic of it, in real 3-D: a bearing
# formation is a spatial object and a flat projection hides the axis it is worst
# determined along. Marker shape carries the agent's domain, which is the other thing
# a plain scatter throws away on a mixed formation.
# (matplotlib code, what the shape is called, what the domain means). The shape has
# to be named in words: the matplotlib code means nothing to someone reading the card.
DOMAIN_MARKER = {
    "R^2":     ("s", "square",   "moves in a plane, no heading"),
    "R^2xS^1": ("^", "triangle", "moves in a plane, and has a heading"),
    "R^3":     ("o", "circle",   "moves in 3-D, no heading"),
    "R^3xS^1": ("D", "diamond",  "moves in 3-D, turns about one axis"),
    "SE(3)":   ("P", "cross",    "moves and turns freely"),
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
    return "Marker shape is what the agent can do: " + "; ".join(
        f"{DOMAIN_MARKER[d][1]} = {d}, {DOMAIN_MARKER[d][2]}" for d in present) + "."


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
    # The poses are centred and unit-normalised, so the numbers on these axes carry no
    # information a reader can use, and three axes of them crowd every panel. The ticks
    # and the grid stay, which is what still reads the box as 3-D.
    ax.tick_params(colors=MUTED, labelsize=6.5, pad=1, length=2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([])
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
    axes_len = np.sqrt(w) * scale
    pts = (V @ (np.sqrt(w)[:, None] * unit.reshape(3, -1))) * scale
    x, y, z = (pts.reshape(3, *unit.shape[1:]) + centre[:, None, None])
    ax.plot_surface(x, y, z, color=color, alpha=0.34, linewidth=0, shade=False,
                    zorder=8)

    # a small shell is a faint smudge without an outline, and the small ones are
    # exactly the panels worth being able to read
    t = np.linspace(0, 2 * np.pi, 3 * res)
    circle = np.stack([np.cos(t), np.sin(t)])
    for a, b in ((0, 1), (0, 2), (1, 2)):
        ring = np.zeros((3, len(t)))
        ring[a] = axes_len[a] * circle[0]
        ring[b] = axes_len[b] * circle[1]
        rx, ry, rz = (V @ ring) + centre[:, None]
        ax.plot(rx, ry, rz, color=color, linewidth=0.9, alpha=0.9, zorder=9)


def _nice_factor(x):
    """Round to something a caption can state: 1, 2, 5, 10, 20, 50, ..."""
    if x <= 1.5:
        return 1
    mag = 10 ** int(np.floor(np.log10(x)))
    return int(min((c * mag for c in (1, 2, 5, 10)), key=lambda c: abs(c - x)))


def _grid_for(count):
    """Panel grid leaving as few empty cells as possible: 1x1, 1x2, 2x2, then 3 or 4 wide.

    Every panel is a fixed width, so more columns is a wider figure rather than smaller
    panels. Ties go to the narrower grid. Seven panels are 4x2 with one hole rather than
    3x3 with two, which is a third of the figure left blank.
    """
    if count <= 2:
        return 1, count
    if count <= 4:
        return 2, 2
    ncols = min((3, 4), key=lambda c: (c * int(np.ceil(count / c)) - count, c))
    return int(np.ceil(count / ncols)), ncols


def _panel_cap_note(rows, episode=0):
    """Say which methods the panel cap left out, rather than letting them vanish."""
    shown = {r["method"] for r in _panel_rows(rows, episode)}
    dropped = [m for m in METHOD_ORDER
               if m not in shown and any(r["method"] == m and r.get("edges") is not None
                                         and r.get("is_IBR") for r in rows
                                         if r.get("episode") == episode)]
    if not dropped:
        return None
    return (f"One panel per method does not fit more than {MAX_FORMATION_PANELS}, so "
            f"{', '.join(dropped)} {'is' if len(dropped) == 1 else 'are'} not drawn here. "
            f"The policy and the reference are kept first. The comparison table carries "
            f"every method.")


def _rank_in(order, method):
    return order.index(method) if method in order else len(order)


def _panel_rows(rows, episode=0):
    """The rows one formation figure draws, in display order.

    A flexible network has no error ellipsoid and no softest mode, so it cannot have a
    panel at all -- that filter is why `initial` is often absent. What survives past it
    is chosen by FORMATION_PRIORITY and then drawn in METHOD_ORDER.
    """
    sel = [r for r in rows if r.get("episode") == episode
           and r.get("edges") is not None and r.get("is_IBR")]
    keep = sorted(sel, key=lambda r: _rank_in(FORMATION_PRIORITY, r["method"]))
    keep = keep[:MAX_FORMATION_PANELS]
    return sorted(keep, key=lambda r: _rank_in(METHOD_ORDER, r["method"]))


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
    cap_note = _panel_cap_note(rows)

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
        f"One formation with one panel per method. The agents and poses are identical "
        f"and only the measured bearings differ. Drawn on the first evaluated network.",
        f"Each shell is where that agent ends up when every bearing carries "
        f"{np.degrees(sigma):.1f} degrees of error. Bigger means the topology pins that "
        f"agent down less well. The percentage above each panel is the worst agent, at "
        f"true scale.",
        f"Shells are drawn {exaggeration}x larger than life so they are visible at all, "
        f"and the same factor is used in every panel so the panels compare.",
        "Arrows run from the measuring agent to the one it measures.",
    ]
    if cap_note:
        notes.append(cap_note)
    dom = _domain_note(net)
    if dom:
        notes.append(dom)

    nrows, ncols = _grid_for(len(ep_rows))
    fig, _, card, top, width = _figure(
        header, "Position uncertainty per agent", methods, notes,
        panel_h=5.4, width=6.8 * ncols, panels=(nrows, ncols), axes3d=True)
    band = (top, card[1])

    for k, (row, per_agent, worst) in enumerate(zip(ep_rows, blocks, worsts)):
        ax = _formation_axes(fig, nrows, ncols, k + 1, P, band)
        work = copy.deepcopy(net)
        work.edges = row["edges"].copy()
        style = method_style(row["method"])
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
    cap_note = _panel_cap_note(rows)

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
        "The softest mode is the way of deforming the formation that changes the "
        "bearings least. Drawn on the first evaluated network.",
        "The mode is normalised and each panel is scaled to its own largest arrow, so "
        "the arrows show which agents move and in which direction. Arrow lengths do not "
        "compare between panels.",
        "The rigidity eigenvalue above each panel says how soft that mode is. Smaller "
        "means the bearings resist the deformation less. Larger is better.",
    ]
    if cap_note:
        notes.append(cap_note)
    dom = _domain_note(net)
    if dom:
        notes.append(dom)

    nrows, ncols = _grid_for(len(ep_rows))
    fig, _, card, top, width = _figure(
        header, "Softest mode of each method", methods, notes,
        panel_h=5.4, width=6.8 * ncols, panels=(nrows, ncols), axes3d=True)
    band = (top, card[1])

    for k, (row, disp, lam, scale) in enumerate(zip(ep_rows, modes, lams, scales)):
        ax = _formation_axes(fig, nrows, ncols, k + 1, P, band)
        work = copy.deepcopy(net)
        work.edges = row["edges"].copy()
        style = method_style(row["method"])
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
    cap_note = _panel_cap_note(rows)

    P = np.array([a.pose.position for a in net.agents], dtype=float)
    methods = [r["method"] for r in ep_rows]
    notes = [
        "Noise on one bearing propagates into the recovered shape by a known amount, and "
        "those amounts add up to the total error exactly. So every measurement has a "
        "share of the error and the shares sum to 100%. Drawn on the first evaluated "
        "network.",
        "Arrow thickness is that single bearing's share. Marker size is the share "
        "contributed by every bearing that agent takes, so a large marker is an agent "
        "whose own sensing carries much of the total.",
    ]
    if cap_note:
        notes.append(cap_note)
    dom = _domain_note(net)
    if dom:
        notes.append(dom)

    nrows, ncols = _grid_for(len(ep_rows))
    fig, _, card, top, width = _figure(
        header, "Share of the total error per measurement", methods, notes,
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
        style = method_style(row["method"])

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
        "The x axis is the error the rigidity matrix predicts for that network at that "
        "noise level. The y axis is the error measured by perturbing every bearing and "
        "recovering the formation.",
        "Points on the dashed diagonal are where prediction and measurement agree. "
        "Points below it are where the noise is past the range the linear prediction "
        "covers.",
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
        style = method_style(m)
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
                 "points on the diagonal are where prediction and measurement agree")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


def plot_repair_choice(run_dir, spread, header, filename="repair_choice"):
    """Among repairs of the same size, how much does the choice matter?

    `spread` is what outputs.py collected: one record per broken instance with
    every minimum-size repair's shape error and where greedy's pick landed.
    """
    if not spread:
        return None

    notes = [
        "After a formation breaks, several different edge sets of the same minimum "
        "size restore rigidity. Each grey dot is one of them, on one broken network.",
        "Height is the shape error that repair leaves behind, relative to the best "
        "repair available on that network. One is the best available and ten is ten "
        "times worse, for the same number of edges.",
        "The marked point is what marginal-gain greedy picked. The height of the column "
        "is the spread across repairs of equal size.",
    ]
    methods = ["greedy"]
    fig, axes, card, top, width = _figure(
        header, "Shape error of each minimum-size repair", methods, notes,
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
                       color=method_style("greedy")["color"], edgecolors=SURFACE,
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


DECISION_FIELDS = ["model", "episode", "step", "kind", "phi_pct", "err_pct", "share_pct",
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
    models = sorted({d.get("model", "learned") for d in decisions},
                    key=lambda m: _rank_in(METHOD_ORDER, m))
    groups = [(m, k) for m in models for k in kinds]
    notes = [
        "At every step, every legal single-edge change is scored and the one the "
        "policy made is ranked among them. 100% is the best available edit and the "
        "dashed line at 50% is what picking at random scores.",
        "The policy is trained on phi, which rewards rank and charges for edges. It is "
        "not trained on shape error, so the two panels are ranked against different "
        "criteria.",
        "The third panel is whether the edit touches an agent already carrying much of "
        "the error. Adds and removes are separated because acting on a loaded agent "
        "means the opposite thing in each case, and a filled diamond marks an add "
        "against a hollow one for a remove.",
    ]
    fig, axes, card, top, width = _figure(
        header, "Rank of the edits each policy applied", models, notes,
        panel_h=4.0, width=13.0, panels=(1, 3))

    for ax, (key, title, sub) in zip(axes, panels):
        data = [[d[key] for d in decisions
                 if d.get("model", "learned") == m and d["kind"] == k
                 and d.get(key) is not None]
                for m, k in groups]
        if not any(data):
            _style_axes(ax)
            _panel_title(ax, title, "no edits to rank")
            continue

        ax.axhline(50.0, color=INK_2, linestyle="--", linewidth=1.1, zorder=1)
        rng = np.random.default_rng(0)
        for pos, ((m, k), vals) in enumerate(zip(groups, data)):
            if not vals:
                continue
            # colour is the model; an add is filled and a remove hollow, so the two
            # kinds stay apart without spending a second hue on them
            colour = method_style(m)["color"]
            face = colour if k == "add" else SURFACE
            ax.scatter(pos + (rng.random(len(vals)) - 0.5) * 0.5, vals, s=16,
                       color=colour, alpha=0.40, linewidths=0, zorder=2)
            ax.scatter([pos], [np.mean(vals)], s=110, marker="D", facecolors=face,
                       edgecolors=colour, linewidths=1.6, zorder=5)
            ax.annotate(f"{np.mean(vals):.0f}%", (pos, np.mean(vals)), fontsize=8.5,
                        color=INK, ha="center", va="bottom",
                        xytext=(0, 11), textcoords="offset points", zorder=6)

        ax.set_xticks(range(len(groups)))
        labels = [f"{k} ({len(v)})" if len(models) == 1 else f"{m} {k} ({len(v)})"
                  for (m, k), v in zip(groups, data)]
        # one model fits flat; past that the groups are twice as many and twice as wide
        if len(groups) > 3:
            ax.set_xticklabels(labels, fontsize=7.6, rotation=30, ha="right",
                               rotation_mode="anchor")
        else:
            ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(-4, 108)
        ax.set_ylabel("percentile among all legal edits")
        _style_axes(ax)
        _panel_title(ax, title, sub)

    return _finish(fig, card, models, notes, width, top, run_dir, filename)


def plot_summary(run_dir, rows, header):
    """Spread across networks of the outcome each method is scored on."""
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    colors = [method_style(m)["color"] for m in methods]

    notes = [
        "One value per network per method. The line is the median, the box the middle "
        "50%, the whiskers reach 1.5x that range and the dots are networks outside it.",
        "Scored on the best network each run visited, which is what the comparison "
        "table reports.",
        "A rigid network has its shape fully determined by its bearing measurements. "
        "Also minimal means it does that with the fewest possible edges.",
        "Steps to best counts steps for the rollout methods and applied edge changes for "
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
        _method_ticklabels(ax, labels)
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
    _method_ticklabels(ax, methods)
    ax.set_ylim(0, 112)
    _style_axes(ax)
    _panel_title(ax, "Networks solved",
                 "% rigid (solid) and % also using the fewest edges (faded)")

    ax = axes[3]
    roll = [m for m in methods if m not in ("initial", "optimal")]
    if roll:
        box(ax, [[r.get("best_at", 0) for r in rows if r["method"] == m] for m in roll],
            roll, [method_style(m)["color"] for m in roll],
            "Steps to the best network",
            "how long each method took to reach its best - lower converges sooner")
    else:
        _style_axes(ax)
        _panel_title(ax, "Steps to the best network", "no rollout method was run")

    return _finish(fig, card, methods, notes, width, top, run_dir, "summary")


def plot_cost(run_dir, rows, header, filename="cost"):
    """What each method spent to get its answer, and what that bought.

    Counts are machine independent; the milliseconds are not, and are shown beside them
    because a call count alone weighs an eigendecomposition of a 6n x 6n matrix the same
    as a rank of a 3 x 6n slice.
    """
    import matplotlib
    matplotlib.use("Agg")

    agg = aggregate(rows)
    metered = {m: v for m, v in agg.items() if v.get("cost") and v["cost"]["total"] > 0}
    if not metered:
        return None
    methods = list(metered)
    colors = [method_style(m)["color"] for m in methods]
    totals = [metered[m]["cost"]["total"] for m in methods]
    times = [metered[m]["cost"]["ms_gmean"] or 0.0 for m in methods]
    edges = [metered[m]["edges_mean"] for m in methods]
    work = [max(metered[m]["work_mean"], 1.0) for m in methods]

    notes = [
        "Counts are calls to the rigidity primitives, summed over the ones that call no "
        "other so nothing is counted twice. They are the same on any machine.",
        "Milliseconds are a geometric mean over the networks. They depend on the machine "
        "and its load, which is why the counts are shown beside them.",
        "A method whose whole cost is reporting rather than searching is left out of "
        "these panels: initial computes nothing, and optimal is exhaustive search.",
        "Cost per change divides the total by the number of edge changes the method "
        "applied, separating an expensive decision from many cheap ones.",
    ]
    fig, axes, cardspec, top, width = _figure(header, "What each method costs",
                                              methods, notes)

    def hbar(ax, values, title, note, fmt="{:.0f}"):
        y = np.arange(len(methods))
        ax.barh(y, values, color=colors, alpha=0.9, zorder=3, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=8.5)
        ax.invert_yaxis()
        for yi, v in zip(y, values):
            ax.annotate(fmt.format(v), xy=(v, yi), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
        _style_axes(ax, logx=True)
        ax.set_xlim(right=max(values) * 3.0)
        _panel_title(ax, title, note)

    hbar(axes[0], totals, "Rigidity computations per network",
         "calls to the primitives - lower is cheaper")
    hbar(axes[1], times, "Time per network",
         "milliseconds, geometric mean - lower is cheaper", fmt="{:.1f}")

    ax = axes[2]
    ax.scatter(totals, edges, s=64, c=colors, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    for m, x, y in zip(methods, totals, edges):
        ax.annotate(m, xy=(x, y), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=INK_2)
    ax.set_xlabel("rigidity computations per network", color=INK_2, fontsize=8.5)
    ax.set_ylabel("edges", color=INK_2, fontsize=8.5)
    _style_axes(ax, logx=True)
    _panel_title(ax, "What the computation buys",
                 "edges against cost - down and to the left is better")

    hbar(axes[3], [t / w for t, w in zip(totals, work)], "Cost per edge change",
         "computations per change applied - lower decides more cheaply", fmt="{:.1f}")

    return _finish(fig, cardspec, methods, notes, width, top, run_dir, filename)


# What the comparison figure shows, in the order the panels are drawn.
# (key, title, note, formatter). `pct` panels share an axis to 100.
COMPARISON_PANELS = [
    ("edges_mean",   "Edges used",        "mean over the networks, fewer is better", "{:.2f}"),
    ("rigid_pct",    "Rigid",             "% of networks whose shape their bearings determine", "{:.0f}%"),
    ("at_bound_pct", "Rigid at the bound", "% using exactly the fewest edges the poses allow", "{:.0f}%"),
    ("minimal_pct",  "Rigid and minimal", "% by the minimality heuristic", "{:.0f}%"),
    ("score_mean",   "Objective score",   "phi, what every method is scored by, higher is better", "{:.1f}"),
]


def plot_comparison(run_dir, rows, header, filename="comparison"):
    """Every method on one instance set, one bar panel per quantity.

    The last panel is per network rather than a mean, because a mean of 17.4 edges hides
    whether that is every network at 17 or half of them at 20.
    """
    import matplotlib
    matplotlib.use("Agg")

    agg = aggregate(rows)
    if not agg:
        return None
    methods = list(agg)
    colors = [method_style(m)["color"] for m in methods]
    m_reqs = [r["m_req"] for r in rows if r.get("m_req") not in (None, "")]
    bound = float(np.mean([int(x) for x in m_reqs])) if m_reqs else None

    notes = [
        "Every method saw the same networks, so the rows compare directly. "
        "constructive is the exception. It discards the initial edges and builds from "
        "the empty graph, being a construction rather than an edit.",
        "Rigid at the bound counts a network rigid on exactly m_req edges, the fewest "
        "these poses admit. Rigid and minimal is the same idea through the repository's "
        "heuristic, which can disagree either way on mixed-domain networks.",
        "The last panel is one dot per network with a tick at the mean, so the spread "
        "behind each bar is visible. The dashed line is m_req.",
    ]
    fig, axes, card, top, width = _figure(header, "Method comparison", methods, notes,
                                          panel_h=3.1, width=13.4, panels=(2, 3))

    for ax, (key, title, note, fmt) in zip(axes, COMPARISON_PANELS):
        vals = [agg[m].get(key) for m in methods]
        shown = [0.0 if v is None else float(v) for v in vals]
        y = np.arange(len(methods))
        ax.barh(y, shown, color=colors, alpha=0.9, zorder=3, height=0.62)
        hi = max(shown + [1e-9])
        for yi, (v, raw) in enumerate(zip(shown, vals)):
            ax.annotate("-" if raw is None else fmt.format(v), xy=(v, yi), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color=INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 118 if key.endswith("_pct") else hi * 1.30)
        if key == "edges_mean" and bound is not None:
            ax.axvline(bound, color=INK_2, ls="--", lw=1.1, zorder=4)
        _style_axes(ax)
        _panel_title(ax, title, note)

    ax = axes[5]
    rng = np.random.default_rng(0)
    for yi, m in enumerate(methods):
        vals = [r["m"] for r in rows if r["method"] == m]
        if not vals:
            continue
        colour = method_style(m)["color"]
        ax.scatter(vals, yi + (rng.random(len(vals)) - 0.5) * 0.34, s=20, color=colour,
                   alpha=0.55, linewidths=0, zorder=3)
        ax.scatter([float(np.mean(vals))], [yi], marker="|", s=170, color=colour,
                   linewidths=2.0, zorder=5)
    if bound is not None:
        ax.axvline(bound, color=INK_2, ls="--", lw=1.1, zorder=2)
        ax.annotate("m_req", xy=(bound, len(methods) - 0.5), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=INK_2, va="bottom")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=8.5)
    ax.invert_yaxis()
    _style_axes(ax)
    _panel_title(ax, "Edges per network", "one dot per network, the tick is the mean")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


def plot_estimation(run_dir, rows, header, filename="estimation"):
    """What each method's graph costs the shape estimate, four ways.

    Shape error is a property of the graph and not only of its edge count, so two
    methods on the same number of edges can land far apart here.
    """
    import matplotlib
    matplotlib.use("Agg")

    agg = aggregate(rows)
    methods = [m for m in agg if agg[m]["shape_err_gmean"] or agg[m]["min_eig_gmean"]]
    if not methods:
        return None
    colors = [method_style(m)["color"] for m in methods]

    notes = [
        "Shape error is the RMS state error per radian of bearing noise, position in "
        "formation radii and attitude in radians. Stiffness is the smallest nonzero "
        "eigenvalue of B'B. Both are geometric means, since both span decades.",
        "The third panel perturbs every bearing by the noise on its x axis, recovers "
        "the formation and compares it against the truth. The dashed line is what the "
        "rigidity matrix predicts for the same graph.",
        "A method that was not rigid on some networks is marked, because a flexible "
        "network has stiffness exactly 0 and infinite shape error and can enter "
        "neither mean.",
    ]
    # 2x2 rather than 1x4: on one row the noise panel is a quarter of the width and its
    # legend covers the curves it is labelling
    fig, axes, card, top, width = _figure(header, "Estimation quality", methods, notes,
                                          panel_h=3.9, width=11.6, panels=(2, 2))

    def bars(ax, key, title, note, fmt):
        vals = [agg[m][f"{key}_gmean"] for m in methods]
        shown = [0.0 if v is None else float(v) for v in vals]
        y = np.arange(len(methods))
        ax.barh(y, shown, color=colors, alpha=0.9, zorder=3, height=0.62)
        for yi, (v, raw, m) in enumerate(zip(shown, vals, methods)):
            mark = "*" if agg[m][f"{key}_n_pos"] < agg[m][f"{key}_n"] else ""
            ax.annotate("-" if raw is None else fmt.format(v) + mark, xy=(v, yi),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, max(shown + [1e-9]) * 1.34)
        _style_axes(ax)
        _panel_title(ax, title, note)

    bars(axes[0], "shape_err", "Shape error", "per radian of bearing noise, lower is better",
         "{:.1f}")
    bars(axes[1], "min_eig", "Stiffness", "higher survives more noise", "{:.1e}")

    ax = axes[3]
    sweep = {m: agg[m]["noise"] for m in methods if agg[m]["noise"]}
    if sweep:
        for m, v in sweep.items():
            style = method_style(m)
            sig = np.degrees(np.array(sorted(v)))
            ax.plot(sig, [v[s][0] for s in sorted(v)], "-", color=style["color"],
                    marker=style.get("marker", "o"), markersize=4, linewidth=1.8,
                    label=m, zorder=style["z"] + 2)
            ax.plot(sig, [v[s][1] for s in sorted(v)], "--", color=style["color"],
                    alpha=0.45, linewidth=1.4, zorder=style["z"])
        from matplotlib.ticker import FuncFormatter, NullFormatter
        ax.set_xscale("log")
        ax.set_yscale("log")
        # the sigmas are a handful of chosen values, so label exactly those; the default
        # log locator writes 6x10^-1 next to 10^0 next to 2x10^0 and they run together
        ticks = sorted({s for v in sweep.values() for s in v})
        ax.set_xticks(np.degrees(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("bearing noise (degrees)", color=INK_2, fontsize=8.5)
        ax.set_ylabel("RMS position error (formation radii)", color=INK_2, fontsize=8.5)
        # the only panel where identity is not a y tick label, so it needs the key.
        # Outside the axes, because inside it sits on top of the curves it labels.
        ax.legend(frameon=False, fontsize=7.6, labelcolor=INK_2, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), ncol=1, handlelength=1.6)
    _style_axes(ax)
    _panel_title(ax, "Error under bearing noise",
                 "solid is measured, dashed is predicted" if sweep
                 else "run with --noise-sweep to measure this")

    ax = axes[2]
    rng = np.random.default_rng(0)
    for yi, m in enumerate(methods):
        vals = [r["shape_err"] for r in rows
                if r["method"] == m and r.get("shape_err") and r["shape_err"] > 0]
        if not vals:
            continue
        colour = method_style(m)["color"]
        ax.scatter(vals, yi + (rng.random(len(vals)) - 0.5) * 0.34, s=20, color=colour,
                   alpha=0.55, linewidths=0, zorder=3)
        ax.scatter([_gmean(vals)], [yi], marker="|", s=170, color=colour, linewidths=2.0,
                   zorder=5)
    ax.set_xscale("log")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("shape error", color=INK_2, fontsize=8.5)
    _style_axes(ax)
    _panel_title(ax, "Shape error per network",
                 "one dot per network, the tick is the geometric mean")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


def plot_topology(run_dir, instances, rows, header, filename="topology"):
    """The graph each method built, and what that graph costs the estimate.

    Two rows per method: the edge set on its own, then the same edges in grey behind
    the per-agent position uncertainty and the softest deformation mode. Splitting them
    is what makes the graph legible; drawn together the arrows and shells hide it.
    """
    from rigidity import (error_covariance, extended_bearing_rigidity_matrix,
                          nullspace_and_softest, rigidity_decomposition,
                          scaled_rigidity_matrix, characteristic_length,
                          estimation_error)
    if not instances:
        return None
    net = instances[0]
    ep_rows = _panel_rows(rows)
    if not ep_rows:
        return None
    cap_note = _panel_cap_note(rows)

    P = np.array([a.pose.position for a in net.agents], dtype=float)
    span = float(np.abs(P - P.mean(axis=0)).max()) or 1.0
    rank_K = int(np.linalg.matrix_rank(
        extended_bearing_rigidity_matrix(net.fully_connected())))

    # one geometry pass per method, so the shells share a scale and the numbers on the
    # panel titles come from the same decomposition that drew them
    per = []
    for row in ep_rows:
        work = copy.deepcopy(net)
        work.edges = row["edges"].copy()
        L = characteristic_length(work)
        Bs = scaled_rigidity_matrix(work, None, L)
        rank, s, lam = rigidity_decomposition(Bs, rank_K)
        # (6n, 6n); the per-agent position block is the 3x3 on its own diagonal
        full = error_covariance(Bs, rank_K)
        cov = [full[3 * i:3 * i + 3, 3 * i:3 * i + 3] for i in range(work.n)]
        _, v, _, _ = nullspace_and_softest(Bs, int(rank))
        mode = (v[:3 * work.n, 0].reshape(-1, 3) if v is not None and v.shape[1]
                else np.zeros_like(P))
        a_opt = estimation_error(s, rank_K, rank)[0]
        err = float(np.sqrt(a_opt / work.n)) if np.isfinite(a_opt) else float("nan")
        per.append({"row": row, "cov": cov, "mode": mode, "lam": lam, "err": err})

    sd_max = max((float(np.sqrt(max(np.linalg.eigvalsh(c).max(), 0.0)))
                  for p in per for c in p["cov"]), default=1.0) or 1.0
    shell = 0.42 * span / sd_max

    methods = [p["row"]["method"] for p in per]
    notes = [
        "The top row is the graph each method built. The bottom row keeps that graph in "
        "grey and adds, per agent, the 1-sigma position uncertainty at one radian of "
        "bearing noise as a shell, and the softest deformation mode as an arrow.",
        "The shells share one scale across every panel, so their sizes compare. Arrows "
        "are scaled per panel, so their lengths do not.",
        "Drawn on the first evaluated network. Shape error and stiffness above each "
        "lower panel are for that graph on that network, not the run average.",
    ]
    if cap_note:
        notes.append(cap_note)
    dom = _domain_note(net)
    if dom:
        notes.append(dom)

    ncols = len(per)
    fig, _, card, top, width = _figure(header, "Topology and what it costs the estimate",
                                       methods, notes, panel_h=4.9,
                                       width=max(4.4 * ncols, 8.0), panels=(2, ncols),
                                       axes3d=True)
    band = (top, card[1])

    for k, p in enumerate(per):
        row, colour = p["row"], method_style(p["row"]["method"])["color"]

        ax = _formation_axes(fig, 2, ncols, k + 1, P, band)
        _draw_edges_3d(ax, P, row["edges"], color=colour, alpha=0.55, width=1.4)
        _draw_agents(ax, net, P)
        bound = row.get("m_req")
        _panel_title_3d(ax, row["method"],
                        f"{row['m']} edges" + (f", the bound is {bound}" if bound else ""))

        ax = _formation_axes(fig, 2, ncols, ncols + k + 1, P, band)
        _draw_edges_3d(ax, P, row["edges"], color=AXIS, alpha=0.5, width=0.8)
        for i in range(net.n):
            _ellipsoid(ax, p["cov"][i], P[i], shell, colour)
        d = p["mode"] * (0.30 * span / (float(np.abs(p["mode"]).max()) or 1.0))
        ax.quiver(P[:, 0], P[:, 1], P[:, 2], d[:, 0], d[:, 1], d[:, 2],
                  color=INK, linewidth=1.6, arrow_length_ratio=0.22, zorder=9)
        _draw_agents(ax, net, P)
        _panel_title_3d(ax, row["method"],
                        f"shape error {p['err']:.2f},  stiffness {p['lam']:.1e}")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename, tight=False)


ABLATION_MODE_HUE = {"shuffle": MODEL_HUES[0], "zero": MODEL_HUES[1],
                     "noise": MODEL_HUES[2]}


def plot_ablation(run_dir, per_mode, header, model, filename="ablation"):
    """What destroying each observation channel costs the policy.

    `per_mode` is {mode: (rows, ref_row)} as ablation.measure returns them. A channel a
    mode could not perturb gets a marker at zero rather than a bar, because a zero there
    would read as evidence the policy ignores it.
    """
    import matplotlib
    matplotlib.use("Agg")

    modes = [m for m in ("shuffle", "zero", "noise") if m in per_mode]
    if not modes:
        return None
    channels = [r["channel"] for r in per_mode[modes[0]][0]]
    # the most depended-on channel at the top, by its worst cost over the modes
    worst = {c: max((_ab_value(per_mode[m][0], c, "d_phi") or 0.0) for m in modes)
             for c in channels}
    channels = sorted(channels, key=lambda c: worst[c])

    notes = [
        "Each channel is destroyed on its own and the episode is re-run. shuffle "
        "permutes it across nodes or pairs and keeps its distribution; zero and noise "
        "also change its scale, so a reaction there can be a response to an input the "
        "network never saw in training.",
        "Cost in phi is the reference run's score minus the ablated one, so positive "
        "means the policy did worse without the channel.",
        "A marker at zero is a channel that mode could not perturb, which happens when "
        "the channel is constant along the shuffled axis. That is not the same as a "
        "measured zero.",
    ]
    # no method key on the card: the modes are not methods, and the axes legend below
    # already maps each one to its colour
    fig, axes, card, top, width = _figure(header, f"Observation channel ablation, {model}",
                                          [], notes, panel_h=0.30 * len(channels) + 1.6,
                                          width=13.0, panels=(1, 2))

    panels = [("d_phi", "Cost in phi of destroying the channel",
               "positive means the policy did worse without it"),
              ("flip_pct", "Steps where the best action changed",
               "% of steps on an unperturbed reference trajectory")]
    h = 0.26
    y = np.arange(len(channels))
    for ax, (key, title, note) in zip(axes, panels):
        for k, mode in enumerate(modes):
            rows = per_mode[mode][0]
            vals = [_ab_value(rows, c, key) for c in channels]
            pos = y + (k - (len(modes) - 1) / 2) * h
            ax.barh([p for p, v in zip(pos, vals) if v is not None],
                    [v for v in vals if v is not None], height=h * 0.92,
                    color=ABLATION_MODE_HUE[mode], alpha=0.9, zorder=3,
                    label=mode if ax is axes[0] else None)
            ax.scatter([0.0] * sum(v is None for v in vals),
                       [p for p, v in zip(pos, vals) if v is None],
                       marker="x", s=14, color=MUTED, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(channels, fontsize=8.5)
        ax.set_xlabel({"d_phi": "phi lost", "flip_pct": "% of steps"}[key],
                      color=INK_2, fontsize=8.5)
        _style_axes(ax)
        _panel_title(ax, title, note)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right",
                   title="mode", title_fontsize=8.5)

    return _finish(fig, card, [], notes, width, top, run_dir, filename)


def _ab_value(rows, channel, key):
    """The measured value, or None when that mode could not perturb the channel.

    `perturbed` is the fraction of states the perturbation actually changed;
    ablation.py calls a channel live at 1% and above, and the same threshold is used
    here so the figure and the table agree on what counts as measured.
    """
    for r in rows:
        if r["channel"] != channel:
            continue
        if r.get("perturbed", 1.0) < 0.01:
            return None
        return float(r["d_phi"] if key == "d_phi" else 100.0 * r["flip"])
    return None


# Four quantities that say whether training worked, in the order they are drawn.
# (tag, title, note, log)
TRAINING_PANELS = [
    ("Reward / Total reward (mean)", "Episode return",
     "the reward is potential based, so this is the total change in phi", False),
    ("Episode/ Best nr edges", "Edges in the best graph of the episode",
     "fewer is better once the graph is rigid", False),
    ("Episode/ Best is min rigid", "Episodes whose best graph was rigid and minimal",
     "fraction of episodes", False),
    ("Probe/ useful (argmax)", "Steps that improved phi, argmax rollout",
     "measured on fixed instances, not on the exploring policy", False),
]


def _smooth(v, frac=0.03):
    """Edge-padded rolling mean. The raw series is drawn behind it."""
    w = max(3, int(len(v) * frac) | 1)
    if len(v) < w:
        return v
    pad = np.pad(v, w // 2, mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def plot_training(run_dir, series, header, panels=None, filename="training"):
    """Learning curves, one line per run.

    `series` is {label: {tag: (steps, values)}}. A run missing a tag is named on the
    card rather than being silently absent from its panel.
    """
    import matplotlib
    matplotlib.use("Agg")

    panels = panels or TRAINING_PANELS
    labels = list(series)
    if not labels:
        return None

    missing = {lab: [t for t, *_ in panels if t not in series[lab]] for lab in labels}
    notes = [
        "The thin line is one point per logged episode and the thick line is a rolling "
        "mean over 3% of the run. The x axis is environment steps.",
        "The first panels are measured on the exploring policy during training. The "
        "probe panel is a separate deterministic rollout on fixed instances, so the "
        "two are not directly comparable to each other.",
    ]
    absent = [f"{lab} has no {', '.join(t.strip() for t in ts)}"
              for lab, ts in missing.items() if ts]
    if absent:
        notes.append("Not every run logged every quantity. " + "; ".join(absent) + ".")

    fig, axes, card, top, width = _figure(header, "Training", labels, notes,
                                          panel_h=3.4, width=12.0, panels=(2, 2))
    for ax, (tag, title, note, log) in zip(axes, panels):
        drawn = False
        for lab in labels:
            if tag not in series[lab]:
                continue
            st, v = series[lab][tag]
            style = method_style(lab)
            ax.plot(st / 1000.0, v, color=style["color"], alpha=0.16, linewidth=0.7,
                    zorder=2)
            ax.plot(st / 1000.0, _smooth(v), color=style["color"], linewidth=2.0,
                    zorder=style["z"], label=lab)
            drawn = True
        ax.set_xlabel("environment steps (thousands)", color=INK_2, fontsize=8.5)
        _style_axes(ax, log=log)
        _panel_title(ax, title, note if drawn else "not logged by any run")
    # one legend for the figure, not one per panel
    handles, names = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, frameon=False, fontsize=9, labelcolor=INK_2,
                   loc="upper right", ncol=min(len(names), 4),
                   bbox_to_anchor=(0.99, 1.0 if _PLAIN else 0.985))
    return _finish(fig, card, labels, notes, width, top, run_dir, filename)


GENERALISATION_PANELS = [
    ("rigid_pct", "Rigid", "% of instances", (0, 118)),
    ("at_bound_pct", "Rigid at the bound", "% using the fewest edges the poses allow",
     (0, 118)),
    ("edges_over_bound", "Edges over the bound", "one is the bound, lower is better",
     None),
    ("phi", "Objective score", "phi, comparable across size and domain", None),
]


def plot_generalisation(run_dir, gen_rows, header, filename="generalisation"):
    """One model against greedy across the instance sets it was evaluated on.

    Each column is a different network, so this says where a policy trained on one
    configuration still works and where it stops working.
    """
    import matplotlib
    matplotlib.use("Agg")

    benches = list(dict.fromkeys(r["benchmark"] for r in gen_rows))
    seen = {}
    for r in gen_rows:
        seen.setdefault(r["method"], set()).add(r["benchmark"])
    # a method evaluated on one set says nothing about generalising, and drawing it
    # puts a "not run" on every other column. Say which were dropped instead.
    series = [m for m in dict.fromkeys(r["method"] for r in gen_rows) if len(seen[m]) > 1]
    thin = [m for m in seen if len(seen[m]) <= 1]
    if not benches or not series:
        return None

    notes = [
        "Every column is a separate evaluation run on a different instance set, "
        "gathered from earlier output directories rather than measured here.",
        "Rigid at the bound counts an instance rigid on exactly m_req edges. Edges over "
        "the bound is the mean edge count divided by m_req, so one is the bound and the "
        "quantity is comparable across sets whose bounds differ.",
        "Those runs were made at different times and possibly against different code. "
        "generalisation.csv carries each row's source directory and git commit.",
    ]
    if thin:
        notes.append("Only evaluated on one set, so not drawn: " + ", ".join(sorted(thin))
                     + ". They are in generalisation.csv.")
    if any(r.get("at_bound_pct") is None for r in gen_rows):
        notes.append("A run made before the edge bound was recorded per instance carries "
                     "no bound, so its columns are blank in the two bound panels rather "
                     "than zero.")
    # Width is capped rather than scaled per benchmark: 3.4 inches each put twelve sets
    # on a forty-inch figure that nothing can display. Past a handful the labels tilt.
    fig, axes, card, top, width = _figure(header, "Across instance sets", series, notes,
                                          panel_h=3.6,
                                          width=min(max(1.0 * len(benches) + 6.0, 11.0), 17.0),
                                          panels=(2, 2))

    x = np.arange(len(benches))
    w = 0.8 / max(len(series), 1)
    for ax, (key, title, note, ylim) in zip(axes, GENERALISATION_PANELS):
        for k, name in enumerate(series):
            vals, pos = [], []
            for i, b in enumerate(benches):
                hit = next((r for r in gen_rows
                            if r["benchmark"] == b and r["method"] == name), None)
                v = None if hit is None else hit.get(key)
                if v is not None:
                    vals.append(float(v))
                    pos.append(i + (k - (len(series) - 1) / 2) * w)
            ax.bar(pos, vals, width=w * 0.9, color=method_style(name)["color"],
                   alpha=0.9, zorder=3, label=name if ax is axes[0] else None)
        if key == "edges_over_bound":
            ax.axhline(1.0, color=INK_2, ls="--", lw=1.1, zorder=4)
        ax.set_xticks(x)
        names = [b.replace("bench_", "") for b in benches]
        if len(benches) > 5:
            ax.set_xticklabels(names, fontsize=7.6, rotation=35, ha="right",
                               rotation_mode="anchor")
        else:
            ax.set_xticklabels(names, fontsize=8.5)
        # a set where a method was never evaluated is said so, not left as a gap that
        # reads as a measured zero
        for i, b in enumerate(benches):
            if not any(r["benchmark"] == b and r.get(key) is not None for r in gen_rows):
                continue
            for k, name in enumerate(series):
                hit = next((r for r in gen_rows if r["benchmark"] == b
                            and r["method"] == name), None)
                if hit is None:
                    ax.annotate("not run", xy=(i + (k - (len(series) - 1) / 2) * w, 0),
                                rotation=90, fontsize=6.2, color=MUTED, ha="center",
                                va="bottom", xytext=(0, 2), textcoords="offset points")
        if ylim:
            ax.set_ylim(*ylim)
        _style_axes(ax)
        _panel_title(ax, title, note)

    handles, names = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, frameon=False, fontsize=9, labelcolor=INK_2,
                   loc="upper right", ncol=min(len(names), 4),
                   bbox_to_anchor=(0.99, 1.0 if _PLAIN else 0.985))
    return _finish(fig, card, series, notes, width, top, run_dir, filename)


def plot_run_info(run_dir, info, header, filename="run_info"):
    """What this run was: the environment, the instances, and every model in it.

    The other figures carry only their own title. This is where the run is written down,
    so a figure pulled into a document a month later can be traced back to it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blocks = [(title, [(k, v) for k, v in rows if v not in (None, "")])
              for title, rows in info]
    blocks = [(t, r) for t, r in blocks if r]
    if not blocks:
        return None

    width = 12.0
    line_h = 0.20                       # inches per row, matching the 9pt text below
    rows_total = sum(len(r) + 1.6 for _, r in blocks)
    fig_h = 0.5 + line_h * rows_total
    fig = plt.figure(figsize=(width, fig_h), facecolor=SURFACE)
    if not _PLAIN:
        fig.text(0.008, 1.0 - 0.26 / fig_h, "What this run was", color=INK,
                 fontsize=12.5, ha="left", va="top")

    y = 1.0 - (0.62 if not _PLAIN else 0.22) / fig_h
    label_w = max((len(k) for _, r in blocks for k, _ in r), default=10)
    for title, rows in blocks:
        fig.text(0.012, y, title.upper(), color=INK_2, fontsize=8.0, ha="left",
                 va="top", family="monospace")
        y -= line_h * 1.1 / fig_h
        for key, value in rows:
            # one line per field, wrapped rather than truncated: these are long
            # generated names and a clipped one is worse than a wrapped one
            for i, part in enumerate(_wrap(str(value), width * 0.72, 9.0)):
                fig.text(0.020, y, key if i == 0 else "", color=MUTED, fontsize=9.0,
                         ha="left", va="top", family="monospace")
                fig.text(0.020 + 0.0062 * (label_w + 2), y, part, color=INK,
                         fontsize=9.0, ha="left", va="top", family="monospace")
                y -= line_h / fig_h
        y -= line_h * 0.5 / fig_h

    return _save(fig, run_dir, filename)


# ── the table, as a figure ────────────────────────────────────────────────────────────
# Same numbers as summary.txt, laid out rather than printed: the direction of each column
# is a subtitle instead of a "(fewer)" tag, reference rows are drawn as reference rows,
# and it drops into a slide next to the other figures without a monospace dump.
TABLE_COLUMNS = [
    dict(key="method",  title="method",   unit="",                 w=1.05, align="left"),
    dict(key="edges",   title="edges",    unit="fewer is better",  w=1.15, align="right"),
    dict(key="score",   title="score  φ", unit="higher is better", w=1.20, align="right"),
    dict(key="rigid",   title="rigid",    unit="% of networks",    w=0.85, align="right"),
    dict(key="at_bound", title="at bound", unit="% of networks",   w=0.90, align="right"),
    dict(key="minimal", title="minimal",  unit="% of networks",    w=0.85, align="right"),
    dict(key="margin_geo", title="stiffness", unit="gmean, higher is better",
         w=1.30, align="right"),
    dict(key="shape_err", title="shape error", unit="gmean, lower is better",
         w=1.35, align="right"),
    dict(key="work",    title="work",     unit="edits applied",    w=1.05, align="right"),
    dict(key="best_at", title="best at",  unit="step reached",     w=1.10, align="right"),
    dict(key="opt",     title="= best",   unit="% matched",        w=0.80, align="right"),
]

TABLE_NOTES = [
    "edges is how many directed bearing measurements the network needs. Each one is a "
    "sensing or communication link, so fewer is better.",
    "score φ is the objective every method is scored with. It rewards rigidity and "
    "charges for each extra edge. Higher is better.",
    "rigid means the network's shape is fully determined by its bearing measurements. "
    "at bound means rigid using exactly the fewest edges these poses allow, counted "
    "against a proven bound. minimal is the same idea through the repository's "
    "minimality heuristic, which can disagree with the bound either way on "
    "mixed-domain networks, so both are shown.",
    "stiffness is how strongly the bearings react to a change in shape. Higher is "
    "better. It is shown as a geometric mean and spread because it ranges over orders of "
    "magnitude, where 'a ×/÷ b' means the typical network sits between a/b and a·b. Its "
    "absolute size depends on how far apart the agents are, so compare rows rather than "
    "the number itself.",
    "shape error is how far the recovered formation lands from the true one, per radian "
    "of error in the bearing measurements, with position counted in formation radii and "
    "attitude in radians. Lower is better. 8.0 means one degree of bearing error "
    "(0.017 rad) displaces the shape by about 14% of its own size. Unlike stiffness it is "
    "comparable across network sizes, domains and pose ranges. A '*' on either column "
    "marks rows where non-rigid networks had to be left out, since their stiffness is "
    "exactly 0 and their shape error infinite and neither can enter a geometric mean.",
    "work is how many changes to the network the method applied. best at is the step its "
    "best network was reached, so a lower number means the rest of the budget added "
    "nothing.",
    "= best is the share of networks where the method tied the exhaustive optimum.",
    "initial and optimal are reference rows rather than competing methods. Every method "
    "starts from initial and optimal is the best achievable. All methods see the same "
    "networks. constructive is the exception, since it discards the initial edges and "
    "builds from empty, being a construction algorithm rather than an edit one.",
    "Every value is a mean over the networks and ± is the standard deviation across "
    "them. The percentage columns carry no ± because they are already means of a yes/no "
    "outcome, whose spread is fixed by the percentage itself.",
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

    head_h = 0.0 if _PLAIN else _header_height(header, width)
    card_h = 0.0 if _PLAIN else _card_height(methods, notes, width)
    row_h, head_row_h = 0.30, 0.52
    table_h = head_row_h + row_h * len(methods) + 0.30
    fig_h = head_h + table_h + card_h
    fig = plt.figure(figsize=(width, fig_h), facecolor=SURFACE)
    top = 1.0 if _PLAIN else _draw_header(fig, header, "Baseline comparison")

    def cell(method, key, v):
        ref = method in ("initial", "optimal")
        if key == "edges":
            return f"{v['edges_mean']:.2f} ±{v['edges_sd']:.2f}"
        if key == "score":
            return f"{v['score_mean']:.2f} ±{v['score_sd']:.2f}"
        if key == "rigid":
            return f"{v['rigid_pct']:.0f}"
        if key == "at_bound":
            return "-" if v["at_bound_pct"] is None else f"{v['at_bound_pct']:.0f}"
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
        style = method_style(m)
        mx0 = edges[0][0]
        fig.add_artist(_fig_line(fig, [mx0 + 0.006, mx0 + 0.030], [yc, yc],
                                 style["color"], style["ls"]))
        fig.text(mx0 + 0.038, yc, m, color=INK_2 if ref else INK, fontsize=9, va="center")
        for c, (cx0, cx1) in list(zip(cols, edges))[1:]:
            fig.text(cx1 - 0.006, yc, cell(m, c["key"], v), color=INK_2 if ref else INK,
                     fontsize=9, ha="right", va="center", family="monospace")

    if not _PLAIN:
        _draw_card(fig, methods, notes, width, card_h)
    return _save(fig, run_dir, filename)


def _save(fig, run_dir, name):
    """Every figure ships in both formats, filed by format: plots/pdf/ and plots/png/.
    PDF is what goes in the thesis, PNG is what you actually look at, and mixing them in
    one directory meant scrolling past each figure twice."""
    import matplotlib.pyplot as plt
    name = f"{name}-plain" if _PLAIN else name
    out = []
    for ext in PLOT_FORMATS:
        directory = os.path.join(run_dir, "plots", ext)
        os.makedirs(directory, exist_ok=True)   # callers may pass a bare --out-dir
        path = os.path.join(directory, f"{name}.{ext}")
        fig.savefig(path, facecolor=SURFACE, dpi=200, bbox_inches="tight")
        out.append(path)
    plt.close(fig)
    return out
