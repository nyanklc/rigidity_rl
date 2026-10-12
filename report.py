"""Rendering for the experiments: the comparison table, the CSVs and the plots.

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
SURFACE = "#ffffff"   # the page is white, so the panel is white
INK = "#000000"
INK_2 = "#1a1a1a"
MUTED = "#4d4d4d"
GRID = "#cccccc"
AXIS = "#000000"
CARD = "#f0f0f0"   # one step off the surface, so the notes card reads as a panel

# Text is scaled once, here, rather than at each call site. The figures are drawn a
# little wider than the page and every explicit fontsize goes through _fs, so a label
# lands near the body text size once the figure is scaled into the document.
FONT_SCALE = 1.0


def _fs(size):
    return size * FONT_SCALE


# matplotlib's own cycle, so a figure looks like a plain matplotlib output. Colour
# separates the series and the marker and dash repeat that separation in print.
C_BLUE, C_ORANGE, C_GREEN, C_RED = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
C_PURPLE, C_BROWN, C_PINK, C_GREY = "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
C_OLIVE, C_CYAN = "#bcbd22", "#17becf"

# The hues that belong to the trained models, kept clear of the baselines above.
MODEL_HUES = [C_ORANGE, C_PINK, C_RED, C_PURPLE, C_CYAN, C_OLIVE, C_BROWN, C_GREEN]

BASELINE_STYLE = {
    "greedy":       {"color": C_BLUE,   "ls": "-",                    "marker": "s", "z": 4},
    "spectral":     {"color": C_GREEN,  "ls": (0, (6, 1.6, 1, 1.6)),  "marker": "D", "z": 3},
    "constructive": {"color": C_BROWN,  "ls": "-",                    "marker": "^", "z": 3},
    "anneal":       {"color": C_PURPLE, "ls": (0, (4, 1.6, 1, 1.6)),  "marker": "v", "z": 2},
    "optimal":      {"color": C_RED,    "ls": "--",                   "marker": "*", "z": 5},
    "degree":       {"color": C_OLIVE,  "ls": "-",                    "marker": "P", "z": 2},
    "random":       {"color": C_CYAN,   "ls": (0, (5, 2)),            "marker": "X", "z": 2},
    "initial":      {"color": C_GREY,   "ls": ":",                    "marker": ".", "z": 1},
}

# With one policy the tables are the same, with the policy taking the remaining hue.
SINGLE_MODEL_STYLE = {
    "greedy":  {"color": C_BLUE,   "ls": "-", "marker": "s", "z": 3},
    "learned": {"color": C_ORANGE, "ls": "-", "marker": "o", "z": 6},
    "random":  {"color": C_CYAN,   "ls": "-", "marker": "X", "z": 2},
    "constructive": {"color": C_BROWN, "ls": "-", "marker": "^", "z": 3},
    "degree":  {"color": C_OLIVE,  "ls": "-", "marker": "P", "z": 2},
    "spectral": {"color": C_GREEN, "ls": "-", "marker": "D", "z": 3},
    "anneal":  {"color": C_PURPLE, "ls": "-", "marker": "v", "z": 2},
    "initial": {"color": C_GREY,   "ls": ":", "marker": ".", "z": 1},
    "optimal": {"color": C_RED,    "ls": "--", "marker": "*", "z": 5},
}

CLASSICAL_ORDER = ["initial", "random", "degree", "greedy", "spectral", "anneal",
                   "constructive"]

METHOD_STYLE = {}
METHOD_ORDER = []

# The formation figures draw one 3-D panel per method. Every panel is a fixed width, so
# `_grid_for` widens the figure rather than shrinking them and this cap bounds height.
MAX_FORMATION_PANELS = 4

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
    FORMATION_PRIORITY[:] = (labels[:1] + ["greedy", "anneal", "constructive", "initial",
                                           "degree", "spectral", "optimal", "random"]
                             + labels[1:])


@contextlib.contextmanager
def restyle(name, **style):
    """Draw one method with a different style for the duration of one figure.

    `MODEL_HUES` and `BASELINE_STYLE` share colours, so the seventh model is drawn in
    `constructive`'s brown and the fourth in `anneal`'s purple. That is invisible while
    every series is on one figure with the baselines in their own panel region, and
    fatal on a figure that carries the baselines and one model together. Rather than
    repaint the whole palette, the caller says which method to move and to what.
    """
    was = METHOD_STYLE.get(name)
    METHOD_STYLE[name] = dict(was or {}, **style)
    try:
        yield
    finally:
        if was is None:
            METHOD_STYLE.pop(name, None)
        else:
            METHOD_STYLE[name] = was


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
# The quantities a paired comparison is worth running on, and which direction wins.
PAIRED_KEYS = (("m", "edges", "lower"),
               ("score", "phi", "higher"),
               ("shape_err", "shape err", "lower"),
               ("is_IBR", "rigid", "higher"),
               ("at_bound", "at bound", "higher"))


def _paired_value(row, key):
    """One quantity off a row. `at_bound` is derived, not stored, so a live run has no
    such key and the column would silently vanish from the block."""
    if key == "at_bound":
        return at_bound(row)
    return row.get(key)


def paired_stats(rows, reference="greedy", keys=PAIRED_KEYS, boots=10000, seed=0):
    """Per-instance differences against `reference`, with a bootstrap CI.

    Every method is run on the same instances, so the pairing is free and throwing it
    away costs real power: `aggregate` reports mean +- sd across networks, and the
    network-to-network spread is far larger than the difference between two methods on
    one network. This resamples the per-instance DIFFERENCES, which is the quantity the
    comparison is actually about.

    -> {method: {key: {n, mean, lo, hi, wins, losses, ties}}}. `lo`/`hi` are the 2.5th
    and 97.5th percentiles of the bootstrap, so an interval clear of zero is a
    difference the instances support. Nothing here is a p-value: with 20 instances and
    one seed it is a spread, not a test.
    """
    rng = np.random.default_rng(seed)
    ref = {r["episode"]: r for r in rows if r["method"] == reference}
    if not ref:
        return {}
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    methods += [m for m in dict.fromkeys(r["method"] for r in rows) if m not in methods]

    out = {}
    for method in methods:
        if method == reference:
            continue
        sel = [r for r in rows if r["method"] == method and r["episode"] in ref]
        per_key = {}
        for key, _, direction in keys:
            diffs = []
            for r in sel:
                a, b = _paired_value(r, key), _paired_value(ref[r["episode"]], key)
                if a is None or b is None or a == "" or b == "":
                    continue
                # geometric for the quantity that spans decades, as the table is
                d = (np.log10(float(a)) - np.log10(float(b))
                     if key == "shape_err" and float(a) > 0 and float(b) > 0
                     else float(a) - float(b))
                if np.isfinite(d):
                    diffs.append(d)
            if len(diffs) < 3:
                continue
            d = np.asarray(diffs, dtype=float)
            draws = rng.integers(0, len(d), size=(boots, len(d)))
            means = d[draws].mean(axis=1)
            better = (d < 0) if direction == "lower" else (d > 0)
            worse = (d > 0) if direction == "lower" else (d < 0)
            per_key[key] = {
                "n": len(d),
                "mean": float(d.mean()),
                "lo": float(np.percentile(means, 2.5)),
                "hi": float(np.percentile(means, 97.5)),
                "wins": int(better.sum()),
                "losses": int(worse.sum()),
                "ties": int((d == 0).sum()),
                "direction": direction,
                "log10": key == "shape_err",
            }
        if per_key:
            out[method] = per_key
    return out


def format_paired(rows, reference="greedy", brief=False, keys=PAIRED_KEYS):
    """paired_stats as text. Empty string when there is no reference to pair against."""
    if not reference:
        return ""
    stats = paired_stats(rows, reference=reference, keys=keys)
    if not stats:
        return ""
    labels = {k: lab for k, lab, _ in keys}
    lines = ["", "=" * 78,
             f"PAIRED AGAINST {reference.upper()}, PER INSTANCE",
             "=" * 78,
             f"  {'method':<14}{'quantity':<10}{'n':>4}{'mean diff':>14}"
             f"{'95% CI':>24}{'w/l/t':>12}"]
    for method, per_key in stats.items():
        first = True
        for key, _, _ in keys:
            st = per_key.get(key)
            if st is None:
                continue
            name = method if first else ""
            first = False
            mean = f"{st['mean']:+.3g}" + (" dex" if st["log10"] else "")
            ci = f"[{st['lo']:+.3g}, {st['hi']:+.3g}]"
            wlt = f"{st['wins']}/{st['losses']}/{st['ties']}"
            lines.append(f"  {name:<14}{labels[key]:<10}{st['n']:>4}{mean:>14}"
                         f"{ci:>24}{wlt:>12}")
    if not brief:
        lines += ["", "  HOW TO READ IT", "",
                  "  Every method saw the same networks, so the difference can be taken on",
                  "  each network and then averaged. That is a much smaller spread than the",
                  "  table's +-, which is how much the NETWORKS differ from each other.",
                  "",
                  "  mean diff  method minus " + reference + " on the same network, so the sign",
                  "             follows the quantity: fewer edges and lower shape error are",
                  "             better, higher phi and more rigid are better.",
                  "  shape err  is a log10 difference (dex), since the quantity spans decades:",
                  "             -1 is ten times lower than " + reference + " on the same network.",
                  "  95% CI     percentile bootstrap over the per-network differences. An",
                  "             interval clear of zero is a difference these instances support.",
                  "             It is a spread, not a test, and one seed cannot make it one.",
                  "  w/l/t      networks where the method was better, worse and exactly tied."]
    return "\n".join(lines) + "\n"


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
        lines.append("  columns are the per-axis sigma; the RMS angle is sqrt(2) times it")
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
        lines.append("  The noise block perturbs every bearing by that many degrees of")
        lines.append("  per-axis sigma -- the RMS angle between the true and the noisy")
        lines.append("  bearing is sqrt(2) times it, the tangent plane having two")
        lines.append("  components -- then")
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
# best_m / best_score / best_final_gap are filled only by a policy that chooses when to
# stop, where the gap between the graph it halted on and the best one it saw is the
# measurement. Blank for every other method, whose row is one or the other by definition.
RESULT_FIELDS = ["episode", "method", "m", "m_req", "score", "is_IBR", "is_MBR",
                 "at_bound", "min_eig", "shape_err", "work", "best_at",
                 "best_m", "best_score", "best_final_gap", "edges_are", "edges"]
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


def encode_edges(edges):
    """The adjacency matrix as `i>j` tokens in row-major order. "" for no edges.

    Every row already carries the graph it reports; without this the csv drops it and
    any question about the structure of an answer needs the run done again.
    """
    if edges is None:
        return None
    e = np.asarray(edges, dtype=bool)
    return " ".join(f"{i}>{j}" for i, j in zip(*np.nonzero(e)))


def decode_edges(text, n):
    """encode_edges inverted. None when the column is absent, which a run written
    before it was added is."""
    if text is None:
        return None
    e = np.zeros((n, n), dtype=bool)
    for token in str(text).split():
        i, _, j = token.partition(">")
        e[int(i), int(j)] = True
    return e


def write_csvs(run_dir, rows, traces):
    with open(os.path.join(run_dir, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(dict(r, at_bound=at_bound(r),
                             edges=encode_edges(r.get("edges"))))
    if traces:
        with open(os.path.join(run_dir, "trajectories.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=TRACE_FIELDS, extrasaction="ignore")
            wr.writeheader()
            for t in traces:
                wr.writerow(t)


NOISE_FIELDS = ["episode", "method", "sigma_rad", "sigma_deg", "measured", "predicted",
                "failed"]


def write_noise(run_dir, rows):
    """One row per (episode, method, noise level). -> whether anything was written.

    The sweep's per-level numbers live only on the row dicts, and results.csv cannot
    hold them because they are one value per sigma rather than one per row. Without
    this the noise and prediction figures cannot be redrawn without repeating the
    whole sweep, which is the most expensive part of an evaluation run.
    """
    out = []
    for r in rows:
        pred = r.get("pred_err")
        for sigma, measured in sorted((r.get("noise") or {}).items()):
            out.append({
                "episode": r.get("episode"), "method": r["method"],
                "sigma_rad": sigma, "sigma_deg": float(np.degrees(sigma)),
                "measured": measured,
                # the analytic counterpart is a slope, so the prediction at this level
                # is sigma times it; pred_err is kept so the slope itself survives too
                "predicted": (None if pred is None or not np.isfinite(pred)
                              else sigma * float(pred)),
                "failed": (r.get("noise_failed") or {}).get(sigma),
            })
    if not out:
        return False
    with open(os.path.join(run_dir, "noise.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=NOISE_FIELDS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(out)
    return True


def write_instances(run_dir, instances, meta=None):
    """The Network objects the formation figures are drawn from.

    Every csv here is a summary of a graph; the formation, uncertainty, softest-mode
    and sensitivity figures need the poses and domains themselves, so they are stored
    in the benchmark's own format and read back with benchmark.read_networks.
    """
    if not instances:
        return False
    import benchmark
    benchmark.save_networks(instances, os.path.join(run_dir, "instances.npz"),
                            meta=meta)
    return True


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
    fig.text(0.008, y, title, color=INK, fontsize=_fs(12.5), ha="left", va="top")
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

    # where the blurb starts, set by the longest method name rather than by a constant:
    # at a fixed offset a name like `constructive` runs straight into its own blurb
    longest = max((len(n) for n, _, st in left if st not in (None, "note")), default=0)
    blurb_x = col_l + 0.046 + longest * _fs(7.8) * 0.55 / 72.0 / width_in

    for i, (name, blurb, style) in enumerate(left):
        y = top - i * step
        if style == "note":                     # notes that spilled over from the right
            fig.text(col_l, y, name, color=INK_2, fontsize=_fs(7.5), va="center")
            continue
        if style is None:                       # column heading (or a blank spacer)
            fig.text(col_l, y, name, color=INK_2, fontsize=_fs(7.0), va="center",
                     family="monospace")
            continue
        fig.add_artist(_fig_line(fig, [col_l + 0.002, col_l + 0.030], [y, y],
                                 style["color"], style["ls"]))
        fig.text(col_l + 0.040, y, name, color=INK, fontsize=_fs(7.8), va="center")
        if blurb:
            fig.text(blurb_x, y, _clip(blurb, width_in * (col_r - blurb_x), 7.5),
                     color=INK_2, fontsize=_fs(7.5), va="center")

    for i, (line, heading) in enumerate(right):
        y = top - i * step
        fig.text(col_r, y, line, color=INK_2, fontsize=_fs(7.0) if heading else 7.5,
                 va="center", family="monospace" if heading else None)


def _fig_line(fig, xs, ys, color, ls):
    from matplotlib.lines import Line2D
    return Line2D(xs, ys, transform=fig.transFigure, color=color, linestyle=ls,
                  linewidth=2.4, solid_capstyle="round", zorder=3)


def _method_ticklabels(ax, methods, fontsize=_fs(8.5)):
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
    """Title, with the reading direction on a muted second line.

    The note is wrapped to the panel rather than to a fixed character count, since the
    same note sits under a half-width panel in one figure and a third-width one in
    another, and the title band is grown to fit however many lines that took.
    """
    if _PLAIN:
        ax.set_title(title, color=INK, fontsize=_fs(10), loc="left", pad=6)
        return
    size = _fs(7.8)
    w_in = ax.figure.get_size_inches()[0] * ax.get_position().width
    chars = max(16, int(w_in * 72.0 / (size * 0.52)))
    lines = textwrap.wrap(note, chars) or [""]
    pad = 8 + size * 1.25 * len(lines)
    ax.set_title(title, color=INK, fontsize=_fs(10), loc="left", pad=pad)
    ax.annotate("\n".join(lines), xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 5),
                textcoords="offset points", color=MUTED, fontsize=size, ha="left",
                va="bottom", annotation_clip=False, linespacing=1.25)


# two lines of title, in inches, reserved above every 3-D panel
PANEL_TITLE_BAND = 0.60


def _title_band(fig):
    return PANEL_TITLE_BAND / fig.get_figheight()


def _panel_title_3d(ax, title, note):
    """Titles for a 3-D panel, in the band _formation_axes leaves free above it.

    The note is wrapped to the panel, since a 3-D panel is narrow and an unwrapped note
    runs across the panel beside it.
    """
    fig = ax.figure
    x0, y0, w, h = ax._panel_rect
    band = _title_band(fig)
    size = _fs(8.2)
    chars = max(14, int(w * fig.get_figwidth() * 72.0 / (size * 0.54)))
    fig.text(x0 + 0.02 * w, y0 + h, title, color=INK, fontsize=_fs(10),
             ha="left", va="top")
    fig.text(x0 + 0.02 * w, y0 + h - band * 0.42, "\n".join(textwrap.wrap(note, chars)),
             color=MUTED, fontsize=size, ha="left", va="top", linespacing=1.25)


def _style_axes(ax, log=False, logx=False):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=_fs(8), length=3.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
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
                    textcoords="offset points", color=INK_2, fontsize=_fs(7.5),
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
                    fontsize=_fs(7.5), zorder=20, bbox=_label_box())

    _style_axes(ax, log=log)
    # a non-rigid graph has a zero eigenvalue, which drags a log axis to 1e-300;
    # floor it just under the smallest value that is actually meaningful
    if log and positives:
        ax.set_ylim(bottom=10 ** np.floor(np.log10(np.percentile(positives, 1))))
    ax.set_xlabel("step", color=INK_2, fontsize=_fs(8.5))
    ax.margins(x=0.14)
    _place_end_labels(ax, end_labels, x_at=max_len - 1)


def _figure(header, kind, methods, notes, panel_h=3.0, width=7.2, panels=(2, 2),
            axes3d=False):
    """A figure laid out as: title block, panels, notes card. Sized to fit all three."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    head_h = 0.0 if _PLAIN else _header_height(header, width)
    card_h = 0.0 if _PLAIN else _card_height(methods, notes, width)
    # the plain variant carries a legend instead of the card, and the 3-D figures place
    # their panels by hand, so the band it needs is reserved here rather than at the end
    legend_h = _legend_height(methods) if _PLAIN else 0.0
    nrows, ncols = panels
    fig_h = nrows * panel_h + head_h + card_h + legend_h
    fig = plt.figure(figsize=(width, fig_h), facecolor=SURFACE)
    # a 3-D caller adds its own projection="3d" panels
    axes = ([] if axes3d
            else [fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)])
    top = 1.0 if _PLAIN else _draw_header(fig, header, kind)
    return fig, axes, (card_h, (card_h + legend_h) / fig_h), top, width


def _legend_height(methods):
    """Inches the plain variant's method key needs, at five entries per row."""
    n = len([m for m in methods]) or 0
    if not n:
        return 0.0
    return 0.34 + 0.24 * int(np.ceil(n / 5))


def _method_legend(fig, methods, width):
    """One key for the whole figure, drawn only in the plain variant.

    The annotated variant carries the method key on its notes card. The plain one drops
    the card, which is what a document wants, and without this the colours would have
    nothing to say which method they are.
    """
    from matplotlib.lines import Line2D
    names = ([m for m in METHOD_ORDER if m in methods]
             + [m for m in methods if m not in METHOD_ORDER])
    if not names:
        return 0.0
    # a figure that already labels its series, in a panel or at the figure level, does
    # not need a second key
    if fig.legends or any(ax.get_legend() is not None for ax in fig.axes):
        return 0.0
    handles = []
    for m in names:
        st = method_style(m)
        handles.append(Line2D([0], [0], color=st["color"], linestyle=st.get("ls", "-"),
                              marker=st.get("marker"), markersize=_fs(4.5),
                              linewidth=1.8, label=m))
    ncol = min(len(handles), 5)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=_fs(8.5), handlelength=2.6, columnspacing=1.8,
               bbox_to_anchor=(0.5, 0.004))


def _finish(fig, card, methods, notes, width, top, run_dir, name, tight=True,
            legend=True):
    card_h, bottom = card
    if _PLAIN and legend:
        _method_legend(fig, methods, width)
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
                                          methods, notes, panel_h=3.0, width=7.2,
                                          panels=(1, 2))
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

        # value labels, so the three alphas never have to be told apart by eye alone.
        # Three bars per method over ten methods is thirty labels, which collide at any
        # readable size, so past a handful of methods the axis carries the values.
        if len(methods) <= 5:
            for vals, errs, stat in drawn:
                k = STAT_ORDER.index(stat)
                for xi, (v, e) in enumerate(zip(vals, errs)):
                    if not np.isfinite(v):
                        continue
                    ax.annotate(panel["fmt"].format(v), xy=(xi + (k - 1) * w, v + e),
                                xytext=(0, 3), textcoords="offset points", ha="center",
                                fontsize=_fs(6.5), color=INK_2,
                                rotation=90 if panel["log"] else 0,
                                va="bottom", zorder=6)

        ax.set_xticks(x)
        _method_ticklabels(ax, methods)
        # which bar is which -- one line under the axis beats a legend or a rotated
        # label per bar, and the shading order is the same in every panel
        ax.set_xlabel("final  ·  best  ·  mean", color=MUTED, fontsize=_fs(7.5),
                      labelpad=6)
        _panel_title(ax, panel["title"], panel["note"])

    return _finish(fig, card, methods, notes, width, top, run_dir, filename, legend=False)


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
        panel_h=3.4, width=7.2, panels=(1, 1))
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
    ax.set_xlabel("bearing noise per measurement (degrees, per-axis sigma)")
    ax.set_ylabel("RMS position error (formation radii)")
    _style_axes(ax, log=True)
    _panel_title(ax, "Measured against predicted",
                 "solid is measured, dashed is predicted from the rigidity matrix")
    ax.legend(frameon=False, fontsize=_fs(8))

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
    # A cube around the formation wastes most of the panel, since a formation is wider
    # than it is tall. The box follows the data instead, and the box aspect follows the
    # same ranges, so one unit is still one unit on every axis.
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    rng = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    lo, hi = lo - pad * rng, hi + pad * rng
    for setter, a, b in ((ax.set_xlim, lo[0], hi[0]), (ax.set_ylim, lo[1], hi[1]),
                         (ax.set_zlim, lo[2], hi[2])):
        setter(a, b)
    ax.set_box_aspect(tuple(hi - lo), zoom=1.12)
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
    ax.tick_params(colors=MUTED, labelsize=_fs(6.5), pad=1, length=2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([])
    # matplotlib leaves a wide margin round a 3-D box; the drawing is the point
    title_h = _title_band(fig)
    ax.set_position([x0 + 0.005 * w, y0 + 0.010, w * 0.99, h - title_h - 0.010])
    for lab, setter in (("x", ax.set_xlabel), ("y", ax.set_ylabel), ("z", ax.set_zlabel)):
        setter(lab, color=MUTED, fontsize=_fs(7), labelpad=-11)
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
        f"{np.degrees(sigma):.1f} degrees of per-axis error. Bigger means the topology pins that "
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
        panel_h=3.1, width=max(3.1 * ncols, 6.8), panels=(nrows, ncols), axes3d=True)
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
                        f"worst agent {worst * sigma / radius:.1%} of the formation")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename, tight=False, legend=False)


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
        panel_h=3.1, width=max(3.1 * ncols, 6.8), panels=(nrows, ncols), axes3d=True)
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
                        f"rigidity eigenvalue {lam:.2e}")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename, tight=False, legend=False)


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
        panel_h=3.1, width=max(3.1 * ncols, 6.8), panels=(nrows, ncols), axes3d=True)
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
        panel_h=3.4, width=7.2, panels=(1, 1))
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
    ax.legend(frameon=False, fontsize=_fs(8), loc="upper left")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


def plot_repair_choice(run_dir, spread, header, filename="repair_choice"):
    """Among repairs of the same size, how much does the choice matter?

    `spread` is what repair.measure collected: one record per damaged instance with
    every enumerated repair's shape error, and what each method's answer scored.
    """
    if not spread:
        return None
    usable = [r for r in spread if len(_finite_positive(r.get("errors"))) >= 2]
    if not usable:
        return None

    methods = _repair_choice_methods(usable)
    notes = [
        "After a formation breaks, several different edge sets of the same minimum "
        "size restore rigidity. Each grey dot is one of them, on one broken network.",
        "Height is the shape error that repair leaves behind, relative to the best "
        "repair available on that network. One is the best available and ten is ten "
        "times worse, for the same number of edges.",
        "The marked points are what each method picked. The height of the column is "
        "the spread across repairs of equal size.",
    ]
    fig, axes, card, top, width = _figure(
        header, "Shape error of each same-size repair", methods, notes,
        panel_h=3.4, width=7.2, panels=(1, 1))
    ax = axes[0]

    rng = np.random.default_rng(0)
    seen = set()
    for k, rec in enumerate(usable):
        errs = _finite_positive(rec["errors"])
        rel = errs / errs.min()
        jitter = k + (rng.random(len(rel)) - 0.5) * 0.55
        ax.scatter(jitter, rel, s=16, color=MUTED, alpha=0.45, linewidths=0, zorder=2)
        for name, err in _repair_choice_picks(rec).items():
            style = method_style(name)
            ax.scatter([k], [err / errs.min()], s=70, marker=style.get("marker", "D"),
                       color=style["color"], edgecolors=SURFACE, linewidths=1.0,
                       zorder=5, label=name if name not in seen else None)
            seen.add(name)

    import matplotlib.ticker as mticker
    ax.axhline(1.0, color=INK_2, linestyle="--", linewidth=1.1, zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("broken network")
    ax.set_ylabel("shape error / best same-size repair")
    ax.set_xticks([])
    _style_axes(ax, log=True)
    # a log axis over one decade labels only the decade, which reads as "no scale"
    # every half-decade over four decades is a wall of labels; the ticks stay, the
    # labels thin out to the decades and one step between them
    ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 3.0)))
    ax.yaxis.set_minor_locator(mticker.LogLocator(subs=(2.0, 5.0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:g}x" if v >= 1 else ""))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    _panel_title(ax, "Every same-size repair, on each broken network",
                 "1x is the best repair available; higher is worse for the same edge count")
    if seen:
        ax.legend(frameon=False, fontsize=_fs(8), loc="upper left")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


def _finite_positive(values):
    """The errors a log axis can draw. A non-rigid answer is infinite, not large."""
    if values is None:
        return np.empty(0)
    a = np.asarray(list(values), dtype=float)
    return a[np.isfinite(a) & (a > 0)]


def _repair_choice_picks(rec):
    """{method: shape error} for one record, in either spread format."""
    picks = rec.get("picks")
    if picks is None:
        # the older single-method record, kept so a run written before repair.py redraws
        picks = {"greedy": rec.get("greedy")}
    return {k: float(v) for k, v in picks.items()
            if v is not None and np.isfinite(float(v)) and float(v) > 0}


def _repair_choice_methods(spread):
    order = {name: i for i, name in enumerate(METHOD_ORDER)}
    names = {name for rec in spread for name in _repair_choice_picks(rec)}
    return sorted(names, key=lambda x: (order.get(x, len(order)), x))


def plot_control_trajectories(run_dir, traces, rows, header,
                              filename="control_trajectories"):
    """Error against time as the controller drives each topology to the goal shape.

    One column per goal shape, so a method is compared against the others on the same
    pair of formations rather than against an average over different ones.
    """
    if not traces:
        return None
    cases = sorted({r["case"] for r in traces})[:3]
    methods = _control_methods(rows or traces)
    panels = (("bearing_rms", "bearing error", "RMS radians per edge"),
              ("pos_err", "position error", "formation radii, gauge removed"))

    tol = next((r.get("tol") for r in (rows or []) if r.get("tol")), None)
    notes = [
        "Columns are goal shapes. Within a column the goal and the start formation are "
        "the same for every method, so the edge set is the only difference.",
        "The top row is what the controller minimises, the RMS bearing error per edge. "
        "The bottom row is the position error against the goal SHAPE, with translation "
        "and scale projected out, since bearing control cannot observe them.",
        "The position error is a small-error metric, so the start of a trajectory is "
        "approximate. A line that stops early left the finite range.",
    ]
    if tol:
        notes.append(f"The dashed line is the tolerance, {tol:g} radians per edge.")

    fig, axes, card, top, width = _figure(
        header, "Error over time", methods, notes,
        panel_h=2.2, width=min(2.4 * max(len(cases), 1), 7.4), panels=(2, len(cases)))

    for r_i, (key, label, unit) in enumerate(panels):
        for c_i, case in enumerate(cases):
            ax = axes[r_i * len(cases) + c_i]
            for method in methods:
                sel = sorted((t for t in traces
                              if t["case"] == case and t["method"] == method),
                             key=lambda t: t["step"])
                xs = [t["t"] for t in sel]
                ys = [t[key] for t in sel]
                keep = [(x, y) for x, y in zip(xs, ys)
                        if y is not None and np.isfinite(y) and y > 0]
                if len(keep) < 2:
                    continue
                style = method_style(method)
                ax.plot([x for x, _ in keep], [y for _, y in keep],
                        color=style["color"], linestyle=style.get("linestyle", "-"),
                        linewidth=1.6, label=method if (r_i == 0 and c_i == 0) else None,
                        zorder=3)
            if key == "bearing_rms" and tol:
                ax.axhline(tol, color=INK_2, linestyle="--", linewidth=1.0, zorder=1)
            ax.set_yscale("log")
            ax.set_xlabel("simulated time (s)")
            ax.set_ylabel(unit)
            _panel_title(ax, f"goal shape {case} - {label}",
                         "lower is closer to the goal")
            _style_axes(ax, log=True)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False, fontsize=_fs(7.5), loc="upper right")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


def plot_control_summary(run_dir, rows, header, filename="control_summary"):
    """What each topology cost: edges, conditioning, and how far it got.

    The edge count and the time to reach the goal are the trade the thesis is about,
    so they sit next to each other rather than in separate figures.
    """
    if not rows:
        return None
    methods = _control_methods(rows)
    tol = next((r.get("tol") for r in rows if r.get("tol")), None)

    notes = [
        "Bars are means over the goal shapes, with one dot per shape behind them.",
        "Time to reach the goal counts only the runs that got there; the label says how "
        "many did, and a method that never did has no bar.",
        "Lambda is the rigidity eigenvalue of the goal formation over that topology. "
        "The linearised error dynamics are xdot = -B'B x, so the slowest mode decays at "
        "lambda and a softer graph takes longer.",
    ]
    fig, axes, card, top, width = _figure(
        header, "What each topology cost", methods, notes,
        panel_h=2.6, width=7.4, panels=(1, 3))

    def series(key, only_reached=False):
        out = {}
        for r in rows:
            if only_reached and not r.get("reached"):
                continue
            v = r.get(key)
            if v is not None and np.isfinite(float(v)):
                out.setdefault(r["method"], []).append(float(v))
        return out

    reached = {}
    for r in rows:
        reached.setdefault(r["method"], []).append(bool(r.get("reached")))

    time_to_goal = {m: [v * next(r["dt"] for r in rows if r["method"] == m)
                        for v in vals]
                    for m, vals in series("steps_to_tol", only_reached=True).items()}

    for ax, (data, label, direction, log) in zip(axes, (
            (series("m"), "edges", "fewer is better", False),
            (series("lam"), "lambda", "higher is stiffer", True),
            (time_to_goal, "time to reach the goal (s)", "lower is better", False))):
        present = [m for m in methods if data.get(m)]
        vals = [float(np.mean(data[m])) for m in present]
        ax.bar(range(len(present)), vals, 0.62, zorder=3,
               color=[method_style(m)["color"] for m in present])
        rng = np.random.default_rng(0)
        for i, m in enumerate(present):
            pts = data[m]
            ax.scatter(i + (rng.random(len(pts)) - 0.5) * 0.28, pts, s=18,
                       color=SURFACE, edgecolors=INK_2, linewidths=0.7, zorder=4)
        if data is time_to_goal:
            for i, m in enumerate(present):
                got, total = sum(reached[m]), len(reached[m])
                ax.text(i, vals[i], f"{got}/{total}", ha="center", va="bottom",
                        fontsize=_fs(7.5), color=INK_2)
        if log and vals:
            ax.set_yscale("log")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=25, ha="right", fontsize=_fs(8))
        _panel_title(ax, label, direction)
        _style_axes(ax, log=log)

    missed = [m for m in methods if not time_to_goal.get(m)]
    if missed and tol:
        notes.append(f"Never reached {tol:g} radians per edge within the step budget: "
                     + ", ".join(missed) + ".")
    return _finish(fig, card, methods, notes, width, top, run_dir, filename, legend=False)


def plot_control_formations(run_dir, traces, rows, header, goals=None,
                            filename="control_formations"):
    """The goal formation, the start formation, and the path each agent took.

    One panel per goal shape and method. `goals` is {case: network}, used to draw the
    domain markers and the edge set; without it the panels still draw the paths.
    """
    if not traces:
        return None
    cases = sorted({t["case"] for t in traces})[:3]
    methods = _control_methods(rows or traces)
    if not methods:
        return None

    by_run = {}
    for t in traces:
        if t.get("positions") is None:
            continue
        by_run.setdefault((t["case"], t["method"]), []).append(t)
    if not by_run:
        return None
    for key in by_run:
        by_run[key].sort(key=lambda t: t["step"])

    from convergence import normalised, rotation_is_gauge

    notes = [
        "Columns are methods and rows are goal shapes. Hollow markers are the goal "
        "formation, the cross is where the run started, and the filled marker is where "
        "it ended. The thin line is the path that agent took.",
        "Arrows are the edge set that method chose. It carries the goal bearings and "
        "builds the controller's B, so the same set is used for both formations.",
        "Every formation is centred and scaled to unit radius. Translation and scale "
        "are unobservable to bearing control, so without that the paths keep whatever "
        "offset the start had and no run would appear to reach the goal.",
    ]
    if goals and any(rotation_is_gauge(g) for g in goals.values()):
        notes.append("A global rotation is unobservable in these domains too, so a "
                     "converged formation can still sit rotated from the goal.")

    ncols, nrows = len(methods), len(cases)
    fig, _, card, top, width = _figure(
        header, "Formations and the paths between them", methods, notes,
        panel_h=min(2.2, max(6.9 / max(ncols, 1), 1.15)),
        width=min(2.2 * ncols, 6.9), panels=(nrows, ncols), axes3d=True)
    band = (top, card[1])

    for r_i, case in enumerate(cases):
        goal_net = (goals or {}).get(case)
        for c_i, method in enumerate(methods):
            index = r_i * ncols + c_i + 1
            trace = by_run.get((case, method))
            if not trace:
                continue
            G = normalised([a.pose.position for a in goal_net.agents]) if goal_net \
                else normalised(trace[-1]["positions"])
            paths = np.stack([normalised(t["positions"]) for t in trace])  # (T, n, 3)
            allp = np.vstack([G, paths.reshape(-1, 3)])

            ax = _formation_axes(fig, nrows, ncols, index, allp, band)
            colour = method_style(method)["color"]
            edges = next((r.get("edges") for r in (rows or [])
                          if r["case"] == case and r["method"] == method
                          and r.get("edges") is not None), None)
            if edges is not None:
                # a dense graph is n(n-1) arrows over ten markers, which hides the
                # thing the panel is about; fade them as the count grows
                m = int(edges.sum())
                _draw_edges_3d(ax, G, edges, color=AXIS,
                               alpha=float(np.clip(1.6 * G.shape[0] / max(m, 1),
                                                   0.10, 0.45)),
                               width=0.8)

            # alpha ramps along the trajectory, so the direction of travel reads
            steps = max(paths.shape[0] - 1, 1)
            for i in range(paths.shape[1]):
                for k in range(steps):
                    ax.plot(paths[k:k + 2, i, 0], paths[k:k + 2, i, 1],
                            paths[k:k + 2, i, 2], color=colour, linewidth=1.3,
                            alpha=0.20 + 0.75 * (k / steps), zorder=4,
                            solid_capstyle="round")
            ax.scatter(paths[0, :, 0], paths[0, :, 1], paths[0, :, 2], s=34,
                       marker="x", color=INK_2, linewidths=1.2, depthshade=False,
                       zorder=5)
            ax.scatter(paths[-1, :, 0], paths[-1, :, 1], paths[-1, :, 2], s=44,
                       marker="o", color=colour, edgecolors=SURFACE, linewidths=0.8,
                       depthshade=False, zorder=7)
            if goal_net is not None:
                _draw_agents(ax, goal_net, G)
            else:
                ax.scatter(G[:, 0], G[:, 1], G[:, 2], s=52, marker="o",
                           facecolors=SURFACE, edgecolors=INK, linewidths=1.1,
                           depthshade=False, zorder=6)

            row = next((r for r in (rows or [])
                        if r["case"] == case and r["method"] == method), None)
            note = (f"{row['m']} edges,  end {row['pos_err_end']:.3f}" if row
                    else f"{paths.shape[0]} samples")
            _panel_title_3d(ax, f"{method}, goal shape {case}", note)

    return _finish(fig, card, methods, notes, width, top, run_dir, filename, tight=False)


def _control_methods(rows):
    order = {name: i for i, name in enumerate(METHOD_ORDER)}
    seen = list(dict.fromkeys(r["method"] for r in rows if r.get("method")))
    return sorted(seen, key=lambda x: (order.get(x, len(order)), x))


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
        panel_h=2.6, width=7.0, panels=(2, 1))

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
            ax.annotate(f"{np.mean(vals):.0f}%", (pos, np.mean(vals)), fontsize=_fs(8.5),
                        color=INK, ha="center", va="bottom",
                        xytext=(0, 11), textcoords="offset points", zorder=6)

        ax.set_xticks(range(len(groups)))
        labels = [f"{k} ({len(v)})" if len(models) == 1 else f"{m} {k} ({len(v)})"
                  for (m, k), v in zip(groups, data)]
        # one model fits flat; past that the groups are twice as many and twice as wide
        if len(groups) > 3:
            ax.set_xticklabels(labels, fontsize=_fs(7.6), rotation=30, ha="right",
                               rotation_mode="anchor")
        else:
            ax.set_xticklabels(labels, fontsize=_fs(8))
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
                    textcoords="offset points", ha="center", fontsize=_fs(7.5), color=INK_2)
        ax.annotate(f"{b:.0f}", xy=(xi + 0.19, b), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=_fs(7.5), color=INK_2)
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
        ax.set_yticklabels(methods, fontsize=_fs(8.5))
        ax.invert_yaxis()
        for yi, v in zip(y, values):
            ax.annotate(fmt.format(v), xy=(v, yi), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=_fs(7.5), color=INK_2)
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
                    fontsize=_fs(8), color=INK_2)
    ax.set_xlabel("rigidity computations per network", color=INK_2, fontsize=_fs(8.5))
    ax.set_ylabel("edges", color=INK_2, fontsize=_fs(8.5))
    _style_axes(ax, logx=True)
    _panel_title(ax, "What the computation buys",
                 "edges against cost - down and to the left is better")

    hbar(axes[3], [t / w for t, w in zip(totals, work)], "Cost per edge change",
         "computations per change applied - lower decides more cheaply", fmt="{:.1f}")

    return _finish(fig, cardspec, methods, notes, width, top, run_dir, filename, legend=False)


# What the comparison figure shows, in the order the panels are drawn.
# (key, title, note, formatter). `pct` panels share an axis to 100.
COMPARISON_PANELS = [
    ("edges_mean",   "Edges used",         "mean over the networks, fewer is better", "{:.2f}"),
    ("at_bound_pct", "Rigid at the bound", "% using exactly the fewest edges the poses allow", "{:.0f}%"),
    ("score_mean",   "Objective score",    "phi, what every method is scored by, higher is better", "{:.1f}"),
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
                                          panel_h=2.9, width=7.2, panels=(2, 2))

    for ax, (key, title, note, fmt) in zip(axes, COMPARISON_PANELS):
        vals = [agg[m].get(key) for m in methods]
        shown = [0.0 if v is None else float(v) for v in vals]
        y = np.arange(len(methods))
        ax.barh(y, shown, color=colors, alpha=0.9, zorder=3, height=0.62)
        hi = max(shown + [1e-9])
        for yi, (v, raw) in enumerate(zip(shown, vals)):
            ax.annotate("-" if raw is None else fmt.format(v), xy=(v, yi), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=_fs(8),
                        color=INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=_fs(8.5))
        ax.invert_yaxis()
        ax.set_xlim(0, 132 if key.endswith("_pct") else hi * 1.42)
        if key == "edges_mean" and bound is not None:
            ax.axvline(bound, color=INK_2, ls="--", lw=1.1, zorder=4)
        _style_axes(ax)
        _panel_title(ax, title, note)

    # the last panel is per network rather than a mean, so it follows however many
    # quantity panels COMPARISON_PANELS carries
    ax = axes[len(COMPARISON_PANELS)]
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
                    textcoords="offset points", fontsize=_fs(8), color=INK_2, va="bottom")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=_fs(8.5))
    ax.invert_yaxis()
    _style_axes(ax)
    _panel_title(ax, "Edges per network", "one dot per network, the tick is the mean")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename, legend=False)


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
                                          panel_h=3.0, width=7.0, panels=(2, 2))

    def bars(ax, key, title, note, fmt):
        vals = [agg[m][f"{key}_gmean"] for m in methods]
        shown = [0.0 if v is None else float(v) for v in vals]
        y = np.arange(len(methods))
        ax.barh(y, shown, color=colors, alpha=0.9, zorder=3, height=0.62)
        for yi, (v, raw, m) in enumerate(zip(shown, vals, methods)):
            mark = "*" if agg[m][f"{key}_n_pos"] < agg[m][f"{key}_n"] else ""
            ax.annotate("-" if raw is None else fmt.format(v) + mark, xy=(v, yi),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=_fs(8), color=INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=_fs(8.5))
        ax.invert_yaxis()
        ax.set_xlim(0, max(shown + [1e-9]) * 1.45)
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
        ax.set_xlabel("bearing noise (degrees, per-axis sigma)", color=INK_2, fontsize=_fs(8.5))
        ax.set_ylabel("RMS position error (formation radii)", color=INK_2, fontsize=_fs(8.5))
        # the only panel where identity is not a y tick label, so it needs the key.
        # Outside the axes, because inside it sits on top of the curves it labels.
        ax.legend(frameon=False, fontsize=_fs(7.6), labelcolor=INK_2, loc="upper left",
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
    ax.set_yticklabels(methods, fontsize=_fs(8.5))
    ax.invert_yaxis()
    ax.set_xlabel("shape error", color=INK_2, fontsize=_fs(8.5))
    _style_axes(ax)
    _panel_title(ax, "Shape error per network",
                 "one dot per network, the tick is the geometric mean")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


def plot_topology(run_dir, instances, rows, header, filename="topology"):
    """The graph each method built, one panel per method and nothing else on it.

    The shells and the softest mode have their own figures. Drawn over the graph they
    hide the edges, which are what this figure exists to show.
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

    # wrapped rather than one row per method: nine methods side by side is a figure four
    # times wider than it is tall, which no page can show at a readable size
    ncols = min(2, len(per))
    nrows = int(np.ceil(len(per) / ncols))
    fig, _, card, top, width = _figure(header, "The graph each method built",
                                       methods, notes, panel_h=3.4,
                                       width=max(3.4 * ncols, 6.8), panels=(nrows, ncols),
                                       axes3d=True)
    band = (top, card[1])

    for k, p in enumerate(per):
        row, colour = p["row"], method_style(p["row"]["method"])["color"]

        ax = _formation_axes(fig, nrows, ncols, k + 1, P, band)
        _draw_edges_3d(ax, P, row["edges"], color=colour, alpha=0.55, width=1.4)
        _draw_agents(ax, net, P)
        bound = row.get("m_req")
        _panel_title_3d(ax, row["method"],
                        f"{row['m']} edges" + (f", the bound is {bound}" if bound else ""))

    return _finish(fig, card, methods, notes, width, top, run_dir, filename, tight=False, legend=False)


def _strip(ax, y, values, colour, jitter=0.34, rng=None):
    """One dot per value on row `y`, with a tick at the median."""
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return
    rng = rng or np.random.default_rng(0)
    ax.scatter(vals, y + (rng.random(len(vals)) - 0.5) * jitter, s=14, color=colour,
               alpha=0.5, linewidths=0, zorder=3)
    # ink, not the series colour: a dense strip buries a tick drawn in its own hue
    ax.scatter([float(np.median(vals))], [y], marker="|", s=240, color=INK,
               linewidths=2.0, zorder=6)


def plot_topology_structure(run_dir, rows, nodes, pairs, header,
                            filename="topology_structure"):
    """What the graphs look like, against what the rigidity theory predicts of them.

    `rows`/`nodes`/`pairs` are topology.measure's three grains. Domain is carried by
    marker shape, never by colour, which belongs to the method.
    """
    import matplotlib
    matplotlib.use("Agg")

    import topology

    agg = topology.summarize(rows)
    if not agg:
        return None
    methods = [v["method"] for v in agg]
    colors = [method_style(m)["color"] for m in methods]

    notes = [
        "Reciprocity is the fraction of edges whose reverse also exists. The reverse "
        "adds no rank at all between two frameless agents and 1 or 2 once either end "
        "carries a frame, so the two are drawn apart. The tick is the level a uniform "
        "random directed graph of the same size reaches by accident.",
        "The out-degree floor r_i is the bearings an agent must measure for its own "
        "attitude to be observable: 0 with no frame, 1 turning about a fixed axis, 2 "
        "turning freely. Marker shape is the domain, colour the method.",
        "c1 is the class of ordered pairs with a planar agent at each end, where an "
        "edge carries rank 1 against 2 elsewhere. Spread is the smallest singular value "
        "of the bearings one agent measures, zero when they are degenerate. Edge "
        "lengths are in formation radii.",
    ]
    fig, axes, card, top, width = _figure(header, "Topology properties", methods, notes,
                                          panel_h=3.0, width=6.9, panels=(3, 2))
    rng = np.random.default_rng(0)

    # ---- 1. reciprocity, split by whether either end carries a frame
    ax = axes[0]
    y = np.arange(len(methods))
    for offset, key, alpha in ((-0.17, "recip_free", 0.95), (0.17, "recip_framed", 0.45)):
        ax.barh(y + offset, [v[key] or 0.0 for v in agg], color=colors, alpha=alpha,
                zorder=3, height=0.30)
    for yi, v in enumerate(agg):
        if v["recip_random"] is not None:
            ax.plot([v["recip_random"]] * 2, [yi - 0.36, yi + 0.36], color=INK,
                    lw=1.6, zorder=6)
        for offset, key in ((-0.17, "recip_free"), (0.17, "recip_framed")):
            ax.annotate(f"{(v[key] or 0):.2f}", xy=(v[key] or 0.0, yi + offset),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=_fs(7.5), color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=_fs(8.5))
    ax.invert_yaxis()
    ax.set_xlim(0, max([0.05] + [(v[k] or 0) for v in agg
                                 for k in ("recip_free", "recip_framed", "recip_random")])
                * 1.35)
    _style_axes(ax)
    _panel_title(ax, "Reciprocity",
                 "solid: both ends frameless, faded: a frame at one end, tick: random")

    # ---- 2. mean out-degree by domain, against the floor
    ax = axes[1]
    dom = topology.by_domain(nodes)
    present = [d for d in DOMAIN_ORDER if any(r["domain"] == d for r in dom)]
    xs = {d: k for k, d in enumerate(present)}
    for m, colour in zip(methods, colors):
        for d in present:
            rec = next((r for r in dom if r["method"] == m and r["domain"] == d), None)
            if rec is None:
                continue
            ax.scatter([xs[d]], [rec["out_deg_mean"]], marker=DOMAIN_MARKER[d][0],
                       s=52, color=colour, zorder=4)
            ax.scatter([xs[d]], [rec["out_deg_min"]], marker=DOMAIN_MARKER[d][0],
                       s=52, facecolors="none", edgecolors=colour, linewidths=1.2,
                       zorder=4)
    if present:
        ax.step([xs[d] for d in present] + [len(present) - 0.5],
                [topology.R_MIN[d] for d in present] + [topology.R_MIN[present[-1]]],
                where="mid", color=INK, ls="--", lw=1.2, zorder=2)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, fontsize=_fs(8))
    ax.set_ylabel("out-degree", color=INK_2, fontsize=_fs(8.5))
    _style_axes(ax)
    _panel_title(ax, "Out-degree by domain",
                 "filled is the mean, open the minimum; dashed is the floor r_i")

    # ---- 3. out-degree above the floor, per agent
    ax = axes[2]
    for yi, (m, colour) in enumerate(zip(methods, colors)):
        _strip(ax, yi, [r["out_deg"] - r["r_i"] for r in nodes if r["method"] == m],
               colour, rng=rng)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=_fs(8.5))
    ax.invert_yaxis()
    ax.set_xlabel("out-degree minus r_i", color=INK_2, fontsize=_fs(8.5))
    _style_axes(ax)
    _panel_title(ax, "Out-degree above the floor",
                 "one dot per agent, tick at the median; 0 is the floor exactly")

    # ---- 4. edge share by rank class
    ax = axes[3]
    for m, colour in zip(methods, colors):
        v = next(r for r in agg if r["method"] == m)
        have = [(k, v[f"share_c{k}"]) for k in (1, 2) if v[f"share_c{k}"] is not None]
        if not have:
            continue
        ax.plot([k for k, _ in have], [s for _, s in have], color=colour, lw=1.2,
                marker="o", ms=5, zorder=4)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["c_k = 1\nboth ends planar", "c_k = 2\nevery other pair"],
                       fontsize=_fs(8))
    ax.set_xlim(0.7, 2.3)
    ax.set_ylabel("edges / pairs available", color=INK_2, fontsize=_fs(8.5))
    _style_axes(ax)
    _panel_title(ax, "Edge share by rank class",
                 "how much of each class a method uses, one line per method")

    # ---- 5. mean edge length per network
    ax = axes[4]
    for yi, (m, colour) in enumerate(zip(methods, colors)):
        _strip(ax, yi, [r["len_mean"] for r in rows if r["method"] == m], colour, rng=rng)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=_fs(8.5))
    ax.invert_yaxis()
    ax.set_xlabel("mean edge length, formation radii", color=INK_2, fontsize=_fs(8.5))
    _style_axes(ax)
    _panel_title(ax, "Edge length",
                 "one dot per network, tick at the median; compare at a similar m")

    # ---- 6. bearing spread per agent
    ax = axes[5]
    for yi, (m, colour) in enumerate(zip(methods, colors)):
        _strip(ax, yi, [r["spread"] for r in nodes if r["method"] == m], colour, rng=rng)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=_fs(8.5))
    ax.invert_yaxis()
    ax.set_xlabel("smallest singular value of the outgoing bearings", color=INK_2,
                  fontsize=_fs(8.5))
    _style_axes(ax)
    _panel_title(ax, "Bearing spread",
                 "one dot per agent measuring two or more; 0 is degenerate")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


def plot_budget(run_dir, rows, header, filename="budget"):
    """What each method's answer is worth when it is only allowed so many moves.

    `rows` is one record per (episode, method, budget), as budget.measure returns them.
    Every panel shares the budget axis except the last two, which say where the answer
    was found.
    """
    import matplotlib
    matplotlib.use("Agg")

    import budget as budget_lib

    summary = budget_lib.summarize(rows)
    if not summary:
        return None
    methods = list(dict.fromkeys(s["method"] for s in summary))
    budgets = sorted({s["budget"] for s in summary})
    by = {(s["method"], s["budget"]): s for s in summary}

    notes = [
        "The budget is graph modifications the method was allowed. For a policy that is "
        "a step of the episode; for greedy it is an applied edit, there being no greedy "
        "step that does nothing.",
        "Every point is the best graph visited within the budget, which is the rule every "
        "rollout in this work is scored by. The observation carries no step counter, so "
        "one rollout per network settles every budget exactly rather than approximately.",
        "At answer is the share of networks where the budget already holds the graph the "
        "full run reports, so it says how much budget is actually needed rather than how "
        "good the answer is. The last panel is the distribution behind it.",
    ]
    fig, axes, card, top, width = _figure(header, "What a budget buys", methods, notes,
                                          panel_h=2.9, width=7.2, panels=(2, 2))

    def series(method, key):
        xs, ys = [], []
        for b in budgets:
            rec = by.get((method, b))
            if rec is not None and rec[key] is not None:
                xs.append(b)
                ys.append(rec[key])
        return xs, ys

    # the two stages come first, because what a budget buys is a phase change rather
    # than a slope: reach rigidity, then work down toward the bound
    panels = [
        ("at_bound_pct", "Rigid at the proven bound",
         "percent on exactly m_req edges", False),
        ("edges_mean", "Edges in the best graph", "mean over the networks", False),
        ("final_pct", "Already at the full-budget answer",
         "percent of networks; where this saturates is how much budget is needed", False),
    ]
    for ax, (key, title, note, log) in zip(axes, panels):
        for m in methods:
            st = method_style(m)
            xs, ys = series(m, key)
            if not xs:
                continue
            ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=1.7, marker=st["marker"],
                    ms=4.5, zorder=st.get("z", 3))
        ax.set_xlabel("moves allowed", color=INK_2, fontsize=_fs(8.5))
        if key.endswith("_pct"):
            ax.set_ylim(-4, 104)
        _style_axes(ax, log=log)
        _panel_title(ax, title, note)

    # ---- where the answer was actually found: the distribution behind the panel above
    # the distribution panel follows however many curve panels there are
    ax = axes[len(panels)]
    rng = np.random.default_rng(0)
    steps = budget_lib.best_steps(rows)
    for yi, m in enumerate(methods):
        vals = [v for v in steps.get(m, []) if v is not None]
        if not vals:
            continue
        colour = method_style(m)["color"]
        ax.scatter(vals, yi + (rng.random(len(vals)) - 0.5) * 0.34, s=20, color=colour,
                   alpha=0.55, linewidths=0, zorder=3)
        ax.scatter([float(np.median(vals))], [yi], marker="|", s=240, color=INK,
                   linewidths=2.0, zorder=6)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=_fs(8.5))
    ax.invert_yaxis()
    ax.set_xlabel("move the best graph was found at", color=INK_2, fontsize=_fs(8.5))
    _style_axes(ax)
    _panel_title(ax, "Where the answer was found",
                 "one dot per network, tick at the median")

    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


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
                                          width=7.2, panels=(1, 2))

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
        ax.set_yticklabels(channels, fontsize=_fs(8.5))
        ax.set_xlabel({"d_phi": "phi lost", "flip_pct": "% of steps"}[key],
                      color=INK_2, fontsize=_fs(8.5))
        _style_axes(ax)
        _panel_title(ax, title, note)
    axes[0].legend(frameon=False, fontsize=_fs(8.5), labelcolor=INK_2, loc="lower right",
                   title="mode", title_fontsize=_fs(8.5))

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
                                          panel_h=2.8, width=7.0, panels=(2, 2))
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
        ax.set_xlabel("environment steps (thousands)", color=INK_2, fontsize=_fs(8.5))
        _style_axes(ax, log=log)
        _panel_title(ax, title, note if drawn else "not logged by any run")
    # one legend for the figure, not one per panel
    handles, names = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, frameon=False, fontsize=_fs(9), labelcolor=INK_2,
                   loc="upper right", ncol=min(len(names), 4),
                   bbox_to_anchor=(0.99, 1.0 if _PLAIN else 0.985))
    return _finish(fig, card, labels, notes, width, top, run_dir, filename)


# `m_over_mreq` is drawn from a baseline of one rather than of zero: every value sits
# between 1.00 and about 1.10, so bars from zero put the whole signal in the top few
# percent of the panel and show nothing. From the bound, the bar length is the excess.
GENERALISATION_PANELS = [
    ("at_bound_pct", "Rigid at the bound", "% using the fewest edges the poses allow",
     (0, 118)),
    ("m_over_mreq", "Edges over the bound", "one is the bound, lower is better", None),
    ("rigid_pct", "Made rigid", "% of instances left rigid", (0, 118)),
]

# panels whose bars grow from a reference value instead of from zero
PANEL_BASELINE = {"m_over_mreq": 1.0}


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
        "the bound is the edge count divided by m_req, averaged per instance, so one is "
        "the bound and the quantity is comparable across sets whose bounds differ. Those "
        "bars are drawn from the bound rather than from zero, so their length is the "
        "excess over it.",
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
    npanels = len(GENERALISATION_PANELS)
    grid = (int(np.ceil(npanels / 2)), 2) if npanels > 1 else (1, 1)
    fig, axes, card, top, width = _figure(header, "Across instance sets", series, notes,
                                          panel_h=2.8,
                                          width=min(max(0.6 * len(benches) + 4.0, 7.0), 9.0),
                                          panels=grid)
    for spare in axes[npanels:]:          # an odd panel count leaves an empty cell
        fig.delaxes(spare)
    axes = axes[:npanels]

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
            base = PANEL_BASELINE.get(key, 0.0)
            ax.bar(pos, [v - base for v in vals], bottom=base, width=w * 0.9,
                   color=method_style(name)["color"], alpha=0.9, zorder=3,
                   label=name if ax is axes[0] else None)
        if key in PANEL_BASELINE:
            base = PANEL_BASELINE[key]
            ax.axhline(base, color=INK_2, ls="--", lw=1.1, zorder=4)
            vals = [float(r[key]) for r in gen_rows if r.get(key) is not None]
            if vals:
                hi = max(vals)
                ax.set_ylim(base - (hi - base) * 0.08, base + (hi - base) * 1.15)
        ax.set_xticks(x)
        names = [b.replace("bench_", "") for b in benches]
        if len(benches) > 5:
            ax.set_xticklabels(names, fontsize=_fs(7.6), rotation=35, ha="right",
                               rotation_mode="anchor")
        else:
            ax.set_xticklabels(names, fontsize=_fs(8.5))
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
                                rotation=90, fontsize=_fs(6.2), color=MUTED, ha="center",
                                va="bottom", xytext=(0, 2), textcoords="offset points")
        if ylim:
            ax.set_ylim(*ylim)
        _style_axes(ax)
        _panel_title(ax, title, note)

    handles, names = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, frameon=False, fontsize=_fs(9), labelcolor=INK_2,
                   loc="upper right", ncol=min(len(names), 4),
                   bbox_to_anchor=(0.99, 1.0 if _PLAIN else 0.985))
    return _finish(fig, card, series, notes, width, top, run_dir, filename)


def _scale_series(rows, only=None):
    """[(method, [row, ...] ordered by n)], methods measured at more than one size.

    `only` restricts and orders the series. Past about eight lines the palette starts
    repeating hues and the figure stops being readable, so the callers draw two figures
    over subsets rather than one over everything.
    """
    by = {}
    for r in rows:
        if r.get("n") is None or (only is not None and r["method"] not in only):
            continue
        by.setdefault(r["method"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["n"])
    if only is not None:
        order = [m for m in only if m in by]
    else:
        order = ([m for m in METHOD_ORDER if m in by]
                 + [m for m in by if m not in METHOD_ORDER])
    return [(m, by[m]) for m in order if len(by[m]) > 1]


def _scale_draw(ax, series, key, logy=False, sizes=None):
    """One line per method over the sizes it was measured at."""
    from matplotlib.ticker import ScalarFormatter
    for name, pts in series:
        xs = [p["n"] for p in pts if p.get(key) is not None]
        ys = [float(p[key]) for p in pts if p.get(key) is not None]
        if not xs:
            continue
        st = method_style(name)
        ax.plot(xs, ys, color=st["color"], ls=st.get("ls", "-"),
                marker=st.get("marker", "o"), ms=4.0, lw=1.5, zorder=st.get("z", 5),
                label=name)
    ax.set_xscale("log", base=2)
    if sizes:
        ax.set_xticks(sizes)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("agents in the formation")
    if logy:
        ax.set_yscale("log")
    _style_axes(ax)


def _scale_legend(fig, ax, series):
    """The annotated variant's own key. The plain one gets the standard bottom legend,
    whose band is reserved for it, so drawing one here would land on the panel titles."""
    if _PLAIN:
        return
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=_fs(8.0), labelcolor=INK_2,
                   loc="upper right", ncol=min(len(labels), 5),
                   bbox_to_anchor=(0.99, 0.985))


def plot_scale(run_dir, rows, header, filename="scale", only=None):
    """What each method's answer is worth as the formation grows.

    The edge count is read as m/m_req, since the bound moves with the size and the
    domain mix and a raw count is not comparable between two sizes. Every quantity here
    is a property of the graph, so a policy trained at one conditioning weight and one
    trained at another sit on the same axes without either being scored by the other's
    objective.
    """
    import matplotlib
    matplotlib.use("Agg")

    series = _scale_series(rows, only)
    if not series:
        return None
    names = [m for m, _ in series]
    sizes = sorted({p["n"] for _, pts in series for p in pts})

    notes = [
        "Every point is one evaluation run on a frozen instance set at that size. Each "
        "set is drawn separately, so a difference between two sizes carries the "
        "difference between two instance distributions as well as the size itself.",
        "m/m_req is the edge count over the proven lower bound for those poses, averaged "
        "per instance. One is the bound and no method can go below it.",
        "The shape error is the predicted RMS state error per radian of bearing noise, a "
        "geometric mean over the instances. Lower is better.",
        "Neither edges nor error is the objective on its own, and the weight that trades "
        "them is what separates the policies, so the two are read side by side rather "
        "than combined into one number here.",
        "A line that stops short means the method was not run past that size.",
    ]
    fig, axes, card, top, width = _figure(header, "Against formation size", names, notes,
                                          panel_h=2.9, width=8.4, panels=(2, 2))

    _scale_draw(axes[0], series, "m_over_mreq", sizes=sizes)
    axes[0].axhline(1.0, color=INK_2, ls=":", lw=1.1, zorder=2)
    axes[0].set_ylabel("edges / $m_{req}$")
    _panel_title(axes[0], "Edges against the bound",
                 "one is the proven bound, lower is better")

    _scale_draw(axes[1], series, "shape_err", logy=True, sizes=sizes)
    axes[1].set_ylabel("shape error")
    _panel_title(axes[1], "Estimation error",
                 "per radian of bearing noise, lower is better")

    _scale_draw(axes[2], series, "rigid_pct", sizes=sizes)
    axes[2].set_ylim(-3, 103)
    axes[2].set_ylabel("rigid, %")
    _panel_title(axes[2], "Instances made rigid",
                 "% of instances the method left infinitesimally bearing rigid")

    _scale_draw(axes[3], series, "at_bound_pct", sizes=sizes)
    axes[3].set_ylim(-3, 103)
    axes[3].set_ylabel("at the bound, %")
    _panel_title(axes[3], "Instances solved at the bound",
                 "% rigid on exactly m_req edges")

    _scale_legend(fig, axes[0], series)
    return _finish(fig, card, names, notes, width, top, run_dir, filename)


def plot_scale_cost(run_dir, rows, header, filename="scale_cost", only=None):
    """What each method spends to get its answer, as the formation grows.

    Counted calls to the named linear algebra primitives, not wall time: a count is a
    property of the method and wall time is a property of this machine under this load.
    The wall time is carried in the last panel so the two can be read against each other.
    """
    import matplotlib
    matplotlib.use("Agg")

    series = _scale_series(rows, only)
    if not series:
        return None
    names = [m for m, _ in series]
    sizes = sorted({p["n"] for _, pts in series for p in pts})

    notes = [
        "A count weighs a full decomposition and a rank of one row block equally, which "
        "is what the wall time panel is there to correct.",
        "Cost to the best graph is derived: the full-horizon count scaled by the "
        "fraction of the horizon the method took to reach its answer. A hill climb stops "
        "on its own so the two coincide for one, while a search policy with no stop "
        "action spends the rest of its horizon whatever it found.",
        "The exponent beside each method in the first panel is the slope of a straight "
        "line fitted to the count against the size on log axes.",
        "Wall time is a geometric mean per network with the linear algebra pinned to one "
        "thread, and it depends on the machine in a way the counts do not.",
    ]
    fig, axes, card, top, width = _figure(header, "Cost against formation size", names,
                                          notes, panel_h=2.9, width=8.4, panels=(2, 2))

    _scale_draw(axes[0], series, "calls", logy=True, sizes=sizes)
    axes[0].set_ylabel("primitive calls per network")
    _panel_title(axes[0], "Cost of one network", "counted calls, log axes")
    # the fitted exponent, written beside the last point of each line. Two lines can end
    # close together, so the labels are laid out top down in the order the lines finish
    fits = []
    for name, pts in series:
        xs = [p["n"] for p in pts if p.get("calls")]
        ys = [float(p["calls"]) for p in pts if p.get("calls")]
        if len(xs) < 3:
            continue
        fits.append((ys[-1], xs[-1], name,
                     float(np.polyfit(np.log(xs), np.log(ys), 1)[0])))
    fits.sort(reverse=True)
    lo, hi = axes[0].get_ylim()
    span = np.log10(hi) - np.log10(lo)
    placed = []
    # methods that scale alike get one label between them: seven lines reading n^1.0 on
    # top of each other say nothing the single label does not
    shown = set()
    shared = {k for k in (f"{sl:.1f}" for *_, sl in fits)
              if sum(1 for *_, s2 in fits if f"{s2:.1f}" == k) > 1}
    for y, x, name, slope in fits:
        if f"{slope:.1f}" in shown:
            continue
        shown.add(f"{slope:.1f}")
        # one label standing for several methods is not any one method's colour
        ink = MUTED if f"{slope:.1f}" in shared else method_style(name)["color"]
        frac = (np.log10(y) - np.log10(lo)) / span if span > 0 else 0.5
        while any(abs(frac - q) < 0.055 for q in placed):
            frac -= 0.055
        placed.append(frac)
        axes[0].annotate(f"$n^{{{slope:.1f}}}$", xy=(1.005, frac),
                         xycoords="axes fraction", fontsize=_fs(6.6),
                         color=ink, va="center", ha="left", annotation_clip=False)

    _scale_draw(axes[1], series, "calls_to_best", logy=True, sizes=sizes)
    axes[1].set_ylabel("primitive calls to the best graph")
    _panel_title(axes[1], "Cost of the answer",
                 "counted calls up to the graph the method returns")

    _scale_draw(axes[2], series, "calls_per_edit", logy=True, sizes=sizes)
    axes[2].set_ylabel("primitive calls per applied edit")
    _panel_title(axes[2], "Cost of one decision",
                 "what the method spends before it changes one edge")

    _scale_draw(axes[3], series, "ms", logy=True, sizes=sizes)
    axes[3].set_ylabel("wall time per network, ms")
    _panel_title(axes[3], "Wall time", "this machine, one thread, lower is better")

    _scale_legend(fig, axes[0], series)
    return _finish(fig, card, names, notes, width, top, run_dir, filename)


KAPPA_SERIES = ("learned", "greedy", "anneal", "spectral")


def plot_kappa(run_dir, rows, header, filename="kappa"):
    """What the conditioning weight buys, with every method optimizing the same weight.

    Each point is one evaluation run in the environment the policy on it was trained for,
    so the classical methods hill climb the objective the policy was trained on rather
    than one fixed objective none of them shares. Both axes are properties of the graph,
    not of any reward, so the two fronts compare directly.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.ticker import ScalarFormatter

    by = {}
    for r in rows:
        if r.get("kappa") is None or r.get("edges") is None:
            continue
        by.setdefault(r["method"], []).append(r)
    series = [(m, sorted(by[m], key=lambda r: r["kappa"]))
              for m in KAPPA_SERIES if len(by.get(m, [])) > 1]
    if not series:
        return None
    names = [m for m, _ in series]

    notes = [
        "Each point is one run in the environment the policy on it was trained for, so "
        "the classical methods are hill climbing that same objective and not a fixed one.",
        "Edges and shape error are properties of the graph, so a policy trained at one "
        "weight and one trained at another sit on the same axes without either being "
        "scored by the other's objective.",
        "The shape error is the predicted RMS state error per radian of bearing noise, a "
        "geometric mean over the instances.",
        "The weights are spaced evenly and labelled by their values rather than placed on "
        "a logarithmic axis, which has no room for a weight of zero.",
        "In the last panel only one series carries the weight labels, since both series "
        "are drawn at the same four weights.",
    ]
    fig, axes, card, top, width = _figure(header, "Against the conditioning weight", names,
                                          notes, panel_h=3.0, width=9.4, panels=(1, 3))

    # kappa = 0 has no place on a log axis, so the weights are drawn evenly spaced and
    # labelled by their values
    ks = sorted({r["kappa"] for _, pts in series for r in pts})
    pos = {k: i for i, k in enumerate(ks)}

    def draw(ax, key, logy=False):
        for name, pts in series:
            st = method_style(name)
            xs = [pos[r["kappa"]] for r in pts if r.get(key) is not None]
            ys = [float(r[key]) for r in pts if r.get(key) is not None]
            if not xs:
                continue
            ax.plot(xs, ys, color=st["color"], ls=st.get("ls", "-"),
                    marker=st.get("marker", "o"), ms=5.0, lw=1.7,
                    zorder=st.get("z", 5), label=name)
        ax.set_xticks(list(pos.values()))
        ax.set_xticklabels([("%g" % k) for k in ks])
        ax.set_xlabel(r"conditioning weight $\kappa$")
        if logy:
            ax.set_yscale("log")
        _style_axes(ax)

    draw(axes[0], "edges")
    axes[0].set_ylabel("edges")
    _panel_title(axes[0], "Measurements kept", "fewer is better")

    draw(axes[1], "shape_err", logy=True)
    axes[1].set_ylabel("shape error")
    _panel_title(axes[1], "Estimation error", "per radian of bearing noise, lower is better")

    ax = axes[2]
    for k, (name, pts) in enumerate(series):
        st = method_style(name)
        ok = [r for r in pts if r.get("edges") and r.get("shape_err")]
        if not ok:
            continue
        ax.plot([r["edges"] for r in ok], [r["shape_err"] for r in ok], color=st["color"],
                ls=st.get("ls", "-"), marker=st.get("marker", "o"), ms=5.0, lw=1.7,
                zorder=st.get("z", 5))
        # every series sits at the same weights, so labelling more than one writes each
        # value twice on top of itself
        if k:
            continue
        for r in ok:
            ax.annotate("%g" % r["kappa"], xy=(r["edges"], r["shape_err"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=_fs(6.8),
                        color=MUTED)
    ax.set_yscale("log")
    ax.margins(x=0.12)
    ax.set_xlabel("edges")
    ax.set_ylabel("shape error")
    _style_axes(ax)
    _panel_title(ax, "What one buys with the other", "down and left is better")

    _scale_legend(fig, axes[0], series)
    return _finish(fig, card, names, notes, width, top, run_dir, filename)


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
                 fontsize=_fs(12.5), ha="left", va="top")

    y = 1.0 - (0.62 if not _PLAIN else 0.22) / fig_h
    label_w = max((len(k) for _, r in blocks for k, _ in r), default=10)
    for title, rows in blocks:
        fig.text(0.012, y, title.upper(), color=INK_2, fontsize=_fs(8.0), ha="left",
                 va="top", family="monospace")
        y -= line_h * 1.1 / fig_h
        for key, value in rows:
            # one line per field, wrapped rather than truncated: these are long
            # generated names and a clipped one is worse than a wrapped one
            for i, part in enumerate(_wrap(str(value), width * 0.72, 9.0)):
                fig.text(0.020, y, key if i == 0 else "", color=MUTED, fontsize=_fs(9.0),
                         ha="left", va="top", family="monospace")
                fig.text(0.020 + 0.0062 * (label_w + 2), y, part, color=INK,
                         fontsize=_fs(9.0), ha="left", va="top", family="monospace")
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


def plot_table(run_dir, rows, header, filename="table", width=8.0):
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
        fig.text(x, y_top - 0.16 / fig_h, c["title"], color=INK, fontsize=_fs(9.5),
                 ha=ha, va="center")
        if c["unit"]:
            fig.text(x, y_top - 0.34 / fig_h, c["unit"], color=MUTED, fontsize=_fs(7.0),
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
        fig.text(mx0 + 0.038, yc, m, color=INK_2 if ref else INK, fontsize=_fs(9), va="center")
        for c, (cx0, cx1) in list(zip(cols, edges))[1:]:
            fig.text(cx1 - 0.006, yc, cell(m, c["key"], v), color=INK_2 if ref else INK,
                     fontsize=_fs(9), ha="right", va="center", family="monospace")

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


# ---------------------------------------------------------------------------- repair

def _repair_series(rows, key):
    """{(event, method): [values]} over rows that carry `key`.

    A policy reports three stopping rules from one rollout, so take the one the event
    calls for -- repair.headline_rule -- and `final` for the classical methods, which
    stop when they are done. Averaging the three would be meaningless.
    """
    from repair import _headline
    out = {}
    for r in _headline(rows):
        if r.get(key) is None:
            continue
        out.setdefault((r["event"], r["method"]), []).append(r[key])
    return out


def _repair_methods(rows):
    seen = [r["method"] for r in rows if r.get("method")]
    return list(dict.fromkeys(seen))


def plot_repair_percentile(run_dir, rows, header, filename="repair_percentile"):
    """Where each method's repair sits among every repair of the same size.

    The headline. 100 is the best repair available, 50 is no better than picking one
    at random, and the classical marginal-gain method was measured at 39-55% in
    THEORY 19.4's own convention -- which is the complement of this one.
    """
    series = _repair_series(rows, "pctile")
    if not series:
        return None
    events = list(dict.fromkeys(e for e, _ in series))
    # only methods that actually appear: one that overspent is excluded from the
    # population by design, and a legend entry with no dots reads as a missing series
    scored = {m for _, m in series}
    methods = [m for m in _repair_methods(rows) if m in scored]
    # an event where almost nothing spent the minimum draws an empty panel, which takes
    # the same room as a full one and says less than the sentence that reports it
    events = [e for e in events
              if sum(1 for m in methods if (e, m) in series) >= max(2, len(methods) // 2)]
    if not events:
        return None

    notes = [
        "After the event, several different edge sets of the same minimum size restore "
        "rigidity, and they differ by a lot in the error they leave. Each dot is one "
        "network; height is where that method's answer sits among all of them.",
        "100 is the best repair available. 50 is no better than choosing one at random, "
        "which is the dashed line.",
        "A method that spent more than the minimum number of edges is not in the "
        "population and is not plotted: more edges almost always means lower error.",
    ]
    rows_, cols = (len(events) + 1) // 2, min(2, len(events))
    fig, axes, card, top, width = _figure(
        header, "Repair quality, as a percentile among equal-size repairs",
        methods, notes, panel_h=3.4, width=7.2, panels=(rows_, cols))

    rng = np.random.default_rng(0)
    for ax, event in zip(axes, events):
        present = [m for m in methods if (event, m) in series]
        for x, method in enumerate(present):
            vals = np.asarray(series[(event, method)], dtype=float)
            ax.scatter(x + (rng.random(len(vals)) - 0.5) * 0.35, vals, s=26,
                       color=method_style(method)["color"], alpha=0.55,
                       linewidths=0, zorder=3)
            ax.scatter([x], [vals.mean()], s=90, marker="_", linewidths=2.4,
                       color=method_style(method)["color"], zorder=4)
        ax.axhline(50, color=INK_2, linestyle="--", linewidth=1.0, zorder=1)
        ax.axhline(100, color=MUTED, linestyle=":", linewidth=1.0, zorder=1)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=20, ha="right", fontsize=_fs(8))
        ax.set_ylim(-4, 108)
        _panel_title(ax, event, "higher is a better repair; 50 is chance")
        _style_axes(ax)
    for ax in axes[len(events):]:
        ax.axis("off")
    return _finish(fig, card, methods, notes, width, top, run_dir, filename, legend=False)


def plot_repair_locality(run_dir, rows, header, filename="repair_locality"):
    """How much of the network each method disturbed to put it right."""
    churn = _repair_series(rows, "churn")
    if not churn:
        return None
    collateral = _repair_series(rows, "collateral")
    events = list(dict.fromkeys(e for e, _ in churn))
    methods = _repair_methods(rows)

    notes = [
        "A repair should be local. Churn is how many edges differ between the damaged "
        "network and the answer; collateral is the part of that which was untouched by "
        "the event and got removed anyway.",
        "The `none` event is the control: nothing happened, so anything above zero is a "
        "method that cannot leave a converged network alone.",
        "Rebuilding from the empty graph reaches a good answer with total churn, which "
        "is the number that says repair is a different problem from construction.",
    ]
    fig, axes, card, top, width = _figure(
        header, "How much the network was disturbed", methods, notes,
        panel_h=3.4, width=7.2, panels=(1, 2))

    for ax, (series, label) in zip(axes, ((churn, "edges changed"),
                                          (collateral, "untouched edges removed"))):
        width_ = 0.8 / max(len(methods), 1)
        for m_i, method in enumerate(methods):
            vals = [np.mean(series.get((e, method), [np.nan])) for e in events]
            ax.bar([x + m_i * width_ for x in range(len(events))], vals, width_ * 0.9,
                   color=method_style(method)["color"], label=method, zorder=3)
        ax.set_xticks([x + 0.4 - width_ / 2 for x in range(len(events))])
        ax.set_xticklabels(events, rotation=20, ha="right", fontsize=_fs(8))
        _panel_title(ax, label, "lower is a more local repair")
        _style_axes(ax)
    # headroom first, then one flat row above the bars: a two-column legend inside the
    # axes sits on top of whichever method rebuilt the graph, which is every panel here
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi * 1.28)
    axes[0].legend(fontsize=_fs(7.5), frameon=False, ncol=len(methods),
                   loc="upper center", handlelength=1.2, columnspacing=1.0,
                   handletextpad=0.4)
    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


def plot_repair_count(run_dir, rows, header, filename="repair_count"):
    """How many edges each repair spent, against the proven minimum.

    The count is the part THEORY 19 settles: `repair_edge_count` is a lower bound and
    greedy attains it wherever c_max = 1, so this figure is expected to be flat and
    exists to show that it is. Where it is not flat, the percentile figure beside it
    is not comparable, since a method that spent more edges is not in the population.
    """
    live = [r for r in rows if r.get("method") is not None]
    if not live:
        return None
    from repair import BREAKING, _headline
    events = [e for e in dict.fromkeys(r["event"] for r in live) if e in BREAKING]
    if not events:
        return None
    # `rebuild` is a construction, not a repair: it spends 13-14 edges over the bound
    # by design, which flattens every other method to the axis
    methods = [m for m in _repair_methods(rows) if m != "rebuild"]

    over = _repair_series(rows, "added_over_bound")
    added = _repair_series(rows, "added")
    exact = {}
    for r in _headline(rows):
        if r["event"] in events and r.get("exact_size") is not None:
            exact.setdefault((r["event"], r["method"]), []).append(bool(r["exact_size"]))

    notes = [
        "Only the events that broke rigidity are drawn: where the graph stayed rigid "
        "the minimum repair is zero edges and there is nothing to count.",
        "The bound is repair_edge_count, which is proven, so zero over the bound means "
        "the method used the fewest edges that could possibly have worked.",
        "The right panel is how often a method spent exactly that many. Only those "
        "repairs are in the percentile population, so a low bar there is a warning "
        "that the quality figure is reading a small slice of that method's answers.",
        "`rebuild` is left out. It builds from the empty graph rather than repairing, "
        "so it is 13-14 edges over the bound by construction and flattens the rest.",
    ]
    fig, axes, card, top, width = _figure(
        header, "Edges each repair spent", methods, notes,
        panel_h=3.4, width=7.2, panels=(1, 2))

    for ax, (series, label, direction) in zip(
            axes, ((over, "edges over the proven bound", "0 is optimal"),
                   (exact, "spent exactly the minimum, %", "higher is better"))):
        step = 0.8 / max(len(methods), 1)
        for m_i, method in enumerate(methods):
            vals = [np.mean(series[(e, method)]) * (100 if series is exact else 1)
                    if (e, method) in series else np.nan for e in events]
            ax.bar([x + m_i * step for x in range(len(events))], vals, step * 0.9,
                   color=method_style(method)["color"], label=method, zorder=3)
        if series is over:
            ax.axhline(0.0, color=INK_2, linestyle="--", linewidth=1.0, zorder=1)
        ax.set_xticks([x + 0.4 - step / 2 for x in range(len(events))])
        ax.set_xticklabels(events, rotation=20, ha="right", fontsize=_fs(8))
        _panel_title(ax, label, direction)
        _style_axes(ax)
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi + (hi - lo) * 0.28)
    axes[0].legend(fontsize=_fs(7.5), frameon=False, ncol=len(methods),
                   loc="upper center", handlelength=1.2, columnspacing=1.0,
                   handletextpad=0.4)
    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


def plot_repair_kappa(run_dir, rows, header, kappas, filename="repair_kappa"):
    """Repair quality against the conditioning weight the policy was trained at.

    `kappas` is {method: stiffness_kappa}, from each model's own manifest. THEORY 15.6
    measured that kappa selects among equally sparse graphs rather than buying edges,
    and a repair is exactly that situation, so this is where a benefit would show. It
    refuses to draw below two distinct kappa, since one point is not a trend.
    """
    live = [r for r in rows if r.get("method") is not None]
    known = {m: k for m, k in (kappas or {}).items() if k is not None}
    if not live or len({round(float(k), 6) for k in known.values()}) < 2:
        return None
    from repair import BREAKING
    events = [e for e in dict.fromkeys(r["event"] for r in live) if e in BREAKING]
    if not events:
        return None

    series = _repair_series(rows, "pctile")
    notes = [
        "One line per event. The x axis is the stiffness_kappa each policy was TRAINED "
        "at, read from its manifest, not a setting of this run.",
        "Every row here is scored by the same phi, so a rise with kappa is the trained "
        "objective paying off on a task it never saw.",
        "Only the policies appear: a classical method has no kappa of its own.",
        "Each dot is one policy. Where two share a kappa the line goes through their "
        "mean, and they differ in more than kappa -- training length, n, and the score "
        "they were trained on -- so this is a check for a strong effect, not a "
        "controlled sweep. Conditioning one run on kappa is what would make it one.",
    ]
    methods = sorted(known, key=lambda m: known[m])
    fig, axes, card, top, width = _figure(
        header, "Repair quality against training kappa", methods, notes,
        panel_h=3.4, width=6.6, panels=(1, 1))
    ax = axes[0]
    for event in events:
        pts = [(float(known[m]), float(np.mean(series[(event, m)])))
               for m in methods if series.get((event, m))]
        if len(pts) < 2:
            continue
        # one dot per model, and the line through the mean at each kappa -- two models
        # can share a kappa, and joining them would draw a vertical jump that reads as
        # a steep dependence on the one axis they agree about
        line = ax.plot([], [])[0]
        ax.scatter([x for x, _ in pts], [y for _, y in pts], s=34,
                   color=line.get_color(), zorder=4)
        by_k = {}
        for x, y in pts:
            by_k.setdefault(x, []).append(y)
        xs = sorted(by_k)
        if len(xs) >= 2:
            ax.plot(xs, [float(np.mean(by_k[x])) for x in xs], color=line.get_color(),
                    linewidth=1.6, label=event, zorder=3)
    ax.axhline(50.0, color=INK_2, linestyle="--", linewidth=1.1, zorder=1)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("stiffness_kappa the policy was trained at")
    ax.set_ylabel("repair percentile, 100 = best available")
    _panel_title(ax, "Repair percentile against training kappa",
                 "higher is a better repair; the dashed line is chance")
    ax.legend(frameon=False, fontsize=_fs(8))
    _style_axes(ax)
    return _finish(fig, card, methods, notes, width, top, run_dir, filename)


def plot_repair_events(run_dir, rows, header, filename="repair_events"):
    """One row per event, over the three things that separate the methods."""
    live = [r for r in rows if r.get("method") is not None]
    if not live:
        return None
    events = list(dict.fromkeys(r["event"] for r in live))
    methods = _repair_methods(rows)
    panels = (("repaired", "rigid again, %", "higher is better"),
              ("added_over_bound", "edges over the proven bound", "0 is optimal"),
              ("err_ratio_to_pre", "shape error / before the event", "below 1 is better"))

    notes = [
        "Rows are events, columns the three things that separate the methods.",
        "The bound is repair_edge_count, which is proven, so 0 means the method used "
        "the fewest edges that could possibly have worked.",
        "The last column is the only quality number defined for every event, including "
        "the ones where rigidity survived and there was nothing to count.",
    ]
    fig, axes, card, top, width = _figure(
        header, "Repair, by event", methods, notes,
        panel_h=1.9, width=7.2, panels=(len(events), 3))

    for r_i, event in enumerate(events):
        for c_i, (key, label, direction) in enumerate(panels):
            ax = axes[r_i * 3 + c_i]
            series = _repair_series(rows, key)
            present = [m for m in methods if (event, m) in series]
            vals = [float(np.mean(series[(event, m)])) * (100 if key == "repaired" else 1)
                    for m in present]
            ax.bar(range(len(present)), vals, 0.62, zorder=3,
                   color=[method_style(m)["color"] for m in present])
            if key == "err_ratio_to_pre":
                ax.axhline(1.0, color=INK_2, linestyle="--", linewidth=1.0, zorder=1)
            ax.set_xticks(range(len(present)))
            ax.set_xticklabels(present, rotation=25, ha="right", fontsize=_fs(7))
            _panel_title(ax, f"{event} - {label}" if c_i == 0 else label, direction)
            _style_axes(ax)
    return _finish(fig, card, methods, notes, width, top, run_dir, filename)
