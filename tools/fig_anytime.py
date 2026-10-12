"""Quality against the number of graph modifications applied.

The budget run covers greedy and the four policies under a hard cap on applied
edits. The classical opponents are not in it, so their curves come from the main
comparison run's per-step traces on the same frozen instances (same benchmark
digest), read as best-so-far. Greedy appears in both sources and agrees, which is
the check that the two are on the same footing.

    uv run tools/fig_anytime.py <outdir>
"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUDGET = "runs_outputs/20260913-043413__addremove__4models__budget/"
MAIN = "runs_outputs/20260912-162935__addremove__3models__main/"
GRID = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64, 88, 118]

STYLE = {"greedy": ("#555555", "-", "o"), "k10": ("#1f77b4", "-", "s"),
         "k0": ("#7fb2d6", "--", "v"), "k50": ("#9467bd", "--", "^"),
         "k200": ("#c5b0d5", "--", "D"), "anneal": ("#d1603d", "-", "P"),
         "spectral": ("#4c9f70", ":", "x"), "degree": ("#8c8c8c", ":", "+"),
         "constructive": ("#bcbd22", ":", "*")}
LABEL = {"k10": "policy $\\kappa{=}10$", "k0": "policy $\\kappa{=}0$",
         "k50": "policy $\\kappa{=}50$", "k200": "policy $\\kappa{=}200$"}


def style(ax):
    ax.grid(True, color="#cccccc", lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(0.8)


def from_budget():
    rows = list(csv.DictReader(open(BUDGET + "budget.csv")))
    out = {}
    for m in dict.fromkeys(r["method"] for r in rows):
        per = {}
        for r in rows:
            if r["method"] != m:
                continue
            per.setdefault(int(r["budget"]), []).append(
                (int(r["episode"]), float(r["score"]), r["at_bound"] == "True"))
        out[m] = per
    return out


def from_traces():
    """best-so-far within k recorded steps, per episode, for the classical opponents."""
    rows = list(csv.DictReader(open(MAIN + "trajectories.csv")))
    out = {}
    for m in ("anneal", "spectral", "degree", "constructive", "greedy"):
        sel = [r for r in rows if r["method"] == m]
        eps = sorted({int(r["episode"]) for r in sel})
        per = {}
        for k in GRID:
            vals = []
            for e in eps:
                s = [float(r["score"]) for r in sel
                     if int(r["episode"]) == e and int(r["step"]) <= k]
                if s:
                    vals.append((e, max(s), False))
            per[k] = vals
        out[m] = per
    return out


def main(outdir):
    bud, tra = from_budget(), from_traces()

    g_b = np.mean([v for _, v, _ in bud["greedy"][1]])
    g_t = np.mean([v for _, v, _ in tra["greedy"][1]])
    print(f"consistency check, greedy at budget 1: budget run {g_b:.2f}, traces {g_t:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))

    ax = axes[0]
    xs = [b for b in GRID if b in bud["greedy"]]
    ys = [np.mean([v for _, v, _ in bud["greedy"][b]]) for b in xs]
    ax.plot(xs, ys, "-", color="#555555", lw=4.0, alpha=0.35, zorder=1)      # halo
    ax.plot(xs, ys, "-", marker="o", color="#555555", lw=1.6, ms=5,
            label="greedy", zorder=4)
    for m in ("k10", "k0", "k50", "k200"):
        c, ls, mk = STYLE[m]
        xs = [b for b in GRID if b in bud[m]]
        ys = [np.mean([v for _, v, _ in bud[m][b]]) for b in xs]
        ax.plot(xs, ys, ls, marker=mk, color=c, lw=1.5, ms=4.5,
                label=LABEL[m], zorder=3 if m == "k10" else 2)
    for m in ("anneal", "spectral", "degree", "constructive"):
        c, ls, mk = STYLE[m]
        xs = GRID
        ys = [np.mean([v for _, v, _ in tra[m][b]]) for b in xs]
        ax.plot(xs, ys, ls, marker=mk, color=c, lw=1.2, ms=4, alpha=0.85, label=m)
    ax.set_xscale("log")
    ax.set_ylim(52, 86)
    ax.annotate("constructive starts at $\\varphi=4$", xy=(1.05, 53.2), fontsize=6.5,
                color="#8a8a20")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 118])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "118"])
    ax.set_xlabel("graph modifications applied")
    ax.set_ylabel("$\\varphi$ of the best graph reached")
    ax.set_title("Quality against budget", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.95)
    style(ax)

    # paired policy - greedy, with a bootstrap band
    ax = axes[1]
    rng = np.random.default_rng(0)
    for m in ("k10", "k0", "k50", "k200"):
        c, ls, mk = STYLE[m]
        xs = [b for b in GRID if b in bud[m]]
        mid, lo, hi = [], [], []
        for b in xs:
            p = {e: v for e, v, _ in bud[m][b]}
            g = {e: v for e, v, _ in bud["greedy"][b]}
            d = np.array([p[e] - g[e] for e in sorted(set(p) & set(g))])
            bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(2000)])
            mid.append(d.mean()); lo.append(np.percentile(bs, 5)); hi.append(np.percentile(bs, 95))
        ax.plot(xs, mid, ls, marker=mk, color=c, lw=1.5, ms=4.5, label=LABEL[m])
        if m == "k10":
            ax.fill_between(xs, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 118])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "118"])
    ax.set_xlabel("graph modifications applied")
    ax.set_ylabel("$\\varphi$, policy $-$ greedy")
    ax.set_title("Paired difference against greedy", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.95)
    style(ax)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/anytime.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {outdir}/anytime.pdf")


if __name__ == "__main__":
    main(sys.argv[1])
