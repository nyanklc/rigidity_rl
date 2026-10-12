"""The trained policies on formations far larger than they were trained on.

Greedy is not affordable past about twenty agents, so the spectral hill climb
carries the classical reference. The estimation error is not comparable across
sizes, so the right panel plots it relative to spectral on the same instances
rather than in absolute terms.

    uv run tools/fig_large_n.py <large_n.csv> <outdir>
"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# n = 10 anchor, 20 frozen instances, from the kappa=10 front run
ANCHOR = {"k0": (1.038, 11.30), "k10": (1.025, 6.37),
          "k200": (1.177, 5.22), "spectral": (1.245, 5.47)}
COL = {"k0": "#7fb2d6", "k10": "#1f77b4", "k200": "#9467bd", "spectral": "#4c9f70"}
LAB = {"k0": "policy $\\kappa{=}0$", "k10": "policy $\\kappa{=}10$",
       "k200": "policy $\\kappa{=}200$", "spectral": "spectral"}


def style(ax):
    ax.grid(True, color="#cccccc", lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(0.8)


def main(src, outdir):
    rows = list(csv.DictReader(open(src)))
    ns = sorted({int(r["n"]) for r in rows})
    gm = lambda v: np.exp(np.mean(np.log([x for x in v if x and x > 0]))) if any(v) else np.nan

    agg = {}
    for n in ns:
        for lab in COL:
            s = [r for r in rows if int(r["n"]) == n and r["method"] == lab]
            if not s:
                continue
            agg[(n, lab)] = (np.mean([float(r["m_over_mreq"]) for r in s]),
                             gm([float(r["shape_err"]) for r in s if r["shape_err"]]))
    for lab, v in ANCHOR.items():
        agg[(10, lab)] = v
    xs = [10] + ns

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7))

    ax = axes[0]
    for lab in COL:
        pts = [(n, agg[(n, lab)][0]) for n in xs if (n, lab) in agg]
        if pts:
            ax.plot(*zip(*pts), "o-", color=COL[lab], lw=1.6, ms=5.5, label=LAB[lab])
    ax.axhline(1.0, color="black", lw=1.0, ls=":")
    ax.annotate("the proven bound", (xs[0], 1.005), fontsize=7, color="#333333")
    ax.axvspan(9, 11, color="#ffd28a", alpha=0.35, zorder=0)
    ax.annotate("trained", (10, ax.get_ylim()[1]), fontsize=6.5, ha="center",
                va="top", color="#8a6d1f")
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("agents in the formation")
    ax.set_ylabel("edges, as a multiple of $m_{\\rm req}$")
    ax.set_title("Edge count against size", fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.95)
    style(ax)

    ax = axes[1]
    for lab in COL:
        if lab == "spectral":
            continue
        pts = [(n, agg[(n, lab)][1] / agg[(n, "spectral")][1])
               for n in xs if (n, lab) in agg and (n, "spectral") in agg]
        if pts:
            ax.plot(*zip(*pts), "o-", color=COL[lab], lw=1.6, ms=5.5, label=LAB[lab])
    ax.axhline(1.0, color="#4c9f70", lw=1.4, ls="-")
    ax.annotate("spectral", (xs[-1], 1.02), fontsize=7.5, color="#4c9f70", ha="right")
    ax.axvspan(9, 11, color="#ffd28a", alpha=0.35, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xticks([], minor=True)
    ax.set_yticks([0.3, 0.5, 1.0, 2.0, 4.0, 8.0])
    ax.set_yticklabels(["0.3", "0.5", "1", "2", "4", "8"]); ax.set_yticks([], minor=True)
    ax.set_xlabel("agents in the formation")
    ax.set_ylabel("estimation error, relative to spectral")
    ax.set_title("Estimation error, against the classical reference", fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.95)
    style(ax)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/large_n.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {outdir}/large_n.pdf\n")
    print(f"{'n':>5s}" + "".join(f"{LAB[l][:12]:>14s}" for l in COL))
    for n in xs:
        print(f"{n:5d}" + "".join(
            f"{agg[(n,l)][0]:6.3f}/{agg[(n,l)][1]:7.2f}" if (n, l) in agg else f"{'-':>14s}"
            for l in COL))
    print("  (cells are edges/m_req and estimation error)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
