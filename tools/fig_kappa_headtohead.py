"""Figures for the kappa head-to-head: where the policy wins, where it loses, and why.

Reads the decomposition `tools/kappa_headtohead.py --csv` writes.

    uv run tools/fig_kappa_headtohead.py <terms.csv> <outdir>
"""
import csv
import math
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MATCHED = {0.0: "k0", 10.0: "k10", 50.0: "k50", 200.0: "k200"}
POL, GRE = "#1f77b4", "#555555"


def style(ax):
    ax.grid(True, color="#cccccc", lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#000000")
        s.set_linewidth(0.8)


def gmean(v):
    v = [x for x in v if x and float(x) > 0]
    return math.exp(st.mean(math.log(float(x)) for x in v)) if v else float("nan")


def main(src, outdir):
    rows = list(csv.DictReader(open(src)))
    ks = sorted({float(r["kappa"]) for r in rows})
    sel = lambda k, m: [r for r in rows if float(r["kappa"]) == k and r["method"] == m]
    x = np.arange(len(ks))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))

    # --- where the gap comes from -------------------------------------------------
    ax = axes[0]
    d = {t: [] for t in ("sparsity", "robustness", "rigidity")}
    tot = []
    for k in ks:
        p = {int(r["episode"]): r for r in sel(k, MATCHED[k])}
        g = {int(r["episode"]): r for r in sel(k, "greedy")}
        e = sorted(set(p) & set(g))
        for t in d:
            d[t].append(np.mean([float(p[i][t]) - float(g[i][t]) for i in e]))
        tot.append(np.mean([float(p[i]["phi"]) - float(g[i]["phi"]) for i in e]))
    bot_p = np.zeros(len(ks))
    bot_n = np.zeros(len(ks))
    for t, c, lab in [("sparsity", "#4c9f70", "sparsity, $-w_e m c_{\\max}/\\rho_K$"),
                      ("robustness", "#d1603d", "robustness, $\\kappa\\,\\omega\\,q$"),
                      ("rigidity", "#8a8a8a", "rigidity, $w_r\\,\\mathrm{rk}/\\rho_K$")]:
        v = np.array(d[t])
        base = np.where(v >= 0, bot_p, bot_n)
        ax.bar(x, v, 0.6, bottom=base, color=c, label=lab, edgecolor="none")
        bot_p = bot_p + np.where(v >= 0, v, 0)
        bot_n = bot_n + np.where(v < 0, v, 0)
    ax.plot(x, tot, "o-", color="black", lw=1.4, ms=5, label="total $\\Delta\\varphi$")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{k:.0f}" for k in ks])
    ax.set_xlabel("$\\kappa$ of the environment")
    ax.set_ylabel("policy $-$ greedy")
    ax.set_title("Objective gap, by term", fontsize=10)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.95)
    style(ax)

    # --- shape error, split -------------------------------------------------------
    ax = axes[1]
    for meth, col, lab in [("pol", POL, "policy"), ("greedy", GRE, "greedy")]:
        for half, ls, mk in [("eps_pos", "-", "o"), ("eps_att", "--", "s")]:
            v = [gmean([r[half] for r in sel(k, MATCHED[k] if meth == "pol" else "greedy")])
                 for k in ks]
            ax.plot(x, v, ls, marker=mk, color=col, lw=1.4, ms=5,
                    label=f"{lab}, {'position' if half=='eps_pos' else 'attitude'}")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{k:.0f}" for k in ks])
    ax.set_xlabel("$\\kappa$ of the environment")
    ax.set_ylabel("shape error")
    ax.set_title("Estimation error, both halves", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.95)
    style(ax)

    # --- edges and at-bound -------------------------------------------------------
    ax = axes[2]
    for meth, col, lab in [("pol", POL, "policy"), ("greedy", GRE, "greedy")]:
        v = [st.mean(float(r["m"]) for r in sel(k, MATCHED[k] if meth == "pol" else "greedy"))
             for k in ks]
        ax.plot(x, v, "o-", color=col, lw=1.4, ms=5, label=f"{lab}, edges")
    ax.set_xticks(x); ax.set_xticklabels([f"{k:.0f}" for k in ks])
    ax.set_xlabel("$\\kappa$ of the environment")
    ax.set_ylabel("edges")
    ax2 = ax.twinx()
    for meth, col in [("pol", POL), ("greedy", GRE)]:
        v = [100 * sum(1 for r in sel(k, MATCHED[k] if meth == "pol" else "greedy")
                       if r["at_bound"] == "True") / 20 for k in ks]
        ax2.plot(x, v, ":", marker="^", color=col, lw=1.2, ms=5, alpha=0.75)
    ax2.set_ylabel("at the bound, %  (dotted)")
    ax2.set_ylim(-5, 105)
    ax.set_title("Edges and minimality", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.95)
    style(ax)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/kappa_headtohead.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {outdir}/kappa_headtohead.pdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
