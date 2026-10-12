"""What each method buys for what it spends.

Three costs are kept apart because they answer different questions: counted calls
to the linear-algebra primitives, calls to the objective itself, and the number
of graph modifications applied. Wall clock is reported but not plotted against
quality, since it compares a Python hill climb with a Python environment that
also runs a neural network.

    uv run tools/fig_cost_quality.py <outdir>
"""
import csv
import statistics as st
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "runs_outputs/20260912-162935__addremove__3models__main/"
ORDER = ["degree", "spectral", "constructive", "greedy", "anneal", "s0"]
NICE = {"s0": "policy", "degree": "degree", "spectral": "spectral",
        "constructive": "constructive", "greedy": "greedy", "anneal": "anneal"}
COL = {"degree": "#8c8c8c", "spectral": "#4c9f70", "constructive": "#bcbd22",
       "greedy": "#555555", "anneal": "#d1603d", "s0": "#1f77b4"}


def style(ax):
    ax.grid(True, color="#cccccc", lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(0.8)


def load():
    cost = {(x["method"], x["episode"]): x for x in csv.DictReader(open(RUN + "cost.csv"))}
    res = list(csv.DictReader(open(RUN + "results.csv")))
    out = {}
    for m in ORDER:
        R = [x for x in res if x["method"] == m]
        C = [cost[(m, x["episode"])] for x in R if (m, x["episode"]) in cost]
        f = lambda k, S: st.mean(float(s[k]) for s in S if s[k] not in ("", "nan"))
        work, best = f("work", R), f("best_at", R)
        out[m] = dict(phi=f("score", R), work=work, best=best,
                      calls=f("total", C), phi_evals=f("score_network", C),
                      ms=f("ms", C), per_edit=f("total", C) / max(work, 1e-9))
    return out


def main(outdir):
    d = load()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7))

    # --- quality against what it cost to get there --------------------------------
    ax = axes[0]
    for m in ORDER:
        v = d[m]
        ax.scatter(v["calls"], v["phi"], s=70, color=COL[m], zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(NICE[m], (v["calls"], v["phi"]), textcoords="offset points",
                    xytext=(7, -3), fontsize=8, color=COL[m])
    # the policy as it would cost if it stopped on its own best graph
    v = d["s0"]
    stopped = v["per_edit"] * v["best"]
    ax.scatter(stopped, v["phi"], s=70, facecolor="none", edgecolor=COL["s0"],
               linewidth=1.6, zorder=3)
    ax.annotate("policy, stopped\non its best graph", (stopped, v["phi"]),
                textcoords="offset points", xytext=(6, -34), fontsize=7.5,
                color=COL["s0"], ha="left")
    ax.annotate("", xy=(stopped, v["phi"]), xytext=(v["calls"], v["phi"]),
                arrowprops=dict(arrowstyle="->", color=COL["s0"], lw=1.1, ls=":"))
    ax.set_xscale("log")
    ax.set_xlabel("counted primitive calls per network")
    ax.set_ylabel("$\\varphi$ reached")
    ax.set_title("Quality against cost", fontsize=10)
    style(ax)

    # --- what the cost is made of -------------------------------------------------
    ax = axes[1]
    x = np.arange(len(ORDER))
    per_edit = [d[m]["per_edit"] for m in ORDER]
    edits = [d[m]["work"] for m in ORDER]
    ax.bar(x - 0.2, per_edit, 0.4, color=[COL[m] for m in ORDER], label="calls per edit")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, edits, 0.4, color=[COL[m] for m in ORDER], alpha=0.42,
            hatch="///", edgecolor="white", label="edits applied")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[m] for m in ORDER], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("calls per applied edit  (solid, log)")
    ax2.set_ylabel("edits applied  (hatched)")
    ax.set_title("Cost per decision, and how many decisions", fontsize=10)
    style(ax)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/cost_quality.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {outdir}/cost_quality.pdf")

    print(f"\n{'method':14s}{'phi':>7s}{'calls':>9s}{'/edit':>8s}{'edits':>7s}"
          f"{'phi evals':>10s}{'ms':>8s}")
    for m in ORDER:
        v = d[m]
        print(f"{NICE[m]:14s}{v['phi']:7.1f}{v['calls']:9.0f}{v['per_edit']:8.1f}"
              f"{v['work']:7.1f}{v['phi_evals']:10.0f}{v['ms']:8.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
