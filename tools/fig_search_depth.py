"""Algorithm and backbone at kappa = 50, and why the ordering is what it is.

Section 4.7.2 says the conditioning is monotone but not submodular, so a search
that commits early has no guarantee on it. The four arms differ in how long they
keep searching, and that is what separates them: the two PPO arms hold out to
step 70-80 of 118, the two DQN arms settle by step 15, and greedy stops at 12.

    uv run tools/fig_search_depth.py <outdir>
"""
import csv
import math
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "runs_outputs/20260912-170134__addremove-n10_randdom__4models__arm_2x2/"
ARMS = ["dqn_gine", "dqn_equi", "ppo_equi", "ppo_gine"]
NICE = {"dqn_gine": "DQN + GINE", "dqn_equi": "DQN + EGNN",
        "ppo_gine": "PPO + GINE", "ppo_equi": "PPO + EGNN"}
COL = {"dqn_gine": "#1f77b4", "dqn_equi": "#7fb2d6",
       "ppo_gine": "#d1603d", "ppo_equi": "#e8a08a"}


def style(ax):
    ax.grid(True, color="#cccccc", lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(0.8)


def main(outdir):
    rows = list(csv.DictReader(open(RUN + "results.csv")))
    by = {(x["method"], x["episode"]): x for x in rows}
    eps = sorted({x["episode"] for x in rows})
    rng = np.random.default_rng(1)

    stat = {}
    for m in ARMS + ["greedy"]:
        lr = np.array([math.log(float(by[(m, e)]["shape_err"]) /
                                float(by[("greedy", e)]["shape_err"])) for e in eps
                       if by[(m, e)]["shape_err"] and by[("greedy", e)]["shape_err"]])
        bs = np.array([rng.choice(lr, len(lr), replace=True).mean() for _ in range(4000)])
        stat[m] = dict(ratio=math.exp(lr.mean()),
                       lo=math.exp(np.percentile(bs, 5)), hi=math.exp(np.percentile(bs, 95)),
                       depth=np.mean([float(by[(m, e)]["best_at"]) for e in eps]),
                       wins=int(sum(1 for e in eps
                                    if float(by[(m, e)]["shape_err"]) <
                                    float(by[("greedy", e)]["shape_err"]))))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    ax = axes[0]
    y = np.arange(len(ARMS))
    for i, m in enumerate(ARMS):
        s = stat[m]
        ax.barh(i, s["ratio"] - 1, 0.55, left=1, color=COL[m], edgecolor="white")
        ax.plot([s["lo"], s["hi"]], [i, i], color="black", lw=1.3, zorder=4)
        ax.plot([s["lo"], s["hi"]], [i, i], "|", color="black", ms=6, zorder=4)
        ax.annotate(f"{s['wins']}/20", (max(s["hi"], 1.0) + 0.012, i), fontsize=7.5,
                    va="center", color="#333333")
    ax.axvline(1, color="black", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels([NICE[m] for m in ARMS], fontsize=8.5)
    ax.set_xlabel("shape error, relative to greedy")
    ax.set_title("Estimation error against greedy, $\\kappa=50$", fontsize=10)
    ax.annotate("better than greedy", xy=(0.90, -0.72), fontsize=7, color="#2a6", ha="center")
    ax.annotate("worse", xy=(1.16, -0.72), fontsize=7, color="#a33", ha="center")
    style(ax)

    ax = axes[1]
    for m in ARMS:
        s = stat[m]
        ax.scatter(s["depth"], s["ratio"], s=95, color=COL[m], zorder=3,
                   edgecolor="white", linewidth=0.9)
        ax.annotate(NICE[m], (s["depth"], s["ratio"]), textcoords="offset points",
                    xytext=(8, 3), fontsize=8, color=COL[m])
    g = stat["greedy"]
    ax.scatter(g["depth"], 1.0, s=95, color="#555555", marker="D", zorder=3,
               edgecolor="white", linewidth=0.9)
    ax.annotate("greedy", (g["depth"], 1.0), textcoords="offset points",
                xytext=(8, -10), fontsize=8, color="#555555")
    ax.axhline(1, color="black", lw=0.9, ls=":")
    ax.set_xlabel("step at which the best graph is first reached")
    ax.set_ylabel("shape error, relative to greedy")
    ax.set_title("Deeper search, better conditioning", fontsize=10)
    style(ax)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/search_depth.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {outdir}/search_depth.pdf\n")
    for m in ARMS:
        s = stat[m]
        print(f"  {NICE[m]:12s} err x{s['ratio']:.3f} [{s['lo']:.3f},{s['hi']:.3f}] "
              f"wins {s['wins']:2d}/20, best graph at step {s['depth']:.0f}")
    print(f"  {'greedy':12s} err x1.000 (reference),          best graph at step {g['depth']:.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
