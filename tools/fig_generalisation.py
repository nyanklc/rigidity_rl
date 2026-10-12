"""Transfer of one policy across ten frozen instance sets.

The policy is trained on ten-agent formations with random domains. The sets vary
the size, the domain mix and the global frame. Both panels are paired per
instance, so the bars are differences on the same formation and not two means
placed side by side.

    uv run tools/fig_generalisation.py <terms.csv> <outdir>
"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NICE = {"bench_n8_R3": "8, $R^3$", "bench_n8_R3_rot": "8, $R^3$ rot.",
        "bench_n8_R3xS1": "8, $R^3\\!\\times\\!S^1$", "bench_n8_SE3": "8, SE(3)",
        "bench_n8_randdom": "8, mixed", "bench_n10_randdom": "10, mixed",
        "bench_n16_R3": "16, $R^3$", "bench_n16_SE3": "16, SE(3)",
        "bench_n16_randdom": "16, mixed", "bench_n20_randdom": "20, mixed"}
MIXED = ["bench_n8_randdom", "bench_n10_randdom", "bench_n16_randdom", "bench_n20_randdom"]
NMIX = {"bench_n8_randdom": 8, "bench_n10_randdom": 10,
        "bench_n16_randdom": 16, "bench_n20_randdom": 20}


def style(ax):
    ax.grid(True, color="#cccccc", lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(0.8)


def main(src, outdir):
    rows = list(csv.DictReader(open(src)))
    order = [r["bench"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))

    # --- minimality, per set -------------------------------------------------------
    ax = axes[0]
    x = np.arange(len(order))
    pol = [100 * int(r["at_pol"]) / int(r["n_inst"]) for r in rows]
    gre = [100 * int(r["at_gre"]) / int(r["n_inst"]) for r in rows]
    ax.bar(x - 0.2, pol, 0.4, color="#1f77b4", label="policy", edgecolor="white")
    ax.bar(x + 0.2, gre, 0.4, color="#555555", label="greedy", edgecolor="white", alpha=0.85)
    trained = order.index("bench_n10_randdom")
    ax.axvspan(trained - 0.5, trained + 0.5, color="#ffd28a", alpha=0.35, zorder=0)
    ax.annotate("trained here", (trained, 97), fontsize=6.5, ha="center", color="#8a6d1f")
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[b] for b in order], rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("at the proven bound, %")
    ax.set_ylim(0, 105)
    ax.set_title("Minimality across the instance sets", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    style(ax)

    # --- estimation error against size, mixed family -------------------------------
    ax = axes[1]
    by = {r["bench"]: r for r in rows}
    xs = [NMIX[b] for b in MIXED]
    mid = [float(by[b]["err_ratio"]) for b in MIXED]
    lo = [float(by[b]["r_lo"]) for b in MIXED]
    hi = [float(by[b]["r_hi"]) for b in MIXED]
    ax.errorbar(xs, mid, yerr=[np.array(mid) - np.array(lo), np.array(hi) - np.array(mid)],
                fmt="o-", color="#1f77b4", lw=1.6, ms=6, capsize=3, label="mixed domains")
    hom = [b for b in order if b not in MIXED]
    hx = [8, 8, 8, 8, 16, 16][:len(hom)]
    ax.scatter([int(NICE[b].split(",")[0]) for b in hom],
               [float(by[b]["err_ratio"]) for b in hom],
               s=42, facecolor="none", edgecolor="#8c8c8c", linewidth=1.2,
               label="fixed domain", zorder=3)
    ax.axhline(1, color="black", lw=1.0, ls=":")
    ax.annotate("policy better", (20.4, 0.80), fontsize=7, color="#2a6", ha="right")
    ax.annotate("greedy better", (20.4, 1.22), fontsize=7, color="#a33", ha="right")
    ax.set_xticks([8, 10, 16, 20])
    ax.set_xlabel("agents in the formation")
    ax.set_ylabel("estimation error, policy / greedy")
    ax.set_title("Estimation error against formation size", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    style(ax)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/generalisation.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {outdir}/generalisation.pdf")
    print("  mixed family, error ratio against greedy:",
          ", ".join(f"n={NMIX[b]}: {float(by[b]['err_ratio']):.3f}" for b in MIXED))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
