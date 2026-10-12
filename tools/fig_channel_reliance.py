"""What the policy leans on, at the two ends of the conditioning weight.

Destroying one observation channel and re-running the episode gives the cost of
that channel in the objective. Comparing the policy trained at kappa = 0 with the
one trained at kappa = 200 says whether the conditioning weight changes what the
policy reads, which is the other half of the question the kappa head-to-head asks.

The exactness column is from Chapter 4: the rank channels answer their question
exactly, the conditioning channels are ranking priors.

    uv run tools/fig_channel_reliance.py <outdir>
"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {"$\\kappa=0$": "runs_outputs/20260912-183108__addremove__ref_k0_dqn_gine",
        "$\\kappa=200$": "runs_outputs/20260912-183357__addremove__ref_k200_dqn_gine"}
# channel -> (family, what Chapter 4 says its fidelity is)
KIND = {
    "add_rank":         ("rank", "exact"),
    "remove_rank":      ("rank", "exact"),
    "add_independence": ("rank", "exact"),
    "pair_max_rank":    ("rank", "exact"),
    "add_stiffness":    ("conditioning", "prior, $+0.93$"),
    "remove_stiffness": ("conditioning", "exact"),
    "node_slack":       ("conditioning", "prior"),
    "node_freedom":     ("conditioning", "prior"),
    "bearings":         ("geometry", "raw"),
    "coord_features":   ("geometry", "raw"),
    "degree":           ("graph", "exact"),
    "common_nbrs":      ("graph", "exact"),
    "reciprocity":      ("graph", "exact"),
}
FAMCOL = {"rank": "#1f77b4", "conditioning": "#d1603d",
          "geometry": "#4c9f70", "graph": "#8c8c8c"}


def style(ax):
    ax.grid(True, axis="x", color="#cccccc", lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(0.8)


def load(d, mode="shuffle"):
    out = {}
    for x in csv.DictReader(open(f"{d}/ablation/learned-{mode}.csv")):
        if x["channel"] == "(reference)" or x["feeds_action_mask"] == "yes":
            continue                       # adj feeds the mask, its cost is not comparable
        if x["d_phi"] not in ("", None):
            out[x["channel"]] = float(x["d_phi"])
    return out


def main(outdir):
    data = {k: load(v) for k, v in RUNS.items()}
    chans = [c for c in KIND if any(c in d for d in data.values())]
    chans.sort(key=lambda c: max(data[k].get(c, 0) for k in data))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(chans))
    h = 0.38
    for i, (lab, off, alpha) in enumerate([(list(RUNS)[0], +h / 2, 1.0),
                                           (list(RUNS)[1], -h / 2, 0.55)]):
        vals = [data[lab].get(c, 0.0) for c in chans]
        ax.barh(y + off, vals, h, color=[FAMCOL[KIND[c][0]] for c in chans],
                alpha=alpha, edgecolor="white", linewidth=0.6,
                label=f"policy trained at {lab}")
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  ({KIND[c][1]})" for c in chans], fontsize=8)
    ax.set_xlabel("cost in $\\varphi$ of destroying the channel")
    ax.set_title("What each policy reads, and how exact that reading is", fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMCOL[f]) for f in FAMCOL]
    leg1 = ax.legend(handles, list(FAMCOL), fontsize=7.5, loc="lower right",
                     title="channel family", title_fontsize=7.5, framealpha=0.95)
    ax.add_artist(leg1)
    solid = plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=1.0)
    faded = plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.55)
    ax.legend([solid, faded], [f"trained at {k}" for k in RUNS], fontsize=7.5,
              loc="center right", framealpha=0.95)
    style(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/channel_reliance.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {outdir}/channel_reliance.pdf")

    for lab in RUNS:
        top = sorted(data[lab].items(), key=lambda kv: -kv[1])[:4]
        print(f"  {lab:12s} leans on: " + ", ".join(f"{c} ({v:.2f})" for c, v in top))


if __name__ == "__main__":
    main(sys.argv[1])
