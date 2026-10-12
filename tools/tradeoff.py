"""The trade-off curve: edges against shape error as kappa rises.

Every arm optimizes a different phi, so scoring them all by one of them answers the
wrong question. Both axes here are objective free. Edges is a count and shape error is
the RMS state error per radian of bearing noise, so a policy trained at one kappa and a
policy trained at another can be read on the same pair of axes whatever each was
rewarded for.

Each policy is read in its own environment, which is the diagonal of the four runs: a
rollout is scored on the best state it visited, so the graph a checkpoint reports
depends on the phi doing the scoring, and the arm's own phi is the one it was trained
for. greedy is drawn at every kappa from the same runs, since it hill-climbs whatever
phi is configured and is therefore conditioning-aware for free.

    PYTHONPATH=. uv run tools/tradeoff.py <run_dir_k0> <run_dir_k10> ... --kappas 0,10,50,200

With no run directories it takes the newest run tagged tradeoff_k<kappa> for each kappa,
falling back to the older front_k<kappa> spelling.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import report

FIELDS = ("kappa", "method", "label", "episodes", "edges_mean", "edges_sd",
          "shape_err_gmean", "shape_err_gsd", "at_bound_pct", "rigid_pct")


def newest(kappa):
    """The newest run for this kappa. `front_k*` is the tag this experiment used to
    carry and is still accepted, so runs made before the rename still load."""
    dirs = sorted(glob.glob(os.path.join("runs_outputs", f"*__tradeoff_k{kappa}"))
                  + glob.glob(os.path.join("runs_outputs", f"*__front_k{kappa}")))
    return dirs[-1] if dirs else None


def rows_of(run_dir):
    with open(os.path.join(run_dir, "results.csv")) as f:
        return list(csv.DictReader(f))


def point(rows, method):
    """(edges mean, edges sd, shape error gmean, gsd, at-bound %, rigid %, n)."""
    sel = [r for r in rows if r["method"] == method]
    if not sel:
        return None
    edges = [float(r["m"]) for r in sel]
    err = [float(r["shape_err"]) for r in sel
           if r["shape_err"] not in ("", "nan") and float(r["shape_err"]) > 0]
    if not err:
        return None
    return {
        "episodes": len(sel),
        "edges_mean": float(np.mean(edges)), "edges_sd": float(np.std(edges)),
        "shape_err_gmean": float(np.exp(np.mean(np.log(err)))),
        "shape_err_gsd": float(np.exp(np.std(np.log(err)))),
        "at_bound_pct": 100.0 * float(np.mean([r["at_bound"] == "True" for r in sel])),
        "rigid_pct": 100.0 * float(np.mean([r["is_IBR"] == "True" for r in sel])),
    }


def collect(dirs, kappas):
    """One record per (kappa, series). The policy of arm k is read only in run k."""
    out = []
    for kappa, run_dir in zip(kappas, dirs):
        rows = rows_of(run_dir)
        for method, label in ((f"k{kappa}", "policy"), ("greedy", "greedy")):
            p = point(rows, method)
            if p is not None:
                out.append(dict(p, kappa=kappa, method=method, label=label))
    return out


def draw(run_dir, records, kappas, header):
    import matplotlib
    matplotlib.use("Agg")

    series = {lab: sorted([r for r in records if r["label"] == lab],
                          key=lambda r: r["kappa"]) for lab in ("policy", "greedy")}
    if not series["policy"]:
        return None
    style = {"policy": report.method_style("learned"), "greedy": report.method_style("greedy")}
    report.METHOD_BLURB.setdefault(
        "policy", "the trained policy of each arm, read in its own environment")
    notes = [
        "Both axes are objective free, so arms trained on different phi compare "
        "directly. Edges is the mean over the instances and shape error the geometric "
        "mean, which is why its spread is a multiplicative factor.",
        "Each policy is read in its own environment. A rollout is scored on the best "
        "state it visited, so the graph a checkpoint reports depends on the phi doing "
        "the scoring, and its own is the one it was trained for.",
        "greedy hill-climbs whatever phi is configured, so it traces a curve of its own "
        "at no training cost. Down and to the left is better on both axes.",
    ]
    fig, axes, card, top, width = report._figure(
        header, "Trading edges for accuracy", ["policy", "greedy"], notes,
        panel_h=3.0, width=7.0, panels=(1, 2))

    ax = axes[0]
    for lab, recs in series.items():
        if not recs:
            continue
        st = style[lab]
        x = [r["edges_mean"] for r in recs]
        y = [r["shape_err_gmean"] for r in recs]
        ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.6, marker=st["marker"],
                ms=7, zorder=5, label=lab)
        # the two series sit on top of each other at low kappa, so their labels go
        # opposite ways rather than overlapping
        dx, dy, ha = ((6, 6, "left") if lab == "policy" else (-6, -12, "right"))
        for r, xi, yi in zip(recs, x, y):
            ax.annotate(f"$\\kappa$={r['kappa']:g}", xy=(xi, yi), xytext=(dx, dy),
                        textcoords="offset points", fontsize=8, ha=ha,
                        color=report.INK_2 if lab == "greedy" else st["color"])
    ax.set_xlabel("edges (mean over the instances)", color=report.INK_2, fontsize=8.5)
    ax.set_ylabel("shape error (geometric mean)", color=report.INK_2, fontsize=8.5)
    report._style_axes(ax, log=True)
    report._panel_title(ax, "Edges against shape error",
                        "one point per kappa; down and left is better on both")

    ax = axes[1]
    for lab, recs in series.items():
        if not recs:
            continue
        st = style[lab]
        ax.plot([r["kappa"] for r in recs], [r["at_bound_pct"] for r in recs],
                color=st["color"], ls=st["ls"], lw=1.6, marker=st["marker"], ms=7,
                zorder=5)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("kappa", color=report.INK_2, fontsize=8.5)
    ax.set_ylabel("rigid at the proven bound, %", color=report.INK_2, fontsize=8.5)
    ax.set_ylim(-4, 104)
    report._style_axes(ax)
    report._panel_title(ax, "What the accuracy costs in sparsity",
                        "percent of instances rigid on exactly m_req edges")

    return report._finish(fig, card, ["policy", "greedy"], notes, width, top, run_dir,
                          "tradeoff")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="*", help="one run directory per kappa, in order")
    ap.add_argument("--kappas", default="0,10,50,200")
    ap.add_argument("--out-dir", default=None, help="where to write; default the k=0 run")
    args = ap.parse_args(argv)

    kappas = [int(k) for k in args.kappas.split(",")]
    dirs = args.run_dirs or [newest(k) for k in kappas]
    missing = [k for k, d in zip(kappas, dirs) if not d]
    if missing:
        print(f"no run tagged tradeoff_k{missing[0]} in runs_outputs/")
        return 1
    for k, d in zip(kappas, dirs):
        print(f"  kappa {k:>3}  {d}")

    records = collect(dirs, kappas)
    out = args.out_dir or dirs[0]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "tradeoff.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS), extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    meta = json.load(open(os.path.join(dirs[0], "meta.json")))
    header = {"short": report.short_env_name(meta["args"]["environment_name"]),
              "env": meta["args"]["environment_name"], "model": None,
              "network": (meta.get("context") or {}).get("network", ""),
              "episodes": meta["args"].get("episodes"),
              "seed": meta["args"].get("seed"),
              "benchmark": meta["args"].get("benchmark")}

    draw(out, records, kappas, header)
    with report.plain():
        draw(out, records, kappas, header)

    print()
    print(f"  {'kappa':>6}{'series':>9}{'edges':>14}{'shape err':>16}{'at bound':>10}")
    for r in sorted(records, key=lambda r: (r["label"], r["kappa"])):
        print(f"  {r['kappa']:>6}{r['label']:>9}"
              f"{r['edges_mean']:>9.2f}+-{r['edges_sd']:<4.2f}"
              f"{r['shape_err_gmean']:>10.2e} x/{r['shape_err_gsd']:<4.1f}"
              f"{r['at_bound_pct']:>9.0f}%")
    print(f"\nwrote {out}/tradeoff.csv and plots/*/tradeoff.*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
