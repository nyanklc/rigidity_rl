"""Which spectral criterion ranks topologies the way the measured error does?

A-optimality *is* the Cramer-Rao error, so ranking by it at small noise is
circular. The question that is not: at realistic bearing noise, where the
linearisation no longer settles the answer, which criterion still agrees with the
measurement? Topologies are compared within one pose set, which is the choice a
policy faces.

    PYTHONPATH=. uv run tools/functional_vs_error.py [--poses 8] [--graphs 12]
"""
import argparse
import copy

import numpy as np

import estimation as E
from rigidity import (estimation_error_of, extended_bearing_rigidity_matrix as B_of,
                      greedy_rigid_construction)
from scenario import random_scenario

DOMAINS = {
    "R^3": ["R^3"] * 8,
    "SE(3)": ["SE(3)"] * 8,
    "R^2xS^1": ["R^2xS^1"] * 8,
    "mixed": ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"] * 2,
}


def spearman(x, y):
    """Rank correlation, written out rather than pulled from scipy."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / den) if den > 0 else np.nan


def rigid_variants(base, rank_K, count, rng):
    """`count` distinct rigid topologies on ONE pose set: greedy orders, then edges."""
    out = []
    while len(out) < count:
        work = copy.deepcopy(base)
        greedy_rigid_construction(work, rank_K, rng)
        for extra in range(rng.integers(0, 4)):
            i, j = rng.choice(work.n, 2, replace=False)
            work.edges[i, j] = True
        if np.linalg.matrix_rank(B_of(work)) >= rank_K:
            out.append(work)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, default=8, help="pose sets per domain")
    ap.add_argument("--graphs", type=int, default=12, help="topologies per pose set")
    ap.add_argument("--trials", type=int, default=60, help="noise draws per topology")
    ap.add_argument("--sigmas", default="0.0001,0.0175,0.0873",
                    help="radians; defaults are ~0, 1 deg and 5 deg")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",")]
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"{args.poses} pose sets x {args.graphs} topologies, {args.trials} noise draws")
    print("Spearman rank correlation between the criterion and the MEASURED position")
    print("error, computed within each pose set and averaged over them. 1.0 = the")
    print("criterion orders topologies exactly as the measured error does.\n")

    head = f"{'domain':9s} {'sigma':>8s} {'deg':>5s} | {'A (trace)':>10s} {'E (1/lam)':>10s} {'D (logdet)':>11s} | {'CRLB ratio':>10s}"
    print(head)
    print("-" * len(head))

    for tag, doms in DOMAINS.items():
        rows = {s: {"A": [], "E": [], "D": [], "ratio": []} for s in sigmas}

        for _ in range(args.poses):
            base, _ = random_scenario(len(doms), doms, edge_count=0)
            rank_K = int(np.linalg.matrix_rank(B_of(base.fully_connected())))
            nets = rigid_variants(base, rank_K, args.graphs, rng)

            crit = np.array([estimation_error_of(g, rank_K) for g in nets])  # a, e, d
            pred = np.array([E.predicted_error(g, rank_K)[0] for g in nets])

            for s in sigmas:
                meas = np.array([
                    E.monte_carlo_error(g, s, trials=args.trials,
                                        rng=np.random.default_rng(args.seed + 1),
                                        rank_K=rank_K)["position"]["rms"]
                    for g in nets])
                ok = np.isfinite(meas) & np.isfinite(crit).all(axis=1)
                if ok.sum() < 3:
                    continue
                rows[s]["A"].append(spearman(crit[ok, 0], meas[ok]))
                rows[s]["E"].append(spearman(crit[ok, 1], meas[ok]))
                rows[s]["D"].append(spearman(crit[ok, 2], meas[ok]))
                rows[s]["ratio"].append(float(np.median(meas[ok] / (s * pred[ok]))))

        for s in sigmas:
            r = rows[s]
            if not r["A"]:
                continue
            print(f"{tag:9s} {s:8.4f} {np.degrees(s):5.1f} | "
                  f"{np.nanmean(r['A']):10.3f} {np.nanmean(r['E']):10.3f} "
                  f"{np.nanmean(r['D']):11.3f} | {np.nanmean(r['ratio']):10.3f}")

    print("\nCRLB ratio is measured / predicted: 1.0 means the linearisation still")
    print("holds at that noise level, and below 1.0 means it has broken down.")


if __name__ == "__main__":
    main()
