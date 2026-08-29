"""Do the A-, D- and E-optimality criteria rank graphs differently?

All three come off B's spectrum and all three are monotone in edges. If they
order graphs the same way, swapping one for another in the state score cannot
change what a policy learns.

    PYTHONPATH=. uv run tools/spectral_criteria.py
"""
import argparse
import copy

import numpy as np

from rigidity import (estimation_error_of, extended_bearing_rigidity_matrix as B_of,
                      greedy_rigid_construction)
from scenario import random_scenario

DOMAINS = {
    "n8 R^3": ["R^3"] * 8,
    "n8 SE(3)": ["SE(3)"] * 8,
    "n8 R^2xS^1": ["R^2xS^1"] * 8,
    "mixed n=10": ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"] * 2,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, default=12)
    ap.add_argument("--graphs", type=int, default=6, help="greedy constructions per pose set")
    ap.add_argument("--extra", type=int, default=3, help="random extra edges after each")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("A = tr((B^T B)^+), E = 1/lambda, D = -sum log w.  All on the")
    print("length-normalised B, so none of them tracks the pose range.\n")
    head = (f"{'config':12s} {'N':>5s} | {'corr(logE,logA)':>16s} {'corr(logE,D)':>13s} "
            f"| {'p10-p90 logA':>13s} {'p10-p90 logE':>13s} {'p10-p90 D':>11s} "
            f"| {'median A*lam':>12s}")
    print(head)
    print("-" * len(head))

    for tag, doms in DOMAINS.items():
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        A, E, D = [], [], []

        for _ in range(args.poses):
            base, _ = random_scenario(len(doms), doms, edge_count=0)
            rank_K = int(np.linalg.matrix_rank(B_of(base.fully_connected())))
            for _ in range(args.graphs):
                work = copy.deepcopy(base)
                greedy_rigid_construction(work, rank_K, rng)
                for _ in range(args.extra + 1):
                    a, e, d = estimation_error_of(work, rank_K)
                    if np.isfinite(a):
                        A.append(a)
                        E.append(e)
                        D.append(d)
                    i, j = rng.choice(work.n, 2, replace=False)
                    work.edges[i, j] = True

        la, le, dd = np.log10(A), np.log10(E), np.array(D)

        def corr(x, y):
            return float(np.corrcoef(x, y)[0, 1])

        def spread(v):
            return float(np.percentile(v, 90) - np.percentile(v, 10))

        # A is dominated by the softest mode exactly when this ratio is near 1
        ratio = float(np.median(np.array(A) / np.array(E)))
        print(f"{tag:12s} {len(A):5d} | {corr(le, la):16.4f} {corr(le, dd):13.4f} "
              f"| {spread(la):13.2f} {spread(le):13.2f} {spread(dd):11.2f} "
              f"| {ratio:12.2f}")

    print("\nmedian A*lam is the share of the whole trace the softest mode carries:")
    print("near 1 means A and E are the same statistic wearing different clothes.")


if __name__ == "__main__":
    main()
