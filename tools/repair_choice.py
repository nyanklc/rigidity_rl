"""Does it matter WHICH minimum-size repair you pick?

After a formation breaks, several different edge sets of the same minimum size
restore rigidity. If they all recover the shape about as well, the choice is free
and a learned policy has nothing to win here beyond the edge count. If they
differ by decades, the choice is the whole point.

Enumerates every minimum-size repair at small n, scores each by shape error, and
reports the spread together with where the marginal-gain greedy lands in it.

    PYTHONPATH=. uv run tools/repair_choice.py [--trials 12] [--drop 2]
"""
import argparse
import copy
import itertools

import numpy as np

from rigidity import (estimation_error_of, extended_bearing_rigidity_matrix as B_of,
                      greedy_rigid_construction, greedy_rigid_repair,
                      repair_edge_count)
from scenario import random_scenario

CONFIGS = {
    "R^2": ["R^2"] * 6,
    "R^3": ["R^3"] * 6,
    "R^2xS^1": ["R^2xS^1"] * 5,
    "SE(3)": ["SE(3)"] * 5,
    "mixed": ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"],
}


def shape_err(net, rank_K):
    a_opt, _, _ = estimation_error_of(net, rank_K)
    return np.sqrt(a_opt / net.n) if np.isfinite(a_opt) else np.inf


def valid_repairs(net, rank_K, size, cap=20000):
    """Every set of `size` absent pairs that restores rigidity."""
    n = net.n
    absent = [(i, j) for i in range(n) for j in range(n)
              if i != j and not net.edges[i, j]]
    out = []
    for count, sub in enumerate(itertools.combinations(absent, size)):
        if count >= cap:
            break
        work = copy.deepcopy(net)
        for i, j in sub:
            work.edges[i, j] = True
        if np.linalg.matrix_rank(B_of(work)) >= rank_K:
            out.append((sub, work))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--drop", type=int, default=2, help="edges removed to break it")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"breaking a minimal graph by dropping {args.drop} edges, then enumerating")
    print("every minimum-size repair. shape error is RMS state error per radian of")
    print("bearing noise, so the ratio is what choosing badly costs.\n")
    head = (f"{'config':10s} {'cases':>6s} {'repairs':>9s} | {'worst/best':>17s} "
            f"| {'greedy pctile':>14s} {'greedy/best':>12s}")
    print(head)
    print("-" * len(head))

    for tag, doms in CONFIGS.items():
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        n = len(doms)
        ratios, pctiles, greedy_ratio, counts = [], [], [], []

        for t in range(args.trials):
            net, _ = random_scenario(n, doms, edge_count=0)
            rank_K = int(np.linalg.matrix_rank(B_of(net.fully_connected())))
            greedy_rigid_construction(net, rank_K, rng)

            present = list(zip(*np.nonzero(net.edges)))
            if len(present) <= args.drop:
                continue
            for idx in rng.choice(len(present), args.drop, replace=False):
                net.edges[present[idx]] = False
            if np.linalg.matrix_rank(B_of(net)) >= rank_K:
                continue

            size = repair_edge_count(net, rank_K=rank_K)
            found = valid_repairs(net, rank_K, size)
            if len(found) < 2:
                continue

            errs = np.array([shape_err(w, rank_K) for _, w in found])
            errs = errs[np.isfinite(errs) & (errs > 0)]
            if len(errs) < 2:
                continue

            work = copy.deepcopy(net)
            _, added = greedy_rigid_repair(work, rank_K, rng=np.random.default_rng(t))
            g = shape_err(work, rank_K)

            counts.append(len(errs))
            ratios.append(errs.max() / errs.min())
            if len(added) == size and np.isfinite(g):
                pctiles.append(100.0 * (errs < g).mean())
                greedy_ratio.append(g / errs.min())

        if ratios:
            print(f"{tag:10s} {len(ratios):6d} {np.mean(counts):9.0f} | "
                  f"{np.exp(np.mean(np.log(ratios))):6.1f}x  "
                  f"(max {max(ratios):6.1f}x) | "
                  f"{np.mean(pctiles) if pctiles else float('nan'):13.0f}% "
                  f"{np.exp(np.mean(np.log(greedy_ratio))) if greedy_ratio else float('nan'):11.2f}x")

    print("\nworst/best is the geometric mean over instances of the spread among")
    print("equally-sized repairs; lower shape error is better. greedy pctile is where")
    print("the marginal-gain repair sits in that distribution -- 0% would be optimal,")
    print("50% is no better than picking at random among valid repairs.")


if __name__ == "__main__":
    main()
