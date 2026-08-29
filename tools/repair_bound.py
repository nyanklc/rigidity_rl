"""Is the repair bound sound, and is it attained?

repair_edge_count bounds below how many edges could restore rigidity to a broken
graph. Subadditivity says nothing smaller can work; it does not say something
that size exists. Both are checked against exhaustive search.

    PYTHONPATH=. uv run tools/repair_bound.py [--trials 20] [--cap 6]
"""
import argparse
import copy
import itertools

import numpy as np

from rigidity import extended_bearing_rigidity_matrix as B_of, repair_edge_count
from scenario import random_scenario

CONFIGS = {
    "R^2": ["R^2"] * 5,
    "R^3": ["R^3"] * 5,
    "R^2xS^1": ["R^2xS^1"] * 5,
    "R^3xS^1": ["R^3xS^1"] * 5,
    "SE(3)": ["SE(3)"] * 5,
    "mix R2/R3/SE3": ["R^2", "R^3", "SE(3)", "R^3xS^1"],
    "mix planar": ["R^2", "R^2", "R^2xS^1", "R^3"],
}


def true_minimum(net, rank_K, cap):
    """Smallest number of added edges that restores rigidity, by exhaustive search."""
    n = net.n
    absent = [(i, j) for i in range(n) for j in range(n)
              if i != j and not net.edges[i, j]]
    for k in range(1, min(cap, len(absent)) + 1):
        for sub in itertools.combinations(absent, k):
            work = copy.deepcopy(net)
            for i, j in sub:
                work.edges[i, j] = True
            if np.linalg.matrix_rank(B_of(work)) >= rank_K:
                return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--cap", type=int, default=6, help="deepest exhaustive level")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("bound vs exhaustive search over broken graphs.")
    print("sound = no smaller set restores rigidity; attained = the bound IS the minimum.\n")
    head = (f"{'config':16s} {'n':>2s} {'cases':>6s} {'sound':>11s} {'attained':>11s} "
            f"{'mean gap':>9s}")
    print(head)
    print("-" * len(head))

    for tag, doms in CONFIGS.items():
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        n = len(doms)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        cases = sound = attained = 0
        gaps = []

        for _ in range(args.trials):
            net, _ = random_scenario(n, doms, edge_count=0)
            rank_K = int(np.linalg.matrix_rank(B_of(net.fully_connected())))
            net.edges[:] = False
            keep = int(rng.integers(1, len(pairs) // 2 + 2))
            for idx in rng.choice(len(pairs), keep, replace=False):
                net.edges[pairs[idx]] = True

            brm = B_of(net)
            if np.linalg.matrix_rank(brm) >= rank_K:
                continue
            lb = repair_edge_count(net, rank_K=rank_K, brmat=brm)
            tm = true_minimum(net, rank_K, args.cap)
            if tm is None:
                continue
            cases += 1
            sound += int(lb <= tm)
            attained += int(lb == tm)
            gaps.append(tm - lb)

        if cases:
            print(f"{tag:16s} {n:2d} {cases:6d} {sound:>6d}/{cases:<4d} "
                  f"{attained:>6d}/{cases:<4d} {np.mean(gaps):9.2f}")

    print("\nSoundness is the property the bound must have. A gap of 0 means it is also")
    print("the true minimum, which Karimian and Tron prove in homogeneous 2-D and which")
    print("is evidence rather than proof anywhere else.")


if __name__ == "__main__":
    main()
