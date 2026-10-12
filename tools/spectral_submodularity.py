"""Are the spectral criteria submodular once the edge set is already rigid?

Shames & Summers (2015, Thm. 5) state that above a minimally rigid edge set the trace of
the Gramian pseudo-inverse (A) and the log product of its nonzero eigenvalues (D) are
submodular, for distance rigidity in the plane. This checks the same statement on bearing
frameworks, alongside the rigidity eigenvalue (E):

    f(S + e) - f(S)  >=  f(T + e) - f(T)     for S subset of T, e not in T,
                                              with S (hence T) containing a rigid set

A = -tr((B_s^T B_s)^+), D = sum log w, E = lambda = min w, with w the rank_K nonzero
eigenvalues of B_s^T B_s and B_s the length-normalised matrix. All three are defined on
rigid sets only, so the flexible-to-rigid switch is outside this check by construction.

    PYTHONPATH=. uv run tools/spectral_submodularity.py
    PYTHONPATH=. uv run tools/spectral_submodularity.py --trials 30 --triples 40
"""
import argparse

import numpy as np

from rigidity import extended_bearing_rigidity_matrix, scaled_rigidity_matrix
from scenario import random_scenario

DOMAINS = ["R^3", "SE(3)", "R^2xS^1",
           ["R^2", "SE(3)", "R^3", "R^2xS^1", "R^3xS^1", "R^3"]]


def criteria(net, E, rank_K):
    """(A, D, E) of the edge set E, oriented so that larger is better."""
    net.edges = E
    w = np.linalg.svd(scaled_rigidity_matrix(net), compute_uv=False) ** 2
    w = np.sort(w)[::-1][:rank_K]
    return np.array([-np.sum(1 / w), np.sum(np.log(w)), w[-1]])


def rigid_base(net, pairs, rank_K, rng):
    """Random-order construction: keep an edge when it raises the rank, stop at rank_K."""
    S = np.zeros((net.n, net.n), dtype=bool)
    r = 0
    for k in rng.permutation(len(pairs)):
        S[pairs[k]] = True
        net.edges = S
        rr = np.linalg.matrix_rank(extended_bearing_rigidity_matrix(net))
        if rr > r:
            r = rr
        else:
            S[pairs[k]] = False
        if r == rank_K:
            return S
    raise RuntimeError("complete graph not reached rank_K")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--trials", type=int, default=15, help="formations per domain")
    ap.add_argument("--triples", type=int, default=20, help="(S, T, e) per formation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # random_scenario draws from the global stream
    names = ["A: -tr(pinv)", "D: logdet+", "E: lambda"]
    viol = np.zeros(3, int)
    worst = np.zeros(3)
    total = 0
    n = args.n
    for dom in DOMAINS:
        for _ in range(args.trials):
            net, _ = random_scenario(n, dom)
            net.edges = ~np.eye(n, dtype=bool)
            rank_K = np.linalg.matrix_rank(extended_bearing_rigidity_matrix(net))
            pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
            S0 = rigid_base(net, pairs, rank_K, rng)
            for _ in range(args.triples):
                free = [p for p in pairs if not S0[p]]
                order = rng.permutation(len(free))
                a = rng.integers(0, len(free) - 2)
                b = rng.integers(a, len(free) - 1)
                S = S0.copy()
                for t in order[:a]:
                    S[free[t]] = True
                T = S.copy()
                for t in order[a:b]:
                    T[free[t]] = True
                e = free[order[-1]]
                Se, Te = S.copy(), T.copy()
                Se[e] = Te[e] = True
                left = criteria(net, Se, rank_K) - criteria(net, S, rank_K)
                right = criteria(net, Te, rank_K) - criteria(net, T, rank_K)
                bad = left < right - 1e-9 * np.maximum(1.0, np.abs(right))
                viol += bad
                gap = (left - right) / np.maximum(np.abs(right), 1e-12)
                worst = np.where(bad, np.minimum(worst, gap), worst)
                total += 1

    print(f"n = {n}, {len(DOMAINS)} domain settings x {args.trials} formations x "
          f"{args.triples} triples = {total} triples, each S containing a rigid set")
    for k in range(3):
        print(f"  {names[k]:14s} violations {viol[k]:4d}/{total}"
              f"   worst (left-right)/|right| = {worst[k]:+.3f}")


if __name__ == "__main__":
    main()
