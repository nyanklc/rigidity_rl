"""Are the two objectives submodular? Reproduces THEORY.md section 14.

A set function has diminishing returns (is submodular) when an element is worth
less once you already have more:

    f(S + e) - f(S)  >=  f(T + e) - f(T)        for S subset of T, e not in T

It matters because minimum-cost cover of a monotone submodular function is a named
problem whose greedy algorithm has a proven approximation ratio. rank(B_S) has that
structure and the rigidity margin does not, which is why greedy is near-optimal on
edge count and unguaranteed on the margin.

    PYTHONPATH=. uv run tools/submodularity.py
    PYTHONPATH=. uv run tools/submodularity.py --trials 80   # tighter, slower
"""
import argparse

import numpy as np

from rigidity import extended_bearing_rigidity_matrix as B_of
from scenario import random_scenario

DOMAINS = ["R^2", "R^3", "R^2xS^1", "R^3xS^1", "SE(3)"]
MIX = ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"]


def set_edges(net, S):
    E = np.zeros((net.n, net.n), dtype=bool)
    for (i, j) in S:
        E[i, j] = True
    net.edges = E


def f_rank(net, S, rank_K=None):
    set_edges(net, S)
    return 0 if not S else int(np.linalg.matrix_rank(B_of(net)))


def f_margin(net, S, rank_K):
    """The rigidity eigenvalue: eigenvalue 6n - rank_K of B^T B, 0 when not rigid."""
    set_edges(net, S)
    B = B_of(net)
    if B.size == 0:
        return 0.0
    w = np.linalg.eigvalsh(B.T @ B)
    idx = 6 * net.n - rank_K
    return float(w[idx]) if 0 <= idx < len(w) else 0.0


def sweep(f, dense, trials, tol, seed=0):
    """(#tested, #violations, worst gap) over random S subset T and e outside T."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    tested = viol = 0
    worst = 0.0
    for dom in DOMAINS + [MIX]:
        n = 5 if isinstance(dom, list) else 6
        for _ in range(8):
            net, _ = random_scenario(n, dom if isinstance(dom, str) else list(dom))
            pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
            net.edges = np.ones((n, n), dtype=bool)
            np.fill_diagonal(net.edges, False)
            rank_K = int(np.linalg.matrix_rank(B_of(net)))

            for _ in range(trials):
                perm = list(rng.permutation(len(pairs)))
                lo = len(pairs) // 2 if dense else 1
                k = int(rng.integers(lo, len(pairs) - 2))
                T = [pairs[p] for p in perm[:k]]
                S = T[:len(T) - max(1, len(T) // 5)] if dense else T[:max(1, k // 2)]
                e = pairs[perm[k]]

                fS, fT = f(net, S, rank_K), f(net, T, rank_K)
                # a margin comparison is only meaningful where both are rigid
                if dense and min(fS, fT) <= 1e-9:
                    continue
                gap = (f(net, S + [e], rank_K) - fS) - (f(net, T + [e], rank_K) - fT)
                tested += 1
                viol += gap < -tol
                worst = min(worst, gap)
    return tested, viol, worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40, help="triples per instance")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("diminishing returns:  f(S+e) - f(S)  >=  f(T+e) - f(T)   for S subset of T\n")
    print(f"  {'objective':22s} {'tested':>8s} {'violations':>12s} {'worst gap':>12s}")

    t, v, w = sweep(f_rank, dense=False, trials=args.trials, tol=0, seed=args.seed)
    print(f"  {'rank(B_S)':22s} {t:8d} {f'{v} ({100*v/max(t,1):.1f}%)':>12s} {w:12.3e}")

    t2, v2, w2 = sweep(f_margin, dense=True, trials=args.trials, tol=1e-9, seed=args.seed)
    print(f"  {'rigidity margin':22s} {t2:8d} {f'{v2} ({100*v2/max(t2,1):.1f}%)':>12s} {w2:12.3e}")

    print("\n  rank(B_S) is submodular (proved in THEORY.md 14.2, confirmed here), so minimum-edge")
    print("  rigidity is minimum submodular cover and greedy is an H(c_max) approximation.")
    print("  The margin is not, so greedy carries no guarantee there. See THEORY.md section 14.")
