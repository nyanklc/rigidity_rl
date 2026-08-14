"""Randomized constructive greedy: the honest classical baseline.

Starts from the empty graph and adds any edge that raises rank(B), stopping at
rank_K. The order is randomised, so restarts explore different maximal
independent sets. With enough restarts this is a strong reference point, and it
is the one a learned policy has to beat. phi-greedy hill-climbing from a random
start is a much weaker opponent because it gets trapped where phi has local
optima.

Two things it measures that matter for the thesis:

  - whether the problem is a matroid. If every restart terminates at the same
    edge count, any greedy solves it optimally and there is nothing for RL to
    win. This happens exactly in the c_max = 1 domains.
  - how the gap to the optimum grows with n, which is where a learned policy
    could earn its place.

    PYTHONPATH=. uv run tools/constructive_greedy.py --domain R^3 --n 8
    PYTHONPATH=. uv run tools/constructive_greedy.py --scenario mixed5
    PYTHONPATH=. uv run tools/constructive_greedy.py --sweep
"""
import argparse
import time

import numpy as np

from rigidity import extended_bearing_rigidity_matrix as B_of, required_edge_count
from scenario import load_scenario, random_scenario


def build(net, rank_K, rng):
    """One restart. Returns (edge set, edge count, reached rank_K)."""
    n = net.n
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    E = np.zeros((n, n), dtype=bool)
    r, m, progress = 0, 0, True
    while r < rank_K and progress:
        progress = False
        rng.shuffle(pairs)
        for i, j in pairs:
            if E[i, j]:
                continue
            E[i, j] = True
            net.edges = E.copy()
            new_r = np.linalg.matrix_rank(B_of(net))
            if new_r > r:
                r, m, progress = new_r, m + 1, True
            else:
                E[i, j] = False
    return E, m, r == rank_K


def solve(net, restarts=20, rng=None):
    """Best of `restarts` restarts. Returns (best edge set, counts per restart)."""
    rng = rng or np.random.default_rng(0)
    K = net.fully_connected()
    rank_K = np.linalg.matrix_rank(B_of(K))
    best_E, counts = None, []
    for _ in range(restarts):
        E, m, ok = build(net, rank_K, rng)
        counts.append(m if ok else None)
        if ok and (best_E is None or m < min(c for c in counts if c is not None)):
            best_E = E.copy()
    net.edges = best_E if best_E is not None else np.zeros_like(net.edges)
    return best_E, counts


def report(net, label, restarts, rng):
    K = net.fully_connected()
    BK = B_of(K)
    rank_K = np.linalg.matrix_rank(BK)
    m_req = required_edge_count(net, rank_K=rank_K, brmat_K=BK)

    t0 = time.time()
    _, counts = solve(net, restarts=restarts, rng=rng)
    dt = time.time() - t0
    ok = [c for c in counts if c is not None]
    if not ok:
        print(f"  {label:22s} no restart reached rank_K")
        return

    best = [min(ok[:k]) for k in (1, 5, restarts) if k <= len(ok)]
    matroid = len(set(ok)) == 1
    print(f"  {label:22s} rank_K {rank_K:3d}  m_req {m_req:3d}  "
          f"k=1 {best[0]:3d}  k=5 {best[1] if len(best) > 1 else best[-1]:3d}  "
          f"k={restarts} {best[-1]:3d}  "
          f"spread {min(ok)}-{max(ok)}  {'matroid' if matroid else 'order matters'}"
          f"  {dt:5.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="R^3")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--scenario", default=None, help="use scenarios/<name>.json instead")
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="all five domains at several n, to show where it is a matroid")
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("randomized constructive greedy, best of k restarts\n")
    if args.sweep:
        for d in ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"]:
            for n in (6, 8, 12):
                net, _ = random_scenario(n, d)
                report(net, f"{d} n={n}", args.restarts, rng)
        print("\n  c_max = 1 domains (R^2, R^2xS^1) terminate at m_req on every restart:")
        print("  the independent sets form a matroid there and greedy is already optimal.")
    elif args.scenario:
        net, _ = load_scenario(f"scenarios/{args.scenario}.json")
        doms = [a.domain for a in net.agents]
        for k in range(args.instances):
            fresh, _ = random_scenario(net.n, doms)
            report(fresh, f"{args.scenario} #{k}", args.restarts, rng)
    else:
        for k in range(args.instances):
            net, _ = random_scenario(args.n, args.domain)
            report(net, f"{args.domain} n={args.n} #{k}", args.restarts, rng)
