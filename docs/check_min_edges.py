"""Verify every number in `min_edges_note.pdf`.

Four checks, all deterministic under one seed:
  1. Trinh et al. Thm 4.1's f(n,d) equals ceil((dn-d-1)/(d-1))
  2. the greedy block-rank bound reduces to f(n,d) on homogeneous R^d
  3. the bound is attained at n=4, by exhaustive search over edge sets
  4. the current-graph variant of the bound can report a false minimum

    PYTHONPATH=.. python3 check_min_edges.py
    PYTHONPATH=.. python3 check_min_edges.py --quick   # skip the exhaustive search
"""
import argparse
import itertools
import math

import numpy as np

from rigidity import (MBR_required_Rd, edge_block_ranks, required_edge_count,
                      extended_bearing_rigidity_matrix as B_of)
from scenario import random_scenario

ROWS = []
MIXES = [
    (['R^2', 'R^2', 'R^3', 'R^3'], 'R^2 x2 + R^3 x2'),
    (['R^2', 'R^3', 'R^2xS^1', 'SE(3)'], 'one of four kinds'),
    (['R^2', 'R^2', 'R^2', 'SE(3)'], 'R^2 x3 + SE(3)'),
    (['R^3', 'R^3xS^1', 'SE(3)', 'R^2xS^1'], 'three spatial kinds'),
]
HOMOG = ['R^2', 'R^3', 'R^2xS^1', 'R^3xS^1', 'SE(3)']


def claim(what, stated, measured, ok):
    ROWS.append((what, str(stated), str(measured), "ok" if ok else "FAIL"))


def greedy_bound(block_ranks, rank_K):
    """eq. (1) of the note: accumulate the largest per-edge contributions."""
    total = m = 0
    for c in sorted(block_ranks, reverse=True):
        if c == 0:
            break
        total += c
        m += 1
        if total >= rank_K:
            break
    return max(m, 1)


def check_closed_form():
    print("\n1. f(n,d) of Trinh Thm 4.1 vs ceil((dn-d-1)/(d-1))")
    agree = total = 0
    for d in (2, 3):
        for n in range(3, 25):
            total += 1
            agree += MBR_required_Rd(n, d) == math.ceil((d*n - d - 1) / (d - 1))
    print(f"   d in {{2,3}}, n = 3..24: {agree}/{total} agree")
    claim("f(n,d) = ceil((dn-d-1)/(d-1))", "44/44", f"{agree}/{total}", agree == total)


def check_reduces(rng):
    print("\n2. the greedy bound reduces to f(n,d) on homogeneous R^d")
    agree = total = 0
    for dom, d in (('R^2', 2), ('R^3', 3)):
        for n in (4, 5, 6, 8):
            for _ in range(3):
                net, _ = random_scenario(n, dom)
                BK = B_of(net.fully_connected())
                rK = int(np.linalg.matrix_rank(BK))
                total += 1
                agree += greedy_bound(edge_block_ranks(BK), rK) == MBR_required_Rd(n, d)
    print(f"   R^2 and R^3 at n = 4,5,6,8: {agree}/{total} agree")
    claim("greedy bound reduces to f(n,d)", "24/24", f"{agree}/{total}", agree == total)


def min_rigid_m(net, rank_K, m_req, extra=3):
    """Smallest edge set admitting rigidity, searched from m_req upward."""
    n = net.n
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    for m in range(m_req, m_req + extra + 1):
        for S in itertools.combinations(pairs, m):
            E = np.zeros((n, n), dtype=bool)
            for (i, j) in S:
                E[i, j] = True
            net.edges = E
            if np.linalg.matrix_rank(B_of(net)) == rank_K:
                return m
    return None


def check_attained():
    print("\n3. is the bound attained?  exhaustive search at n = 4")
    print(f"   {'configuration':24s} {'rank_K':>6s} {'m_req':>6s} {'min m':>6s}  verdict")
    attained = total = 0
    for dom in HOMOG + [m[0] for m in MIXES]:
        label = dom if isinstance(dom, str) else \
            next(m[1] for m in MIXES if m[0] == dom)
        for t in range(2):
            net, _ = random_scenario(4, dom if isinstance(dom, str) else list(dom))
            BK = B_of(net.fully_connected())
            rK = int(np.linalg.matrix_rank(BK))
            mr = int(required_edge_count(net, rank_K=rK, brmat_K=BK))
            got = min_rigid_m(net, rK, mr)
            total += 1
            attained += got == mr
            print(f"   {label if t == 0 else '':24s} {rK:6d} {mr:6d} "
                  f"{str(got):>6s}  {'attained' if got == mr else 'NOT attained'}")
    claim("m_req attained at n=4", "18/18", f"{attained}/{total}", attained == total)


def check_current_graph_variant(rng):
    print("\n4. the current-graph variant of the bound: does it report a false minimum?")
    false_min = rigid = 0
    for doms in (['R^2', 'R^2', 'R^3', 'R^3', 'SE(3)'],
                 ['R^2', 'R^2', 'R^2', 'R^3', 'R^3']):
        for _ in range(60):
            net, _ = random_scenario(5, list(doms))
            n = 5
            BK = B_of(net.fully_connected())
            rK = int(np.linalg.matrix_rank(BK))
            m_true = int(required_edge_count(net, rank_K=rK, brmat_K=BK))
            pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
            size = int(rng.integers(m_true, m_true + 4))
            E = np.zeros((n, n), dtype=bool)
            for k in rng.choice(len(pairs), size=size, replace=False):
                E[pairs[k]] = True
            net.edges = E
            B = B_of(net)
            if np.linalg.matrix_rank(B) != rK:
                continue
            rigid += 1
            m_local = greedy_bound(edge_block_ranks(B), rK)
            if int(E.sum()) == m_local and int(E.sum()) > m_true:
                false_min += 1
    print(f"   {false_min} false minima in {rigid} rigid graphs")
    claim("current-graph variant false minima", "0/48", f"{false_min}/{rigid}",
          false_min == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the exhaustive search")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    check_closed_form()
    check_reduces(rng)
    if not args.quick:
        check_attained()
    check_current_graph_variant(rng)

    print("\n" + "=" * 72)
    print(f"  {'claim':44s} {'note':>10s} {'measured':>10s}")
    print("-" * 72)
    for what, stated, got, status in ROWS:
        print(f"  {what:44.44s} {stated:>10s} {got:>10s}  {status}")
    fails = [r for r in ROWS if r[3] == "FAIL"]
    print("-" * 72)
    print(f"  {len(ROWS)-len(fails)}/{len(ROWS)} reproduce"
          + (f"   FAILURES: {len(fails)}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
