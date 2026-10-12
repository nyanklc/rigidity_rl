"""What is the reverse of an edge worth, and why does the answer depend on the domain?

    OMP_NUM_THREADS=1 PYTHONPATH=. uv run tools/reverse_edge.py

In a frameless domain the row block of j -> i is exactly -b_ij: the projector and the
scale are shared, P(-p_hat) = P(p_hat) and s_ji = s_ij, and R_i = R_j = I, so the two
blocks differ only in which endpoint carries the sign. A reciprocal pair therefore adds
no rank over one of its two edges. Once an agent carries a frame the attitude term sits
in the measuring node's columns alone, those columns differ between the two directions,
and the cancellation fails.

Three columns per domain, all against rebuilt-matrix ground truth:

  max |b_ji + b_ij|   0 exactly in R^2 and R^3, order 1 in the oriented domains
  reverse adds        rank gained by adding j -> i when i -> j is already present
  fresh pair adds     rank gained by an edge between two untouched agents, i.e. c_max

Reproduces the table in the thesis chapter on rigidity analysis, and the claim the
proposition there makes. Exit code is nonzero if any measured value contradicts it.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from rigidity import (candidate_block, candidate_gain,                # noqa: E402
                      characteristic_length, extended_bearing_rigidity_matrix,
                      nullspace, nullspace_in_scaled_units)
from scenario import random_scenario                                  # noqa: E402

DOMAINS = ("R^2", "R^3", "R^2xS^1", "R^3xS^1", "SE(3)")
# what the proposition and the per-domain reading predict
REVERSE = {"R^2": 0, "R^3": 0, "R^2xS^1": 1, "R^3xS^1": 1, "SE(3)": 2}
FRESH = {"R^2": 1, "R^3": 2, "R^2xS^1": 1, "R^3xS^1": 2, "SE(3)": 2}


def one_edge_network(n, domain, seed):
    """A formation carrying the single edge 0 -> 1."""
    np.random.seed(seed)
    net, _ = random_scenario(n, domain, edge_count=0)
    net.edges = np.zeros((n, n), dtype=bool)
    net.edges[0, 1] = True
    return net


def rank_of(net):
    B = extended_bearing_rigidity_matrix(net)
    return int(np.linalg.matrix_rank(B)) if B.size else 0


def main(n=6, trials=8):
    print(f"{'domain':10s} {'max|b_ji + b_ij|':>17s} {'reverse adds':>13s} "
          f"{'fresh pair adds':>16s} {'predicted':>20s}")
    bad = 0
    for dom in DOMAINS:
        worst, rev, fresh = 0.0, set(), set()
        for sd in range(trials):
            net = one_edge_network(n, dom, sd)

            # the two row blocks, built by the matrix routine itself
            b_ij = candidate_block(net, 0, 1)
            b_ji = candidate_block(net, 1, 0)
            worst = max(worst, float(np.abs(b_ji + b_ij).max()))

            # the exact addition criterion, and ground truth by rebuilding
            r0 = rank_of(net)
            L = characteristic_length(net)
            Z = nullspace_in_scaled_units(nullspace(
                extended_bearing_rigidity_matrix(net), r0), net.n, L)
            gain = candidate_gain(net, Z, length_scale=L)[1]
            for (i, j), store in (((1, 0), rev), ((2, 3), fresh)):
                store.add(int(gain[i, j]))
                net.edges[i, j] = True
                store.add(rank_of(net) - r0)          # ground truth
                net.edges[i, j] = False

        ok = rev == {REVERSE[dom]} and fresh == {FRESH[dom]}
        bad += int(not ok)
        print(f"{dom:10s} {worst:17.2e} {str(sorted(rev)):>13s} {str(sorted(fresh)):>16s} "
              f"{'ok' if ok else 'MISMATCH':>20s}")

    print()
    print("A set with more than one entry means the criterion and the rebuilt matrix")
    print("disagreed, or that the value moved between configurations. Both would")
    print("contradict the proposition.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
