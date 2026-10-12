"""The per-node attitude-observability condition, and what it catches.

An agent's own rotation is a motion of the framework like any other. If agent i's
outgoing bearings cannot see it, that rotation sits in ker(B) without being a trivial
motion, so the framework is not IBR. The condition is local: it reads only i's own
domain and its own outgoing bearings.

    range(P_i)  cap  intersection_{j in O_i} span{p_hat_ij}  =  {0}          (necessary)

equivalently  rank( vstack_{j in O_i} [p_hat_ij]_x P_i ) == rank(P_i), with the empty
intersection read as R^3 so a node with no out-edges and P_i != 0 always fails.

Reproduces the numbers quoted in THEORY.md and in the thesis chapter on rigidity
analysis. Presenza et al. (arXiv:2604.11754) prove the homogeneous SE(d) case as
"dim B_i >= d - 1"; this is that condition written per node, so it also covers the
oriented and mixed domains.

    OMP_NUM_THREADS=1 PYTHONPATH=. uv run tools/attitude_observability.py
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from scenario import random_scenario                        # noqa: E402
from rigidity import (node_dof_projectors, is_IBR,           # noqa: E402
                      extended_bearing_rigidity_matrix, required_edge_count)
from util import skew_symmetric                             # noqa: E402

TOL = 1e-8
DOMAINS = ["R^2", "R^3", "R^2xS^1", "R^3xS^1", "SE(3)"]
MIXES = dict({d: d for d in DOMAINS}, **{
    "mixed": DOMAINS,
    "R^2+SE(3)": ["R^2", "SE(3)"] * 3,
    "R^2xS^1+R^3xS^1": ["R^2xS^1", "R^3xS^1"] * 3,
})
# minimum out-degree the condition forces, per domain (0 = no rotational DOF)
MIN_OUTDEG = {"R^2": 0, "R^3": 0, "R^2xS^1": 1, "R^3xS^1": 1, "SE(3)": 2}


def attitude_pinned(network, i):
    """Can agent i's own rotation be seen by the bearings agent i measures?"""
    _, P_i = node_dof_projectors(network.agents[i])
    rank_P = np.linalg.matrix_rank(P_i, tol=TOL)
    if rank_P == 0:                      # R^2 / R^3: no attitude to pin
        return True
    out = [j for j in range(network.n) if network.edges[i, j] and j != i]
    if not out:
        return False
    blocks = []
    for j in out:
        p_ij = network.agents[j].pose.position - network.agents[i].pose.position
        blocks.append(skew_symmetric(p_ij / np.linalg.norm(p_ij)) @ P_i)
    return np.linalg.matrix_rank(np.vstack(blocks), tol=TOL) == rank_P


def condition_holds(network):
    return all(attitude_pinned(network, i) for i in range(network.n))


def main(trials=200, n=6, seed=0):
    # random_scenario draws the poses from the global stream, so seeding only the
    # local generator left the census a different number on every run while the
    # invariant it measures stayed put. Both are seeded now.
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    tot = viol = ibr = ibr_and_viol = held_not_ibr = 0
    at_bound_not_ibr = at_bound_caught = 0

    for name, doms in MIXES.items():
        k = n if isinstance(doms, str) else len(doms)
        for _ in range(trials):
            m = int(rng.integers(1, k * (k - 1)))
            net, _ = random_scenario(k, doms if isinstance(doms, str) else list(doms),
                                     edge_count=m)
            brmat_K = extended_bearing_rigidity_matrix(net.fully_connected())
            rank_K = np.linalg.matrix_rank(brmat_K)
            is_rigid = bool(is_IBR(net, rank_K=rank_K)[0])
            ok = condition_holds(net)

            tot += 1
            viol += (not ok)
            ibr += is_rigid
            ibr_and_viol += (is_rigid and not ok)
            held_not_ibr += (ok and not is_rigid)

            # what the local test catches that the count bound cannot
            m_req = required_edge_count(net, rank_K=rank_K, brmat_K=brmat_K)
            if m >= m_req and not is_rigid:
                at_bound_not_ibr += 1
                at_bound_caught += (not ok)

    print(f"networks                     {tot}")
    print(f"  IBR                        {ibr}")
    print(f"  condition violated         {viol}")
    print(f"  IBR and violated           {ibr_and_viol}   <- must be 0 (the claim)")
    print(f"  condition held, not IBR    {held_not_ibr}   <- necessary, not sufficient")
    print()
    print("non-rigid graphs that already have m >= m_req, so the count bound says nothing:")
    print(f"  {at_bound_not_ibr}, of which the local test rejects {at_bound_caught} "
          f"({100.0 * at_bound_caught / max(at_bound_not_ibr, 1):.0f}%)")
    print()
    print("forced minimum out-degree, and Sum_i r_i against m_req at n = 8, 16:")
    for d in DOMAINS:
        r = MIN_OUTDEG[d]
        row = f"  {d:9s} r_i = {r}"
        for nn in (8, 16):
            net, _ = random_scenario(nn, d, edge_count=nn)
            brmat_K = extended_bearing_rigidity_matrix(net.fully_connected())
            rank_K = np.linalg.matrix_rank(brmat_K)
            row += (f"   n={nn}: sum r_i = {r * nn:3d}  m_req = "
                    f"{required_edge_count(net, rank_K=rank_K, brmat_K=brmat_K):3d}")
        print(row)
    return 1 if ibr_and_viol else 0


if __name__ == "__main__":
    sys.exit(main())
