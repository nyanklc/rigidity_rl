"""Closed forms for rank_K, c_max and m_req on an arbitrary domain mix, checked
exactly against the matrix computation.

    OMP_NUM_THREADS=1 PYTHONPATH=.:tools uv run tools/domain_closed_forms.py

Needed because the complete graph's rigidity matrix allocates a dense (3m, 3m)
block at m = n^2, which is 7.2 GB at n=100. If these hold, the scale analysis can
go to any n.

    dim D_i   R^2 2, R^3 3, R^2xS^1 3, R^3xS^1 4, SE(3) 6
    rank_K  = sum_i dim D_i - (3 if any agent is planar else 4)
    c_k     = 1 iff both endpoints are planar, else 2
    m_req   = greedy accumulation of the sorted c_k until rank_K is reached
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from scenario import random_scenario
from network import DOMAINS
import rigidity as R

DOF = {"R^2": 2, "R^3": 3, "R^2xS^1": 3, "R^3xS^1": 4, "SE(3)": 6}
PLANAR = {"R^2", "R^2xS^1"}


FRAMELESS = {"R^2", "R^3"}


def trivial_dim(domains):
    """Translations + uniform scaling + whatever of SO(3) every frame absorbs.

    A global rotation is trivial only if every agent's frame turns with the world.
    An R^d agent has none, so a rotation changes its global-frame bearings; an
    R^dxS^1 agent has one axis, so only rotations about that shared axis survive.
    Assumes the S^1 axes agree, which is the only case any scenario builds.
    """
    translations = 2 if any(d in PLANAR for d in domains) else 3
    if any(d in FRAMELESS for d in domains):
        rotations = 0
    elif all(d == "SE(3)" for d in domains):
        rotations = 3
    else:
        rotations = 1
    return translations + 1 + rotations


def predict(domains):
    n = len(domains)
    planar = [d in PLANAR for d in domains]
    rank_K = sum(DOF[d] for d in domains) - trivial_dim(domains)
    c_max = 1 if all(planar) else 2
    p = sum(planar)
    n_one = p * (p - 1)                      # ordered pairs with both ends planar
    n_two = n * (n - 1) - n_one
    # greedy takes the rank-2 blocks first
    got, m = 0, 0
    for c, count in ((2, n_two), (1, n_one)):
        if got >= rank_K:
            break
        need = int(np.ceil((rank_K - got) / c))
        take = min(need, count)
        got += take * c
        m += take
    return rank_K, c_max, max(m, 1)


def measure(net):
    K = net.fully_connected()
    BK = R.extended_bearing_rigidity_matrix(K)
    rank_K = int(np.linalg.matrix_rank(BK))
    blocks = R.edge_block_ranks(BK)
    c_max = R.max_edge_rank(net, brmat_K=BK)
    m_req = R.required_edge_count(net, rank_K=rank_K, brmat_K=BK, block_ranks=blocks)
    return rank_K, int(c_max), int(m_req), blocks


def _validate():
    rng = np.random.default_rng(0)
    bad = {"rank_K": 0, "c_max": 0, "m_req": 0, "c_k rule": 0}
    cases = 0
    print(f"{'n':>4s} {'composition':38s} {'rank_K':>13s} {'c_max':>7s} {'m_req':>11s}")
    ORIENTED = ["R^2xS^1", "R^3xS^1", "SE(3)"]
    SPATIAL = ["R^3", "R^3xS^1", "SE(3)"]
    for n in (4, 5, 6, 8, 10, 12, 16, 20, 24):
        for trial in range(24):
            np.random.seed(n * 100 + trial)
            if trial < 5:
                doms = [DOMAINS[trial]] * n                       # the five corners
            elif trial < 11:
                doms = [d for d in rng.choice(ORIENTED, size=n)]  # no frameless agent
            elif trial < 15:
                doms = [d for d in rng.choice(SPATIAL, size=n)]   # nothing planar
            elif trial < 18:
                doms = [d for d in rng.choice(["R^3xS^1", "SE(3)"], size=n)]
            elif trial < 21:
                doms = [d for d in rng.choice(["R^2", "R^2xS^1"], size=n)]
            else:
                doms = [d for d in rng.choice(DOMAINS, size=n)]
            net, _ = random_scenario(n, doms, edge_count=n)
            mk, mc, mm, blocks = measure(net)
            pk, pc, pm = predict(doms)
            # the per-pair rule, checked against the real block ranks
            ii, jj = np.nonzero(net.fully_connected().edges)
            pred_blocks = [1 if (doms[i] in PLANAR and doms[j] in PLANAR) else 2
                           for i, j in zip(ii, jj)]
            ok_ck = (list(map(int, blocks)) == pred_blocks)
            cases += 1
            bad["rank_K"] += mk != pk
            bad["c_max"] += mc != pc
            bad["m_req"] += mm != pm
            bad["c_k rule"] += not ok_ck
            if trial in (0, 5, 15) or mk != pk or mm != pm or not ok_ck:
                tag = "" if (mk == pk and mc == pc and mm == pm and ok_ck) else "   <-- MISMATCH"
                label = "".join(sorted(d[0] + ("p" if d in PLANAR else "") for d in set(doms)))
                print(f"{n:4d} {str(sorted(set(doms)))[:37]:38s} "
                      f"{mk:6d}/{pk:<6d} {mc:3d}/{pc:<3d} {mm:5d}/{pm:<5d}{tag}")
    
    print(f"\n{cases} configurations checked")
    for k, v in bad.items():
        print(f"  {k:10s} mismatches: {v}")
    

if __name__ == "__main__":
    _validate()
