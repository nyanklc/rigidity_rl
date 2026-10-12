"""Every theoretical claim Chapter 4 of the thesis states, checked against the code.

    PYTHONPATH=. OMP_NUM_THREADS=1 uv run tools/verify_chapter4.py

A proof is the evidence. This is the guard against a proof that is correct about a different
matrix than the one the repository builds, which is the failure mode a derivation cannot
catch by itself. One block per numbered result:

  Prop. rank_K     q_t = (2 if any planar else 3) + 1 + q_r, and rank_K = sum(c_i) - q_t
  Prop. c_k        c_k = rank[P S_i | P S_j | [p]_x P_i], and its per-domain values
  Prop. attitude   no IBR framework violates the per-node condition; r_i = 0, 0, 1, 1, 2
  Prop. homog      B_per = B_node whenever S_i = S_j on every edge
  Thm.  rank test  rank(B) = 6n - q_v - q_i
  Prop. leverage   rank lost by deleting an edge = unit eigenvalues of b (B^T B)^+ b^T
  Prop. downdate   lambda after deletion = mu_{q+1}(B^T B - b^T b), exactly
  Thm.  addition   rank gained by adding i->j = rank(b_ij Z)
  Prop. dlambda    d/dt mu_{q+1}(G + t b^T b) at t=0 = ||b v||^2
  Prop. m_req      the edge-count bound is sound (exhaustive at n = 4)
  Prop. repair     the repair bound is sound (exhaustive at n = 4)
  counterexample   the three-agent formation, rigid but reported flexible by B_per
"""
import itertools

import numpy as np

from network import DOMAINS, Network
from rigidity import (absent_pairs, candidate_block, candidate_gain,
                      characteristic_length, edge_block_ranks,
                      extended_bearing_rigidity_matrix, node_dof_projectors,
                      nullspace, nullspace_in_scaled_units, repair_edge_count,
                      required_edge_count, rigidity_decomposition)
from scenario import random_scenario
from util import Pose, orthogonal_projection_matrix, skew_symmetric

PLANAR = {"R^2", "R^2xS^1"}
FRAMELESS = {"R^2", "R^3"}
CDIM = {"R^2": 2, "R^3": 3, "R^2xS^1": 3, "R^3xS^1": 4, "SE(3)": 6}
RMIN = {"R^2": 0, "R^3": 0, "R^2xS^1": 1, "R^3xS^1": 1, "SE(3)": 2}


def rank_of(net):
    B = extended_bearing_rigidity_matrix(net)
    return (int(np.linalg.matrix_rank(B)) if B.size else 0), B


def admissible_basis(net):
    """A, an orthonormal basis of the admissible subspace, and c = dim A."""
    n = net.n
    cols = []
    for i, agent in enumerate(net.agents):
        S_i, P_i = node_dof_projectors(agent)
        for M, off in ((S_i, 3 * i), (P_i, 3 * n + 3 * i)):
            w, V = np.linalg.eigh(M)
            for t in range(3):
                if w[t] > 0.5:
                    col = np.zeros(6 * n)
                    col[off:off + 3] = V[:, t]
                    cols.append(col)
    A = np.array(cols).T
    return A, A.shape[1]


def predicted_rank_K(doms):
    c = sum(CDIM[d] for d in doms)
    q_t = (2 if any(d in PLANAR for d in doms) else 3) + 1
    if any(d in FRAMELESS for d in doms):
        q_r = 0
    elif all(d == "SE(3)" for d in doms):
        q_r = 3
    else:
        q_r = 1
    return c - (q_t + q_r)


def mixtures(n, k, rng):
    out = [[d] * n for d in DOMAINS]
    for _ in range(k):
        out.append([DOMAINS[i] for i in rng.integers(0, 5, n)])
    return out


def grow_rigid(net, rank_K, rng):
    """A rigid graph on net's poses, grown in random order. False if none was reached."""
    n = net.n
    order = [(i, j) for i in range(n) for j in range(n) if i != j]
    rng.shuffle(order)
    E = np.zeros((n, n), dtype=bool)
    rank = 0
    for i, j in order:
        if rank >= rank_K:
            break
        E[i, j] = True
        net.edges = E
        r = rank_of(net)[0]
        if r > rank:
            rank = r
        else:
            E[i, j] = False
    net.edges = E
    return rank >= rank_K


# ---------------------------------------------------------------- rank_K
def check_rank_K():
    bad = tot = 0
    for n in (5, 7, 9):
        for doms in mixtures(n, 30, np.random.default_rng(n)):
            np.random.seed(abs(hash(tuple(doms))) % 2 ** 31)
            net, _ = random_scenario(n, doms, edge_count=0)
            got = int(np.linalg.matrix_rank(
                extended_bearing_rigidity_matrix(net.fully_connected())))
            tot += 1
            bad += int(got != predicted_rank_K([a.domain for a in net.agents]))
    print(f"rank_K closed form      {tot:5d} mixtures, disagreements: {bad}")


# ---------------------------------------------------------------- c_k
def check_block_rank():
    bad = tot = 0
    ranges = {}
    for dom in DOMAINS:
        np.random.seed(3)
        net, _ = random_scenario(6, dom, edge_count=30)
        BK = extended_bearing_rigidity_matrix(net)
        blocks = edge_block_ranks(BK)
        ranges[dom] = (int(min(blocks)), int(max(blocks)))
        p = [a.pose.position for a in net.agents]
        for k, (i, j) in enumerate(zip(*np.nonzero(net.edges))):
            ph = (p[j] - p[i]) / np.linalg.norm(p[j] - p[i])
            P = orthogonal_projection_matrix(ph)
            S_i, P_i = node_dof_projectors(net.agents[i])
            S_j, _ = node_dof_projectors(net.agents[j])
            pred = int(np.linalg.matrix_rank(
                np.hstack([P @ S_i, P @ S_j, skew_symmetric(ph) @ P_i])))
            tot += 1
            bad += int(pred != blocks[k])
    print(f"per-edge block rank     {tot:5d} edges, disagreements: {bad}")
    print(f"                              c_k range per domain: {ranges}")


# ---------------------------------------------------------------- attitude
def check_attitude():
    viol = tot = 0
    seen = {d: 99 for d in DOMAINS}
    for dom in list(DOMAINS) + ["mix"]:
        for sd in range(120):
            np.random.seed(7000 + sd)
            doms = dom if dom != "mix" else [DOMAINS[i]
                                             for i in np.random.randint(0, 5, 6)]
            net, _ = random_scenario(6, doms,
                                     edge_count=int(np.random.randint(6, 26)))
            rank_K = int(np.linalg.matrix_rank(
                extended_bearing_rigidity_matrix(net.fully_connected())))
            if rank_of(net)[0] != rank_K:
                continue
            tot += 1
            p = [a.pose.position for a in net.agents]
            for i, agent in enumerate(net.agents):
                _, P_i = node_dof_projectors(agent)
                O_i = [j for j in range(net.n) if net.edges[i, j]]
                if O_i:
                    M = np.vstack([
                        skew_symmetric((p[j] - p[i]) / np.linalg.norm(p[j] - p[i])) @ P_i
                        for j in O_i])
                    ok = np.linalg.matrix_rank(M) == np.linalg.matrix_rank(P_i)
                else:
                    ok = np.linalg.matrix_rank(P_i) == 0
                viol += int(not ok)
                seen[agent.domain] = min(seen[agent.domain], len(O_i))
    print(f"attitude observability  {tot:5d} IBR frameworks, violations: {viol}")
    print(f"                              min out-degree seen: "
          f"{ {d: seen[d] for d in DOMAINS} }   predicted: {RMIN}")


# ---------------------------------------------------------------- B_per = B_node
def build_per_edge(net, table_iii=False):
    """The edge-indexed construction. table_iii mirrors Michieletto Table III, which
    sets U_ij = I3 on an edge where a planar agent measures a spatial one."""
    n, m = net.n, int(net.edges.sum())
    p = [a.pose.position for a in net.agents]
    R = [a.pose.rotation_mat() for a in net.agents]
    E = np.zeros((n, m)); Eo = np.zeros((n, m))
    Dp = np.zeros((3 * m, 3 * m)); Da = np.zeros((3 * m, 3 * m))
    U = np.zeros((3 * m, 3 * m)); V = np.zeros((3 * m, 3 * m))
    planar = lambda a: a.domain in PLANAR
    for k, (i, j) in enumerate(zip(*np.nonzero(net.edges))):
        E[i, k] = -1; E[j, k] = 1; Eo[i, k] = -1
        s = 1.0 / np.linalg.norm(p[j] - p[i])
        ph = s * (p[j] - p[i])
        Dp[3*k:3*k+3, 3*k:3*k+3] = s * R[i].T @ orthogonal_projection_matrix(ph)
        Da[3*k:3*k+3, 3*k:3*k+3] = -R[i].T @ skew_symmetric(ph)
        S_i, P_i = node_dof_projectors(net.agents[i])
        U[3*k:3*k+3, 3*k:3*k+3] = (np.eye(3) if table_iii and planar(net.agents[i])
                                   and not planar(net.agents[j]) else S_i)
        V[3*k:3*k+3, 3*k:3*k+3] = P_i
    Eb = np.kron(E, np.eye(3)); Eob = np.kron(Eo, np.eye(3))
    return np.hstack([Dp @ U @ Eb.T, Da @ V @ Eob.T])


def check_homog():
    worst = 0.0; cases = 0
    for dom in DOMAINS:
        for sd in range(6):
            np.random.seed(sd)
            net, _ = random_scenario(6, dom, edge_count=18)
            worst = max(worst, float(np.abs(
                build_per_edge(net) - extended_bearing_rigidity_matrix(net)).max()))
            cases += 1
    print(f"B_per = B_node          {cases:5d} homogeneous cases, "
          f"max abs difference: {worst:.1e}")


# ---------------------------------------------------------------- rank identity
def check_rank_identity():
    bad = tot = 0
    for n in (5, 7):
        for doms in mixtures(n, 12, np.random.default_rng(n + 100)):
            np.random.seed(n * 31 + len(doms))
            net, _ = random_scenario(n, doms, edge_count=int(2.5 * n))
            B = extended_bearing_rigidity_matrix(net)
            A, c = admissible_basis(net)
            q_v = 6 * n - c
            q_i = c - int(np.linalg.matrix_rank(B @ A))
            tot += 1
            bad += int(int(np.linalg.matrix_rank(B)) != 6 * n - q_v - q_i)
    print(f"rank = 6n - q_v - q_i   {tot:5d} frameworks, disagreements: {bad}")


# ---------------------------------------------------------------- removal
def check_removal():
    bad_r = bad_l = tot = 0
    worst = 0.0
    for dom in DOMAINS:
        for sd in range(12):
            np.random.seed(400 + sd)
            net, _ = random_scenario(6, dom,
                                     edge_count=int(np.random.randint(12, 26)))
            rank_K = int(np.linalg.matrix_rank(
                extended_bearing_rigidity_matrix(net.fully_connected())))
            rank, B = rank_of(net)
            G = B.T @ B
            w, V = np.linalg.eigh(G)
            tol = max(float(w.max()), 1e-30) * 1e-10
            Minv = (V * np.where(w > tol, 1.0 / np.maximum(w, 1e-300), 0.0)) @ V.T
            lam = rigidity_decomposition(B, rank_K)[2]
            for k, (i, j) in enumerate(zip(*np.nonzero(net.edges))):
                b = B[3*k:3*k+3, :]
                drop = int((np.linalg.eigvalsh(b @ Minv @ b.T) > 1.0 - 1e-6).sum())
                keep = net.edges.copy()
                net.edges = keep.copy(); net.edges[i, j] = False
                true_rank, B2 = rank_of(net)
                net.edges = keep
                tot += 1
                bad_r += int(rank - true_rank != drop)
                if drop == 0 and lam > 0:
                    pred = float(np.linalg.eigvalsh(
                        G - b.T @ b)[B.shape[1] - rank_K])
                    true = rigidity_decomposition(B2, rank_K)[2]
                    rel = abs(pred - true) / max(true, 1e-300)
                    worst = max(worst, rel); bad_l += int(rel > 1e-6)
    print(f"block leverage          {tot:5d} deletions, rank disagreements: {bad_r}")
    print(f"rank-3 downdate               stiffness disagreements: {bad_l}, "
          f"worst relative error {worst:.1e}")


# ---------------------------------------------------------------- addition
def check_addition():
    bad = tot = 0
    for dom in DOMAINS:
        for sd in range(8):
            np.random.seed(900 + sd)
            net, _ = random_scenario(6, dom,
                                     edge_count=int(np.random.randint(6, 18)))
            rank, B = rank_of(net)
            L = characteristic_length(net)
            Z = nullspace_in_scaled_units(nullspace(B, rank), net.n, L)
            rk = candidate_gain(net, Z, length_scale=L)[1]
            for (i, j) in absent_pairs(net):
                keep = net.edges.copy()
                net.edges[i, j] = True
                new_rank = rank_of(net)[0]
                net.edges = keep
                tot += 1
                bad += int(new_rank - rank != int(rk[i, j]))
    print(f"addition criterion      {tot:5d} candidate pairs, disagreements: {bad}")


# ---------------------------------------------------------------- derivative
def check_dlambda():
    worst = 0.0; tot = 0
    for dom in DOMAINS:
        for sd in range(6):
            np.random.seed(200 + sd)
            net, _ = random_scenario(6, dom,
                                     edge_count=int(np.random.randint(14, 26)))
            rank_K = int(np.linalg.matrix_rank(
                extended_bearing_rigidity_matrix(net.fully_connected())))
            rank, B = rank_of(net)
            if rank < rank_K:
                continue
            G = B.T @ B
            q = B.shape[1] - rank_K
            w, V = np.linalg.eigh(G)
            lam, v = w[q], V[:, q]
            if abs(w[q + 1] - lam) < 1e-6 * max(lam, 1e-12):
                continue                       # the proposition assumes lambda simple
            for (i, j) in absent_pairs(net)[:4]:
                b = candidate_block(net, i, j)
                pred = float(np.linalg.norm(b @ v) ** 2)
                h = 1e-6
                num = (np.linalg.eigvalsh(G + h * b.T @ b)[q] - lam) / h
                worst = max(worst, abs(num - pred) / max(abs(pred), 1e-12))
                tot += 1
    print(f"first-order gain        {tot:5d} pairs, worst relative error against a "
          f"finite difference: {worst:.1e}")


# ---------------------------------------------------------------- the bounds
def check_bounds():
    bad_m = bad_rep = tot = 0
    for dom in DOMAINS:
        for sd in range(4):
            np.random.seed(sd)
            net, _ = random_scenario(4, [dom] * 4, edge_count=0)
            rank_K = int(np.linalg.matrix_rank(
                extended_bearing_rigidity_matrix(net.fully_connected())))
            m_req = required_edge_count(net)
            pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
            smallest = winner = None
            for size in range(1, len(pairs) + 1):
                for sub in itertools.combinations(pairs, size):
                    E = np.zeros((4, 4), dtype=bool)
                    for (i, j) in sub:
                        E[i, j] = True
                    net.edges = E
                    if rank_of(net)[0] == rank_K:
                        smallest, winner = size, sub
                        break
                if smallest:
                    break
            tot += 1
            bad_m += int(m_req > smallest)
            # damage the winner, then bound the repair
            net.edges = np.zeros((4, 4), dtype=bool)
            for (i, j) in winner:
                net.edges[i, j] = True
            for _ in range(min(2, int(net.edges.sum()))):
                net.edges[list(zip(*np.nonzero(net.edges)))[0]] = False
            bound = repair_edge_count(net, rank_K=rank_K)
            broken = net.edges.copy()
            absent = [(i, j) for i in range(4) for j in range(4)
                      if i != j and not broken[i, j]]
            true_rep = None
            for size in range(0, len(absent) + 1):
                for sub in itertools.combinations(absent, size):
                    E = broken.copy()
                    for (i, j) in sub:
                        E[i, j] = True
                    net.edges = E
                    if rank_of(net)[0] == rank_K:
                        true_rep = size
                        break
                if true_rep is not None:
                    break
            bad_rep += int(bound > true_rep)
    print(f"m_req bound             {tot:5d} exhaustive cases, unsound: {bad_m}")
    print(f"repair bound            {tot:5d} exhaustive cases, unsound: {bad_rep}")


# ---------------------------------------------------------------- counterexample
def check_counterexample():
    pos = np.array([[0., 0., 0.], [2., 0., 0.], [1., 1., 1.]])
    net = Network(pos, np.zeros((3, 3)), np.zeros((3, 3), dtype=bool))
    for i, d in enumerate(["R^2", "R^2", "R^3"]):
        net.agents[i].set_domain(d)
    out = {}
    for name, edges in (("G", [(0, 2), (1, 2)]),
                        ("K", [(i, j) for i in range(3) for j in range(3) if i != j])):
        net.edges = np.zeros((3, 3), dtype=bool)
        for i, j in edges:
            net.edges[i, j] = True
        A, c = admissible_basis(net)
        out[name] = (int(np.linalg.matrix_rank(build_per_edge(net, table_iii=True))),
                     int(np.linalg.matrix_rank(extended_bearing_rigidity_matrix(net))),
                     c - int(np.linalg.matrix_rank(
                         extended_bearing_rigidity_matrix(net) @ A)))
    print(f"counterexample          rank B_per: G {out['G'][0]}, K {out['K'][0]}   "
          f"rank B_node: G {out['G'][1]}, K {out['K'][1]}   "
          f"q_i {out['G'][2]}, q_t {out['K'][2]}  ->  IBR "
          f"{out['G'][2] == out['K'][2]}")


if __name__ == "__main__":
    check_rank_K()
    check_block_rank()
    check_attitude()
    check_homog()
    check_rank_identity()
    check_removal()
    check_addition()
    check_dlambda()
    check_bounds()
    check_counterexample()
