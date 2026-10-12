"""Is B^T B a weighted graph Laplacian, and how does lambda relate to algebraic connectivity?

    PYTHONPATH=. OMP_NUM_THREADS=1 uv run tools/laplacian_structure.py

Three questions, in order.

1. STRUCTURE.  Writing W_k = s_k^2 P(p_hat_k), M_k = P(p_hat_k) and C_k = -s_k [p_hat_k]_x,

       B^T B = [ S_bar L_W S_bar     S_bar C P_bar ]
               [ P_bar C^T S_bar     P_bar D P_bar ]

   where L_W = Ebar W Ebar^T is a MATRIX-WEIGHTED graph Laplacian, D = Ebar_o M Ebar_o^T is
   BLOCK DIAGONAL, and C = Ebar C_blk Ebar_o^T couples the two halves. The attitude block is
   diagonal because Ebar_o registers only the measuring node, which is the same fact that
   makes attitude observability a per-node condition.

2. A SPECTRAL BOUND.  P(p_hat) <= I in the PSD order, so L_W <= L_s (x) I_3 with L_s the
   SCALAR Laplacian of edge weights s_k^2 = 1/||p_ij||^2. Weyl monotonicity plus the index
   count gives, in the frameless domains,

       lambda <= lambda_2(L_s),                                                     (*)

   the algebraic connectivity of the distance-weighted graph. In particular a rigid framework
   is connected, which is weaker than the two-vertex connectivity Trinh et al. prove.

3. HOW TIGHT.  (*) is sound and loose. At the minimum edge count lambda sits around 2% of the
   bound, rising to about 28% on the complete graph, and log lambda correlates with
   log lambda_2 at 0.1 to 0.3. Connectivity bounds the conditioning without predicting it.
"""
import numpy as np

from rigidity import (extended_bearing_rigidity_matrix, node_dof_projectors,
                      rigidity_decomposition)
from scenario import random_scenario
from util import orthogonal_projection_matrix, skew_symmetric

DOMAINS = ("R^2", "R^3", "R^2xS^1", "R^3xS^1", "SE(3)")


def laplacian_blocks(network):
    """(L_W, D, C) from the definitions, built without reference to B."""
    n = network.n
    p = [a.pose.position for a in network.agents]
    L_W = np.zeros((3 * n, 3 * n))
    D = np.zeros((3 * n, 3 * n))
    C = np.zeros((3 * n, 3 * n))
    for i, j in zip(*np.nonzero(network.edges)):
        if i == j:
            continue
        s = 1.0 / np.linalg.norm(p[j] - p[i])
        ph = s * (p[j] - p[i])
        W_k = s ** 2 * orthogonal_projection_matrix(ph)      # Dp_k^T Dp_k
        M_k = orthogonal_projection_matrix(ph)               # Da_k^T Da_k
        C_k = -s * skew_symmetric(ph)                        # Dp_k^T Da_k
        bi, bj = slice(3*i, 3*i+3), slice(3*j, 3*j+3)
        L_W[bi, bi] += W_k; L_W[bj, bj] += W_k               # Ebar W Ebar^T
        L_W[bi, bj] -= W_k; L_W[bj, bi] -= W_k
        D[bi, bi] += M_k                                     # Ebar_o M Ebar_o^T: tail only
        C[bi, bi] += C_k
        C[bj, bi] -= C_k
    return L_W, D, C


def dof_projectors(network):
    n = network.n
    S = np.zeros((3 * n, 3 * n))
    P = np.zeros((3 * n, 3 * n))
    for i, agent in enumerate(network.agents):
        S[3*i:3*i+3, 3*i:3*i+3], P[3*i:3*i+3, 3*i:3*i+3] = node_dof_projectors(agent)
    return S, P


def scalar_laplacian(network, weighted=True):
    """L_s on the underlying undirected graph, weights 1/||p_ij||^2 or 1."""
    n = network.n
    p = [a.pose.position for a in network.agents]
    L = np.zeros((n, n))
    for i, j in zip(*np.nonzero(network.edges)):
        if i == j:
            continue
        w = 1.0 / np.linalg.norm(p[j] - p[i]) ** 2 if weighted else 1.0
        L[i, i] += w; L[j, j] += w; L[i, j] -= w; L[j, i] -= w
    return L


def rigid_graph(n, domain, seed, extra=0):
    """A rigid graph grown in random order, plus `extra` edges. (network, rank_K) or None."""
    np.random.seed(seed)
    net, _ = random_scenario(n, domain, edge_count=0)
    rank_K = int(np.linalg.matrix_rank(
        extended_bearing_rigidity_matrix(net.fully_connected())))
    order = [(i, j) for i in range(n) for j in range(n) if i != j]
    np.random.shuffle(order)
    E = np.zeros((n, n), dtype=bool)
    rank = 0
    for i, j in order:
        if rank >= rank_K:
            break
        E[i, j] = True
        net.edges = E
        r = int(np.linalg.matrix_rank(extended_bearing_rigidity_matrix(net)))
        if r > rank:
            rank = r
        else:
            E[i, j] = False
    if rank < rank_K:
        return None
    for i, j in order:
        if E.sum() >= E.sum() + 0 and extra <= 0:
            break
        if not E[i, j]:
            E[i, j] = True
            extra -= 1
            if extra <= 0:
                break
    net.edges = E
    return net, rank_K


def check_structure(n=7, seeds=6):
    print("1. B^T B against the block construction")
    print(f"   {'domain':10s} {'max|B^T B - blocks|':>21s} {'attitude off-diagonal':>23s}")
    for dom in DOMAINS:
        worst = offdiag = 0.0
        for sd in range(seeds):
            np.random.seed(sd)
            net, _ = random_scenario(n, dom, edge_count=3 * n)
            G = extended_bearing_rigidity_matrix(net)
            G = G.T @ G
            L_W, D, C = laplacian_blocks(net)
            S, P = dof_projectors(net)
            pred = np.block([[S @ L_W @ S, S @ C @ P], [P @ C.T @ S, P @ D @ P]])
            worst = max(worst, float(np.abs(G - pred).max()))
            att = P @ D @ P
            mask = np.ones((3 * n, 3 * n), dtype=bool)
            for i in range(n):
                mask[3*i:3*i+3, 3*i:3*i+3] = False
            offdiag = max(offdiag, float(np.abs(att[mask]).max()))
        print(f"   {dom:10s} {worst:21.2e} {offdiag:23.2e}")


def check_bound(n=8, seeds=60):
    print("\n2. lambda against lambda_2(L_s), the distance-weighted algebraic connectivity")
    print(f"   {'domain':10s} {'sound':>10s} {'median ratio':>13s} {'p90':>8s} "
          f"{'corr(log,log)':>14s}")
    for dom in DOMAINS:
        ratios, lams, fied, ok = [], [], [], 0
        for sd in range(seeds):
            got = rigid_graph(n, dom, 1000 + sd)
            if got is None:
                continue
            net, rank_K = got
            lam = rigidity_decomposition(extended_bearing_rigidity_matrix(net), rank_K)[2]
            if lam <= 0:
                continue
            f = float(np.linalg.eigvalsh(scalar_laplacian(net))[1])
            ok += int(lam <= f * (1 + 1e-9))
            ratios.append(lam / f); lams.append(lam); fied.append(f)
        r = np.array(ratios)
        c = float(np.corrcoef(np.log10(lams), np.log10(fied))[0, 1])
        print(f"   {dom:10s} {ok:5d}/{len(r):<4d} {np.median(r):13.4f} "
              f"{np.percentile(r, 90):8.4f} {c:14.3f}")


def check_tightness(n=8, seeds=40):
    print("\n3. how the bound tightens with density, R^3")
    print(f"   {'m/m_req':>8s} {'sound':>9s} {'median ratio':>13s} {'max ratio':>10s}")
    m_req = 10
    for mult in (1.0, 1.5, 2.0, 3.0, 5.6):
        ratios, ok = [], 0
        for sd in range(seeds):
            got = rigid_graph(n, "R^3", sd, extra=max(0, int(round(mult * m_req)) - m_req))
            if got is None:
                continue
            net, rank_K = got
            lam = rigidity_decomposition(extended_bearing_rigidity_matrix(net), rank_K)[2]
            if lam <= 0:
                continue
            f = float(np.linalg.eigvalsh(scalar_laplacian(net))[1])
            ok += int(lam <= f * (1 + 1e-9))
            ratios.append(lam / f)
        r = np.array(ratios)
        print(f"   {mult:8.1f} {ok:4d}/{len(r):<4d} {np.median(r):13.4f} {r.max():10.4f}")


def check_connectivity(n=6, seeds=300):
    """The bound says a disconnected framework cannot be rigid."""
    bad = 0
    for sd in range(seeds):
        np.random.seed(5000 + sd)
        net, _ = random_scenario(n, "R^3", edge_count=int(np.random.randint(5, 16)))
        A = net.edges | net.edges.T
        seen, stack = {0}, [0]
        while stack:
            u = stack.pop()
            for v in range(n):
                if A[u, v] and v not in seen:
                    seen.add(v); stack.append(v)
        rank_K = int(np.linalg.matrix_rank(
            extended_bearing_rigidity_matrix(net.fully_connected())))
        rigid = int(np.linalg.matrix_rank(
            extended_bearing_rigidity_matrix(net))) == rank_K
        bad += int(rigid and len(seen) < n)
    print(f"\n4. rigid but disconnected, over {seeds} random graphs: {bad}")


if __name__ == "__main__":
    check_structure()
    check_bound()
    check_tightness()
    check_connectivity()
