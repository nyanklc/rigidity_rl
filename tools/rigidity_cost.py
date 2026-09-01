"""Which rigidity primitive costs what, and which one blocks a larger n?

Three tables. The first times every primitive a step can reach on a near-minimal
graph. The second isolates `removal_costs`, whose cost is not a function of n
alone: it does one eigh(6n) per *redundant* edge, so it grows with density, and a
near-minimal graph hides that. The third is the complete-graph rigidity matrix,
which `reset()` builds once and which sets the memory ceiling on n.

Pin BLAS to one thread, for the reason in tools/flag_cost.py.

    OMP_NUM_THREADS=1 PYTHONPATH=. uv run tools/rigidity_cost.py
"""
import argparse
import time

import numpy as np

import rigidity as R
from scenario import random_scenario


def timeit(f, reps=5):
    f()
    t0 = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t0) / reps * 1e3


def components(n, domain, seed=3):
    """Every primitive a step can reach, on a greedy (near-minimal) graph."""
    np.random.seed(seed)
    net, _ = random_scenario(n, domain)
    netK = net.fully_connected()
    BK = R.extended_bearing_rigidity_matrix(netK)
    rank_K = int(np.linalg.matrix_rank(BK))
    c_max = R.max_edge_rank(net, brmat_K=BK)

    net.edges, _, _ = R.greedy_rigid_construction(net, rank_K, np.random.default_rng(seed))
    B = R.extended_bearing_rigidity_matrix(net)
    rank, _, lam = R.rigidity_decomposition(B, rank_K)
    L = R.characteristic_length(net)
    Z, _, w, V = R.nullspace_and_softest(B, rank)
    Zs = R.nullspace_in_scaled_units(Z, n, L)
    ZK = R.nullspace_in_scaled_units(R.nullspace(BK, rank_K), n, L)

    out = {"n": n, "m": int(net.edges.sum())}
    out["build B            (3m x 6n)"] = timeit(
        lambda: R.extended_bearing_rigidity_matrix(net))
    out["rigidity_decomp    svd(B)"] = timeit(
        lambda: R.rigidity_decomposition(B, rank_K))
    out["is_MBR             m x rank(3 x 6n)"] = timeit(
        lambda: R.is_MBR(net, rank_K=rank_K, brmat=B, rank_brm=rank))
    out["nullspace          eigh(6n)"] = timeit(lambda: R.nullspace(B, rank))
    out["nullspace_softest  eigh(6n)"] = timeit(lambda: R.nullspace_and_softest(B, rank))
    out["candidate_gain     n^2 pairs x k"] = timeit(
        lambda: R.candidate_gain(net, Zs, length_scale=L))
    out["flex_space + magn  svd(6n x k)"] = timeit(
        lambda: R.node_flex_magnitude(R.flex_space(Zs, ZK), n))
    out["removal_costs      m x eigh(6n)"] = timeit(
        lambda: R.removal_costs(B, net, rank_K, lam=lam, w=w, V=V, c_max=c_max), reps=3)
    out["estimation_error_of"] = timeit(
        lambda: R.estimation_error_of(net, rank_K, brmat=B))
    out["[reset] build B_K"] = timeit(
        lambda: R.extended_bearing_rigidity_matrix(netK), reps=3)
    out["[reset] edge_block_ranks(B_K)"] = timeit(lambda: R.edge_block_ranks(BK), reps=3)
    out["[reset] max_edge_rank"] = timeit(lambda: R.max_edge_rank(net, brmat_K=BK), reps=3)
    out["[reset] nullspace(B_K)"] = timeit(lambda: R.nullspace(BK, rank_K), reps=3)
    out["closeness          n^3 python"] = timeit(
        lambda: net.get_closeness_centrality_features())
    out["brandes            twice per step"] = timeit(lambda: net._brandes_betweenness())
    out["eigenvector"] = timeit(lambda: net.get_eigenvector_centrality_features())
    out["all_pairs_bearings n^2 python"] = timeit(lambda: net.get_all_pairs_bearings())
    return out


def removal_vs_density(ns, domain, seed=5):
    print("removal_costs, and how many edges reach the eigh(6n) downdate branch")
    print(f"{'n':>4}{'m/m_req':>9}{'m':>6}{'redundant':>11}{'ms':>10}")
    for n in ns:
        np.random.seed(seed)
        net, _ = random_scenario(n, domain)
        BK = R.extended_bearing_rigidity_matrix(net.fully_connected())
        rank_K = int(np.linalg.matrix_rank(BK))
        c_max = R.max_edge_rank(net, brmat_K=BK)
        base, _, _ = R.greedy_rigid_construction(net, rank_K, np.random.default_rng(seed))
        m_req = int(base.sum())

        for mult in (1.0, 1.5, 2.0):
            E = base.copy()
            absent = [(i, j) for i in range(n) for j in range(n)
                      if i != j and not E[i, j]]
            rng = np.random.default_rng(1)
            for k in rng.permutation(len(absent))[:max(0, int(m_req * mult) - m_req)]:
                E[absent[k]] = True
            net.edges = E
            B = R.extended_bearing_rigidity_matrix(net)
            rank, _, lam = R.rigidity_decomposition(B, rank_K)
            _, _, w, V = R.nullspace_and_softest(B, rank)
            rank_lost, _ = R.removal_costs(B, net, rank_K, lam=lam, w=w, V=V, c_max=c_max)
            # an edge carrying no rank of its own is the one that pays for a downdate
            redundant = int(((rank_lost == 0) & net.edges).sum())
            ms = timeit(lambda: R.removal_costs(B, net, rank_K, lam=lam, w=w, V=V,
                                                c_max=c_max), reps=3)
            print(f"{n:>4}{mult:>9.1f}{int(E.sum()):>6}{redundant:>11}{ms:>10.2f}")
    print()


def complete_graph_matrix(ns, domain, seed=5, mb_limit=8000):
    # Dp and Da are dense (3m, 3m). On the complete graph m = n^2, so this is the
    # Theta(n^4) allocation that sets the ceiling on n.
    print("extended_bearing_rigidity_matrix on the COMPLETE graph (once per reset)")
    print(f"{'n':>4}{'m=n^2':>8}{'Dp shape':>16}{'Dp+Da MB':>10}{'ms':>10}")
    for n in ns:
        np.random.seed(seed)
        net, _ = random_scenario(n, domain)
        netK = net.fully_connected()
        m = int(netK.edges.sum())
        mb = 2 * (3 * m) ** 2 * 8 / 1e6
        ms = (timeit(lambda: R.extended_bearing_rigidity_matrix(netK), reps=1)
              if mb < mb_limit else float("nan"))
        print(f"{n:>4}{m:>8}{str((3 * m, 3 * m)):>16}{mb:>10.0f}{ms:>10.1f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="R^3")
    ap.add_argument("--n", default="8,16,32")
    ap.add_argument("--complete-n", default="8,16,32,64")
    args = ap.parse_args()
    ns = [int(x) for x in args.n.split(",")]

    rows = [components(n, args.domain) for n in ns]
    keys = [k for k in rows[0] if k not in ("n", "m")]
    print(f"domain={args.domain}")
    print(f"{'component (ms)':<36}"
          + "".join(f"{'n=' + str(r['n']) + ' m=' + str(r['m']):>16}" for r in rows))
    for k in keys:
        print(f"{k:<36}" + "".join(f"{r[k]:>16.3f}" for r in rows))
    print()

    removal_vs_density(ns, args.domain)
    complete_graph_matrix([int(x) for x in args.complete_n.split(",")], args.domain)


if __name__ == "__main__":
    main()
