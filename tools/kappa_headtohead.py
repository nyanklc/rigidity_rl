"""Policy against greedy across the four kappa environments, term by term.

The objective (5.8) is a sum of three parts,

    phi = w_rank*rank/rank_K  -  w_edge*m*c_max/rank_K  +  kappa*one_edge*1[IBR]*q

and a method can win on any of them. This splits the gap between the matched
policy and greedy into those three terms at each kappa, which is what says
*where* the difference comes from rather than only how big it is.

Everything is recomputed from the frozen instances and the final edge sets each
run already stores, and the recomputed phi is checked against the stored `score`
before anything is reported.

    PYTHONPATH=. uv run tools/kappa_headtohead.py [--csv out.csv]
"""
import argparse
import csv
import glob
import json
import os
import numpy as np

from network import Network
import rigidity as R

W_RANK, W_EDGE = 100.0, 25.0
FRONT = "runs_outputs/*front_k*/"


def rebuild(inst, ep):
    pos = inst["positions"][ep]
    n = len(pos)
    net = Network(pos, np.zeros((n, 3)), np.zeros((n, n), dtype=bool))
    for i in range(n):
        net.agents[i].pose.rotation = inst["orientations"][ep][i]
        net.agents[i].set_domain(str(inst["domains"][ep][i]))
        if net.agents[i].domain == "R^3xS^1":
            net.agents[i].rotation_axis = inst["rotation_axes"][ep][i]
    return net


def terms(net, rank_K, c_max, kappa):
    """(rigidity, sparsity, robustness, eps_pos, eps_att, at_bound) for the current edges."""
    B = R.extended_bearing_rigidity_matrix(net)
    rank = np.linalg.matrix_rank(B)
    m = int(net.edges.sum())
    n = net.n
    rig = W_RANK * rank / rank_K
    spa = -W_EDGE * m * c_max / rank_K
    rob, e_pos, e_att = 0.0, None, None
    if rank == rank_K:                                   # only defined on a rigid framework
        Bs = R.scaled_rigidity_matrix(net)
        a_pos, a_att = R.estimation_error_blocks(Bs, rank_K, n)
        e_pos, e_att = np.sqrt(a_pos / n), np.sqrt(a_att / n)
        q = R.shape_error_quality(float(np.sqrt((a_pos + a_att) / n)), n)
        if q is not None:
            rob = kappa * (W_EDGE * c_max / rank_K) * q
    return rig, spa, rob, e_pos, e_att, m, rank == rank_K


def run():
    out = []
    for d in sorted(glob.glob(FRONT), key=lambda p: float(p.split("front_k")[1].rstrip("/"))):
        kappa = float(d.split("front_k")[1].rstrip("/"))
        inst = np.load(d + "instances.npz", allow_pickle=True)
        rows = list(csv.DictReader(open(d + "results.csv")))
        cache = {}
        for row in rows:
            ep = int(row["episode"])
            net = rebuild(inst, ep)
            n = net.n
            if ep not in cache:
                full = np.ones((n, n), dtype=bool)
                np.fill_diagonal(full, False)
                net.edges = full
                BK = R.extended_bearing_rigidity_matrix(net)
                cache[ep] = (np.linalg.matrix_rank(BK), R.max_edge_rank(net, BK),
                             R.required_edge_count(net, rank_K=np.linalg.matrix_rank(BK)))
            rank_K, c_max, m_req = cache[ep]
            E = np.zeros((n, n), dtype=bool)
            for tok in row["edges"].split():
                i, j = tok.split(">")
                E[int(i), int(j)] = True
            net.edges = E
            rig, spa, rob, ep_, ea, m, rigid = terms(net, rank_K, c_max, kappa)
            out.append(dict(kappa=kappa, method=row["method"], episode=ep,
                            phi=rig + spa + rob, stored=float(row["score"]),
                            rigidity=rig, sparsity=spa, robustness=rob,
                            eps_pos=ep_, eps_att=ea, m=m, m_req=m_req,
                            rigid=rigid, at_bound=bool(rigid and m == m_req)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows = run()

    bad = [r for r in rows if abs(r["phi"] - r["stored"]) > 1e-6 * max(1, abs(r["stored"]))]
    print(f"rows {len(rows)}, phi mismatches against the stored score: {len(bad)}")
    if bad:
        for r in bad[:5]:
            print(f"   k={r['kappa']} {r['method']} ep{r['episode']}: "
                  f"recomputed {r['phi']:.4f} stored {r['stored']:.4f}")
        print("  refusing to report a decomposition that does not reproduce phi")
        return

    kappas = sorted({r["kappa"] for r in rows})
    matched = {0.0: "k0", 10.0: "k10", 50.0: "k50", 200.0: "k200"}
    print("\nMatched policy minus greedy, per instance, split by objective term")
    print(f"{'kappa':>6s} {'d phi':>8s} {'d rigidity':>11s} {'d sparsity':>11s} "
          f"{'d robust':>9s} {'d edges':>8s} {'W/L/T':>9s} {'at-bnd p/g':>11s}")
    for k in kappas:
        pol = {r["episode"]: r for r in rows if r["kappa"] == k and r["method"] == matched[k]}
        gre = {r["episode"]: r for r in rows if r["kappa"] == k and r["method"] == "greedy"}
        eps = sorted(set(pol) & set(gre))
        d = lambda f: np.mean([pol[e][f] - gre[e][f] for e in eps])
        w = sum(1 for e in eps if pol[e]["phi"] > gre[e]["phi"] + 1e-9)
        l = sum(1 for e in eps if pol[e]["phi"] < gre[e]["phi"] - 1e-9)
        t = len(eps) - w - l
        abp = sum(1 for e in eps if pol[e]["at_bound"])
        abg = sum(1 for e in eps if gre[e]["at_bound"])
        print(f"{k:6.0f} {d('phi'):8.2f} {d('rigidity'):11.2f} {d('sparsity'):11.2f} "
              f"{d('robustness'):9.2f} {d('m'):8.2f} {w:3d}/{l:2d}/{t:2d} {abp:5d}/{abg:<5d}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
