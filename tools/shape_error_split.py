"""Split the reported shape error into its position and attitude halves.

`results.csv` records only the combined eps = sqrt(a_opt(B_s)/n). The two halves
come from `estimation_error_blocks`, and every run directory already stores what
is needed to recompute them: `instances.npz` holds the poses and domains and the
`edges` column of `results.csv` holds each method's final graph. So no experiment
has to be re-run.

    PYTHONPATH=. uv run tools/shape_error_split.py <run_dir> [--csv out.csv]

The combined value it recomputes is checked against the stored one; a mismatch
means the reconstruction is wrong and the split must not be trusted.
"""
import argparse
import csv
import os
import numpy as np

from network import Network
import rigidity as R


def rebuild(inst, episode):
    """The network of one frozen instance, edges left empty."""
    pos = inst["positions"][episode]
    n = len(pos)
    net = Network(pos, np.zeros((n, 3)), np.zeros((n, n), dtype=bool))
    for i in range(n):
        net.agents[i].pose.rotation = inst["orientations"][episode][i]
        net.agents[i].set_domain(str(inst["domains"][episode][i]))
        if net.agents[i].domain == "R^3xS^1":
            net.agents[i].rotation_axis = inst["rotation_axes"][episode][i]
    return net


def parse_edges(field, n):
    E = np.zeros((n, n), dtype=bool)
    for token in field.split():
        i, j = token.split(">")
        E[int(i), int(j)] = True
    return E


def split_run(run_dir):
    inst = np.load(os.path.join(run_dir, "instances.npz"), allow_pickle=True)
    rows = list(csv.DictReader(open(os.path.join(run_dir, "results.csv"))))
    out, rank_K = [], {}
    for row in rows:
        ep = int(row["episode"])
        net = rebuild(inst, ep)
        n = net.n
        if ep not in rank_K:                      # rank_K is per instance, not per method
            full = np.ones((n, n), dtype=bool)
            np.fill_diagonal(full, False)
            net.edges = full
            rank_K[ep] = np.linalg.matrix_rank(R.extended_bearing_rigidity_matrix(net))
        net.edges = parse_edges(row["edges"], n)
        Bs = R.scaled_rigidity_matrix(net)
        if np.linalg.matrix_rank(Bs) < rank_K[ep]:
            out.append((row["method"], ep, None, None, None))   # flexible: eps is unbounded
            continue
        a_pos, a_att = R.estimation_error_blocks(Bs, rank_K[ep], n)
        stored = float(row["shape_err"]) if row["shape_err"] else float("nan")
        out.append((row["method"], ep, stored,
                    float(np.sqrt(a_pos / n)), float(np.sqrt(a_att / n))))
    return out


def gmean(v):
    v = [x for x in v if x is not None and x > 0]
    return float(np.exp(np.mean(np.log(v)))) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--csv")
    args = ap.parse_args()

    rows = split_run(args.run_dir)
    bad = [r for r in rows if r[2] is not None
           and abs(np.hypot(r[3], r[4]) - r[2]) / r[2] > 1e-6]
    print(f"rows {len(rows)}, reconstruction mismatches {len(bad)}")
    if bad:
        print("  refusing to report a split that does not reproduce the stored value")
        return

    methods = list(dict.fromkeys(r[0] for r in rows))
    print(f"\n{'method':22s} {'eps':>8s} {'eps_pos':>8s} {'eps_att':>8s} {'att share':>9s}")
    for m in methods:
        sel = [r for r in rows if r[0] == m]
        e, ep_, ea = gmean([r[2] for r in sel]), gmean([r[3] for r in sel]), gmean([r[4] for r in sel])
        share = ea**2 / (ep_**2 + ea**2) if ea == ea and ep_ == ep_ else float("nan")
        print(f"{m:22s} {e:8.2f} {ep_:8.2f} {ea:8.2f} {share:9.2f}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "episode", "shape_err", "shape_err_pos", "shape_err_att"])
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
