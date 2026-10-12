"""Score every method's own search at any conditioning weight, without re-running it.

A method's trajectory is a sequence of visited graphs, and the environment returns
the best of them under phi.

A POLICY's trajectory does not depend on kappa: it acts from observations, which
carry no kappa, so only which visited graph is called best changes. Its whole
kappa axis can therefore be swept offline, including weights it was never trained
or evaluated at.

The CLASSICAL searches are different and must NOT be swept this way: they hill
climb on phi itself, so their trajectory changes with kappa. Greedy's path at
kappa=200 diverges from its kappa=0 path at step 5 and ends 21 steps long against
8. Use tools/kappa_extreme.py, which re-runs them, for those.

The sweep is validated by rescoring each run at its own kappa and checking the
result against the stored `score` in `results.csv`.

    PYTHONPATH=. uv run tools/kappa_extrapolate.py [--csv out.csv]
"""
import argparse
import csv
import glob
import numpy as np

from network import Network
import rigidity as R

W_RANK, W_EDGE = 100.0, 25.0
MAIN = "runs_outputs/20260912-162935__addremove__3models__main/"
FRONTS = sorted(glob.glob("runs_outputs/*front_k*/"),
                key=lambda p: float(p.split("front_k")[1].rstrip("/")))
KAPPAS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


def c_max_of(run, ep):
    inst = np.load(run + "instances.npz", allow_pickle=True)
    pos = inst["positions"][ep]
    n = len(pos)
    net = Network(pos, np.zeros((n, 3)), np.zeros((n, n), dtype=bool))
    for i in range(n):
        net.agents[i].pose.rotation = inst["orientations"][ep][i]
        net.agents[i].set_domain(str(inst["domains"][ep][i]))
        if net.agents[i].domain == "R^3xS^1":
            net.agents[i].rotation_axis = inst["rotation_axes"][ep][i]
    full = np.ones((n, n), dtype=bool)
    np.fill_diagonal(full, False)
    net.edges = full
    BK = R.extended_bearing_rigidity_matrix(net)
    return R.max_edge_rank(net, BK), n


def phi(rank, m, rank_K, c_max, is_IBR, shape_err, n, kappa):
    s = (W_RANK * rank - W_EDGE * m * c_max) / rank_K
    if kappa > 0 and is_IBR and shape_err:
        q = R.shape_error_quality(float(shape_err), n)
        if q is not None:
            s += kappa * (W_EDGE * c_max / rank_K) * q
    return s


def sweep(run, kappas):
    """best-state-visited phi at each kappa, per (method, episode)."""
    rows = list(csv.DictReader(open(run + "trajectories.csv")))
    meta = {}
    out = {}
    for r in rows:
        ep = int(r["episode"])
        if ep not in meta:
            meta[ep] = c_max_of(run, ep)
        c_max, n = meta[ep]
        vals = [phi(int(r["rank"]), int(r["edges"]), int(r["rank_K"]), c_max,
                    r["is_IBR"] == "True", r["shape_err"], n, k) for k in kappas]
        key = (r["method"], ep)
        cur = out.get(key)
        out[key] = vals if cur is None else [max(a, b) for a, b in zip(cur, vals)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    args = ap.parse_args()

    # --- validation: rescoring a run at its own kappa must reproduce results.csv
    print("validation, rescored best-state phi against the stored score")
    ok = True
    for run in FRONTS:
        k = float(run.split("front_k")[1].rstrip("/"))
        got = sweep(run, [k])
        stored = {(r["method"], int(r["episode"])): float(r["score"])
                  for r in csv.DictReader(open(run + "results.csv"))}
        d = [abs(got[key][0] - v) for key, v in stored.items() if key in got]
        bad = sum(1 for x in d if x > 1e-6)
        ok &= bad == 0
        print(f"   kappa={k:6.0f}: {len(d)} rows, mismatches {bad}, worst {max(d):.2e}")
    if not ok:
        print("   refusing to extrapolate from traces that do not reproduce the stored score")
        return

    # policies and greedy come from one front run: the traces are kappa-independent
    pol = sweep(FRONTS[0], KAPPAS)
    cls = sweep(MAIN, KAPPAS)

    print(f"\nBest-state phi against the conditioning weight, mean over 20 instances")
    print("Policies only: a classical search re-plans when kappa changes, so its"
          "\ntrajectory cannot be rescored. See tools/kappa_extreme.py for those.")
    methods = ["k0", "k10", "k50", "k200"]
    extra = []
    print(f"{'kappa':>7s}" + "".join(f"{m:>9s}" for m in methods + extra))
    rows_out = []
    for i, k in enumerate(KAPPAS):
        line = f"{k:7d}"
        for m in methods:
            v = [pol[key][i] for key in pol if key[0] == m]
            line += f"{np.mean(v):9.1f}" if v else f"{'-':>9s}"
            rows_out.append(dict(kappa=k, method=m, phi=float(np.mean(v)) if v else None))
        for m in extra:
            v = [cls[key][i] for key in cls if key[0] == m]
            line += f"{np.mean(v):9.1f}" if v else f"{'-':>9s}"
            rows_out.append(dict(kappa=k, method=m, phi=float(np.mean(v)) if v else None))
        print(line)

    print("\nbest policy at each kappa")
    for i, k in enumerate(KAPPAS):
        cand = {}
        for m in methods:
            v = [pol[key][i] for key in pol if key[0] == m]
            if v:
                cand[m] = np.mean(v)
        for m in extra:
            v = [cls[key][i] for key in cls if key[0] == m]
            if v:
                cand[m] = np.mean(v)
        best = max(cand, key=cand.get)
        spread = cand[best] - min(cand.values())
        print(f"   kappa={k:6d}  best={best:6s} {cand[best]:9.1f}   "
              f"spread over the four policies {spread:8.2f}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["kappa", "method", "phi"])
            w.writeheader(); w.writerows([r for r in rows_out if r["phi"] is not None])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
