"""Does the predicted shape error match the measured one, and where does it stop?

Perturb every bearing by sigma radians, solve for the shape, compare against the
Cramer-Rao prediction read off B's spectrum. The prediction is a linearisation,
so the second question is the noise level above which it stops being one.

    PYTHONPATH=. uv run tools/crlb_validation.py
    PYTHONPATH=. uv run tools/crlb_validation.py --formations 30 --trials 100 --csv out.csv
    PYTHONPATH=. uv run tools/crlb_validation.py --from-csv 'runs_outputs/crlb_validation_20260930/*.csv'

With --formations above one, each domain is measured on that many random formations
(one greedy-constructed rigid graph each), and the summary gives the geometric mean of
the per-formation position ratio with a bootstrap interval over formations.
"""
import argparse
import csv

import numpy as np

import estimation as E
from rigidity import extended_bearing_rigidity_matrix as B_of, greedy_rigid_construction
from scenario import random_scenario

DOMAINS = {
    "R^2": ["R^2"] * 8,
    "R^3": ["R^3"] * 8,
    "R^2xS^1": ["R^2xS^1"] * 8,
    "R^3xS^1": ["R^3xS^1"] * 8,
    "SE(3)": ["SE(3)"] * 8,
    "mixed": ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"] * 2,
}
BOOTSTRAPS = 4000


def formation(doms, seed):
    """A random formation with a greedy-constructed rigid graph on it."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net, _ = random_scenario(len(doms), doms, edge_count=0)
    rank_K = int(np.linalg.matrix_rank(B_of(net.fully_connected())))
    greedy_rigid_construction(net, rank_K, rng)
    return net, rank_K


def geometric_interval(ratios, seed=0):
    """Geometric mean and 95% bootstrap interval over formations."""
    logs = np.log(np.asarray(ratios, dtype=float))
    if len(logs) < 2:
        return float(np.exp(logs.mean())), np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = logs[rng.integers(0, len(logs), size=(BOOTSTRAPS, len(logs)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.exp(logs.mean())), float(np.exp(lo)), float(np.exp(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--sigmas", default="0.0001,0.001,0.01,0.03,0.1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--formations", type=int, default=1,
                    help="random formations per domain; one reproduces the single-graph table")
    ap.add_argument("--domains", default=",".join(DOMAINS),
                    help="comma-separated subset of " + ",".join(DOMAINS))
    ap.add_argument("--csv", default=None, help="write one row per (domain, formation, sigma)")
    ap.add_argument("--from-csv", default=None,
                    help="glob of csv files written by --csv; summarize them instead of measuring")
    args = ap.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]
    domains = args.domains.split(",")

    rows = []
    if args.from_csv:
        import glob
        for path in sorted(glob.glob(args.from_csv)):
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    rows.append({k: (v if k == "domain" else float(v)) for k, v in r.items()})
        domains = [d for d in DOMAINS if any(r["domain"] == d for r in rows)]
        sigmas = sorted({r["sigma"] for r in rows})
    for tag in ([] if args.from_csv else domains):
        doms = DOMAINS[tag]
        for f in range(args.formations):
            seed = args.seed + 1000 * f
            net, rank_K = formation(doms, seed)
            pred_p, pred_a = E.predicted_error(net, rank_K)
            for s in sigmas:
                got = E.monte_carlo_error(net, s, trials=args.trials,
                                          rng=np.random.default_rng(seed + 6), rank_K=rank_K)
                mp, ma = got["position"]["rms"], got["attitude"]["rms"]
                rows.append({"domain": tag, "formation": f, "seed": seed, "n": len(doms),
                             "edges": int(net.edges.sum()), "sigma": s, "sigma_deg": np.degrees(s),
                             "pred_pos": s * pred_p, "meas_pos": mp,
                             "ratio_pos": mp / (s * pred_p) if pred_p > 0 else np.nan,
                             "pred_att": s * pred_a, "meas_att": ma,
                             "ratio_att": ma / (s * pred_a) if pred_a > 0 else np.nan,
                             "eps": float(np.hypot(pred_p, pred_a)), "eps_pos": pred_p,
                             "failed_pos": got["position"]["failed"]})
            print(f"  {tag} formation {f + 1}/{args.formations} done", flush=True)

    if args.csv and not args.from_csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print("\nPosition RMS, measured over predicted. Geometric mean over formations with a")
    print("95% bootstrap interval; 'within 10%' counts formations with a ratio in [0.9, 1.1].")
    print("Position is in formation radii, attitude in radians.\n")
    head = (f"{'domain':8s} {'sigma':>7s} {'deg':>5s} | {'ratio':>6s} {'95% interval':>15s} "
            f"{'within 10%':>10s} {'failed':>7s} | {'median sigma*eps':>16s}")
    print(head)
    print("-" * len(head))
    for tag in domains:
        for s in sigmas:
            cell = [r for r in rows if r["domain"] == tag and r["sigma"] == s]
            ratios = [r["ratio_pos"] for r in cell if np.isfinite(r["ratio_pos"]) and r["ratio_pos"] > 0]
            g, lo, hi = geometric_interval(ratios)
            within = sum(0.9 <= r <= 1.1 for r in ratios)
            failed = np.mean([r["failed_pos"] for r in cell])
            se = np.median([s * r["eps"] for r in cell])
            print(f"{tag:8s} {s:7.4f} {np.degrees(s):5.2f} | {g:6.3f} [{lo:6.3f}, {hi:6.3f}] "
                  f"{within:>4d}/{len(ratios):<5d} {failed:7.3f} | {se:16.4f}")
        print()

    # Pooled over every domain, by the noise level times the position error coefficient.
    x = np.array([r["sigma"] * r["eps_pos"] for r in rows])
    y = np.array([r["ratio_pos"] for r in rows])
    good = np.isfinite(y) & (y > 0)
    x, y = x[good], y[good]
    print(f"{'sigma*eps_pos':>16s} {'cells':>6s} {'ratio':>7s} {'within 10%':>11s}")
    bins = [0, 0.01, 0.03, 0.1, 0.2, 0.5, 1, np.inf]
    for a, b in zip(bins[:-1], bins[1:]):
        m = (x >= a) & (x < b)
        if m.any():
            print(f"{a:6.2f} - {b:6.2f}   {m.sum():6d} {np.exp(np.log(y[m]).mean()):7.3f} "
                  f"{100 * np.mean(np.abs(y[m] - 1) <= 0.1):10.0f}%")
    print()
    print("A ratio near 1 means the linearisation holds at that noise level. It fails")
    print("in both directions: above 1 the nonlinearity amplifies the error, below 1")
    print("the estimator saturates on a wrong-but-bounded configuration instead.")


if __name__ == "__main__":
    main()
