"""What centre and width does a reference-free conditioning term need?

phi's conditioning bonus used to be a sigmoid of log10(lambda / lambda_ref), where
lambda_ref was the median lambda of `stiffness_ref_samples` greedy constructions on
the episode's own poses. lambda has no fixed scale, so it needed a per-episode
yardstick -- and one construction rebuilds B once per candidate edge, which at n=20
is 2290 rigidity-matrix builds per reset against 1.6 per step.

`shape_err = sqrt(tr((B^T B)^+) / n)` on the length-normalised B needs no yardstick:
the formation's size is already divided out by the length normalisation and the agent
count by the `/n`, so what is left is comparable enough to centre once, in advance.
This derives that centre and width, and prices the one correction that was considered
and dropped.

    OMP_NUM_THREADS=1 PYTHONPATH=. uv run tools/shape_error_scale.py [--quick]
"""
import sys

import numpy as np

from scenario import random_scenario
import rigidity as R

MIX5 = ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"]

CONFIGS = [(8, "R^3"), (10, "R^3"), (12, "R^3"), (16, "R^3"), (20, "R^3"),
           (8, "SE(3)"), (10, "SE(3)"), (16, "SE(3)"),
           (10, "R^2"), (10, "R^2xS^1"), (10, "R^3xS^1"),
           (8, MIX5), (10, MIX5 * 2), (16, MIX5 * 4), (20, MIX5 * 4)]


def instances(quick=False):
    """Per instance: log10 shape_err over near-minimal rigid graphs on its own poses.

    Near-minimal is the population the term has to discriminate within, since the
    bonus is gated on rigidity and the edge term already charges for density.
    """
    per_instance, per_config = [], {}
    for n, dom in CONFIGS:
        rng = np.random.default_rng(0)
        np.random.seed(0)
        label = f"n={n:<3d} {'mixed' if isinstance(dom, list) else dom}"
        pool = []
        for _ in range(3 if quick else 8):
            net, _ = random_scenario(n, dom, edge_count=0)
            rank_K = int(np.linalg.matrix_rank(
                R.scaled_rigidity_matrix(net.fully_connected())))
            vals = []
            for _ in range(2 if quick else 4):
                w = net.__class__.__new__(net.__class__)
                w.__dict__.update(net.__dict__)
                w.edges = np.zeros((n, n), dtype=bool)
                R.greedy_rigid_repair(w, rank_K, rng=rng)
                rank, s, _ = R.rigidity_decomposition(
                    R.scaled_rigidity_matrix(w), rank_K)
                if rank < rank_K:
                    continue
                vals.append(float(np.log10(np.sqrt((1.0 / (s[:rank_K] ** 2)).sum() / n))))
            if vals:
                per_instance.append((n, float(np.median(vals)), float(np.ptp(vals))))
                pool += vals
        per_config[label] = (n, np.asarray(pool))
    return per_instance, per_config


def constants(per_instance, alpha):
    """(centre, width) for a given n-correction exponent.

    The logistic has to stay responsive over the band a graph can actually occupy:
    the spread within one instance, plus whatever a fixed centre fails to remove.
    """
    ns = np.array([r[0] for r in per_instance], float)
    med = np.array([r[1] for r in per_instance])
    adj = med - alpha * np.log10(ns)
    centre = float(np.median(adj))
    within = float(np.mean([r[2] for r in per_instance]))
    return centre, (within + float(np.ptp(adj))) / 4.0, adj


def main(quick=False):
    per_instance, per_config = instances(quick)

    print(f"\nlog10 shape_err, median [p10-p90] over near-minimal rigid graphs\n")
    print(f"{'configuration':22s} {'shape_err':>18s}")
    print("-" * 42)
    for label, (n, pool) in per_config.items():
        print(f"{label:22s} {np.median(pool):>11.2f} "
              f"[{np.percentile(pool, 90) - np.percentile(pool, 10):4.2f}]")
    meds = [np.median(p) for _, p in per_config.values()]
    print(f"\n  drift of the centre across configurations: {np.ptp(meds):.2f} decades")

    # An off-centre instance is not a biased one: n is fixed within an episode, so a
    # constant offset cancels in a potential-based reward. What it costs is gradient,
    # and that is what decides whether an n-correction is worth a fitted exponent.
    print(f"\n  {len(per_instance)} instances. An n-correction shifts the centre; what it "
          f"has to buy is gradient.\n")
    print(f"{'exponent':>9s} {'centre':>8s} {'width':>7s} {'|offset| med':>13s} {'max':>6s}"
          f" {'gradient kept med':>18s} {'p10':>6s} {'worst':>6s}")
    for alpha in (0.0, 1.0, 1.88, 2.0):
        centre, width, adj = constants(per_instance, alpha)
        off = np.abs(adj - centre)
        q = 1.0 / (1.0 + np.exp(off / width))
        grad = q * (1 - q) / 0.25          # relative to the logistic's peak slope
        print(f"{alpha:9.2f} {centre:8.2f} {width:7.2f} {np.median(off):13.2f} "
              f"{off.max():6.2f} {np.median(grad):18.2f} "
              f"{np.percentile(grad, 10):6.2f} {grad.min():6.2f}")
    print("\n  gradient kept = the logistic's slope over that instance's own band,")
    print("  against its peak slope. The exponent moves the median and the p10 not at")
    print("  all, because a wider sigmoid absorbs the drift by itself, so it is off.")

    centre, width, _ = constants(per_instance, R.SHAPE_ERR_EXPONENT)
    print(f"\n  at SHAPE_ERR_EXPONENT = {R.SHAPE_ERR_EXPONENT}:")
    print(f"  SHAPE_ERR_CENTRE          = {centre:.2f}")
    print(f"  SHAPE_ERR_SIGMOID_DECADES = {width:.2f}")
    print(f"\n  in rigidity.py: {R.SHAPE_ERR_CENTRE}, {R.SHAPE_ERR_SIGMOID_DECADES}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
