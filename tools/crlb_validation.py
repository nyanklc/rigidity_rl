"""Does the predicted shape error match the measured one, and where does it stop?

Perturb every bearing by sigma radians, solve for the shape, compare against the
Cramer-Rao prediction read off B's spectrum. The prediction is a linearisation,
so the second question is the noise level above which it stops being one.

    PYTHONPATH=. uv run tools/crlb_validation.py
"""
import argparse

import numpy as np

import estimation as E
from rigidity import extended_bearing_rigidity_matrix as B_of, greedy_rigid_construction
from scenario import random_scenario

DOMAINS = {
    "R^3": ["R^3"] * 8,
    "R^2xS^1": ["R^2xS^1"] * 8,
    "R^3xS^1": ["R^3xS^1"] * 8,
    "SE(3)": ["SE(3)"] * 8,
    "mixed": ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"] * 2,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--sigmas", default="0.0001,0.001,0.01,0.03,0.1")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]

    print("Cramer-Rao bounds E[||x||^2], so the RMS over trials is what it predicts;")
    print("the MEAN sits about 1/(4k) below that for k identifiable modes.")
    print("Position is in formation radii, attitude in radians.\n")
    head = (f"{'domain':8s} {'sigma':>8s} {'deg':>5s} | {'pos rms':>10s} {'pos pred':>10s} "
            f"{'ratio':>6s} | {'att rms':>10s} {'att pred':>10s} {'ratio':>6s}")
    print(head)
    print("-" * len(head))

    for tag, doms in DOMAINS.items():
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        net, _ = random_scenario(len(doms), doms, edge_count=0)
        rank_K = int(np.linalg.matrix_rank(B_of(net.fully_connected())))
        greedy_rigid_construction(net, rank_K, rng)
        pred_p, pred_a = E.predicted_error(net, rank_K)

        for s in sigmas:
            got = E.monte_carlo_error(net, s, trials=args.trials,
                                      rng=np.random.default_rng(args.seed + 6),
                                      rank_K=rank_K)
            mp, ma = got["position"]["rms"], got["attitude"]["rms"]
            rp = mp / (s * pred_p) if pred_p > 0 else np.nan
            ra = ma / (s * pred_a) if pred_a > 0 else np.nan
            print(f"{tag:8s} {s:8.4f} {np.degrees(s):5.1f} | {mp:10.3e} {s * pred_p:10.3e} "
                  f"{rp:6.3f} | {ma:10.3e} {s * pred_a:10.3e} {ra:6.3f}")
        print()

    print("A ratio near 1 means the linearisation holds at that noise level. It fails")
    print("in both directions: above 1 the nonlinearity amplifies the error, below 1")
    print("the estimator saturates on a wrong-but-bounded configuration instead.")


if __name__ == "__main__":
    main()
