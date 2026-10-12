"""The generalization shards, read as paired per-instance comparisons.

One policy trained on ten-agent formations with random domains is evaluated on
ten frozen instance sets that vary the size, the domain mix, and the global frame.
The existing table reports means side by side; what a transfer claim needs is the
per-instance difference and how far it is from zero.

    PYTHONPATH=. uv run tools/generalisation_paired.py [--csv out.csv]
"""
import argparse
import csv
import glob
import json
import math
import numpy as np

SHARDS = sorted(glob.glob("runs_outputs/20260912-18*__gen/"))
ORDER = ["bench_n8_R3", "bench_n8_R3_rot", "bench_n8_R3xS1", "bench_n8_SE3",
         "bench_n8_randdom", "bench_n10_randdom", "bench_n16_R3",
         "bench_n16_SE3", "bench_n16_randdom", "bench_n20_randdom"]
NICE = {"bench_n8_R3": "n=8  $\\R^3$", "bench_n8_R3_rot": "n=8  $\\R^3$ rotated",
        "bench_n8_R3xS1": "n=8  $\\R^3{\\times}S^1$", "bench_n8_SE3": "n=8  SE(3)",
        "bench_n8_randdom": "n=8  mixed", "bench_n10_randdom": "n=10 mixed (trained)",
        "bench_n16_R3": "n=16 $\\R^3$", "bench_n16_SE3": "n=16 SE(3)",
        "bench_n16_randdom": "n=16 mixed", "bench_n20_randdom": "n=20 mixed"}


def find(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key and not isinstance(v, (dict, list)):
                return v
            r = find(v, key)
            if r is not None:
                return r
    return None


def load():
    out = {}
    for d in SHARDS:
        bench = find(json.load(open(d + "meta.json")), "benchmark")
        rows = list(csv.DictReader(open(d + "results.csv")))
        by = {(r["method"], r["episode"]): r for r in rows}
        eps = sorted({r["episode"] for r in rows})
        out[bench] = (by, eps)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); a = ap.parse_args()
    data = load()
    rng = np.random.default_rng(0)
    recs = []

    print("Policy minus greedy, paired per instance. CI is a 90% percentile bootstrap.\n")
    print(f"{'instance set':24s} {'d edges':>16s} {'at-bound p/g':>13s} "
          f"{'err ratio':>18s} {'rigid':>7s}")
    for b in ORDER:
        if b not in data:
            continue
        by, eps = data[b]
        de = np.array([float(by[("learned", e)]["m"]) - float(by[("greedy", e)]["m"])
                       for e in eps])
        bs = np.array([rng.choice(de, len(de), replace=True).mean() for _ in range(4000)])
        lo, hi = np.percentile(bs, 5), np.percentile(bs, 95)
        abp = sum(1 for e in eps if by[("learned", e)]["at_bound"] in ("True", "1"))
        abg = sum(1 for e in eps if by[("greedy", e)]["at_bound"] in ("True", "1"))
        lr = np.array([math.log(float(by[("learned", e)]["shape_err"]) /
                                float(by[("greedy", e)]["shape_err"])) for e in eps
                       if by[("learned", e)]["shape_err"] and by[("greedy", e)]["shape_err"]])
        bs2 = np.array([rng.choice(lr, len(lr), replace=True).mean() for _ in range(4000)])
        rlo, rhi = math.exp(np.percentile(bs2, 5)), math.exp(np.percentile(bs2, 95))
        rig = sum(1 for e in eps if by[("learned", e)]["is_IBR"] in ("True", "1"))
        star = "*" if (lo > 0 or hi < 0) else " "
        star2 = "*" if (rlo > 1 or rhi < 1) else " "
        print(f"{NICE[b]:24s} {de.mean():+6.2f} [{lo:+5.2f},{hi:+5.2f}]{star} "
              f"{abp:5d}/{abg:<6d} {math.exp(lr.mean()):6.3f} [{rlo:5.3f},{rhi:5.3f}]{star2} "
              f"{rig:3d}/{len(eps)}")
        recs.append(dict(bench=b, d_edges=de.mean(), lo=lo, hi=hi, at_pol=abp, at_gre=abg,
                         err_ratio=math.exp(lr.mean()), r_lo=rlo, r_hi=rhi,
                         rigid=rig, n_inst=len(eps)))
    print("\n  * = the 90% interval excludes no difference. "
          "err ratio < 1 means the policy's estimation error is lower.")

    tot_p = sum(r["at_pol"] for r in recs); tot_g = sum(r["at_gre"] for r in recs)
    tot_n = sum(r["n_inst"] for r in recs)
    print(f"\n  pooled over all {tot_n} instances: at the bound, "
          f"policy {tot_p} ({100*tot_p/tot_n:.0f}%), greedy {tot_g} ({100*tot_g/tot_n:.0f}%)")
    print(f"  sets where the policy reaches the bound more often: "
          f"{sum(1 for r in recs if r['at_pol'] > r['at_gre'])} of {len(recs)}")
    print(f"  sets where its estimation error is lower: "
          f"{sum(1 for r in recs if r['err_ratio'] < 1)} of {len(recs)}")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
