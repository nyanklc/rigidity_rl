"""How much of `test_feature_scale_*`'s verdict is drift, and how much is the sampler?

`tests/test_scale_invariance.py` asserts that no observation channel's mean changes by
more than 1.6x (n=8 vs 16) or 1.7x (n=8/16/32) as the network grows. It samples fresh
graphs at each size with **no seed**, so the ratio it computes is a random variable and
the pass/fail is a draw from it. This measures that distribution, which is the thing the
test cannot tell you about itself.

What it found, at each test's own sizes and reps, counting how many seeds would fail:

  fast test, n = 8/16, reps 12, gate 1.6, 12 seeds
    degree            1.02 - 1.32     0/12
    node_freedom      1.03 - 1.34     0/12
    add_independence  1.01 - 1.65     1/12
    add_rank          1.05 - 1.79     3/12

  slow test, n = 8/16/32, reps 8, gate 1.7, 8 seeds
    degree            1.10 - 1.62     0/8
    node_freedom      1.06 - 1.46     0/8
    add_independence  1.16 - 2.34     5/8
    add_rank          1.31 - 2.45     4/8

`degree` and `node_freedom` are flat: they never reach either gate, so an intermittent
failure on them is the sampler. The two pair channels genuinely drift, and their spread
STRADDLES both gates -- the fast test passes them about three times in four by luck, and
the slow test fails them about half the time. Seeding would make the verdict reproducible
and would also silently choose it, since one seed is green and another is red on the same
code. That is a decision about the invariant rather than a bug fix, which is why it is
measured here and not applied to the test.

    OMP_NUM_THREADS=1 PYTHONPATH=. uv run tools/feature_scale_noise.py
    OMP_NUM_THREADS=1 PYTHONPATH=. uv run tools/feature_scale_noise.py --seeds 10
"""
import argparse

import numpy as np

# (label, obs key, slice) -- mirrors tests/test_scale_invariance.py:DRIFTING
CHANNELS = [("degree", "node_features", slice(5, 7)),
            ("node_freedom", "node_features", slice(10, 11)),
            ("add_independence", "edge_features", slice(6, 7)),
            ("add_rank", "edge_features", slice(8, 9))]


def rich_env(n, domain):
    """Exactly the environment the test builds.

    Not a re-spelling of it: the channel slices below are offsets into a layout that
    every observation flag moves, so an env built with different defaults measures a
    different column under the same name. The test's own `rich_env` is imported and the
    fixture's defaults are replayed here, so the two cannot drift apart.
    """
    from tests import conftest
    from tests.test_scale_invariance import rich_env as test_rich_env

    return test_rich_env(conftest.make_env.__wrapped__(), n, domain)


def channel_mean(n, domain, key, sl, reps, seed):
    np.random.seed(seed)
    e = rich_env(n, domain)
    e.action_space.seed(seed)
    vals = []
    for _ in range(reps):
        e.reset()
        for _ in range(4):
            e.step(e.action_space.sample())
        a = e._get_obs()[key]
        vals.append(a.reshape(-1, a.shape[-1])[:, sl])
    return float(np.concatenate(vals).mean())


def ratios(sizes, reps, seeds, domain):
    """{label: [ratio per seed]}, the quantity the test thresholds."""
    out = {}
    for label, key, sl in CHANNELS:
        got = []
        for s in range(seeds):
            means = {n: channel_mean(n, domain, key, sl, reps, s + 977 * n)
                     for n in sizes}
            lo, hi = min(means.values()), max(means.values())
            got.append(hi / max(lo, 1e-9))
        out[label] = got
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--domain", default="R^3")
    ap.add_argument("--reps", default="8,32",
                    help="graphs per size, comma separated. The fast test uses 12 and "
                         "the slow one 8")
    ap.add_argument("--sizes", default="8,16;8,16,32",
                    help="size sets, ';' separated. The two the test uses")
    args = ap.parse_args()

    gates = {"8,16": 1.6, "8,16,32": 1.7}
    for size_spec in args.sizes.split(";"):
        sizes = tuple(int(x) for x in size_spec.split(","))
        gate = gates.get(size_spec)
        head = f"sizes {sizes}" + (f", the test's gate is {gate}" if gate else "")
        print(f"\n{head}")
        print(f"  {'channel':18s} {'reps':>4s}  {'min':>5s} {'median':>7s} {'max':>5s}"
              f"   {'over gate':>9s}   ratios")
        for reps in (int(x) for x in args.reps.split(",")):
            for label, got in ratios(sizes, reps, args.seeds, args.domain).items():
                over = (sum(r >= gate for r in got) if gate else 0)
                flag = f"{over}/{len(got)}" if gate else "-"
                print(f"  {label:18s} {reps:4d}  {min(got):5.2f} {np.median(got):7.2f} "
                      f"{max(got):5.2f}   {flag:>9s}   "
                      + " ".join(f"{r:.2f}" for r in got))
    print("\n  `over gate` is how many seeds would FAIL the test. A channel that is")
    print("  neither 0/n nor n/n is one whose verdict the seed decides.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
