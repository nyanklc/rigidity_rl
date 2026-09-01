"""What does each observation flag cost per step and per episode?

One env per flag set, same poses and same action sequence, so the difference
between rows is the flag and nothing else. Step and reset are reported apart:
several flags do their work once per episode in `compute_episode_constants`, and
a table that mixes the two hides which.

Pin BLAS to one thread. Unpinned, a 96x96 `eigh` on a loaded machine times
anywhere from 0.2 to 16 ms and every row becomes noise.

    OMP_NUM_THREADS=1 PYTHONPATH=. uv run tools/flag_cost.py [--domain R^3]
"""
import argparse
import time

import numpy as np

from environment import Environment

FLAGSETS = [
    ("baseline (all off)",   dict()),
    ("graph_features",       dict(graph_features=True)),
    ("rigidity_global",      dict(rigidity_global=True)),
    ("rigidity_quality",     dict(rigidity_quality=True)),
    ("rigidity_flex",        dict(rigidity_flex=True)),
    ("rigidity_edge",        dict(rigidity_edge=True)),
    ("rigidity_stiffness",   dict(rigidity_stiffness=True)),
    ("rigidity_removal",     dict(rigidity_removal=True)),
    ("all six",              dict(rigidity_global=True, rigidity_quality=True,
                                  rigidity_flex=True, rigidity_edge=True,
                                  rigidity_stiffness=True, rigidity_removal=True)),
    ("all six + graph_feat", dict(graph_features=True, rigidity_global=True,
                                  rigidity_quality=True, rigidity_flex=True,
                                  rigidity_edge=True, rigidity_stiffness=True,
                                  rigidity_removal=True)),
]


def make(n, domain, seed, **kw):
    # the global stream is the one poses are drawn from, so reseeding here is what
    # makes every flag set see the same instance
    np.random.seed(seed)
    opts = dict(action_space_type="SelectNodesSequentially", obs_space_type="Dict",
                state_score_type="WeightedNormalized",
                termination_condition_type="MaxSteps", max_steps=10 ** 6,
                track_data_enable=False, skip_is_stop=False,
                random_graph_with_mean_min_edges=True, graph_features=False)
    opts.update(kw)
    env = Environment()
    env.initialize(n, domain, **opts)
    return env


def time_steps(env, steps, seed, reps=3):
    rng = np.random.default_rng(seed)
    actions = [int(rng.integers(env.action_space.n)) for _ in range(steps)]
    out = []
    for _ in range(reps):
        env.reset()
        t0 = time.perf_counter()
        for a in actions:
            env.step(a)
        out.append((time.perf_counter() - t0) / steps * 1e3)
    return float(np.median(out))


def time_reset(env, k, seed):
    np.random.seed(seed)
    t0 = time.perf_counter()
    for _ in range(k):
        env.reset()
    return (time.perf_counter() - t0) / k * 1e3


def table(title, values, ns, fmt="{:>10.2f}"):
    print(title)
    print(f"{'flags':<24}" + "".join(f"{'n=' + str(n):>10}" for n in ns))
    for label, _ in FLAGSETS:
        print(f"{label:<24}" + "".join(fmt.format(values[(label, n)]) for n in ns))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="R^3")
    ap.add_argument("--n", default="8,16")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--resets", type=int, default=3)
    args = ap.parse_args()
    ns = [int(x) for x in args.n.split(",")]

    print(f"domain={args.domain} steps/measurement={args.steps}\n")
    step_ms, reset_ms = {}, {}
    for label, kw in FLAGSETS:
        for n in ns:
            env = make(n, args.domain, seed=7, **kw)
            step_ms[(label, n)] = time_steps(env, args.steps, seed=11)
            reset_ms[(label, n)] = time_reset(env, args.resets, seed=13)

    base = FLAGSETS[0][0]
    ratio = {k: step_ms[k] / step_ms[(base, k[1])] for k in step_ms}
    table("ms per env.step()", step_ms, ns)
    table("x baseline (step)", ratio, ns)
    table("ms per env.reset()  (episode constants)", reset_ms, ns)

    print("Any rigidity flag pays for nullspace() and candidate_gain(), which")
    print("compute_rigidity_features() runs before it branches on which flag is set.")


if __name__ == "__main__":
    main()
