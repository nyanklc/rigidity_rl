"""How does each baseline's cost grow with n, and does the cheaper one give up anything?

One evaluation run measures cost at one n, which cannot show a scaling exponent.
`greedy` rescores all n(n-1) toggles per improvement step and takes O(n) of them,
so it is O(n^6) per network; `spectral` reads the same landscape off one pass of
the rigidity algebra. This is where that difference becomes a number.

Two tables. The first checks that the cheaper method is not cheaper by giving
something up: at stiffness_kappa = 0 the closed form is exact, so spectral should
reach greedy's answer edge for edge in every domain. The rest are the scaling.

Counts come from cost.py and are the same on any machine. The milliseconds are
not, so pin BLAS to one thread; see tools/flag_cost.py.

    OMP_NUM_THREADS=1 PYTHONPATH=. uv run tools/cost_scaling.py
"""
import argparse
import copy

import numpy as np

import cost
import outputs as E
from environment import Environment

METHODS = ("degree", "spectral", "greedy", "constructive", "anneal")


def make(n, domain, seed, **kw):
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


def run_one(name, env, rng, budget):
    if name == "greedy":
        return E.run_greedy(env, verbose=False)
    if name == "spectral":
        return E.run_spectral(env, verbose=False)
    if name == "degree":
        return E.run_degree(env, rng, verbose=False)
    if name == "anneal":
        return E.run_anneal(env, rng, budget=budget, verbose=False)
    if name == "constructive":
        return E.run_constructive(env, rng, restarts=1, verbose=False)
    raise ValueError(name)


def measure(ns, domain, episodes, seed, methods):
    """{method: {n: (calls, ms, edges)}}, every method on the same instances."""
    out = {m: {} for m in methods}
    for n in ns:
        env = make(n, domain, seed)
        acc = {m: [0.0, 0.0, 0.0] for m in methods}
        for ep in range(episodes):
            env.freeze_network = False
            env.reset()
            instance = copy.deepcopy(env.network)
            env.freeze_network = True
            # greedy runs first so the annealer can be given its phi-evaluation count
            budget = 4 * n * (n - 1)
            for name in methods:
                env.network = copy.deepcopy(instance)
                env.reset()
                rng = np.random.default_rng(seed + ep)
                with cost.Meter() as m:
                    res = run_one(name, env, rng, budget)
                if name == "greedy":
                    budget = m.counts.get("score_network", budget)
                acc[name][0] += m.total()
                acc[name][1] += m.ms
                acc[name][2] += res["m"]
        for name in methods:
            out[name][n] = tuple(v / episodes for v in acc[name])
    return out


def equivalence(domains, ns, episodes, seed):
    """Does spectral give up anything for being cheap? At kappa = 0 it should not.

    phi is affine in rank there, so the closed-form landscape is greedy's own landscape
    and the two should agree edge for edge. The cost ratio beside it is what that costs.
    """
    print("stiffness_kappa = 0: does spectral reach greedy's answer, and for how much less?")
    print(f"{'domain':<10}{'n':>3}{'same phi':>11}{'same edges':>12}"
          f"{'greedy':>9}{'spectral':>10}{'cheaper by':>12}")
    for domain in domains:
        for n in ns:
            env = make(n, domain, seed)
            same_phi = same_m = 0
            greedy_calls = spectral_calls = 0
            for ep in range(episodes):
                env.freeze_network = False
                env.reset()
                instance = copy.deepcopy(env.network)
                env.freeze_network = True

                env.network = copy.deepcopy(instance)
                env.reset()
                with cost.Meter() as mg:
                    g = E.run_greedy(env, verbose=False)
                env.network = copy.deepcopy(instance)
                env.reset()
                with cost.Meter() as ms:
                    s = E.run_spectral(env, verbose=False)

                same_phi += abs(g["score"] - s["score"]) < 1e-9
                same_m += g["m"] == s["m"]
                greedy_calls += mg.total()
                spectral_calls += ms.total()
            ratio = greedy_calls / max(spectral_calls, 1)
            print(f"{domain:<10}{n:>3}{same_phi:>8}/{episodes:<3}{same_m:>9}/{episodes:<3}"
                  f"{greedy_calls / episodes:>9.0f}{spectral_calls / episodes:>10.0f}"
                  f"{ratio:>11.1f}x")
    print()


def exponent(series):
    """Slope of log(cost) against log(n): the measured scaling exponent."""
    ns = sorted(series)
    if len(ns) < 2:
        return float("nan")
    x = np.log([float(n) for n in ns])
    y = np.log([max(series[n], 1e-9) for n in ns])
    return float(np.polyfit(x, y, 1)[0])


def table(data, ns, title, index):
    print(title)
    print(f"{'method':<14}" + "".join(f"n={n}".rjust(11) for n in ns) + "exponent".rjust(11))
    for name, series in data.items():
        cells = "".join(f"{series[n][index]:,.0f}".rjust(11) for n in ns)
        e = exponent({n: series[n][index] for n in ns})
        print(f"{name:<14}{cells}{e:>11.2f}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", default="6,8,12,16")
    p.add_argument("--domain", default="R^3")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--domains", default="R^2,R^3,R^2xS^1,R^3xS^1,SE(3)",
                   help="domains for the equivalence table")
    p.add_argument("--skip-equivalence", action="store_true")
    args = p.parse_args()

    ns = [int(x) for x in args.ns.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if not args.skip_equivalence:
        # kappa = 0, where the closed form is exact and the comparison is meaningful
        equivalence([d.strip() for d in args.domains.split(",")],
                    ns[:2], args.episodes, args.seed)

    data = measure(ns, args.domain, args.episodes, args.seed, methods)

    print(f"{args.domain}, {args.episodes} networks per size, same instances per row\n")
    table(data, ns, "rigidity computations per network", 0)
    table(data, ns, "milliseconds per network", 1)
    table(data, ns, "edges in the network produced", 2)
    print("The exponent is the slope of log(cost) against log(n). It understates the")
    print("asymptotic order at these sizes, where per-call overhead still dominates the")
    print("matrix work, and steepens as n grows.")


if __name__ == "__main__":
    main()
