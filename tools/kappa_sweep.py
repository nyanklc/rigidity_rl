"""What does raising stiffness_kappa actually buy, and what does it cost in edges?

Runs one method across several kappa on the SAME instances and prints edges,
margin and rigidity per kappa. This is the Pareto front in miniature: kappa < 1
should move the margin at a flat edge count, kappa > 1 should start buying edges.

`greedy` needs no checkpoint and hill-climbs on whatever phi is configured, so it
gives a reading before any policy is trained; `--model` scores a trained one the
same way.

    PYTHONPATH=. uv run tools/kappa_sweep.py
    PYTHONPATH=. uv run tools/kappa_sweep.py --n 8 --domain SE\(3\) --instances 20
    PYTHONPATH=. uv run tools/kappa_sweep.py --kappas 0,0.5,1,2,4,8
"""
import argparse
import copy

import numpy as np

import baselines as B
from environment import Environment
from rigidity import rigidity_eigenvalue


def build(n, domain, kappa, samples):
    e = Environment()
    e.initialize(n, domain,
                 action_space_type="AddRemoveEdgeDiscreteNoSelfLoops",
                 obs_space_type="Dict",
                 state_score_type="WeightedNormalized",
                 termination_condition_type="MaxSteps",
                 max_steps=4 * n + 10,
                 random_graph_with_mean_min_edges=True,
                 stiffness_kappa=kappa,
                 stiffness_ref_samples=samples)
    return e


def run_one(n, domain, kappa, samples, instances, seed0):
    edges, margins, rigid = [], [], []
    for i in range(instances):
        # same seed per instance across kappa, so every arm sees identical poses
        np.random.seed(seed0 + i)
        e = build(n, domain, kappa, samples)
        e.reset()
        e.margin_rng = np.random.default_rng(0)
        e.compute_episode_constants()
        e.freeze_network = True
        B.run_greedy(e, verbose=False)
        edges.append(int(e.network.edges.sum()))
        margins.append(rigidity_eigenvalue(e.network, rank_K=e.rank_K))
        rigid.append(int(e.network.is_MBR(rank_K=e.rank_K)[1]))
    live = [x for x in margins if x > 0]
    gmean = float(10.0 ** np.mean(np.log10(live))) if live else 0.0
    return np.mean(edges), gmean, np.mean(rigid)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--domain", default="R^3")
    p.add_argument("--kappas", default="0,0.9,2,4")
    p.add_argument("--instances", type=int, default=12)
    p.add_argument("--samples", type=int, default=3, help="stiffness_ref_samples")
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    kappas = [float(k) for k in args.kappas.split(",")]
    rows = [(k,) + run_one(args.n, args.domain, k, args.samples,
                           args.instances, args.seed) for k in kappas]

    base = next((g for k, _, g, _ in rows if k == 0.0), rows[0][2]) or 1.0
    print(f"\ngreedy, n={args.n} {args.domain}, {args.instances} instances, "
          f"identical poses across arms")
    print(f"\n{'kappa':>7} {'edges':>7} {'margin (gmean)':>16} {'vs k=0':>9} {'rigid':>7}")
    for k, m, g, r in rows:
        print(f"{k:>7} {m:>7.2f} {g:>16.2e} {g / base:>8.2f}x {r * 100:>6.0f}%")
    print("\nflat edges with rising margin = tie-break; rising edges = buying margin\n")


if __name__ == "__main__":
    main()
