"""Create frozen mixed-domain evaluation sets for the thesis results."""

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import benchmark
from environment import Environment


ENVIRONMENT = (
    "env_actionAddRemoveEdgeDiscreteNoSelfLoops_"
    "rewardWeightedNormalizedShapeError_termMaxSteps_rigGFE_lean_"
    "n10_randdom_k10"
)
PAIRS = {
    "R2_SE3": ("R^2", "SE(3)"),
    "R3_R2xS1": ("R^3", "R^2xS^1"),
}


def create_set(env, pair, assignment, instances, seed, name):
    """Draw one frozen set with its domain assignment recorded per instance."""
    if os.path.exists(benchmark.path(name)):
        raise FileExistsError(benchmark.path(name))

    n = env.network.n
    if n % 2:
        raise ValueError("fixed-split sets require an even number of agents")

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    env.random_domains = False
    env.freeze_network = False
    nets = []

    for _ in range(instances):
        if assignment == "random":
            domains = rng.choice(pair, size=n).tolist()
        else:
            domains = np.array([pair[0]] * (n // 2) + [pair[1]] * (n // 2))
            domains = domains[rng.permutation(n)].tolist()
        env.domains = domains
        env.reset()
        nets.append(copy.deepcopy(env.network))

    return benchmark.save_networks(
        nets,
        benchmark.path(name),
        meta={
            "source_environment": ENVIRONMENT,
            "seed": seed,
            "domain_pair": list(pair),
            "assignment": assignment,
            "instances": instances,
            "n": n,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default=ENVIRONMENT)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4401)
    args = parser.parse_args()

    env = Environment()
    with benchmark.without_spectral_references():
        env.load(f"environments/{args.environment}.json")

        index = 0
        for pair_name, pair in PAIRS.items():
            for assignment in ("random", "fixed"):
                index += 1
                name = f"b2_n10_mixed_{assignment}_{pair_name}"
                path = create_set(
                    env, pair, assignment, args.instances, args.seed + index, name
                )
                print(
                    f"wrote {path}  ({args.instances} instances, "
                    f"{benchmark.digest(name)})"
                )


if __name__ == "__main__":
    main()
