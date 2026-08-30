"""Frozen evaluation instances, so a number measured today is comparable next month.

Every config regeneration resamples the instance distribution -- which already made
one pair of n=16 evaluations incomparable (52.25 vs 23.70 initial edges). A benchmark
pins poses, domains and initial edges to disk; every method and every checkpoint then
sees literally the same graphs.

    uv run benchmark.py <environment_name> <benchmark_name> [--instances N] [--seed S]
    uv run benchmark.py rotate <source_benchmark> <benchmark_name> [--seed S]
    uv run benchmark.py list
    uv run evaluation.py <environment_name> --benchmark <benchmark_name>
"""
import copy
import hashlib
import json
import os
import sys

import numpy as np
import quaternion

DIR = "benchmarks"


def path(name):
    return os.path.join(DIR, f"{name}.npz")


def save(env, name, instances=20, seed=0):
    """Draw `instances` fresh instances from `env` and store them."""
    np.random.seed(seed)
    env.freeze_network = False

    pos, quats, edges = [], [], []
    for _ in range(instances):
        env.reset()
        net = env.network
        pos.append([a.pose.position for a in net.agents])
        quats.append([quaternion.as_float_array(a.pose.orientation) for a in net.agents])
        edges.append(net.edges.copy())

    net = env.network
    axes = np.array([[np.nan] * 3 if a.rotation_axis is None else a.rotation_axis
                     for a in net.agents], dtype=float)

    os.makedirs(DIR, exist_ok=True)
    np.savez_compressed(
        path(name),
        positions=np.asarray(pos, dtype=float),
        orientations=np.asarray(quats, dtype=float),
        edges=np.asarray(edges, dtype=bool),
        domains=np.asarray([a.domain for a in net.agents]),
        rotation_axes=axes,
        meta=np.asarray([json.dumps({
            "source_environment": getattr(env, "filepath", None) or "programmatic",
            "instances": instances, "seed": seed, "n": net.n,
        })]),
    )
    return path(name)


def load(name):
    """(list of Networks, meta dict). Networks are ready to assign to env.network."""
    from network import Network

    if not os.path.exists(path(name)):
        raise FileNotFoundError(
            f"no benchmark at {path(name)} -- create it with "
            f"`uv run benchmark.py <environment_name> {name}`")
    d = np.load(path(name), allow_pickle=False)
    domains = [str(x) for x in d["domains"]]
    axes = d["rotation_axes"]

    nets = []
    for k in range(d["positions"].shape[0]):
        net = Network(d["positions"][k], np.zeros_like(d["positions"][k]), d["edges"][k])
        for i, agent in enumerate(net.agents):
            agent.pose.position = np.array(d["positions"][k][i], dtype=float)
            agent.pose.orientation = quaternion.from_float_array(d["orientations"][k][i])
            agent.domain = domains[i]
            agent.rotation_axis = None if np.isnan(axes[i]).any() else np.array(axes[i])
        nets.append(net)
    return nets, json.loads(str(d["meta"][0]))


def rotate(source, name, seed=0):
    """A copy of `source` with one random global rotation per instance.

    The task is rotation invariant; the R^d observation is not, so the pair measures
    what that costs.
    """
    import copy
    nets, meta = load(source)
    rng = np.random.default_rng(seed)
    pos, quats, edges = [], [], []
    for net in nets:
        net = copy.deepcopy(net)
        planar = any(a.domain in ("R^2", "R^2xS^1") for a in net.agents)
        axis = np.array([0.0, 0.0, 1.0]) if planar else rng.normal(size=3)
        net.rotate_network(axis / np.linalg.norm(axis), rng.uniform(0, 2 * np.pi))
        pos.append([a.pose.position for a in net.agents])
        quats.append([quaternion.as_float_array(a.pose.orientation) for a in net.agents])
        edges.append(net.edges.copy())

    ref = nets[0]
    os.makedirs(DIR, exist_ok=True)
    np.savez_compressed(
        path(name),
        positions=np.asarray(pos, dtype=float),
        orientations=np.asarray(quats, dtype=float),
        edges=np.asarray(edges, dtype=bool),
        domains=np.asarray([a.domain for a in ref.agents]),
        rotation_axes=np.array([[np.nan] * 3 if a.rotation_axis is None else a.rotation_axis
                                for a in ref.agents], dtype=float),
        meta=np.asarray([json.dumps({"source_environment": f"{source} rotated",
                                     "instances": len(nets), "seed": seed, "n": ref.n})]),
    )
    return path(name)


def digest(name):
    """Short hash of the instances themselves, recorded next to results.

    Over the arrays, not the file: npz is a zip and stores timestamps, so hashing
    the bytes gave a different digest every time an identical benchmark was
    rewritten -- exactly the false mismatch this is meant to detect.
    """
    d = np.load(path(name), allow_pickle=False)
    h = hashlib.sha256()
    for key in ("positions", "orientations", "edges", "domains", "rotation_axes"):
        h.update(np.ascontiguousarray(d[key]).tobytes())
    return h.hexdigest()[:12]


def available():
    if not os.path.isdir(DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(DIR) if f.endswith(".npz"))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "rotate":
        seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
        out = rotate(sys.argv[2], sys.argv[3], seed=seed)
        print(f"wrote {out}  (rotated copy of {sys.argv[2]}, seed {seed}, {digest(sys.argv[3])})")
        sys.exit(0)

    if len(sys.argv) >= 2 and sys.argv[1] == "list":
        for nm in available():
            _, meta = load(nm)
            print(f"  {nm:40s} {meta['instances']:>4d} instances  n={meta['n']:<3d} "
                  f"seed {meta['seed']}  {digest(nm)}")
        sys.exit(0)

    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    import argparse
    from environment import Environment

    ap = argparse.ArgumentParser()
    ap.add_argument("environment_name")
    ap.add_argument("benchmark_name")
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = Environment()
    env.load(f"./environments/{args.environment_name}.json")
    out = save(env, args.benchmark_name, instances=args.instances, seed=args.seed)
    print(f"wrote {out}  ({args.instances} instances, seed {args.seed}, {digest(args.benchmark_name)})")
