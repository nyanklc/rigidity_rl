"""Everything about an environment config in one place.

The switches it was generated with, the observation layout channel by channel
with the statistics each channel actually takes, the episode constants, and what
a run against it would cost in time and in instance coverage.

The observation layout is the part worth checking: it is built by concatenation
in build_dict_obs and the order is mirrored by hand in ablation.py, so a channel
added in one place and not the other is silent.

    PYTHONPATH=. uv run tools/env_report.py <environment_name>
    PYTHONPATH=. uv run tools/env_report.py --all
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from environment import Environment

DIM = {"R^2": 2, "R^3": 3, "R^2xS^1": 3, "R^3xS^1": 4, "SE(3)": 6}

NODE_BLOCKS = [("domain", 5, None), ("degree", 2, None),
               ("closeness", 1, "graph_features"), ("eigenvector", 1, "graph_features"),
               ("node_between", 1, "graph_features"),
               ("rigidity_glob", 3, "rigidity_global"), ("flex_mag", 1, "rigidity_flex")]
EDGE_BLOCKS = [("bearings", 3, None), ("edge_exists", 1, None),
               ("edge_between", 1, "graph_features"), ("reciprocity", 1, None),
               ("common_nbrs", 1, None), ("flex_align", 1, "rigidity_flex"),
               ("block_rank", 1, "rigidity_edge")]


def blocks(spec, cfg, width, label):
    out, i = [], 0
    for name, w, flag in spec:
        if flag is not None and not cfg.get(flag, False):
            continue
        out.append((name, i, i + w))
        i += w
    if i != width:
        print(f"    ! {label} layout says {i} channels, observation has {width}."
              f" build_dict_obs and this table have drifted apart.")
    return out


def stats(env, key, spec, resets=12):
    acc = []
    for _ in range(resets):
        obs, _ = env.reset()
        acc.append(np.asarray(obs[key]))
    A = np.concatenate([a.reshape(-1, a.shape[-1]) for a in acc], 0)
    print(f"\n  {key}   shape {acc[0].shape}   width {A.shape[-1]}")
    print(f"    {'channel':14s} {'cols':>9s} {'mean':>9s} {'sd':>9s} {'min':>9s} {'max':>9s}")
    for name, a, b in blocks(spec, env_cfg, A.shape[-1], key):
        sl = A[:, a:b]
        print(f"    {name:14s} {f'{a}:{b}':>9s} {sl.mean():9.3f} {sl.std():9.3f} "
              f"{sl.min():9.3f} {sl.max():9.3f}")


def report(name, cost=True):
    global env_cfg
    path = f"environments/{name}.json"
    env_cfg = json.load(open(path))
    env = Environment()
    env.load(path)
    env.device = "cpu"
    obs, _ = env.reset()

    doms = env.domains if isinstance(env.domains, list) else [env.domains] * env.n
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"  n {env.n}   domains {dict((d, doms.count(d)) for d in sorted(set(doms)))}")
    print(f"  sum dim D_i {sum(DIM[d] for d in doms)}   rank_K {env.rank_K}   "
          f"c_max {env.c_max}   m_req {env.m_req}")
    phi_opt = (100 * env.rank_K - 25 * env.m_req * env.c_max) / env.rank_K
    print(f"  phi ceiling (at m = m_req) {phi_opt:.2f}   "
          f"one edge costs {25 * env.c_max / env.rank_K:.2f}")

    print("\n  switches")
    keys = ["action_type", "state_score_type", "termination_condition_type", "max_steps",
            "skip_enabled", "skip_is_stop", "time_penalty_value", "rotation_augmentation",
            "random_graph_with_mean_min_edges", "only_randomize_edges",
            "include_candidate_bearings", "graph_features",
            "rigidity_global", "rigidity_flex", "rigidity_edge", "scenario"]
    for k in keys:
        v = env_cfg.get(k, "(absent)")
        note = ""
        if k == "max_steps":
            note = f"   4*m_req+10 = {4 * env.m_req + 10}"
        if k == "skip_enabled" and v:
            note = "   stop-action arm"
        if k == "rotation_augmentation" and v:
            note = "   augmentation arm"
        print(f"    {k:34s} {str(v):10s}{note}")

    stats(env, "node_features", NODE_BLOCKS)
    stats(env, "edge_features", EDGE_BLOCKS)
    for k in ("coord_features", "adj", "selection"):
        if k in obs:
            print(f"\n  {k:14s} shape {np.asarray(obs[k]).shape}")

    if cost:
        env.reset()
        t0 = time.time()
        for _ in range(60):
            env.step(env.action_space.sample())
        step = (time.time() - t0) / 60
        t0 = time.time()
        for _ in range(15):
            env.reset()
        reset = (time.time() - t0) / 15
        ms = env_cfg["max_steps"]
        print(f"\n  cost   step {1000 * step:.2f} ms   reset {1000 * reset:.2f} ms"
              f"   ({1000 * reset / ms:.2f} ms/step amortised)")
        print(f"         400k steps -> {400000 * (step + reset / ms) / 3600:.2f} h,"
              f" {400000 // ms} instances, {10000 * 4 // ms} of them in a 10k x 4 buffer")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("environment", nargs="?", default=None)
    ap.add_argument("--all", action="store_true", help="every config in environments/")
    ap.add_argument("--no-cost", action="store_true")
    args = ap.parse_args()

    if args.all:
        names = sorted(os.path.basename(p)[:-5] for p in glob.glob("environments/*.json"))
    elif args.environment:
        names = [args.environment]
    else:
        print("usage: env_report.py <environment_name> | --all")
        print("\navailable:")
        for p in sorted(glob.glob("environments/*.json")):
            print("   ", os.path.basename(p)[:-5])
        raise SystemExit(1)

    for nm in names:
        try:
            report(nm, cost=not args.no_cost)
        except Exception as e:
            print(f"\n{nm}\n  could not load: {type(e).__name__}: {e}")
