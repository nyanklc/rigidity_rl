"""Reference points for a trained policy.

Every method here is scored with Environment.compute_state_score, i.e. the exact same state
score phi the agent is trained on, so the numbers are directly comparable:

  initial  the graph the sampler produced (what the agent starts from)
  random   uniform random actions through env.step()  -> the floor for this action space
  greedy   hill-climbing on phi, one edge toggle at a time
  learned  a trained policy, acting deterministically
  optimal  exhaustive search for the fewest-edge rigid graph (small n only)

greedy gets stuck exactly where RL should win: states where no single edit improves phi but a
swap of two does. That is the interesting comparison.

usage:
  uv run baselines.py <environment_name> [--episodes N] [--model NAME] [--brute-force] [--steps K] [--tag NAME] [--device cpu|cuda] [--methods a,b,c]
"""

import argparse
import copy
import csv
import itertools
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from environment import Environment
from rigidity import rigidity_eigenvalue, is_IBR_explicit

MAX_BRUTE_FORCE_N = 5


# the only state scores whose value depends on the is_MBR flag
MBR_DEPENDENT_SCORES = {"RigidAndMinRigid", "MinRigid", "MinRigidAndMinEigenvalue"}


def score_network(env, need_mbr=None):
    """(score, is_IBR, is_MBR, rank, m) for whatever edges env.network currently holds.

    is_MBR costs one rank computation *per edge* on top of the rank of the whole matrix.
    greedy evaluates n(n-1) candidates per improvement step, so that is skipped unless the
    configured state score actually reads the flag. At n=16 it is roughly half the cost.
    """
    brm = env.network.extended_bearing_rigidity_matrix()
    if need_mbr is None:
        need_mbr = env.state_score_type in MBR_DEPENDENT_SCORES

    if need_mbr:
        is_MBR, is_IBR, rank = env.network.is_MBR(rank_K=env.rank_K, brm=brm)
    elif int(env.network.edges.sum()) == 0:
        is_MBR, is_IBR, rank = False, False, 0
    else:
        is_MBR = False
        is_IBR, rank = is_IBR_explicit(brm, rank_K=env.rank_K)

    score = env.compute_state_score(brm, is_IBR, is_MBR, rank)
    return score, bool(is_IBR), bool(is_MBR), int(rank), int(env.network.edges.sum())


def result(method, score, is_IBR, is_MBR, m, steps):
    return {
        "method": method,
        "score": float(score),
        "is_IBR": bool(is_IBR),
        "is_MBR": bool(is_MBR),
        "m": int(m),
        "steps": int(steps),
    }


# --------------------------------------------------------------------------------------
def run_initial(env):
    score, is_IBR, is_MBR, _, m = score_network(env)
    return result("initial", score, is_IBR, is_MBR, m, 0)


def run_greedy(env, max_steps=200, verbose=True):
    """Repeatedly apply the single edge toggle that improves phi the most.

    Cost is n(n-1) phi evaluations per improvement step, so this is by far the most
    expensive baseline at large n -- drop it with --methods if you only want the rest.
    """
    n = env.network.n
    candidates = [(i, j) for i in range(n) for j in range(n) if i != j]

    score, is_IBR, is_MBR, _, m = score_network(env)
    steps = 0
    bar = tqdm(desc="    greedy", unit="edit", leave=False) if verbose else None

    for _ in range(max_steps):
        best_delta = 0.0
        best_move = None
        best_eval = None

        for (i, j) in candidates:
            existed = env.network.edge_exists(i, j)
            if existed:
                env.network.remove_edge(i, j)
            else:
                env.network.add_edge(i, j)

            cand = score_network(env)

            # revert
            if existed:
                env.network.add_edge(i, j)
            else:
                env.network.remove_edge(i, j)

            delta = cand[0] - score
            if delta > best_delta:
                best_delta = delta
                best_move = (i, j, existed)
                best_eval = cand

        if best_move is None:  # local optimum, nothing improves phi
            break

        i, j, existed = best_move
        if existed:
            env.network.remove_edge(i, j)
        else:
            env.network.add_edge(i, j)
        score, is_IBR, is_MBR, _, m = best_eval
        steps += 1
        if bar is not None:
            bar.update(1)
            bar.set_postfix(m=m, phi=f"{score:.0f}")

    if bar is not None:
        bar.close()

    # the search may have skipped the is_MBR flag; the reported row needs it
    score, is_IBR, is_MBR, _, m = score_network(env, need_mbr=True)
    return result("greedy", score, is_IBR, is_MBR, m, steps)


def run_random(env, steps):
    """Uniform random actions. Scored on the best state visited, not the final one."""
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    # steps to reach the best graph, not steps burned: a rollout method always runs
    # the whole horizon, so the latter says nothing
    return result("random", env.best_state_score, env.best_stats["is_IBR"],
                  env.best_stats["is_MBR"], env.best_stats["m"], env.best_step)


def run_policy(agent, wrapped_env, raw_env, steps):
    """The trained policy acting as it was trained (PPO stays stochastic), best state visited."""
    obs, _ = wrapped_env.reset()  # freeze_network keeps the instance
    for t in range(steps):
        with torch.no_grad():
            action, _ = agent.act(obs, states=wrapped_env.state(), timestep=t, timesteps=steps)
        obs, _, terminated, truncated, _ = wrapped_env.step(action)
        done = terminated.any().item() if torch.is_tensor(terminated) else terminated
        trunc = truncated.any().item() if torch.is_tensor(truncated) else truncated
        if done or trunc:
            break
    return result("learned", raw_env.best_state_score, raw_env.best_stats["is_IBR"],
                  raw_env.best_stats["is_MBR"], raw_env.best_stats["m"], raw_env.best_step)


def run_brute_force(env, verbose=True):
    """Fewest-edge rigid graph, then the most rigid one at that edge count.

    Scans m ascending and stops at the first level that admits an IBR graph. Because phi
    rewards rank more than it penalises edges, that level contains the phi optimum. Unlike
    the R^d closed form this makes no homogeneity assumption, it is just slow.
    """
    n = env.network.n
    if n > MAX_BRUTE_FORCE_N:
        return None

    all_edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    saved = env.network.edges.copy()

    best = None
    checked = 0
    for m in range(1, len(all_edges) + 1):
        found_at_this_level = False
        best_eig = -np.inf

        # each level is C(n^2-n, m) rank computations, which grows fast: at n=5 a network
        # needing 9 edges costs ~430k of them. Show progress so it does not look hung.
        subsets = itertools.combinations(all_edges, m)
        if verbose:
            subsets = tqdm(subsets, total=math.comb(len(all_edges), m),
                           desc=f"    brute force m={m}", unit="graph",
                           unit_scale=True, leave=False)

        for subset in subsets:
            env.network.set_edges_list(list(subset))
            checked += 1
            score, is_IBR, is_MBR, rank, _ = score_network(env)
            if not is_IBR:
                continue

            found_at_this_level = True
            eig = rigidity_eigenvalue(env.network, rank_K=env.rank_K)
            if eig > best_eig:
                best_eig = eig
                best = result("optimal", score, is_IBR, is_MBR, m, 0)

        if found_at_this_level:
            break

    env.network.set_edges(saved)
    return best


# --------------------------------------------------------------------------------------
def summarize(rows):
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    print()
    # `steps` is edits applied for greedy, and steps taken to reach the best graph for
    # the rollout methods -- in both cases "how long until it had its answer"
    print(f"{'method':<10}{'|E|':>8}{'score':>10}{'%IBR':>8}{'%MBR':>8}{'steps':>8}{'n':>6}")
    print("-" * 58)
    for method in methods:
        sel = [r for r in rows if r["method"] == method]
        print(
            f"{method:<10}"
            f"{np.mean([r['m'] for r in sel]):>8.2f}"
            f"{np.mean([r['score'] for r in sel]):>10.2f}"
            f"{100 * np.mean([r['is_IBR'] for r in sel]):>8.0f}"
            f"{100 * np.mean([r['is_MBR'] for r in sel]):>8.0f}"
            f"{np.mean([r['steps'] for r in sel]):>8.1f}"
            f"{len(sel):>6}"
        )

    # how often does each method actually reach the optimum?
    opt = {r["episode"]: r["score"] for r in rows if r["method"] == "optimal"}
    if opt:
        print()
        for method in methods:
            if method == "optimal":
                continue
            sel = [r for r in rows if r["method"] == method and r["episode"] in opt]
            if not sel:
                continue
            hits = sum(1 for r in sel if r["score"] >= opt[r["episode"]] - 1e-9)
            print(f"  {method:<10} matches optimum on {100 * hits / len(sel):.0f}% of episodes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("environment_name")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model", default=None, help="trained model name from train/")
    parser.add_argument("--brute-force", action="store_true")
    parser.add_argument("--steps", type=int, default=None,
                        help="rollout horizon for random/learned (default: env truncate/max steps)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="device for the --model rollout")
    parser.add_argument("--tag", default=None,
                        help="suffix for the output csv, so separate runs do not overwrite")
    parser.add_argument("--methods", default="initial,greedy,random,learned",
                        help="comma-separated subset of initial,greedy,random,learned "
                             "(greedy is the expensive one at large n)")
    parser.add_argument("--replay-env", action="store_true",
                        help="score every method against the environment --model was trained "
                             "on (from its manifest) instead of the current code")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in ("initial", "greedy", "random", "learned")]
    if unknown:
        print(f"unknown method(s): {unknown}")
        return 1

    filepath = "./environments/" + args.environment_name + ".json"
    if not os.path.exists(filepath):
        print(f"file {filepath} does not exist")
        return 1

    np.random.seed(args.seed)

    env = Environment()
    env.load(filepath)
    # gymnasium spaces carry their own RNG, so np.random.seed does not make
    # action_space.sample() reproducible -- the random baseline needs this too
    env.action_space.seed(args.seed)

    steps = args.steps
    if steps is None:
        steps = int(env.truncate_max_steps if env.truncate_enable else env.max_steps)
    n = env.network.n

    agent = wrapped = None
    if args.model:
        from skrl.envs.wrappers.torch import wrap_env
        from agent_loader import load_agent, load_run

        if args.replay_env:
            # every method is then scored through the archived compute_state_score, which
            # is what keeps the table internally consistent with the run being evaluated
            agent, wrapped, env, info = load_run(
                args.model, env_name=args.environment_name, device=args.device
            )
            algorithm = (info or {}).get("algorithm", "?")
            env.action_space.seed(args.seed)
            n = env.network.n
            if args.steps is None:
                steps = int(env.truncate_max_steps if env.truncate_enable else env.max_steps)
        else:
            # the wrapper reads env.device to decide where to put observations; without
            # this it defaults to cuda while the agent is built on cpu
            env.device = args.device
            wrapped = wrap_env(env)
            wrapped.reset()
            agent, algorithm = load_agent(args.model, wrapped, env, device=args.device)
        print(f"loaded {algorithm} model '{args.model}' on {args.device}"
              f"{' (replaying its archived environment)' if args.replay_env else ''}")

    if args.brute_force and n > MAX_BRUTE_FORCE_N:
        print(f"brute force refused: n={n} > {MAX_BRUTE_FORCE_N} "
              f"({2 ** (n * n - n):.3g} possible graphs)")
        args.brute_force = False

    print(f"env: {args.environment_name} | n={n} | rollout horizon={steps} | episodes={args.episodes}")

    # re-seed here so the episodes drawn below do not depend on how much RNG the setup
    # above consumed: loading a model does an extra reset, which would otherwise make a
    # --model run and a plain run evaluate different instances
    np.random.seed(args.seed)
    env.action_space.seed(args.seed)

    rows = []
    for ep in range(args.episodes):
        env.freeze_network = False
        env.reset()                              # draw a fresh instance
        instance = copy.deepcopy(env.network)
        env.freeze_network = True                # every reset below keeps it

        def restore():
            env.network = copy.deepcopy(instance)
            env.reset()

        episode_rows = []
        if "initial" in methods:
            restore()
            episode_rows.append(run_initial(env))

        if "greedy" in methods:
            restore()
            episode_rows.append(run_greedy(env))

        if "random" in methods:
            restore()
            episode_rows.append(run_random(env, steps))

        if "learned" in methods and agent is not None:
            restore()
            episode_rows.append(run_policy(agent, wrapped, env, steps))

        if args.brute_force:
            restore()
            opt = run_brute_force(env)
            if opt is not None:
                episode_rows.append(opt)

        for r in episode_rows:
            r["episode"] = ep
        rows.extend(episode_rows)

        line = "  ".join(f"{r['method']}: m={r['m']} phi={r['score']:.1f}" for r in episode_rows)
        print(f"  ep {ep:>3}  {line}")

    summarize(rows)

    os.makedirs("runs_baselines", exist_ok=True)
    csv_path = f"runs_baselines/{args.environment_name}{'_' + args.tag if args.tag else ''}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "method", "m", "score", "is_IBR", "is_MBR", "steps"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
