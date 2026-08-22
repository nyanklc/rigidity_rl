"""Reference points for a trained policy.

Every method here is scored with Environment.compute_state_score, i.e. the exact same state
score phi the agent is trained on, so the numbers are directly comparable:

  initial  the graph the sampler produced (what the agent starts from)
  random   uniform random actions through env.step()  -> the floor for this action space
  greedy   hill-climbing on phi, one edge toggle at a time
  learned  a trained policy, sampling actions (--policy-mode greedy for argmax instead)
  optimal  exhaustive search for the fewest-edge rigid graph (small n only)

greedy gets stuck exactly where RL should win: states where no single edit improves phi but a
swap of two does. That is the interesting comparison.

usage:
  uv run baselines.py <environment_name> [--episodes N] [--model NAME] [--brute-force] [--steps K] [--tag NAME] [--device cpu|cuda] [--methods a,b,c] [--restarts K] [--policy-mode sample|greedy]
  [--no-plots] [--plot-episodes N] [--brief] [--out-dir PATH] [--replay-env]

Writes one directory per run under runs_baselines/: the table, per-episode results,
per-step trajectories, provenance, and plots. See report.py.
"""

import argparse
import copy
import csv
import itertools
import json
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from environment import Environment
from rigidity import rigidity_eigenvalue, rigidity_decomposition, greedy_rigid_construction
import benchmark
import manifest
import report

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

    # rank and lam from one SVD: phi needs lam once margin_kappa > 0
    if need_mbr:
        is_MBR, is_IBR, rank = env.network.is_MBR(rank_K=env.rank_K, brm=brm)
        lam = rigidity_decomposition(brm, env.rank_K)[2] if is_IBR else 0.0
    elif int(env.network.edges.sum()) == 0:
        is_MBR, is_IBR, rank, lam = False, False, 0, 0.0
    else:
        is_MBR = False
        rank, _, lam = rigidity_decomposition(brm, env.rank_K)
        is_IBR = rank == env.rank_K

    score = env.compute_state_score(brm, is_IBR, is_MBR, rank, lam=lam)
    return score, bool(is_IBR), bool(is_MBR), int(rank), int(env.network.edges.sum())


def result(method, score, is_IBR, is_MBR, m, work=0, best_at=0, min_eig=None):
    """One method's outcome on one instance.

    `work` counts graph modifications actually applied and `best_at` is the step the best
    graph was reached at -- the old single `steps` column meant different things per method.
    """
    return {
        "method": method,
        "score": float(score),
        "is_IBR": bool(is_IBR),
        "is_MBR": bool(is_MBR),
        "m": int(m),
        "work": int(work),
        "best_at": int(best_at),
        "min_eig": None if min_eig is None else float(min_eig),
    }


def record(trace, method, episode, step, stats):
    """Append one point of a (episode, method) time series."""
    if trace is None or stats is None:
        return
    trace.append({"episode": episode, "method": method, "step": int(step),
                  "score": stats["score"], "edges": stats["m"], "rank": stats["rank"],
                  "rank_K": stats["rank_K"], "is_IBR": stats["is_IBR"],
                  "is_MBR": stats["is_MBR"], "min_eig": stats["min_eig"]})


def stats_now(env, need_mbr=True):
    """Full per-step record for the graph currently in `env`."""
    score, is_IBR, is_MBR, rank, m = score_network(env, need_mbr=need_mbr)
    return {"score": score, "m": m, "rank": rank, "rank_K": int(env.rank_K),
            "is_IBR": is_IBR, "is_MBR": is_MBR,
            "min_eig": float(rigidity_eigenvalue(env.network, rank_K=env.rank_K))}


def step_stats(env, tracing):
    """What the env recorded this step, computed here if it is too old to provide it.

    --replay-env can hand us an archived Environment from before last_stats existed.
    """
    if not tracing:
        return None
    stats = getattr(env, "last_stats", None)
    return stats if stats is not None else stats_now(env)


# --------------------------------------------------------------------------------------
def run_initial(env, trace=None, episode=0):
    # one call per episode, so pay for the is_MBR flag rather than reporting a false one
    st = stats_now(env)
    record(trace, "initial", episode, 0, st)
    return result("initial", st["score"], st["is_IBR"], st["is_MBR"], st["m"],
                  min_eig=st["min_eig"])


def run_greedy(env, max_steps=200, verbose=True, trace=None, episode=0):
    """Repeatedly apply the single edge toggle that improves phi the most.

    Cost is n(n-1) phi evaluations per improvement step, so this is by far the most
    expensive baseline at large n -- drop it with --methods if you only want the rest.
    """
    n = env.network.n
    candidates = [(i, j) for i in range(n) for j in range(n) if i != j]

    score, is_IBR, is_MBR, _, m = score_network(env)
    steps = 0
    if trace is not None:
        record(trace, "greedy", episode, 0, stats_now(env))
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
        if trace is not None:
            record(trace, "greedy", episode, steps, stats_now(env))
        if bar is not None:
            bar.update(1)
            bar.set_postfix(m=m, phi=f"{score:.0f}")

    if bar is not None:
        bar.close()

    # the search may have skipped the is_MBR flag; the reported row needs it
    st = stats_now(env)
    # greedy stops at its own best, so work and best@ coincide
    return result("greedy", st["score"], st["is_IBR"], st["is_MBR"], st["m"],
                  work=steps, best_at=steps, min_eig=st["min_eig"])


def _construct_once(env, order, rng):
    """One restart, on env.network. (edges, additions in order, rank reached).

    The loop is `rigidity.greedy_rigid_construction`, shared with the margin
    reference. `order` is accepted for call compatibility and rebuilt there.
    """
    return greedy_rigid_construction(env.network, env.rank_K, rng)


def run_constructive(env, rng, restarts=20, verbose=True, trace=None, episode=0):
    """Randomized constructive greedy, best of `restarts` independent orders.

    The classical algorithm for this problem, and the one to beat: no rigidity theory
    beyond rank(B), no learning. Unlike every other method it starts from the **empty
    graph** rather than the initial one, because it is a construction and not an edit.
    """
    n = env.network.n
    order = [(i, j) for i in range(n) for j in range(n) if i != j]

    best = None
    bar = tqdm(total=restarts, desc="    constructive", unit="restart",
               leave=False) if verbose else None
    for _ in range(restarts):
        _, added, rank = _construct_once(env, order, rng)
        score, is_IBR, is_MBR, _, m = score_network(env)
        # among rigid graphs phi is monotone decreasing in m, so this picks fewest edges
        if rank == env.rank_K and (best is None or score > best[0]):
            best = (score, list(added))
        if bar is not None:
            bar.update(1)
            bar.set_postfix(m=m, best=f"{best[0]:.0f}" if best else "-")
    if bar is not None:
        bar.close()

    if best is None:                     # no restart reached rank_K
        env.network.edges = np.zeros((n, n), dtype=bool)
        st = stats_now(env)
        record(trace, "constructive", episode, 0, st)
        return result("constructive", st["score"], st["is_IBR"], st["is_MBR"], st["m"])

    # replay the winner, so only it pays for the per-step statistics
    added = best[1]
    E = np.zeros((n, n), dtype=bool)
    env.network.edges = E.copy()
    record(trace, "constructive", episode, 0, stats_now(env))
    for k, (i, j) in enumerate(added, start=1):
        E[i, j] = True
        env.network.edges = E.copy()
        record(trace, "constructive", episode, k, stats_now(env))

    st = stats_now(env)
    # monotone construction: it ends on its own best graph
    return result("constructive", st["score"], st["is_IBR"], st["is_MBR"], st["m"],
                  work=len(added), best_at=len(added), min_eig=st["min_eig"])


def rollout_result(method, env, work):
    """Best graph the rollout visited, not the one it happened to stop on."""
    return result(method, env.best_state_score, env.best_stats["is_IBR"],
                  env.best_stats["is_MBR"], env.best_stats["m"],
                  work=work, best_at=env.best_step,
                  min_eig=env.best_stats.get("min_eig"))


def run_random(env, steps, trace=None, episode=0):
    """Uniform random actions. Scored on the best state visited, not the final one."""
    record(trace, "random", episode, 0, step_stats(env, trace is not None))
    work = 0
    before = env.network.edges.tobytes()
    for t in range(steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        after = env.network.edges.tobytes()
        if after != before:          # count only steps that actually changed the graph
            work += 1
            before = after
        record(trace, "random", episode, t + 1, step_stats(env, trace is not None))
        if terminated or truncated:
            break
    return rollout_result("random", env, work)


def deterministic_action(agent, obs):
    """argmax over the model's own scores, for PPO and DQN alike.

    skrl's CategoricalMixin.act always *samples*, so a PPO agent asked to act behaves as it
    did during training. DQN's act already takes an argmax when no exploration scheduler is
    configured. Going through compute() directly makes both algorithms deterministic in the
    same way, and keeps the models' action masking intact.
    """
    role = "policy" if "policy" in agent.models else "q_network"
    with torch.no_grad():
        scores, _ = agent.models[role].compute({"observations": obs}, role=role)
    return torch.argmax(scores, dim=-1, keepdim=True)


def run_policy(agent, wrapped_env, raw_env, steps, mode="sample", trace=None, episode=0):
    """Roll out a trained policy, scored on the best state visited.

    mode="greedy"  the action the policy considers best -- what you would deploy
    mode="sample"  sampled actions, i.e. the policy used as a sampling-based search over the
                   horizon (PPO only; a DQN q-network has nothing to sample from)
    """
    agent.enable_models_training_mode(False)  # eval mode (skrl 2.x naming)
    obs, _ = wrapped_env.reset()  # freeze_network keeps the instance
    seen = set()
    record(trace, "learned", episode, 0, step_stats(raw_env, trace is not None))
    work = 0
    before = raw_env.network.edges.tobytes()

    for t in range(steps):
        if mode == "greedy":
            action = deterministic_action(agent, obs)
        else:
            with torch.no_grad():
                action, _ = agent.act(obs, states=wrapped_env.state(),
                                      timestep=t, timesteps=steps)
        obs, _, terminated, truncated, _ = wrapped_env.step(action)
        after = raw_env.network.edges.tobytes()
        if after != before:          # count only steps that actually changed the graph
            work += 1
            before = after
        record(trace, "learned", episode, t + 1, step_stats(raw_env, trace is not None))

        done = terminated.any().item() if torch.is_tensor(terminated) else terminated
        trunc = truncated.any().item() if torch.is_tensor(truncated) else truncated
        if done or trunc:
            break

        # a deterministic policy in a deterministic environment is eventually periodic, so
        # once a state repeats nothing new can be found and the rest of the horizon is waste
        if mode == "greedy":
            key = (after, raw_env.selection.tobytes())
            if key in seen:
                break
            seen.add(key)

    return rollout_result("learned", raw_env, work)


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

    best_subset = None
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
                best_subset = list(subset)

        if found_at_this_level:
            break

    # the inner loop runs hundreds of thousands of times and so skips the is_MBR flag;
    # recompute the winner properly before reporting it
    best = None
    if best_subset is not None:
        env.network.set_edges_list(best_subset)
        st = stats_now(env)
        best = result("optimal", st["score"], st["is_IBR"], st["is_MBR"], st["m"],
                      min_eig=st["min_eig"])

    env.network.set_edges(saved)
    return best


# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("environment_name")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model", default=None, help="trained model name from train/")
    parser.add_argument("--brute-force", action="store_true")
    parser.add_argument("--steps", type=int, default=None,
                        help="rollout horizon for random/learned (default: env truncate/max steps)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark", default=None,
                        help="evaluate on a frozen instance set from benchmarks/ instead of "
                             "sampling; makes runs comparable across config regenerations")
    parser.add_argument("--device", default="cpu", help="device for the --model rollout")
    parser.add_argument("--tag", default=None,
                        help="label appended to the run directory name")
    parser.add_argument("--out-dir", default=None,
                        help="write results here instead of runs_baselines/<generated name>")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip plots; also skips per-step tracing, so rollouts are faster")
    parser.add_argument("--plot-episodes", type=int, default=3,
                        help="how many individual episodes get their own detail figure")
    parser.add_argument("--brief", action="store_true",
                        help="print the table without the explanatory legend")
    parser.add_argument("--methods", default="initial,greedy,constructive,random,learned",
                        help="comma-separated subset of initial,greedy,constructive,random,"
                             "learned (greedy and constructive are the expensive ones at large n)")
    parser.add_argument("--restarts", type=int, default=20,
                        help="restarts for the constructive baseline; it reports the best of them")
    parser.add_argument("--policy-mode", default="sample", choices=("sample", "greedy"),
                        help="sample (default): sampled actions, i.e. the policy used as a "
                             "sampling search over the horizon, scored on the best state it "
                             "finds. greedy: the single action the policy considers best, "
                             "which is reproducible and terminates on a cycle. A DQN q-network "
                             "has nothing to sample from and is argmax either way.")
    parser.add_argument("--replay-env", action="store_true",
                        help="score every method against the environment --model was trained "
                             "on (from its manifest) instead of the current code")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods
               if m not in ("initial", "greedy", "constructive", "random", "learned")]
    if unknown:
        print(f"unknown method(s): {unknown}")
        return 1

    filepath = "./environments/" + args.environment_name + ".json"
    if not os.path.exists(filepath):
        print(f"file {filepath} does not exist")
        return 1

    np.random.seed(args.seed)

    with open(filepath, "r") as f:
        env_config_data = json.load(f)

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

    # tracing costs one extra eigendecomposition per step, so it rides with the plots
    tracing = not args.no_plots
    # an archived environment replayed by --replay-env may predate this flag
    if hasattr(env, "trace_min_eig"):
        env.trace_min_eig = tracing
    traces = [] if tracing else None

    # re-seed here so the episodes drawn below do not depend on how much RNG the setup
    # above consumed: loading a model does an extra reset, which would otherwise make a
    # --model run and a plain run evaluate different instances
    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    # a sampling policy draws from torch's global RNG, so this is what makes
    # --policy-mode sample repeatable for a given seed
    torch.manual_seed(args.seed)

    # private to the constructive baseline; see the note in _construct_once
    construct_rng = np.random.default_rng(args.seed)

    frozen = None
    if args.benchmark:
        frozen, bench_meta = benchmark.load(args.benchmark)
        if len(frozen) < args.episodes:
            print(f"  benchmark {args.benchmark} has {len(frozen)} instances; "
                  f"running that many instead of {args.episodes}")
            args.episodes = len(frozen)
        print(f"  instances: benchmark {args.benchmark} "
              f"({benchmark.digest(args.benchmark)}), sampling disabled")

    rows = []
    for ep in range(args.episodes):
        env.freeze_network = False
        if frozen is not None:
            env.network = copy.deepcopy(frozen[ep])
            env.freeze_network = True
            env.reset()                          # bookkeeping only
        else:
            env.reset()                          # draw a fresh instance
        instance = copy.deepcopy(env.network)
        env.freeze_network = True                # every reset below keeps it

        def restore():
            env.network = copy.deepcopy(instance)
            env.reset()

        episode_rows = []
        if "initial" in methods:
            restore()
            episode_rows.append(run_initial(env, trace=traces, episode=ep))

        if "greedy" in methods:
            restore()
            episode_rows.append(run_greedy(env, trace=traces, episode=ep))

        if "constructive" in methods:
            restore()
            episode_rows.append(run_constructive(env, construct_rng, restarts=args.restarts,
                                                 trace=traces, episode=ep))

        if "random" in methods:
            restore()
            episode_rows.append(run_random(env, steps, trace=traces, episode=ep))

        if "learned" in methods and agent is not None:
            restore()
            episode_rows.append(run_policy(agent, wrapped, env, steps,
                                           mode=args.policy_mode, trace=traces, episode=ep))

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

    # ── report ────────────────────────────────────────────────────────────────────────
    domains = env.domains if isinstance(env.domains, list) else [env.domains]
    domain_str = domains[0] if len(set(domains)) == 1 else f"mixed {sorted(set(domains))}"
    context = {
        "environment": args.environment_name,
        "network": f"{n} agents in {domain_str}, action space {env.action_space_type}",
        "objective": f"{env.state_score_type} state score",
        "instances": (f"{args.episodes} networks from benchmark {args.benchmark} "
                      f"({benchmark.digest(args.benchmark)})" if args.benchmark
                      else f"{args.episodes} random networks, seed {args.seed}"),
    }
    if args.model:
        context["policy"] = (f"{args.model} ({algorithm}, --policy-mode {args.policy_mode}, "
                             f"{steps}-step budget)")

    table = report.format_table(rows, context, brief=args.brief)
    print("\n" + table)

    run_dir = report.make_run_dir("runs_baselines", args.environment_name,
                                  model_name=args.model, tag=args.tag, out_dir=args.out_dir,
                                  with_plots=bool(traces))
    report.write_summary(run_dir, table)
    report.write_csvs(run_dir, rows, traces)
    report.write_meta(run_dir, {
        "args": vars(args),
        "environment_config": env_config_data,
        "n": n, "rollout_steps": steps,
        # the instance set, so two runs are only comparable when these agree
        "benchmark": args.benchmark,
        "benchmark_digest": benchmark.digest(args.benchmark) if args.benchmark else None,
        "provenance": manifest.collect_provenance(seed=args.seed, device=args.device),
    })

    written = ["summary.txt", "results.csv"] + (["trajectories.csv"] if traces else [])
    if traces:
        # the full names go in the figure titles (wrapped): a plot pulled into a slide
        # has to say which model and which environment produced it
        header = {
            "short": report.short_env_name(args.environment_name),
            "env": args.environment_name,
            "model": args.model,
            "network": context["network"],
            "episodes": args.episodes,
            "seed": args.seed,
            "benchmark": args.benchmark,
        }
        report.plot_trajectories(run_dir, traces, rows, header)
        report.plot_outcomes(run_dir, traces, rows, header)
        report.plot_summary(run_dir, rows, header)
        # the table itself, so the numbers travel with the figures. The policy line drops
        # the model name -- the header already carries it in full one line above
        policy = (f"{algorithm}, --policy-mode {args.policy_mode}, {steps}-step budget"
                  if args.model else None)
        report.plot_table(run_dir, rows, dict(header, objective=context["objective"],
                                              policy=policy))
        for ep in range(min(args.plot_episodes, args.episodes)):
            sel = [t for t in traces if t["episode"] == ep]
            ep_header = dict(header, episodes=None, subtitle=f"episode {ep}")
            report.plot_trajectories(run_dir, sel, [r for r in rows if r["episode"] == ep],
                                     ep_header, filename=f"episode_{ep:03d}",
                                     aggregate_over_episodes=False)
        figures = 4 + min(args.plot_episodes, args.episodes)
        written.append(f"plots/pdf/ and plots/png/ ({figures} figures each)")

    print(f"\nwrote {run_dir}/")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
