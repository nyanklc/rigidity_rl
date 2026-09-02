"""Is greedy's phi landscape the same thing the observation already computes?

`evaluation.py --methods greedy` scores every one of the n(n-1) single-edge
toggles by rebuilding B and taking an SVD. The observation's `add_rank` and
`remove_rank` channels are the exact rank change of those same toggles, computed
for all pairs at once. So the two can be compared directly, and the answer
decides whether the policy is being handed greedy's own decision as an input.

Three questions, one per section:

  equivalence  is d(phi) exactly the rank channels? (yes at kappa = 0)
  divergence   what does the stiffness term break? (kappa > 0, the thesis case)
  cost         what would computing the landscape from the channels save?

Pin BLAS to one thread; see tools/flag_cost.py.

    OMP_NUM_THREADS=1 PYTHONPATH=. uv run tools/greedy_landscape.py
"""
import argparse
import time

import numpy as np

import rigidity as R
from environment import Environment
from evaluation import phi_landscape, run_greedy, score_network


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


def rigid_state(env, seed):
    """Put the network on a rigid graph, so the stiffness term is live."""
    env.reset()
    env.network.edges, _, _ = R.greedy_rigid_construction(
        env.network, env.rank_K, np.random.default_rng(seed))


def brute_deltas(env):
    """greedy's own landscape, by the route greedy takes: n(n-1) phi evaluations."""
    n = env.network.n
    base = score_network(env)[0]
    D = np.full((n, n), np.nan)
    evals = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            existed = env.network.edge_exists(i, j)
            (env.network.remove_edge if existed else env.network.add_edge)(i, j)
            D[i, j] = score_network(env)[0] - base
            evals += 1
            (env.network.add_edge if existed else env.network.remove_edge)(i, j)
    return D, evals


def rank_deltas(env):
    """The same landscape in closed form, as evaluation.py's spectral baseline reads it.

    Kept as a one-liner on purpose: this script exists to check that function, so a
    second copy of the formula here would make the check meaningless.
    """
    return phi_landscape(env, stiffness=False)


def stiffness_proxy_deltas(env, rank_only):
    """rank_deltas plus the stiffness channels, scaled into phi by hand.

    An addition's true lambda is not available to the observation, so this uses
    `add_stiffness` (a ranking prior, THEORY.md 16) rescaled to the stiffness
    term's own budget. Removals use `remove_stiffness`, which is exact. The hand
    scaling is the weak part and the reason a miss here is not evidence that the
    information is absent: a policy fits that scaling from data.
    """
    n = env.network.n
    B = env.network.extended_bearing_rigidity_matrix()
    rank, _, lam = R.rigidity_decomposition(B, env.rank_K)
    if rank < env.rank_K:
        return rank_only
    L = R.characteristic_length(env.network)
    _, v, w, V = R.nullspace_and_softest(B, int(rank))
    c = max(int(env.c_max), 1)
    budget = env.stiffness_kappa * 25.0 * c / max(int(env.rank_K), 1)

    add_st = np.zeros((n, n))
    if v is not None and v.shape[1] == 1:
        vs = R.nullspace_in_scaled_units(v, n, L)
        add_st = R.candidate_gain(env.network, vs, length_scale=L)[0]
    _, rem_st = R.removal_costs(B, env.network, int(env.rank_K), lam=lam, w=w, V=V,
                                c_max=env.c_max)
    E = env.network.edges.astype(bool)
    bonus = np.where(E, -budget * rem_st,
                     budget * add_st / max(float(add_st.max()), 1e-12))
    return rank_only + bonus


def equivalence(domains, ns, episodes, states, seed):
    print("kappa = 0: is d(phi) exactly the rank channels?")
    print(f"{'domain':<9}{'n':>3}{'phi evals':>11}{'n(n-1)':>8}"
          f"{'max |brute - closed form|':>27}{'top move':>11}")
    for domain in domains:
        for n in ns:
            env = make(n, domain, seed=seed)
            rng = np.random.default_rng(seed)
            err, same, total, evals = 0.0, 0, 0, 0
            for ep in range(episodes):
                rigid_state(env, seed + ep)
                for _ in range(states):
                    Db, evals = brute_deltas(env)
                    Dr = rank_deltas(env)
                    fin = ~np.isnan(Db)
                    err = max(err, float(np.abs(Db[fin] - Dr[fin]).max()))
                    same += int(np.unravel_index(np.nanargmax(Db), Db.shape)
                                == np.unravel_index(np.nanargmax(Dr), Dr.shape))
                    total += 1
                    env.step(int(rng.integers(env.action_space.n)))
            print(f"{domain:<9}{n:>3}{evals:>11}{n * (n - 1):>8}{err:>27.2e}"
                  f"{same:>7}/{total:<3}")
    print()


def divergence(domains, ns, kappas, episodes, states, seed):
    print("kappa > 0: how much of greedy's move survives in the channels")
    print("(dphi lost is in phi; one edge is worth 25*c_max/rank_K, printed as 'edge')")
    print(f"{'domain':<9}{'n':>3}{'kappa':>7}{'rank-only':>11}{'+stiffness':>12}"
          f"{'mean dphi lost':>16}{'worst':>8}{'edge':>7}")
    for domain in domains:
        for n in ns:
            for kappa in kappas:
                env = make(n, domain, seed=seed, stiffness_kappa=kappa)
                rng = np.random.default_rng(seed)
                same_r, same_p, total, gaps = 0, 0, 0, []
                for ep in range(episodes):
                    rigid_state(env, seed + ep)
                    for _ in range(states):
                        Db, _ = brute_deltas(env)
                        Dr = rank_deltas(env)
                        Dp = stiffness_proxy_deltas(env, Dr)
                        best = np.unravel_index(np.nanargmax(Db), Db.shape)
                        pick = np.unravel_index(np.nanargmax(Dr), Dr.shape)
                        same_r += int(pick == best)
                        same_p += int(np.unravel_index(np.nanargmax(Dp), Dp.shape) == best)
                        gaps.append(float(np.nanmax(Db) - Db[pick]))
                        total += 1
                        env.step(int(rng.integers(env.action_space.n)))
                one_edge = 25.0 * max(int(env.c_max), 1) / max(int(env.rank_K), 1)
                print(f"{domain:<9}{n:>3}{kappa:>7.1f}{same_r:>7}/{total:<3}"
                      f"{same_p:>8}/{total:<3}{np.mean(gaps):>16.3f}"
                      f"{np.max(gaps):>8.3f}{one_edge:>7.2f}")
    print()


def cost(ns, domain, seed):
    def timeit(f, reps=3):
        f()
        t0 = time.perf_counter()
        for _ in range(reps):
            f()
        return (time.perf_counter() - t0) / reps * 1e3

    print("one greedy improvement step, ms")
    print(f"{'n':>4}{'phi evals':>11}{'as implemented':>16}{'from channels':>15}{'speedup':>9}")
    for n in ns:
        env = make(n, domain, seed=seed)
        rigid_state(env, seed)
        brute = timeit(lambda: brute_deltas(env), reps=1)
        chan = timeit(lambda: rank_deltas(env))
        print(f"{n:>4}{n * (n - 1):>11}{brute:>16.1f}{chan:>15.2f}{brute / chan:>8.0f}x")
    print()

    print("improvement steps a greedy run actually takes")
    for n in ns:
        env = make(n, domain, seed=seed)
        works = []
        for _ in range(5):
            env.reset()
            works.append(run_greedy(env, verbose=False)["work"])
        print(f"  n={n:>3}  m_req={env.m_req:>3}  steps {np.mean(works):.1f} "
              f"+- {np.std(works):.1f}   phi evals/episode ~ "
              f"{np.mean(works) * n * (n - 1):,.0f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="R^3,SE(3),R^3xS^1")
    ap.add_argument("--n", default="6,8")
    ap.add_argument("--kappas", default="0.9,2.0")
    ap.add_argument("--cost-n", default="6,8,12,16")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--states", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--skip-cost", action="store_true")
    args = ap.parse_args()
    domains = args.domains.split(",")
    ns = [int(x) for x in args.n.split(",")]

    equivalence(domains, ns, args.episodes, args.states, args.seed)
    divergence(domains, ns, [float(k) for k in args.kappas.split(",")],
               args.episodes, args.states, args.seed)
    if not args.skip_cost:
        cost([int(x) for x in args.cost_n.split(",")], domains[0], args.seed)


if __name__ == "__main__":
    main()
