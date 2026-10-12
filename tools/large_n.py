"""The trained policies on formations far larger than they were trained on.

Greedy evaluates all n(n-1) toggles before every edit, so it is not affordable
past about twenty agents and is left out. The spectral hill climb is, because it
reads the same landscape in closed form, so it carries the classical reference
here. The comparison is therefore policy against policy, with spectral beside
them.

Instances are generated in process. Nothing under `environments/`, `benchmarks/`
or `models/` is written.

    PYTHONPATH=. uv run tools/large_n.py --ns 10,16,24,32 --instances 10 --csv out.csv
"""
import argparse
import csv
import time
import numpy as np

from environment import Environment
import baselines as B
import rigidity as R

MODELS = [("ref_k0_dqn_gine", "k0"), ("ref_k10_s0_dqn_gine", "k10"),
          ("ref_k200_dqn_gine", "k200")]
TRAINED_N = 10


def make_env(n, kappa, seed):
    e = Environment()
    e.initialize(
        n, "R^2",
        action_space_type="AddRemoveEdgeDiscreteNoSelfLoops",
        obs_space_type="Dict",
        state_score_type="WeightedNormalizedShapeError",
        termination_condition_type="MaxSteps",
        track_data_enable=False,
        skip_is_stop=False,
        random_domains=True,
        random_graph_with_mean_min_edges=True,
        include_candidate_bearings=True,
        graph_features=False,
        rigidity_edge=True, rigidity_flex=True, rigidity_global=True,
        rigidity_quality=True, rigidity_removal=True, rigidity_stiffness=True,
        rotation_augmentation=True,
        stiffness_kappa=float(kappa), stiffness_ref_samples=3,
        time_penalty_value=0.0, action_rewards_enable=False,
        max_steps=1e6, truncate_enable=False,
    )
    np.random.seed(seed)
    e.reset()
    return e


def summarise(env, label, n, inst, extra=None):
    Bm = R.extended_bearing_rigidity_matrix(env.network)
    rank = np.linalg.matrix_rank(Bm)
    rigid = rank == env.rank_K
    m = int(env.network.edges.sum())
    se = float(R.shape_error_of(env.network, env.rank_K)) if rigid else None
    rec = dict(n=n, instance=inst, method=label, m=m, m_req=int(env.m_req),
               m_over_mreq=m / max(int(env.m_req), 1), rigid=rigid,
               at_bound=bool(rigid and m == env.m_req),
               phi=float(env.compute_state_score(Bm, rigid, False, rank)),
               shape_err=se)
    rec.update(extra or {})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="24,48,96")
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default=None,
                    help="comma-separated NAME=LABEL, overriding the default trio")
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    models = MODELS
    if a.models:
        models = [tuple(x.split("=")) for x in a.models.split(",")]

    from skrl.envs.wrappers.torch import wrap_env
    from agent_loader import load_agent

    recs = []
    for n in ns:
        t_n = time.time()
        env = make_env(n, a.kappa, seed=0)
        env.device = a.device
        wrapped = wrap_env(env)
        wrapped.reset()
        agents = {}
        for name, label in models:
            try:
                ag, _ = load_agent(name, wrapped, env, device=a.device)
                agents[label] = ag
            except Exception as exc:                       # a checkpoint that does not fit
                print(f"  n={n}: {label} would not load: {str(exc)[:90]}")
        steps = int(4 * env.m_req + 10)

        for inst in range(a.instances):
            np.random.seed(1000 + inst)
            env.freeze_network = False
            env.reset()
            env.freeze_network = True
            start = env.network.edges.copy()

            for label, ag in agents.items():
                env.network.edges = start.copy()
                env.begin_episode()
                t = time.time()
                B.run_policy(ag, wrapped, env, steps, mode="sample", label=label)
                recs.append(summarise(env, label, n, inst,
                                      {"seconds": time.time() - t, "steps": steps,
                                       "best_at": int(env.best_step)}))
            env.network.edges = start.copy()
            env.begin_episode()
            t = time.time()
            B.run_spectral(env, verbose=False)
            recs.append(summarise(env, "spectral", n, inst,
                                  {"seconds": time.time() - t, "steps": None,
                                   "best_at": None}))
        print(f"  n={n:3d} done in {time.time()-t_n:6.1f}s  "
              f"(m_req={env.m_req}, horizon={steps})", flush=True)
        if a.csv:                       # flush after each size, not only at the end
            with open(a.csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(recs[0]))
                w.writeheader(); w.writerows(recs)

    labels = [l for _, l in models] + ["spectral"]
    print(f"\n{'n':>4s} {'method':>9s} {'m/m_req':>8s} {'rigid':>7s} {'at bnd':>7s} "
          f"{'shape err':>10s} {'best@':>7s} {'s/inst':>7s}")
    for n in ns:
        for lab in labels:
            s = [r for r in recs if r["n"] == n and r["method"] == lab]
            if not s:
                continue
            se = [r["shape_err"] for r in s if r["shape_err"]]
            rig = 100 * sum(r["rigid"] for r in s) / len(s)
            ab = 100 * sum(r["at_bound"] for r in s) / len(s)
            ba = [r["best_at"] for r in s if r["best_at"] is not None]
            print(f"{n:4d} {lab:>9s} {np.mean([r['m_over_mreq'] for r in s]):8.3f} "
                  f"{rig:6.0f}% {ab:6.0f}% "
                  f"{np.exp(np.mean(np.log(se))) if se else float('nan'):10.2f} "
                  f"{np.mean(ba) if ba else float('nan'):7.1f} "
                  f"{np.mean([r['seconds'] for r in s]):7.2f}")
        print()

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]))
            w.writeheader(); w.writerows(recs)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
