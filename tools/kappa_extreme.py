"""The classical searches at conditioning weights no experiment used.

The policies can be carried to any kappa offline, because their trajectories do
not depend on it: only which visited graph is returned does. The classical
searches cannot, because they hill-climb on phi itself, so their trajectory
changes with kappa (greedy's diverges from its kappa=0 path at step 5). They have
to be re-run, which needs no trained model.

The environment is built in process from the frozen instances, so nothing under
`environments/` is read or written.

    PYTHONPATH=. uv run tools/kappa_extreme.py [--kappas 0,10,50,200,1000,5000] [--csv out.csv]
"""
import argparse
import csv
import numpy as np

from environment import Environment
import baselines as B
import rigidity as R

RUN = "runs_outputs/20260912-162935__addremove__3models__main/"
W_RANK, W_EDGE = 100.0, 25.0


def load_instances():
    d = np.load(RUN + "instances.npz", allow_pickle=True)
    return d


def make_env(kappa, n):
    e = Environment()
    e.initialize(
        n, "R^3",
        action_space_type="AddRemoveEdgeDiscreteNoSelfLoops",
        obs_space_type="Dict",
        state_score_type="WeightedNormalizedShapeError",
        termination_condition_type="MaxSteps",
        max_steps=10 ** 6,
        track_data_enable=False,
        skip_is_stop=False,
        stiffness_kappa=float(kappa),
        graph_features=False,
    )
    return e


def install(env, inst, ep):
    """Put one frozen instance into an in-process environment."""
    pos = inst["positions"][ep]
    n = len(pos)
    for i in range(n):
        a = env.network.agents[i]
        a.pose.position = np.array(pos[i], dtype=float)
        a.pose.rotation = inst["orientations"][ep][i]
        a.set_domain(str(inst["domains"][ep][i]))
        if a.domain == "R^3xS^1":
            a.rotation_axis = np.array(inst["rotation_axes"][ep][i], dtype=float)
    env.network.edges = np.array(inst["edges"][ep], dtype=bool).copy()
    env.begin_episode()
    return n


def phi_of(env):
    B_ = R.extended_bearing_rigidity_matrix(env.network)
    rank = np.linalg.matrix_rank(B_)
    return env.compute_state_score(B_, rank == env.rank_K, False, rank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappas", default="0,10,50,200,1000,5000")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--csv")
    a = ap.parse_args()
    kappas = [float(x) for x in a.kappas.split(",")]
    inst = load_instances()
    n_ag = inst["positions"].shape[1]

    recs = []
    for k in kappas:
        env = make_env(k, n_ag)
        for meth in ("greedy", "spectral", "anneal"):
            for ep in range(a.episodes):
                install(env, inst, ep)
                env.freeze_network = True
                rng = np.random.default_rng(1000 + ep)
                if meth == "greedy":
                    B.run_greedy(env, verbose=False, episode=ep)
                elif meth == "spectral":
                    B.run_spectral(env, verbose=False, episode=ep)
                else:
                    B.run_anneal(env, rng, budget=880, verbose=False, episode=ep)
                m = int(env.network.edges.sum())
                Bm = R.extended_bearing_rigidity_matrix(env.network)
                rank = np.linalg.matrix_rank(Bm)
                rigid = rank == env.rank_K
                se = R.shape_error_of(env.network, env.rank_K) if rigid else None
                # phi of the graph the method returns; greedy and spectral stop when
                # nothing improves, and run_anneal restores its best before returning
                phi = env.compute_state_score(Bm, rigid, False, rank)
                recs.append(dict(kappa=k, method=meth, episode=ep, m=m,
                                 phi=float(phi), rigid=rigid,
                                 at_bound=bool(rigid and m == env.m_req),
                                 shape_err=float(se) if se else None))
        print(f"  kappa={k:>6.0f} done")

    print(f"\n{'kappa':>7s}" + "".join(f"{m:>12s}" for m in ("greedy", "spectral", "anneal")))
    for k in kappas:
        line = f"{k:7.0f}"
        for m in ("greedy", "spectral", "anneal"):
            v = [r["phi"] for r in recs if r["kappa"] == k and r["method"] == m]
            line += f"{np.mean(v):12.1f}" if v else f"{'-':>12s}"
        print(line)
    print(f"\n{'kappa':>7s}{'greedy m':>10s}{'greedy err':>12s}{'anneal m':>10s}{'anneal err':>12s}")
    for k in kappas:
        g = [r for r in recs if r["kappa"] == k and r["method"] == "greedy"]
        an = [r for r in recs if r["kappa"] == k and r["method"] == "anneal"]
        ge = [r["shape_err"] for r in g if r["shape_err"]]
        ae = [r["shape_err"] for r in an if r["shape_err"]]
        print(f"{k:7.0f}{np.mean([r['m'] for r in g]):10.2f}"
              f"{np.exp(np.mean(np.log(ge))) if ge else float('nan'):12.2f}"
              f"{np.mean([r['m'] for r in an]):10.2f}"
              f"{np.exp(np.mean(np.log(ae))) if ae else float('nan'):12.2f}")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
