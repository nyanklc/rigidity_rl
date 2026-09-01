"""What do the observation flags cost on the policy side, not the environment side?

Three tables: how wide each flag makes the observation, what a backbone forward
costs at that width, and how many parameters the width buys. The flags widen the
per-pair edge tensor, which is what both backbones spend their time on and what
the replay buffer stores, so the cost lands in compute and memory rather than in
capacity.

`torch.set_num_threads(1)`, for the reason in tools/flag_cost.py: unpinned, a
batch-1 forward of this size timed at 160 ms against a true 1 ms.

    PYTHONPATH=. uv run tools/policy_cost.py
"""
import argparse
import time

import numpy as np
import torch

from environment import Environment
from policy.gnn_backbone import GNNBackboneEquivariant, GNNBackboneGINE

torch.set_num_threads(1)

FLAGSETS = [
    ("baseline (all off)", dict()),
    ("graph_features",     dict(graph_features=True)),
    ("rigidity_global",    dict(rigidity_global=True)),
    ("rigidity_quality",   dict(rigidity_quality=True)),
    ("rigidity_flex",      dict(rigidity_flex=True)),
    ("rigidity_edge",      dict(rigidity_edge=True)),
    ("rigidity_stiffness", dict(rigidity_stiffness=True)),
    ("rigidity_removal",   dict(rigidity_removal=True)),
    ("all six",            dict(rigidity_global=True, rigidity_quality=True,
                                rigidity_flex=True, rigidity_edge=True,
                                rigidity_stiffness=True, rigidity_removal=True)),
]


def make(n, domain, **kw):
    np.random.seed(7)
    opts = dict(action_space_type="SelectNodesSequentially", obs_space_type="Dict",
                state_score_type="WeightedNormalized",
                termination_condition_type="MaxSteps", max_steps=10 ** 6,
                track_data_enable=False, skip_is_stop=False,
                random_graph_with_mean_min_edges=True, graph_features=False)
    opts.update(kw)
    env = Environment()
    env.initialize(n, domain, **opts)
    env.reset()
    return env


def widths(n, domain, mem_size):
    print(f"observation width, n={n}, {domain}")
    print(f"{'flags':<22}{'node F':>8}{'edge E':>8}{'floats':>9}{'buffer MB':>11}")
    for label, kw in FLAGSETS:
        o = make(n, domain, **kw)._get_obs()
        floats = int(sum(np.prod(v.shape) for v in o.values()))
        # obs and next_obs, float32
        buf = mem_size * floats * 2 * 4 / 1e6
        print(f"{label:<22}{o['node_features'].shape[-1]:>8}"
              f"{o['edge_features'].shape[-1]:>8}{floats:>9}{buf:>11.0f}")
    print()


def forward_cost(ns, batches, hidden, node_f, edge_e):
    print(f"backbone forward, ms, hidden={hidden}, 3 layers, node_F={node_f}, "
          f"edge_E={edge_e}")
    print(f"{'model':<8}{'batch':>7}" + "".join(f"{'n=' + str(n):>10}" for n in ns))
    for name in ("GINE", "EGNN"):
        for B in batches:
            row = []
            for n in ns:
                nodes = torch.randn(B, n, node_f)
                edges = torch.randn(B, n, n, edge_e)
                coors = torch.randn(B, n, 3)
                if name == "GINE":
                    model = GNNBackboneGINE(node_f, edge_e, hidden)
                    f = lambda: model(nodes, edges)
                else:
                    model = GNNBackboneEquivariant(node_f, edge_e, hidden)
                    f = lambda: model(nodes, coors, edges=edges)
                with torch.no_grad():
                    for _ in range(3):
                        f()
                    t0 = time.perf_counter()
                    for _ in range(10):
                        f()
                    row.append((time.perf_counter() - t0) / 10 * 1e3)
            print(f"{name:<8}{B:>7}" + "".join(f"{v:>10.1f}" for v in row))
    print()


def parameters(hidden, node_f, edge_dims):
    print(f"parameter count, hidden={hidden}")
    for E in edge_dims:
        gine = GNNBackboneGINE(node_f, E, hidden)
        egnn = GNNBackboneEquivariant(node_f, E, hidden)
        print(f"edge_feat_dim={E:<3} GINE {sum(p.numel() for p in gine.parameters()):>9,}"
              f"   EGNN {sum(p.numel() for p in egnn.parameters()):>9,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="R^3")
    ap.add_argument("--n", default="8,16,32")
    ap.add_argument("--batches", default="1,256")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mem-size", type=int, default=10000, help="train_dqn.py MEM_SIZE")
    args = ap.parse_args()
    ns = [int(x) for x in args.n.split(",")]

    widths(ns[0], args.domain, args.mem_size)
    forward_cost(ns, [int(x) for x in args.batches.split(",")], args.hidden,
                 node_f=5, edge_e=12)
    parameters(args.hidden, node_f=5, edge_dims=(6, 12))


if __name__ == "__main__":
    main()
