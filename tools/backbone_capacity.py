"""How do the two backbones compare on width and on parameter count?

`EGNN` preserves the feature width, so `GNNBackboneEquivariant` needs an input
embedder to reach `gnn_hidden_dim` at all.
Adding one equalizes the *width* against `GNNBackboneGINE` but not the parameter
count, because `dim` widens every EGNN layer's own MLPs too. The two controls
cannot both hold, so a backbone comparison has to say which one it ran -- this
prints the trade-off so that choice is made on numbers.

Widths are measured by forwarding, not assumed, since that is exactly the thing
that was wrong before WP10.

The `m_dim` column models a configuration the backbone does not currently expose:
`GNNBackboneEquivariant` wires `m_dim = hidden_dim`. Separating them is what a
matched-parameter arm would need, so the sweep builds the EGNN stack directly to
show what it would buy.

    PYTHONPATH=. uv run tools/backbone_capacity.py
    PYTHONPATH=. uv run tools/backbone_capacity.py --node-feat-dim 6 --hidden 64
    PYTHONPATH=. uv run tools/backbone_capacity.py --dims 32,64,128 --layers 2
"""
import argparse
import math

import torch
import torch.nn as nn
from egnn_pytorch import EGNN

from policy.gnn_backbone import GNNBackboneEquivariant, GNNBackboneGINE

N = 6  # only needs to be big enough to forward; nothing here depends on it


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def measured_width(backbone, node_feat_dim, edge_feat_dim):
    """Forward once and read the output width off the result."""
    nodes = torch.randn(1, N, node_feat_dim)
    edges = torch.randn(1, N, N, edge_feat_dim)
    with torch.no_grad():
        if isinstance(backbone, GNNBackboneGINE):
            out = backbone(nodes, edges)
        else:
            out = backbone(feats=nodes, coors=torch.randn(1, N, 3), edges=edges)
    return out.shape[-1]


def egnn_stack(dim, m_dim, edge_feat_dim, layers):
    """The EGNN layers alone, at a (dim, m_dim) the backbone cannot currently be given."""
    return nn.ModuleList([
        EGNN(dim=dim, m_dim=m_dim, edge_dim=edge_feat_dim, init_eps=1e-2,
             m_pool_method="mean", update_coors=False)
        for _ in range(layers)
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node-feat-dim", type=int, default=11,
                   help="11 is `mixed`: 5 domain + 2 degree + 3 rigidity_global + 1 node_freedom")
    p.add_argument("--edge-feat-dim", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128, help="GINE's width, the reference")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dims", default="11,32,48,64,96,128")
    args = p.parse_args()

    F_NODE, F_EDGE, H, L = args.node_feat_dim, args.edge_feat_dim, args.hidden, args.layers
    dims = [int(d) for d in args.dims.split(",")]

    gine = GNNBackboneGINE(F_NODE, F_EDGE, H, num_layers=L)
    gine_w, gine_p = measured_width(gine, F_NODE, F_EDGE), n_params(gine)

    equi = GNNBackboneEquivariant(F_NODE, F_EDGE, H, num_layers=L)
    equi_w, equi_p = measured_width(equi, F_NODE, F_EDGE), n_params(equi)

    print(f"\nnode_feat_dim {F_NODE}, edge_feat_dim {F_EDGE}, {L} layers\n")
    print("as built today")
    print(f"  {'GNNBackboneGINE':28} width {gine_w:>4}   {gine_p:>9,} params   1.0x")
    print(f"  {'GNNBackboneEquivariant':28} width {equi_w:>4}   {equi_p:>9,} params "
          f"  {equi_p / gine_p:.1f}x")
    print(f"  {'  of which the embedder':28} {'':10}   {n_params(equi.embed):>9,} params")

    print(f"\nEGNN layers alone, swept (embedder added where dim != node_feat_dim)")
    print(f"  {'dim':>5} {'m_dim':>6} {'params':>10} {'x GINE':>8}")
    best = None
    for dim in dims:
        for m_dim in sorted({dim, H}):
            total = n_params(egnn_stack(dim, m_dim, F_EDGE, L))
            if dim != F_NODE:
                total += n_params(nn.Sequential(nn.Linear(F_NODE, dim), nn.LeakyReLU(),
                                                nn.Linear(dim, dim)))
            ratio = total / gine_p
            mark = "   <- as built" if (dim == H and m_dim == H) else ""
            print(f"  {dim:>5} {m_dim:>6} {total:>10,} {ratio:>7.1f}x{mark}")
            # distance in log-ratio: 0.5x and 2x are equally far from parity,
            # which |ratio - 1| gets wrong (it prefers being too small)
            if best is None or abs(math.log(ratio)) < abs(math.log(best[2])):
                best = (dim, m_dim, ratio)

    print(f"\n  closest to parameter parity: dim {best[0]}, m_dim {best[1]} "
          f"({best[2]:.1f}x GINE, width {best[0]} against {gine_w})")
    print("  matched width and matched parameters are different experiments; name the one you ran.\n")


if __name__ == "__main__":
    main()
