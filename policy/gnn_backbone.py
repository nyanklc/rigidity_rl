import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINEConv
from egnn_pytorch import EGNN


# num_layers is a constructor argument so checkpoints trained at other depths
# still load. See DESIGN_NOTES.md#backbone-num-layers
class _GNNBackbone(nn.Module):
    def _register_convs(self, convs):
        self.num_layers = len(convs)
        for i, conv in enumerate(convs, start=1):
            setattr(self, f"conv{i}", conv)

    def convs(self):
        return [getattr(self, f"conv{i}") for i in range(1, self.num_layers + 1)]


class GNNBackboneGAT(_GNNBackbone):
    def __init__(self, node_feat_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.init_args = dict(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim)
        self._register_convs([
            GATConv(node_feat_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

    def forward(self, nodes, edge_index):
        batch_size, n, _ = nodes.shape
        h = nodes.reshape(-1, nodes.size(-1))

        for conv in self.convs():
            h = F.leaky_relu(conv(h, edge_index))

        return h.reshape(batch_size, n, -1)


# TODO: hardcoded MLP sizes
class GNNBackboneGINE(_GNNBackbone):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.init_args = dict(
            node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim, hidden_dim=hidden_dim
        )
        self._register_convs([
            GINEConv(
                nn=nn.Sequential(
                    nn.Linear(node_feat_dim if i == 0 else hidden_dim, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, hidden_dim),
                ),
                edge_dim=edge_feat_dim,
            )
            for i in range(num_layers)
        ])

    # complete digraph over the batch, no self loops; row-major (i outer, j inner)
    # so it lines up with a dense (B, N, N, E) edge tensor flattened the same way
    def _complete_edge_index(self, batch_size, n, device):
        key = (batch_size, n, device)
        if getattr(self, "_ei_key", None) != key:
            idx = torch.arange(n, device=device)
            src, dst = idx.repeat_interleave(n), idx.repeat(n)
            keep = src != dst
            src, dst = src[keep], dst[keep]
            offs = (torch.arange(batch_size, device=device) * n).repeat_interleave(src.numel())
            self._ei = torch.stack([src.repeat(batch_size) + offs, dst.repeat(batch_size) + offs])
            self._ei_keep, self._ei_key = keep, key
        return self._ei, self._ei_keep

    # edges: dense (B, N, N, E). Message passing is over every ordered pair, not
    # just existing edges -- the edge features say which is which.
    # See DESIGN_NOTES.md#gine-dense-all-pairs
    def forward(self, nodes, edges):
        batch_size, n, _ = nodes.shape
        h = nodes.reshape(-1, nodes.size(-1))

        edge_index, keep = self._complete_edge_index(batch_size, n, nodes.device)
        edge_attr = edges.reshape(batch_size, n * n, -1)[:, keep].reshape(-1, edges.size(-1))

        # aggregate outward bearings, not inward; DESIGN_NOTES.md#gine-edge-direction
        edge_index = edge_index.flip(0)

        # TODO: relu??
        for conv in self.convs():
            h = conv(h, edge_index, edge_attr=edge_attr)

        return h.reshape(batch_size, n, -1)


class GNNBackboneEquivariant(_GNNBackbone):
    # init_eps is egnn_pytorch's Linear init std. Its 1e-3 default makes the
    # edge-feature path start ~1e-10 against the node residual, so the model is
    # briefly blind to bearings. Kept at the default because that is what the
    # working DQN run used. See DESIGN_NOTES.md#egnn-init-eps
    def __init__(self, node_feat_dim, edge_dim, hidden_dim, num_layers=3, init_eps=1e-3):
        super().__init__()
        self.init_args = dict(
            node_feat_dim=node_feat_dim, edge_dim=edge_dim, hidden_dim=hidden_dim
        )
        self._register_convs([
            EGNN(dim=node_feat_dim, m_dim=hidden_dim, edge_dim=edge_dim, init_eps=init_eps)
            for _ in range(num_layers)
        ])

    def forward(self,
                feats,
                coors,
                adj_mat=None,
                edges=None):
        batch_size = feats.shape[0]
        n = feats.shape[1]

        # adj_mat is accepted but NOT forwarded: egnn_pytorch only reads it in
        # nearest-neighbour mode, so passing it was a silent no-op. Message
        # passing is dense all-pairs by design here -- the graph reaches the
        # model through the edge features. See DESIGN_NOTES.md#egnn-dense-all-pairs
        # TODO: should we recalculate bearings (edges) by using the new coordinates (c_out)?
        for conv in self.convs():
            feats, coors = conv(feats=feats, coors=coors, edges=edges)

        return feats.reshape(batch_size, n, -1)
