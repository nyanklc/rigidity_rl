import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINEConv
from egnn_pytorch import EGNN


# The layer count used to be hardcoded at 3 and was 2 in older runs. It is a
# constructor argument now so that a checkpoint trained at a different depth can
# still be loaded (see agent_loader.rebuild_backbone). Submodules keep the names
# conv1..convN, so state dicts of 3-layer models are unaffected.
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

    def forward(self, nodes, edge_index, edges):
        batch_size, n, _ = nodes.shape
        h = nodes.reshape(-1, nodes.size(-1))

        # IMPORTANT: the GIN(E) message passing adds the inward edge features
        # to the neighbor's features during message passing. however it makes
        # more sense for us to use outward edge
        # ("I have this bearing to this node")
        edge_index = edge_index.flip(0)

        # TODO: relu??
        for conv in self.convs():
            h = conv(h, edge_index, edge_attr=edges)

        return h.reshape(batch_size, n, -1)


class GNNBackboneEquivariant(_GNNBackbone):
    def __init__(self, node_feat_dim, edge_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.init_args = dict(
            node_feat_dim=node_feat_dim, edge_dim=edge_dim, hidden_dim=hidden_dim
        )
        self._register_convs([
            EGNN(dim=node_feat_dim, m_dim=hidden_dim, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])

    def forward(self,
                feats,
                coors,
                adj_mat=None,
                edges=None):
        batch_size = feats.shape[0]
        n = feats.shape[1]

        # TODO: should we recalculate bearings (edges) by using the new coordinates (c_out)?
        for conv in self.convs():
            feats, coors = conv(feats=feats, coors=coors, edges=edges, adj_mat=adj_mat)

        return feats.reshape(batch_size, n, -1)
