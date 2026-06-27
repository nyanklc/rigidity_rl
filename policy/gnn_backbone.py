import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINEConv
from egnn_pytorch import EGNN



class GNNBackboneGAT(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim):
        super().__init__()
        self.conv1 = GATConv(node_dim=node_feat_dim, hidden_dim=hidden_dim)
        self.conv2 = GATConv(node_dim=hidden_dim, hidden_dim=hidden_dim)

    def forward(self, nodes, edge_index):
        batch_size, n, _ = nodes.shape
        x = nodes.reshape(-1, nodes.size(-1))

        h = F.leaky_relu(self.conv1(x, edge_index))
        h = F.leaky_relu(self.conv2(h, edge_index))

        return h.reshape(batch_size, n, -1)


class GNNBackboneGINE(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, 128),
                nn.LeakyReLU(),
                nn.Linear(128, hidden_dim),
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.LeakyReLU(),
                nn.Linear(128, hidden_dim),
            ),
            edge_dim=edge_feat_dim
        )

    def forward(self, nodes, edge_index, edges):
        batch_size, n, _ = nodes.shape
        x = nodes.reshape(-1, nodes.size(-1))
        # e = edges.reshape(-1, edges.size(-1))

        # print(f"inside gnn nodes: {nodes.shape}, edge_index: {edge_index.shape}, edges: {edges.shape}")
        # print(f"inside gnn x: {x.shape}, e: -")

        # TODO: relu??
        h = self.conv1(x, edge_index, edge_attr=edges)
        h = self.conv2(h, edge_index, edge_attr=edges)

        return h.reshape(batch_size, n, -1)


class GNNBackboneEquivariant(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, edge_dim):
        super().__init__()
        self.conv1 = EGNN(dim=node_feat_dim, m_dim=hidden_dim, edge_dim=edge_dim)
        self.conv2 = EGNN(dim=node_feat_dim, m_dim=hidden_dim, edge_dim=edge_dim)

    def forward(self,
                feats,
                coors,
                adj_mat=None,
                edges=None):
        batch_size = feats.shape[0]
        n = feats.shape[1]

        n_out, c_out = self.conv1(feats=feats,
                                  coors=coors,
                                  edges=edges,
                                  adj_mat=adj_mat)
        h, _ = self.conv2(feats=n_out,
                          coors=c_out,
                          edges=edges,
                          adj_mat=adj_mat)

        return h.reshape(batch_size, n, -1)
