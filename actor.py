import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        self.node_dim = 6  # example: position + orientation
        self.hidden_dim = 64

        self.gat1 = GATConv(self.node_dim, self.hidden_dim)
        self.gat2 = GATConv(self.hidden_dim, self.hidden_dim)

        self.fc = nn.Linear(self.hidden_dim, features_dim)

    # obs = [node_features_flat, adj_flat]
    def forward(self, obs):
        n = int(torch.sqrt(torch.tensor(obs.shape[-1])))  # hack, better store n explicitly

        # split observation
        node_feats = obs[..., :n*self.node_dim].reshape(n, self.node_dim)
        adj = obs[..., n*self.node_dim:].reshape(n, n)

        edge_index, _ = dense_to_sparse(adj)

        x = self.gat1(node_feats, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        x = torch.relu(x)

        # global pooling (mean)
        x = x.mean(dim=0)

        return self.fc(x)