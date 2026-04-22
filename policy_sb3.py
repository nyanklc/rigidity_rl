import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


# NOTE: use with discrete action space, trying to make it work with multi discrete requires
# custom actor and critic network definitions which is a hassle with SB3 for this problem.
# multi discrete -> (add/remove/skip, i_idx, j_idx)
class GNNBackbone(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim=64):
        super().__init__()

        # +2 since we add the degrees as additional features to the node
        self.conv1 = GATConv(node_feature_dim + 2, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)

        # # +1 since we add 1/0 depending on if the edge exists in the current graph
        # self.edge_mlp = nn.Sequential(
        #     nn.Linear(hidden_dim * 2 + 1, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 2)  # [add, remove]
        # )

        # +1 since we add 1/0 depending on if the edge exists in the current graph
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # just add an edge
        )

    def forward(self, node_features, adj):
        # node_features: (B, N, F)
        # adj: (B, N, N)

        batch_size, n, node_feature_size = node_features.shape

        actor_outputs = []
        critic_outputs = []
        for b in range(batch_size):
            A = adj[b]
            out_deg = A.sum(dim=1, keepdim=True)
            in_deg  = A.sum(dim=0, keepdim=True).T
            degrees = th.cat([out_deg, in_deg], dim=1)
            x = th.hstack([node_features[b], degrees])

            # # current graph
            # edge_index = A.nonzero(as_tuple=False).T

            # fully connected (with self loops)
            edge_index_actor = (th.ones(A.shape)).nonzero(as_tuple=False).T
            edge_index_critic = A.nonzero(as_tuple=False).T
            self_loops = torch.arange(n)
            self_loops = torch.stack([self_loops, self_loops], dim=0)
            edge_index_critic = torch.cat([edge_index_critic, self_loops], dim=1)

            # GNN
            h_actor = self.conv1(x, edge_index_actor)
            h_actor = torch.relu(h_actor)
            h_actor = self.conv2(h_actor, edge_index_actor)
            h_actor = torch.relu(h_actor)  # (N, hidden)

            h_critic = self.conv1(x, edge_index_critic)
            h_critic = torch.relu(h_critic)
            h_critic = self.conv2(h_critic, edge_index_critic)
            h_critic = torch.relu(h_critic)

            # ACTOR
            # pairwise edges (fully connected)
            h_i = h_actor.unsqueeze(1).expand(n, n, -1)
            h_j = h_actor.unsqueeze(0).expand(n, n, -1)
            edge_exists = A.unsqueeze(-1).float()  # (N, N, 1)
            edge_feat = torch.cat([h_i, h_j, edge_exists], dim=-1)  # (N, N, 2H + 1)

            # edge_logits = self.edge_mlp(edge_feat)  # (N, N, 2)
            # flatten → (N*N*2,)
            # logits = edge_logits.reshape(-1)

            # TODO: remove edge mlp?
            logits = edge_feat.reshape(-1)

            actor_outputs.append(logits)

            # CRITIC
            g = h_critic.mean(dim=0)
            value = self.value_head(g)
            critic_outputs.append(value)

        return torch.stack(actor_outputs, dim=0), torch.stack(critic_outputs, dim=0)
