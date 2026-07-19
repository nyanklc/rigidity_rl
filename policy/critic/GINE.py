import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from skrl.models.torch import Model
from skrl.models.torch import DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *



class PPO_CriticModel_GINE(DeterministicMixin, Model):
    def __init__(
        self,
        n,
        node_feat_dim,
        edge_feat_dim,
        gnn_hidden_dim,
        head_hidden_dim,

        observation_space,
        action_space,
        device,
    ):
        # Model.__init__(self, observation_space, action_space, device)
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)

        self.gnn = GNNBackboneGINE(
            node_feat_dim, edge_feat_dim, gnn_hidden_dim
        )  # output dim = hidden dim

        # input cat[node features, selected node's features(zeros if no selected)]
        self.head = nn.Sequential(
            nn.Linear(2 * node_feat_dim, head_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(head_hidden_dim, 1),
        )

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]
        edge_features = observations["edge_features"]
        adj = observations["adj"]

        n = node_features.shape[1]
        batch_size = node_features.shape[0]

        # batch
        edge_index_list = []
        edge_attr_list = []
        for i in range(batch_size):
            src, dst = adj[i].nonzero(as_tuple=True)

            edge_index = torch.stack([src, dst], dim=0) + i * n
            edge_index_list.append(edge_index)

            edge_attr_list.append(edge_features[i][src, dst])
        full_edge_index = torch.cat(edge_index_list, dim=1).to(self.device)
        full_edge_attr = torch.cat(edge_attr_list, dim=0).to(self.device)

        # current graph pass
        h = self.gnn(node_features, full_edge_index, full_edge_attr)

        # graph embedding
        batch_mapping = torch.arange(batch_size, device=h.device).repeat_interleave(
            n
        )
        graph_latent = global_mean_pool(h.reshape(-1, h.shape[-1]), batch_mapping)

        # print(f"hey h: {h.shape}, new: {new_embeddings.shape}, latent: {graph_latent.shape}")
        value = self.head(graph_latent)

        return value, {}
