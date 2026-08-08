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

        self.head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, head_hidden_dim),
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

        # current graph pass
        h = self.gnn(node_features, edge_features)

        # graph embedding
        batch_mapping = torch.arange(batch_size, device=h.device).repeat_interleave(
            n
        )
        graph_latent = global_mean_pool(h.reshape(-1, h.shape[-1]), batch_mapping)

        # print(f"hey h: {h.shape}, new: {new_embeddings.shape}, latent: {graph_latent.shape}")
        value = self.head(graph_latent)

        return value, {}
