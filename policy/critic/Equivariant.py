import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from skrl.models.torch import Model
from skrl.models.torch import DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *



# compatible with observation type "DictEquivariantNodeFeaturesAndAdjAndSelection"
class PPO_CriticModel_Equivariant(DeterministicMixin, Model):
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

        self.gnn = GNNBackboneEquivariant(
            node_feat_dim, edge_feat_dim, gnn_hidden_dim
        )  # output dim = node_feat_dim

        # input cat[node features, selected node's features(zeros if no selected)]
        self.head = nn.Sequential(
            nn.Linear(node_feat_dim, head_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(head_hidden_dim, 1),
        )

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]  # B, N, 3
        coord_features = observations["coord_features"]  # B, N, 3
        edge_features = observations["edge_features"]  # B, N, N, 3
        adj = observations["adj"]  # B, N, N
        selection = observations["selection"]  # B, N

        # print(f"node_features: {node_features}")
        # print(f"adj: {adj}")
        # print(f"selection: {selection}")

        batch_size = node_features.shape[0]
        n = node_features.shape[1]

        h = self.gnn(feats=node_features, coors=coord_features, edges=edge_features,
                     adj_mat=adj)

        # graph embedding
        batch_mapping = torch.arange(batch_size, device=h.device).repeat_interleave(n)
        graph_latent = global_mean_pool(h.reshape(-1, h.shape[-1]), batch_mapping)

        value = self.head(graph_latent)

        return value, {}
