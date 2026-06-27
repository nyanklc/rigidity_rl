import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from skrl.models.torch import Model
from skrl.models.torch import DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *



class PPO_CriticModel_GINE_Selection(DeterministicMixin, Model):
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

        # +1 since we'll add the degree as a node feature
        self.gnn = GNNBackboneGINE(
            node_feat_dim + 1, edge_feat_dim, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n
        # +1 for selection "bit"
        self.head = nn.Sequential(
            nn.Linear(2*gnn_hidden_dim, head_hidden_dim), nn.LeakyReLU(), nn.Linear(head_hidden_dim, 1)
        )

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]
        edge_features = observations["edge_features"]
        adj = observations["adj"]
        selection = observations["selection"]

        batch_size = node_features.shape[0]

        # batch
        edge_index_list = []
        edge_attr_list = []
        for i in range(batch_size):
            src, dst = adj[i].nonzero(as_tuple=True)

            edge_index = torch.stack([src, dst], dim=0) + i * self.n
            edge_index_list.append(edge_index)

            edge_attr_list.append(edge_features[i][src, dst])
        full_edge_index = torch.cat(edge_index_list, dim=1).to(self.device)
        full_edge_attr = torch.cat(edge_attr_list, dim=0).to(self.device)

        out_degrees = adj.sum(dim=-1, keepdim=True)
        node_features = torch.cat([node_features, out_degrees], dim=-1)

        # current graph pass
        h = self.gnn(node_features, full_edge_index, full_edge_attr)

        # concat selected node's features
        selected = (h * selection.unsqueeze(-1)).sum(dim=1) # zeros if not selected
        selected_repeated = selected.unsqueeze(1).expand(-1, self.n, -1)
        new_embeddings = torch.cat([h, selected_repeated], dim=-1)

        # graph embedding
        batch_mapping = torch.arange(batch_size, device=new_embeddings.device).repeat_interleave(
            self.n
        )
        graph_latent = global_mean_pool(new_embeddings.reshape(-1, new_embeddings.shape[-1]), batch_mapping)

        # print(f"hey h: {h.shape}, new: {new_embeddings.shape}, latent: {graph_latent.shape}")
        value = self.head(graph_latent)

        return value, {}
