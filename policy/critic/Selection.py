import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from skrl.models.torch import Model
from skrl.models.torch import DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *



class PPO_CriticModel_Selection(DeterministicMixin, Model):
    def __init__(
        self,
        n,
        node_feat_dim,
        gnn_hidden_dim,
        head_hidden_dim,

        observation_space,
        action_space,
        device,
    ):
        # Model.__init__(self, observation_space, action_space, device)
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)

        self.gnn = GNNBackboneGAT(node_feat_dim, gnn_hidden_dim)
        self.n = n
        # +1 for selection "bit"
        self.head = nn.Sequential(
            nn.Linear(gnn_hidden_dim + 1, head_hidden_dim), nn.Linear(head_hidden_dim, 1)
        )

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]
        adj = observations["adj"]
        selection = observations["selection"]

        batch_size = node_features.shape[0]

        # adj comes in batched
        batch_edges = []
        for i in range(batch_size):
            env_edges = adj[i].nonzero().t().contiguous()
            env_edges = env_edges + (i * self.n)
            batch_edges.append(env_edges)
        full_edge_index = torch.cat(batch_edges, dim=1).to(self.device)

        # current graph pass
        h = self.gnn(node_features, full_edge_index)

        # concat selected node's features
        selected = (h * selection.unsqueeze(-1)).sum(dim=1) # zeros if not selected
        selected_repeated = selected.unsqueeze(1).expand(-1, self.n, -1)
        new_embeddings = torch.cat([h, selected_repeated], dim=-1)

        # graph embedding
        batch_mapping = torch.arange(batch_size, device=new_embeddings.device).repeat_interleave(
            self.n
        )
        graph_latent = global_mean_pool(new_embeddings.reshape(-1, new_embeddings.shape[-1]), batch_mapping)

        value = self.head(graph_latent)

        return value, {}
