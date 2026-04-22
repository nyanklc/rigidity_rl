import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from skrl.models.torch import Model
from skrl.models.torch import CategoricalMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space, untensorize_space


class GNNBackbone(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim):
        super().__init__()
        self.conv1 = GATConv(node_feat_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)

    def forward(self, nodes, edge_index):
        batch_size, n, _ = nodes.shape
        x = nodes.reshape(-1, nodes.size(-1))

        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)

        return h.reshape(batch_size, n, -1)


class ActorModel(CategoricalMixin, Model):
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
        CategoricalMixin.__init__(self)

        # +1 since we'll add the degree as a node feature
        self.gnn = GNNBackbone(
            node_feat_dim + 1, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        adj_fc = torch.ones((self.n, self.n)) - torch.eye(self.n)
        self.fc_edge_index = adj_fc.nonzero().t().contiguous()

        # +1 since we'll add the adj information on the edge embeddings
        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim + 1, head_hidden_dim),
            nn.Linear(head_hidden_dim, 1),  # output single logit ("add")
        )


    def compute(self, inputs, role):
        # TODO: for some reason untensorize_space doesn't work for us.
        # idk how to properly use the api, and we shouldn't even need to do this
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])
        # observations = untensorize_space(self.observation_space, inputs["observations"])

        # # Don't squeeze, we'll lose the batch that way
        # node_features = observations["node_features"].squeeze()
        # adj = observations["adj"].squeeze()

        node_features = observations["node_features"]
        adj = observations["adj"]

        batch_size = node_features.shape[0]

        batch_fc_edges = []
        for i in range(batch_size):
            batch_fc_edges.append(self.fc_edge_index + (i * self.n))
        full_fc_edge_index = torch.cat(batch_fc_edges, dim=1).to(self.device)

        out_degrees = adj.sum(dim=-1, keepdim=True)
        node_features = torch.cat([node_features, out_degrees], dim=-1)

        # fully connected pass
        h = self.gnn(node_features, full_fc_edge_index)
        h_i = h.unsqueeze(2).expand(-1, -1, self.n, -1)
        h_j = h.unsqueeze(1).expand(-1, self.n, -1, -1)
        exists_flag = adj.unsqueeze(-1)
        edge_embeddings = torch.cat([h_i, h_j, exists_flag], dim=-1)

        logits = self.head(edge_embeddings).squeeze(-1).reshape(batch_size, -1)

        # exclude self loops
        logits = logits.view(batch_size, self.n, self.n)
        mask = ~torch.eye(self.n, dtype=torch.bool, device=logits.device)  # (n, n)
        logits = logits[:, mask]  # (B, n*n - n)

        return logits, {}


class CriticModel(DeterministicMixin, Model):
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

        self.gnn = GNNBackbone(node_feat_dim, gnn_hidden_dim)
        self.n = n
        self.head = self.head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, head_hidden_dim), nn.Linear(head_hidden_dim, 1)
        )

        # self.observation_space = observation_space
        # self.action_space = action_space
        # self.device = device

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]
        adj = observations["adj"]
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

        # graph embedding
        batch_mapping = torch.arange(batch_size, device=h.device).repeat_interleave(
            self.n
        )
        graph_latent = global_mean_pool(h.reshape(-1, h.shape[-1]), batch_mapping)

        value = self.head(graph_latent)

        return value, {}
