import torch
import torch.nn as nn
from skrl.models.torch import Model
from skrl.models.torch import CategoricalMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *


class PPO_ActorModel_AddRemoveEdgeDiscreteNoSelfLoops_FC(CategoricalMixin, Model):
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

        self.gnn = GNNBackboneGAT(
            node_feat_dim, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        adj_fc = torch.ones((self.n, self.n)) - torch.eye(self.n)
        self.fc_edge_index = adj_fc.nonzero().t().contiguous()

        # +1 since we'll add the adj information on the edge embeddings
        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim + 1, head_hidden_dim),
            nn.Linear(head_hidden_dim, 2),  # two logits ("add", "remove")
        )

        self.skip_head = nn.Linear(gnn_hidden_dim, 1)

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        adj = observations["adj"]
        node_features = observations["node_features"]

        batch_size = node_features.shape[0]

        batch_fc_edges = []
        for i in range(batch_size):
            batch_fc_edges.append(self.fc_edge_index + (i * self.n))
        full_fc_edge_index = torch.cat(batch_fc_edges, dim=1).to(self.device)

        # fully connected pass
        h = self.gnn(node_features, full_fc_edge_index)
        h_i = h.unsqueeze(2).expand(-1, -1, self.n, -1)
        h_j = h.unsqueeze(1).expand(-1, self.n, -1, -1)
        exists_flag = adj.unsqueeze(-1)
        edge_embeddings = torch.cat([h_i, h_j, exists_flag], dim=-1) # (b, n, n, edge_feat)

        edge_logits = self.head(edge_embeddings)  # (b, n, n, 2)

        # mask out self loops
        mask = ~torch.eye(self.n, dtype=torch.bool, device=edge_logits.device)
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        edge_logits = edge_logits[mask.expand(batch_size, -1, -1, 2)]
        edge_logits = edge_logits.view(batch_size, self.n*(self.n-1), 2)

        add_logits = edge_logits[:, :, 0]      # (B, E)
        remove_logits = edge_logits[:, :, 1]   # (B, E)
        skip_logit = self.skip_head(torch.mean(h, dim=1))

        logits = torch.cat([
            add_logits,
            remove_logits,
            skip_logit
        ], dim=1)   # (B, 2*ec - 2*n + 1)

        # print(f"probs: {torch.softmax(logits, dim=1)} -> {torch.argmax(torch.softmax(logits, dim=1))}")

        return logits, {}
