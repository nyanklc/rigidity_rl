import torch
import torch.nn as nn
from skrl.models.torch import Model
from skrl.models.torch import CategoricalMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *


class PPO_ActorModel_AddEdgeDiscreteNoSelfLoops(CategoricalMixin, Model):
    def __init__(
        self,
        n,
        node_feat_dim,
        gnn_hidden_dim,
        head_hidden_dim,

        observation_space,
        action_space,
        device,
        allow_skip=True,
    ):
        # Model.__init__(self, observation_space, action_space, device)
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        CategoricalMixin.__init__(self)

        self.gnn = GNNBackboneGAT(
            node_feat_dim, gnn_hidden_dim
        )  # output dim = hidden dim

        adj_fc = torch.ones((n, n)) - torch.eye(n)
        self.fc_edge_index = adj_fc.nonzero().t().contiguous()

        # +1 since we'll add the adj information on the edge embeddings
        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim + 1, head_hidden_dim),
            nn.Linear(head_hidden_dim, 1),  # output single logit ("add")
        )

        self.allow_skip = allow_skip
        self.skip_head = nn.Linear(gnn_hidden_dim, 1)


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

        n = node_features.shape[1]

        batch_size = node_features.shape[0]

        batch_fc_edges = []
        for i in range(batch_size):
            batch_fc_edges.append(self.fc_edge_index + (i * n))
        full_fc_edge_index = torch.cat(batch_fc_edges, dim=1).to(self.device)

        # fully connected pass
        h = self.gnn(node_features, full_fc_edge_index)
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        exists_flag = adj.unsqueeze(-1)
        edge_embeddings = torch.cat([h_i, h_j, exists_flag], dim=-1) # (b, n, n, edge_feat)

        logits = self.head(edge_embeddings).squeeze(-1).reshape(batch_size, -1) # (b, n*n)

        # exclude self loops
        logits = logits.view(batch_size, n, n)
        mask = ~torch.eye(n, dtype=torch.bool, device=logits.device)  # (n, n)
        logits = logits[:, mask]  # (b, n*n - n)

        skip_logit = self.skip_head(torch.mean(h, dim=1))
        if not self.allow_skip:
            skip_logit = torch.full_like(skip_logit, MASK_VALUE)

        logits = torch.cat([
            logits,
            skip_logit
        ], dim=1)

        return logits, {}
