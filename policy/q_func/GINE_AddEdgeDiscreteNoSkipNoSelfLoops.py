from typing import Any
import torch
import torch.nn as nn
from skrl.models.torch import Model
from skrl.models.torch import TabularMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *



class DQN_QNetwork_GINE_AddEdgeDiscreteNoSkipNoSelfLoops(TabularMixin, Model):
    def __init__(
        self,
        n,
        node_feat_dim,
        gnn_hidden_dim,
        edge_feat_dim,
        head_hidden_dim,

        observation_space,
        action_space,
        device,
    ):
        # Model.__init__(self, observation_space, action_space, device)
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        TabularMixin.__init__(self)

        self.gnn = GNNBackboneGINE(
            node_feat_dim, edge_feat_dim, gnn_hidden_dim
        )  # output dim = hidden dim

        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim + 1, head_hidden_dim),
            nn.Linear(head_hidden_dim, 1),
        )

    def random_act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])
        adj = observations["adj"]
        batch_size = adj.shape[0]
        n = adj.shape[1]

        add_mask = (adj == 0)
        add_mask = add_mask[:, ~torch.eye(n, dtype=torch.bool, device=adj.device)].view(batch_size, -1)
        
        actions = torch.multinomial(add_mask.float(), 1)
        return actions, {}

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]
        edge_features = observations["edge_features"]
        adj = observations["adj"]
        # selection = observations["selection"]

        batch_size = node_features.shape[0]

        n = node_features.shape[1]

        # batch
        h = self.gnn(node_features, edge_features)
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        exists_flag = adj.unsqueeze(-1)
        edge_embeddings = torch.cat([h_i, h_j, exists_flag], dim=-1) # (b, n, n, edge_feat)

        q_values = self.head(edge_embeddings)  # (b, n, n, 1)

        # mask out self loops
        mask = ~torch.eye(n, dtype=torch.bool, device=q_values.device)
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        q_values = q_values[mask.expand(batch_size, -1, -1, 1)]
        q_values = q_values.view(batch_size, n*(n-1))

        # q_values = torch.cat([
        #     add_logits,
        # ], dim=1)   # (B, 2*ec - 2*n + 1)

        # mask invalid ADD
        add_mask = (adj == 0)  # only allow add where edge doesn't exist
        add_mask = add_mask[:, ~torch.eye(n, dtype=torch.bool, device=adj.device)]
        add_mask = add_mask.view(batch_size, -1)

        # apply masks
        E = (q_values.shape[-1])
        q_values[:, :E][~add_mask] = -1e9

        return q_values, {}
