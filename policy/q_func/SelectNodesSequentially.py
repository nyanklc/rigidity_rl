import torch
from typing import Any
import torch.nn as nn
from skrl.models.torch import Model
from skrl.models.torch import TabularMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *



# compatible with observation type "DictNodeFeaturesAndAdjAndSelection"
class DQN_QNetwork_SelectNodesSequentially(TabularMixin, Model):
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
        TabularMixin.__init__(self)

        self.gnn = GNNBackboneGAT(
            node_feat_dim, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        # input all node features + selected
        self.head = nn.Sequential(
            nn.Linear(n * gnn_hidden_dim + n, head_hidden_dim),
            nn.Linear(head_hidden_dim, n + 1), # +1 skip
        )

    def random_act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])
        selection = observations["selection"]
        batch_size = selection.shape[0]
        n = selection.shape[1]

        selected_mask = selection.bool().squeeze(-1)
        has_selected = selection.sum(dim=1) > 0
        has_selected = has_selected.unsqueeze(1).expand(-1, selected_mask.size(1))
        invalid_mask = selected_mask & has_selected   # (B, N)
        valid_mask = ~invalid_mask

        skip_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=selection.device)
        full_mask = torch.cat([valid_mask, skip_mask], dim=1)

        actions = torch.multinomial(full_mask.float(), 1)
        return actions, {}

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]
        adj = observations["adj"]
        selection = observations["selection"]

        # print(f"node_features: {node_features}")
        # print(f"adj: {adj}")
        # print(f"selection: {selection}")

        batch_size = node_features.shape[0]

        # adj comes in batched
        batch_edges = []
        for i in range(batch_size):
            env_edges = adj[i].nonzero().t().contiguous()
            env_edges = env_edges + (i * self.n)
            batch_edges.append(env_edges)
        full_edge_index = torch.cat(batch_edges, dim=1).to(self.device)

        # fully connected pass
        h = self.gnn(node_features, full_edge_index)

        # calculate node scores for selection
        q_values = self.head(torch.cat([h.flatten(-2), selection], dim=-1)).squeeze(-1).reshape(batch_size, -1)

        # mask out self loops
        # print(f"q_values before: {q_values}")
        selected_mask = selection.bool().squeeze(-1)
        has_selected = selection.sum(dim=1) > 0
        has_selected = has_selected.unsqueeze(1).expand(-1, selected_mask.size(1))
        mask = selected_mask & has_selected   # (B, N)
        q_values[:, :-1] = q_values[:, :-1].masked_fill(mask, -1e9) # exclude the skip action
        # print(f"selected_mask: {selected_mask}")
        # print(f"has_selected: {has_selected}")
        # print(f"mask: {mask}")
        # print(f"q_values after: {q_values}")

        return q_values, {}
