from typing import Any
from typing import Any
import torch
import torch.nn as nn
from skrl.models.torch import Model
from skrl.models.torch import TabularMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *



class DQN_QNetwork_Equivariant_SelectNodesSequentially(TabularMixin, Model):
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
        allow_skip=True,
    ):
        # Model.__init__(self, observation_space, action_space, device)
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        TabularMixin.__init__(self)

        # An always-available zero-reward action is an absorbing optimum: with
        # SelectNodesSequentially, select->skip is a no-op 2-cycle that never touches
        # the graph. Masked here in both compute() and random_act() so epsilon-greedy
        # exploration cannot reintroduce it.
        self.allow_skip = allow_skip

        self.gnn = GNNBackboneEquivariant(
            node_feat_dim,edge_feat_dim, gnn_hidden_dim
        )  # output dim = node_feat_dim

        # input cat[node features, selected node's features(zeros if no selected)]
        self.head = nn.Sequential(
            nn.Linear(2 * node_feat_dim, head_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(head_hidden_dim, 1),
        )

        # input graph embedding
        if allow_skip:
            self.skip_head = nn.Sequential(
                nn.Linear(node_feat_dim, head_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(head_hidden_dim, 1),
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

        skip_mask = torch.full((batch_size, 1), self.allow_skip, dtype=torch.bool, device=selection.device)
        full_mask = torch.cat([valid_mask, skip_mask], dim=1)

        actions = torch.multinomial(full_mask.float(), 1)
        return actions, {}

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"] # B, N, ...
        coord_features = observations["coord_features"] # B, N, 3
        edge_features = observations["edge_features"] # B, N, N, ...
        adj = observations["adj"] # B, N, N
        selection = observations["selection"] # B, N

        n = node_features.shape[1]

        h = self.gnn(feats=node_features, coors=coord_features, edges=edge_features,
                     adj_mat=adj)

        # concat selected node's features
        selected = (h * selection.unsqueeze(-1)).sum(dim=1) # zeros if not selected
        selected_repeated = selected.unsqueeze(1).expand(-1, n, -1)
        new_embeddings = torch.cat([h, selected_repeated], dim=-1)

        # calculate node scores for selection
        add_remove_logits = self.head(new_embeddings).squeeze(-1)
        if self.allow_skip:
            skip_logit = self.skip_head(torch.mean(h, dim=1))
        else:
            skip_logit = torch.full(
                (add_remove_logits.shape[0], 1), MASK_VALUE, device=add_remove_logits.device
            )
        q_values = torch.cat([add_remove_logits, skip_logit], dim=-1)

        # mask out self loops
        # print(f"q_values before: {q_values}")
        selected_mask = selection.bool().squeeze(-1)
        has_selected = selection.sum(dim=1) > 0
        has_selected = has_selected.unsqueeze(1).expand(-1, selected_mask.size(1))
        mask = selected_mask & has_selected   # (B, N)
        q_values[:, :-1] = q_values[:, :-1].masked_fill(mask, MASK_VALUE) # exclude the skip action

        q_values = unmask_if_all_masked(q_values)

        return q_values, {}
