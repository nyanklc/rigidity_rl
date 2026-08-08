import torch
import torch.nn as nn
from skrl.models.torch import Model
from skrl.models.torch import CategoricalMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *


class PPO_ActorModel_GINE_SelectNodesSequentially(CategoricalMixin, Model):
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
        CategoricalMixin.__init__(self)

        # An always-available zero-reward action is an absorbing optimum for on-policy
        # methods: select->skip is a no-op 2-cycle that never touches the graph, so the
        # reward variance (and with it the policy gradient) collapses to zero.
        self.allow_skip = allow_skip

        self.gnn = GNNBackboneGINE(
            node_feat_dim, edge_feat_dim, gnn_hidden_dim
        )  # output dim = hidden dim

        # input cat[node features, selected node's features(zeros if no selected)]
        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim, head_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(head_hidden_dim, 1),
        )

        # input graph embedding
        if allow_skip:
            self.skip_head = nn.Sequential(
                nn.Linear(gnn_hidden_dim, head_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(head_hidden_dim, 1),
            )

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]
        edge_features = observations["edge_features"]
        adj = observations["adj"]
        selection = observations["selection"]

        batch_size = node_features.shape[0]
        n = node_features.shape[1]

        # batch
        h = self.gnn(node_features, edge_features)

        # concat selected node's features
        selected = (h * selection.unsqueeze(-1)).sum(dim=1) # zeros if not selected
        selected_repeated = selected.unsqueeze(1).expand(-1, n, -1)
        new_embeddings = torch.cat([h, selected_repeated], dim=-1)

        # calculate node scores for selection
        add_remove_logits = self.head(new_embeddings).squeeze(-1)
        if self.allow_skip:
            skip_logit = self.skip_head(torch.mean(h, dim=1))
        else:
            skip_logit = torch.full((batch_size, 1), -1e9, device=add_remove_logits.device)
        logits = torch.cat([add_remove_logits, skip_logit], dim=-1)

        # mask out self loops
        selected_mask = selection.bool().squeeze(-1)
        has_selected = selection.sum(dim=1) > 0
        has_selected = has_selected.unsqueeze(1).expand(-1, selected_mask.size(1))
        mask = selected_mask & has_selected   # (B, N)
        logits[:, :-1] = logits[:, :-1].masked_fill(mask, -1e9) # exclude the skip action

        return logits, {}
