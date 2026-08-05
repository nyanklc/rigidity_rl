import torch
import torch.nn as nn
from skrl.models.torch import Model
from skrl.models.torch import CategoricalMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *



# compatible with observation type "DictEquivariantNodeFeaturesAndAdjAndSelection"
class PPO_ActorModel_Equivariant_SelectNodesSequentially(CategoricalMixin, Model):
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
        allow_skip=True,
    ):
        # Model.__init__(self, observation_space, action_space, device)
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        CategoricalMixin.__init__(self)

        # An always-available zero-reward action is an absorbing optimum for on-policy
        # methods: select->skip is a no-op 2-cycle that never touches the graph, so the
        # reward variance (and with it the policy gradient) collapses to zero.
        self.allow_skip = allow_skip

        # +1 since we'll add the degree as a node feature
        node_feat_dim = node_feat_dim + 1
        self.gnn = GNNBackboneEquivariant(
            node_feat_dim, edge_feat_dim, gnn_hidden_dim
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

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"] # B, N, 3
        coord_features = observations["coord_features"] # B, N, 3
        edge_features = observations["edge_features"] # B, N, N, 3
        adj = observations["adj"] # B, N, N
        selection = observations["selection"] # B, N

        # print(f"node_features: {node_features}")
        # print(f"adj: {adj}")
        # print(f"selection: {selection}")

        batch_size = node_features.shape[0]
        n = node_features.shape[1]

        out_degrees = adj.sum(dim=-1, keepdim=True)
        node_features = torch.cat([node_features, out_degrees], dim=-1)

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
            skip_logit = torch.full((batch_size, 1), -1e9, device=add_remove_logits.device)
        logits = torch.cat([add_remove_logits, skip_logit], dim=-1)

        # mask out self loops
        # print(f"q_values before: {q_values}")
        selected_mask = selection.bool().squeeze(-1)
        has_selected = selection.sum(dim=1) > 0
        has_selected = has_selected.unsqueeze(1).expand(-1, selected_mask.size(1))
        mask = selected_mask & has_selected   # (B, N)
        logits[:, :-1] = logits[:, :-1].masked_fill(mask, -1e9) # exclude the skip action
        # print(f"selected_mask: {selected_mask}")
        # print(f"has_selected: {has_selected}")
        # print(f"mask: {mask}")
        # print(f"q_values after: {q_values}")

        return logits, {}
