import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from skrl.models.torch import Model
from skrl.models.torch import CategoricalMixin, DeterministicMixin, MultiCategoricalMixin
from skrl.models.torch.tabular import TabularMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space, untensorize_space
from torch_geometric.utils import to_edge_index, dense_to_sparse
from egnn_pytorch import EGNN


class GNNBackboneEquivariant(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, edge_dim):
        super().__init__()
        self.conv1 = EGNN(dim=node_feat_dim, m_dim=hidden_dim, edge_dim=edge_dim)
        self.conv2 = EGNN(dim=node_feat_dim, m_dim=hidden_dim, edge_dim=edge_dim)

    def forward(self,
                feats,
                coors,
                adj_mat=None,
                edges=None):
        batch_size = feats.shape[0]
        n = feats.shape[1]

        n_out, c_out = self.conv1(feats=feats,
                                  coors=coors,
                                  edges=edges,
                                  adj_mat=adj_mat)
        h, _ = self.conv2(feats=n_out,
                          coors=c_out,
                          edges=edges,
                          adj_mat=adj_mat)

        return h.reshape(batch_size, n, -1)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# PPO

# compatible with observation type "DictEquivariantNodeFeaturesAndAdjAndSelection"
class PPO_Equivariant_ActorModel_SelectNodesSequentially(CategoricalMixin, Model):
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
        node_feat_dim = node_feat_dim + 1
        self.gnn = GNNBackboneEquivariant(
            node_feat_dim, gnn_hidden_dim, observation_space["edge_features"].shape[-1]
        )  # output dim = hidden dim
        self.n = n

        # input cat[node features, selected node's features(zeros if no selected)]
        self.head = nn.Sequential(
            nn.Linear(2 * node_feat_dim, head_hidden_dim),
            nn.Linear(head_hidden_dim, 1),
        )

        # input graph embedding
        self.skip_head = nn.Sequential(
            nn.Linear(node_feat_dim, head_hidden_dim),
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
        selected_repeated = selected.unsqueeze(1).expand(-1, self.n, -1)
        new_embeddings = torch.cat([h, selected_repeated], dim=-1)

        # calculate node scores for selection
        add_remove_logits = self.head(new_embeddings).squeeze(-1)
        skip_logit = self.skip_head(torch.mean(h, dim=1))
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

# compatible with observation type "DictEquivariantNodeFeaturesAndAdjAndSelection"
class PPO_Equivariant_CriticModel_Selection(DeterministicMixin, Model):
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

        # +1 since we'll add the degree as a node feature
        node_feat_dim = node_feat_dim + 1
        self.gnn = GNNBackboneEquivariant(
            node_feat_dim, gnn_hidden_dim, observation_space["edge_features"].shape[-1]
        )  # output dim = hidden dim
        self.n = n

        # input cat[node features, selected node's features(zeros if no selected)]
        self.head = nn.Sequential(
            nn.Linear(2 * node_feat_dim, head_hidden_dim),
            nn.Linear(head_hidden_dim, 1),
        )

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]  # B, N, 3
        coord_features = observations["coord_features"]  # B, N, 3
        edge_features = observations["edge_features"]  # B, N, N, 3
        adj = observations["adj"]  # B, N, N
        selection = observations["selection"]  # B, N

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
        selected = (h * selection.unsqueeze(-1)).sum(dim=1)  # zeros if not selected
        selected_repeated = selected.unsqueeze(1).expand(-1, self.n, -1)
        new_embeddings = torch.cat([h, selected_repeated], dim=-1)

        # graph embedding
        batch_mapping = torch.arange(batch_size, device=new_embeddings.device).repeat_interleave(
            self.n
        )
        graph_latent = global_mean_pool(new_embeddings.reshape(-1, new_embeddings.shape[-1]), batch_mapping)

        value = self.head(graph_latent)

        return value, {}
