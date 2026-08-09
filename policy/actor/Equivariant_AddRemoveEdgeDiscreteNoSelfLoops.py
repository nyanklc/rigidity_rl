import torch
import torch.nn as nn
from skrl.models.torch import Model
from skrl.models.torch import CategoricalMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *



class PPO_ActorModel_Equivariant_AddRemoveEdgeDiscreteNoSelfLoops(CategoricalMixin, Model):
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

        self.gnn = GNNBackboneEquivariant(
            node_feat_dim, edge_feat_dim, gnn_hidden_dim
        )  # output dim = node_feat_dim

        # input cat[node features, selected node's features(zeros if no selected)]
        self.head = nn.Sequential(
            nn.Linear(2 * node_feat_dim + 1, head_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(head_hidden_dim, 2),
        )

        # input graph embedding
        self.allow_skip = allow_skip
        self.skip_head = nn.Sequential(
            nn.Linear(node_feat_dim, head_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(head_hidden_dim, 1),
        )

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"] # B, N, ...
        coord_features = observations["coord_features"] # B, N, 3
        edge_features = observations["edge_features"] # B, N, N, ...
        adj = observations["adj"] # B, N, N
        # selection = observations["selection"] # B, N

        batch_size = node_features.shape[0]

        n = node_features.shape[1]

        h = self.gnn(feats=node_features, coors=coord_features, edges=edge_features,
                     adj_mat=adj)
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        exists_flag = adj.unsqueeze(-1)
        edge_embeddings = torch.cat([h_i, h_j, exists_flag], dim=-1) # (b, n, n, edge_feat)

        edge_logits = self.head(edge_embeddings)  # (b, n, n, 2)

        # mask out self loops
        mask = ~torch.eye(n, dtype=torch.bool, device=edge_logits.device)
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        edge_logits = edge_logits[mask.expand(batch_size, -1, -1, 2)]
        edge_logits = edge_logits.view(batch_size, n*(n-1), 2)

        add_logits = edge_logits[:, :, 0]      # (B, E)
        remove_logits = edge_logits[:, :, 1]   # (B, E)
        skip_logit = self.skip_head(torch.mean(h, dim=1))
        if not self.allow_skip:
            skip_logit = torch.full_like(skip_logit, MASK_VALUE)

        logits = torch.cat([
            add_logits,
            remove_logits,
            skip_logit
        ], dim=1)   # (B, 2*ec - 2*n + 1)

        # mask invalid ADD
        add_mask = (adj == 0)  # only allow add where edge doesn't exist
        add_mask = add_mask[:, ~torch.eye(n, dtype=torch.bool, device=adj.device)]
        add_mask = add_mask.view(batch_size, -1)

        # mask invalid REMOVE
        remove_mask = (adj == 1)
        remove_mask = remove_mask[:, ~torch.eye(n, dtype=torch.bool, device=adj.device)]
        remove_mask = remove_mask.view(batch_size, -1)

        # apply masks
        E = (logits.shape[-1]-1)//2
        logits[:, :E][~add_mask] = MASK_VALUE
        logits[:, E:2*E][~remove_mask] = MASK_VALUE

        logits = unmask_if_all_masked(logits)

        return logits, {}
