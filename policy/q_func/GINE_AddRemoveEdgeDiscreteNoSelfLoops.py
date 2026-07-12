from typing import Any
import torch
import torch.nn as nn
from skrl.models.torch import Model
from skrl.models.torch import TabularMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *



# compatible with observation type "DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection"
class DQN_QNetwork_GINE_AddRemoveEdgeDiscreteNoSelfLoops(TabularMixin, Model):
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
            nn.Linear(head_hidden_dim, 2),  # two logits ("add", "remove")
        )

        # input graph embedding
        self.skip_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, head_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(head_hidden_dim, 1),
        )

    def random_act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])
        adj = observations["adj"]
        batch_size = adj.shape[0]
        n = adj.shape[1]

        add_mask = (adj == 0)
        add_mask = add_mask[:, ~torch.eye(n, dtype=torch.bool, device=adj.device)].view(batch_size, -1)
        
        remove_mask = (adj == 1)
        remove_mask = remove_mask[:, ~torch.eye(n, dtype=torch.bool, device=adj.device)].view(batch_size, -1)

        skip_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=adj.device)
        full_mask = torch.cat([add_mask, remove_mask, skip_mask], dim=1)
        
        actions = torch.multinomial(full_mask.float(), 1)
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
        edge_index_list = []
        edge_attr_list = []
        for i in range(batch_size):
            src, dst = adj[i].nonzero(as_tuple=True)
            edge_index = torch.stack([src, dst], dim=0) + i * n
            edge_index_list.append(edge_index)
            # we get all possible edges' features from the observation
            # but we only need existing edges'
            edge_attr_list.append(edge_features[i][src, dst])
        full_edge_index = torch.cat(edge_index_list, dim=1).to(self.device)
        full_edge_attr = torch.cat(edge_attr_list, dim=0).to(self.device)

        h = self.gnn(node_features, full_edge_index, full_edge_attr)
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

        q_values = torch.cat([
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
        E = (q_values.shape[-1]-1)//2
        q_values[:, :E][~add_mask] = -1e9
        q_values[:, E:2*E][~remove_mask] = -1e9

        return q_values, {}
