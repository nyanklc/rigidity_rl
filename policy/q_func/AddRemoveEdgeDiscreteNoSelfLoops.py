import torch
from typing import Any
import torch.nn as nn
from skrl.models.torch import Model
from skrl.models.torch import TabularMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *



class DQN_QNetwork_AddRemoveEdgeDiscreteNoSelfLoops(TabularMixin, Model):
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
        TabularMixin.__init__(self)

        self.gnn = GNNBackboneGAT(
            node_feat_dim, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        # +1 since we'll add the adj information on the edge embeddings
        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim + 1, head_hidden_dim),
            nn.Linear(head_hidden_dim, 2),  # two logits ("add", "remove")
        )

        self.allow_skip = allow_skip
        self.skip_head = nn.Linear(gnn_hidden_dim, 1)

    def random_act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])
        adj = observations["adj"]
        batch_size = adj.shape[0]
        n = adj.shape[1]

        add_mask = (adj == 0)
        add_mask = add_mask[:, ~torch.eye(n, dtype=torch.bool, device=adj.device)].view(batch_size, -1)

        remove_mask = (adj == 1)
        remove_mask = remove_mask[:, ~torch.eye(n, dtype=torch.bool, device=adj.device)].view(batch_size, -1)

        skip_mask = torch.full((batch_size, 1), self.allow_skip, dtype=torch.bool, device=adj.device)
        full_mask = torch.cat([add_mask, remove_mask, skip_mask], dim=1)

        actions = torch.multinomial(full_mask.float(), 1)
        return actions, {}

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        adj = observations["adj"]
        node_features = observations["node_features"]

        batch_size = node_features.shape[0]

        # adj comes in batched
        batch_edges = []
        for i in range(batch_size):
            env_edges = adj[i].nonzero().t().contiguous()
            env_edges = env_edges + (i * self.n)
            batch_edges.append(env_edges)
        full_edge_index = torch.cat(batch_edges, dim=1).to(self.device)

        # use current adj
        h = self.gnn(node_features, full_edge_index)
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
        if not self.allow_skip:
            skip_logit = torch.full_like(skip_logit, MASK_VALUE)

        q_values = torch.cat([
            add_logits,
            remove_logits,
            skip_logit
        ], dim=1)   # (B, 2*ec - 2*n + 1)

        # mask invalid ADD
        add_mask = (adj == 0)  # only allow add where edge doesn't exist
        add_mask = add_mask[:, ~torch.eye(self.n, dtype=torch.bool, device=adj.device)]
        add_mask = add_mask.view(batch_size, -1)

        # mask invalid REMOVE
        remove_mask = (adj == 1)
        remove_mask = remove_mask[:, ~torch.eye(self.n, dtype=torch.bool, device=adj.device)]
        remove_mask = remove_mask.view(batch_size, -1)

        # apply masks
        E = (q_values.shape[-1]-1)//2
        q_values[:, :E][~add_mask] = MASK_VALUE
        q_values[:, E:2*E][~remove_mask] = MASK_VALUE

        q_values = unmask_if_all_masked(q_values)

        return q_values, {}
