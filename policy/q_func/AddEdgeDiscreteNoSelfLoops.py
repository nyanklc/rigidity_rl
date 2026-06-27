import torch
import torch.nn as nn
from skrl.models.torch import Model
from skrl.models.torch import TabularMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *



class DQN_QNetwork_AddEdgeDiscreteNoSelfLoops(TabularMixin, Model):
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

        # +1 since we'll add the degree as a node feature
        self.gnn = GNNBackboneGAT(
            node_feat_dim + 1, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        # +1 since we'll add the adj information on the edge embeddings
        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim + 1, head_hidden_dim),
            nn.Linear(head_hidden_dim, 1),  # output single logit ("add")
        )

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

        batch_size = node_features.shape[0]

        # adj comes in batched
        batch_edges = []
        for i in range(batch_size):
            env_edges = adj[i].nonzero().t().contiguous()
            env_edges = env_edges + (i * self.n)
            batch_edges.append(env_edges)
        full_edge_index = torch.cat(batch_edges, dim=1).to(self.device)

        out_degrees = adj.sum(dim=-1, keepdim=True)
        node_features = torch.cat([node_features, out_degrees], dim=-1)

        # fully connected pass
        h = self.gnn(node_features, full_edge_index)
        h_i = h.unsqueeze(2).expand(-1, -1, self.n, -1)
        h_j = h.unsqueeze(1).expand(-1, self.n, -1, -1)
        exists_flag = adj.unsqueeze(-1)
        edge_embeddings = torch.cat([h_i, h_j, exists_flag], dim=-1) # (b, n, n, edge_feat)

        q_values = self.head(edge_embeddings).squeeze(-1).reshape(batch_size, -1) # (b, n*n)

        # exclude self loops
        q_values = q_values.view(batch_size, self.n, self.n)
        mask = ~torch.eye(self.n, dtype=torch.bool, device=q_values.device)  # (n, n)
        q_values = q_values[:, mask]  # (b, n*n - n)

        skip_logit = self.skip_head(torch.mean(h, dim=1))

        q_values = torch.cat([
            q_values,
            skip_logit
        ], dim=1)

        add_mask = (adj == 0)
        add_mask = add_mask[:, ~torch.eye(self.n, dtype=torch.bool, device=adj.device)].view(batch_size, -1)
        q_values[:, :-1][~add_mask] = -5

        # action = torch.argmax(q_values)
        # print(f"==> {q_values}, max: {action}")
        # def dummy():
        #     n = adj.shape[-1]

        #     # skip
        #     if action == n**2 - n:
        #         pass
        #     # add
        #     else:
        #         i_idx = action // (n - 1)
        #         j_idx = action % (n - 1)
        #         if j_idx >= i_idx:
        #             j_idx += 1
        #         print(f"==> ALKJAK {i_idx} -> {j_idx}")
        # dummy()

        return q_values, {}
