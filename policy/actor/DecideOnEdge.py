import torch
import torch.nn as nn
from skrl.models.torch import Model
from skrl.models.torch import CategoricalMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *


class PPO_ActorModel_DecideOnEdge(CategoricalMixin, Model):
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
        CategoricalMixin.__init__(self)

        self.gnn = GNNBackboneGAT(
            node_feat_dim, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        # input selected nodes' embeddings
        self.allow_skip = allow_skip
        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim, head_hidden_dim),
            nn.Linear(head_hidden_dim, 3), # add/remove/noop
        )

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]
        adj = observations["adj"]
        proposed_edge = observations["proposed_edge"].to(torch.int32)

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

        # edge_index, edge_attr = dense_to_sparse(adj)

        h = self.gnn(node_features, full_edge_index)  # B, N, H

        # score selected edge
        i_idx = proposed_edge[:, 0]  # B
        j_idx = proposed_edge[:, 1]  # B
        i_nemb = h[torch.arange(h.shape[0]), i_idx]  # B, 1, H
        j_nemb = h[torch.arange(h.shape[0]), j_idx]  # B, 1, H
        edge_emb = torch.cat([i_nemb, j_nemb], dim=-1).squeeze(1)  # B, 2H
        logits = self.head(edge_emb)

        # mask self loops
        self_loops = torch.argwhere(i_idx == j_idx)
        logits[self_loops, 0] = MASK_VALUE # add
        logits[self_loops, 1] = MASK_VALUE # remove

        # mask existing edges (or the other way)
        # TODO: maybe we shoudn't mask them out

        if not self.allow_skip:
            logits[:, 2] = MASK_VALUE  # index 2 is 'skip'

        logits = unmask_if_all_masked(logits)

        return logits, {}
