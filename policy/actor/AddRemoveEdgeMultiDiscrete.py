import torch
import torch.nn as nn
from skrl.models.torch import Model
from skrl.models.torch import MultiCategoricalMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *



class PPO_ActorModel_AddRemoveEdgeMultiDiscrete(MultiCategoricalMixin, Model):
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
        MultiCategoricalMixin.__init__(self)

        self.gnn = GNNBackboneGAT(
            node_feat_dim, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        # takes in node embeddings and outputs logits for selection of the "i" node
        self.select_i_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, head_hidden_dim),
            nn.Linear(head_hidden_dim, 1),
        )

        # takes in node embeddings and outputs logits for selection of the "j" node
        self.select_j_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, head_hidden_dim),
            nn.Linear(head_hidden_dim, 1),
        )

        # takes in global embedding (mean) and decides to add/remove/skip
        self.allow_skip = allow_skip
        self.action_type_head = nn.Linear(gnn_hidden_dim, 3)

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

        # current graph pass
        h = self.gnn(node_features, full_edge_index)

        # graph embedding
        batch_mapping = torch.arange(batch_size, device=h.device).repeat_interleave(
            self.n
        )
        graph_latent = global_mean_pool(h.reshape(-1, h.shape[-1]), batch_mapping)

        action_type_logits = self.action_type_head(graph_latent)
        if not self.allow_skip:
            action_type_logits = action_type_logits.clone()
            action_type_logits[:, 2] = MASK_VALUE  # add / remove / skip
        i_logits = self.select_i_head(h).squeeze(-1)
        j_logits = self.select_j_head(h).squeeze(-1)

        cat = torch.cat([action_type_logits, i_logits, j_logits], dim=-1)

        return cat, {}
