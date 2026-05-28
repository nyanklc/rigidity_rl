import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from skrl.models.torch import Model
from skrl.models.torch import CategoricalMixin, DeterministicMixin, MultiCategoricalMixin
from skrl.models.torch.tabular import TabularMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space, untensorize_space
from torch_geometric.utils import to_edge_index, dense_to_sparse


class GNNBackbone(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim):
        super().__init__()
        self.conv1 = GATConv(node_feat_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)

    def forward(self, nodes, edge_index):
        batch_size, n, _ = nodes.shape
        x = nodes.reshape(-1, nodes.size(-1))

        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)

        return h.reshape(batch_size, n, -1)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# PPO

class PPO_ActorModel_AddEdgeDiscreteNoSelfLoops(CategoricalMixin, Model):
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
        self.gnn = GNNBackbone(
            node_feat_dim + 1, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        adj_fc = torch.ones((self.n, self.n)) - torch.eye(self.n)
        self.fc_edge_index = adj_fc.nonzero().t().contiguous()

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

        batch_fc_edges = []
        for i in range(batch_size):
            batch_fc_edges.append(self.fc_edge_index + (i * self.n))
        full_fc_edge_index = torch.cat(batch_fc_edges, dim=1).to(self.device)

        out_degrees = adj.sum(dim=-1, keepdim=True)
        node_features = torch.cat([node_features, out_degrees], dim=-1)

        # fully connected pass
        h = self.gnn(node_features, full_fc_edge_index)
        h_i = h.unsqueeze(2).expand(-1, -1, self.n, -1)
        h_j = h.unsqueeze(1).expand(-1, self.n, -1, -1)
        exists_flag = adj.unsqueeze(-1)
        edge_embeddings = torch.cat([h_i, h_j, exists_flag], dim=-1) # (b, n, n, edge_feat)

        logits = self.head(edge_embeddings).squeeze(-1).reshape(batch_size, -1) # (b, n*n)

        # exclude self loops
        logits = logits.view(batch_size, self.n, self.n)
        mask = ~torch.eye(self.n, dtype=torch.bool, device=logits.device)  # (n, n)
        logits = logits[:, mask]  # (b, n*n - n)

        skip_logit = self.skip_head(torch.mean(h, dim=1))

        logits = torch.cat([
            logits,
            skip_logit
        ], dim=1)

        return logits, {}

class PPO_ActorModel_AddEdgeDiscreteNoSkipNoSelfLoops(CategoricalMixin, Model):
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
        self.gnn = GNNBackbone(
            node_feat_dim + 1, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        adj_fc = torch.ones((self.n, self.n)) - torch.eye(self.n)
        self.fc_edge_index = adj_fc.nonzero().t().contiguous()

        # +1 since we'll add the adj information on the edge embeddings
        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim + 1, head_hidden_dim),
            nn.Linear(head_hidden_dim, 1),  # output single logit ("add")
        )


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

        batch_fc_edges = []
        for i in range(batch_size):
            batch_fc_edges.append(self.fc_edge_index + (i * self.n))
        full_fc_edge_index = torch.cat(batch_fc_edges, dim=1).to(self.device)

        out_degrees = adj.sum(dim=-1, keepdim=True)
        node_features = torch.cat([node_features, out_degrees], dim=-1)

        # fully connected pass
        h = self.gnn(node_features, full_fc_edge_index)
        h_i = h.unsqueeze(2).expand(-1, -1, self.n, -1)
        h_j = h.unsqueeze(1).expand(-1, self.n, -1, -1)
        exists_flag = adj.unsqueeze(-1)
        edge_embeddings = torch.cat([h_i, h_j, exists_flag], dim=-1) # (b, n, n, edge_feat)

        logits = self.head(edge_embeddings).squeeze(-1).reshape(batch_size, -1) # (b, n*n)

        # exclude self loops
        logits = logits.view(batch_size, self.n, self.n)
        mask = ~torch.eye(self.n, dtype=torch.bool, device=logits.device)  # (n, n)
        logits = logits[:, mask]  # (b, n*n - n)

        return logits, {}

class PPO_ActorModel_AllEdges(CategoricalMixin, Model):
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
        self.gnn = GNNBackbone(
            node_feat_dim + 1, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        # +1 since we'll add the adj information on the edge embeddings
        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim + 1, head_hidden_dim),
            nn.Linear(head_hidden_dim, 2),  # two logits ("add", "remove")
        )

        self.skip_head = nn.Linear(gnn_hidden_dim, 1)

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        adj = observations["adj"]
        node_features = observations["node_features"]

        batch_size = node_features.shape[0]

        batch_fc_edges = []
        for i in range(batch_size):
            batch_fc_edges.append(self.fc_edge_index + (i * self.n))
        full_fc_edge_index = torch.cat(batch_fc_edges, dim=1).to(self.device)

        out_degrees = adj.sum(dim=-1, keepdim=True)
        node_features = torch.cat([node_features, out_degrees], dim=-1)

        # fully connected pass
        h = self.gnn(node_features, full_fc_edge_index)
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

        logits = torch.cat([
            add_logits,
            remove_logits,
            skip_logit
        ], dim=1)   # (B, 2*ec - 2*n + 1)

        # print(f"probs: {torch.softmax(logits, dim=1)} -> {torch.argmax(torch.softmax(logits, dim=1))}")

        return logits, {}

class PPO_ActorModel_AddRemoveEdgeDiscreteNoSelfLoops_FC(CategoricalMixin, Model):
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
        self.gnn = GNNBackbone(
            node_feat_dim + 1, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        adj_fc = torch.ones((self.n, self.n)) - torch.eye(self.n)
        self.fc_edge_index = adj_fc.nonzero().t().contiguous()

        # +1 since we'll add the adj information on the edge embeddings
        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim + 1, head_hidden_dim),
            nn.Linear(head_hidden_dim, 2),  # two logits ("add", "remove")
        )

        self.skip_head = nn.Linear(gnn_hidden_dim, 1)

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        adj = observations["adj"]
        node_features = observations["node_features"]

        batch_size = node_features.shape[0]

        batch_fc_edges = []
        for i in range(batch_size):
            batch_fc_edges.append(self.fc_edge_index + (i * self.n))
        full_fc_edge_index = torch.cat(batch_fc_edges, dim=1).to(self.device)

        out_degrees = adj.sum(dim=-1, keepdim=True)
        node_features = torch.cat([node_features, out_degrees], dim=-1)

        # fully connected pass
        h = self.gnn(node_features, full_fc_edge_index)
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

        logits = torch.cat([
            add_logits,
            remove_logits,
            skip_logit
        ], dim=1)   # (B, 2*ec - 2*n + 1)

        # print(f"probs: {torch.softmax(logits, dim=1)} -> {torch.argmax(torch.softmax(logits, dim=1))}")

        return logits, {}

class PPO_ActorModel_AddRemoveEdgeDiscreteNoSelfLoops(CategoricalMixin, Model):
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
        self.gnn = GNNBackbone(
            node_feat_dim + 1, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        # +1 since we'll add the adj information on the edge embeddings
        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim + 1, head_hidden_dim),
            nn.Linear(head_hidden_dim, 2),  # two logits ("add", "remove")
        )

        self.skip_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.Linear(gnn_hidden_dim, 1)
        )

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

        out_degrees = adj.sum(dim=-1, keepdim=True)
        node_features = torch.cat([node_features, out_degrees], dim=-1)

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

        logits = torch.cat([
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
        # effectively settings the probability of these actions to 0
        E = (logits.shape[-1]-1)//2
        logits[:, :E][~add_mask] = -1e9
        logits[:, E:2*E][~remove_mask] = -1e9

        # print(f"probs: {torch.softmax(logits, dim=1)} -> {torch.argmax(torch.softmax(logits, dim=1))}")

        return logits, {}

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
    ):
        # Model.__init__(self, observation_space, action_space, device)
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        MultiCategoricalMixin.__init__(self)

        self.gnn = GNNBackbone(
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
        i_logits = self.select_i_head(h).squeeze(-1)
        j_logits = self.select_j_head(h).squeeze(-1)

        print(f"action_type_logits: {action_type_logits.shape}")
        print(f"i_logits: {i_logits.shape}")
        print(f"j_logits: {j_logits.shape}")

        cat = torch.cat([action_type_logits, i_logits, j_logits], dim=-1)
        print(f"cat: {cat.shape}")

        return cat, {}

# compatible with observation type "DictNodeFeaturesAndAdjAndSelection"
class PPO_ActorModel_SelectNodesSequentially(CategoricalMixin, Model):
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
        self.gnn = GNNBackbone(
            node_feat_dim + 1, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        # input cat[node features, selected node's features(zeros if no selected)]
        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim, head_hidden_dim),
            nn.Linear(head_hidden_dim, 1),
        )

        # input graph embedding
        self.skip_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, head_hidden_dim),
            nn.Linear(head_hidden_dim, 1),
        )

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]
        adj = observations["adj"]
        selection = observations["selection"]

        # print(f"node_features: {node_features}")
        # print(f"adj: {adj}")
        # print(f"selection: {selection}")

        batch_size = node_features.shape[0]
        n = node_features.shape[1]

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

# compatible with observation "DictNodeFeaturesAndAdjAndEdgeProposal"
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
    ):
        # Model.__init__(self, observation_space, action_space, device)
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        CategoricalMixin.__init__(self)

        # +1 since we'll add the degree as a node feature TODO: do this in environment?
        self.gnn = GNNBackbone(
            node_feat_dim + 1, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        # input selected nodes' embeddings
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

        out_degrees = adj.sum(dim=-1, keepdim=True)
        node_features = torch.cat([node_features, out_degrees], dim=-1)

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
        logits[self_loops, 0] = -1e9 # add
        logits[self_loops, 1] = -1e9 # remove

        # mask existing edges (or the other way)
        # TODO: maybe we shoudn't mask them out

        return logits, {}

class PPO_CriticModel_Selection(DeterministicMixin, Model):
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

        self.gnn = GNNBackbone(node_feat_dim, gnn_hidden_dim)
        self.n = n
        # +1 for selection "bit"
        self.head = nn.Sequential(
            nn.Linear(gnn_hidden_dim + 1, head_hidden_dim), nn.Linear(head_hidden_dim, 1)
        )

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]
        adj = observations["adj"]
        selection = observations["selection"]

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

        # concat selected node's features
        selected = (h * selection.unsqueeze(-1)).sum(dim=1) # zeros if not selected
        selected_repeated = selected.unsqueeze(1).expand(-1, self.n, -1)
        new_embeddings = torch.cat([h, selected_repeated], dim=-1)

        # graph embedding
        batch_mapping = torch.arange(batch_size, device=new_embeddings.device).repeat_interleave(
            self.n
        )
        graph_latent = global_mean_pool(new_embeddings.reshape(-1, new_embeddings.shape[-1]), batch_mapping)

        value = self.head(graph_latent)

        return value, {}

class PPO_CriticModel(DeterministicMixin, Model):
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

        self.gnn = GNNBackbone(node_feat_dim, gnn_hidden_dim)
        self.n = n
        self.head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, head_hidden_dim), nn.Linear(head_hidden_dim, 1)
        )

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

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

        # current graph pass
        h = self.gnn(node_features, full_edge_index)

        # graph embedding
        batch_mapping = torch.arange(batch_size, device=h.device).repeat_interleave(
            self.n
        )
        graph_latent = global_mean_pool(h.reshape(-1, h.shape[-1]), batch_mapping)

        value = self.head(graph_latent)

        return value, {}


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# DQN


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
    ):
        # Model.__init__(self, observation_space, action_space, device)
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        TabularMixin.__init__(self)

        # +1 since we'll add the degree as a node feature
        self.gnn = GNNBackbone(
            node_feat_dim + 1, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        # +1 since we'll add the adj information on the edge embeddings
        self.head = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim + 1, head_hidden_dim),
            nn.Linear(head_hidden_dim, 2),  # two logits ("add", "remove")
        )

        self.skip_head = nn.Linear(gnn_hidden_dim, 1)

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

        out_degrees = adj.sum(dim=-1, keepdim=True)
        node_features = torch.cat([node_features, out_degrees], dim=-1)

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
        # TODO: masking with a big (negative) number would work with softmax, but these are q values.
        with torch.no_grad():
            dynamic_min = q_values.min() - 1.0
        q_values[:, :E][~add_mask] = dynamic_min
        q_values[:, E:2*E][~remove_mask] = dynamic_min

        return q_values, {}

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
        self.gnn = GNNBackbone(
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

# compatible with observation type "DictNodeFeaturesAndAdjAndSelection"
class DQN_QNetwork_SelectNodesSequentially(TabularMixin, Model):
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
        self.gnn = GNNBackbone(
            node_feat_dim + 1, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        # input all node features + selected
        self.head = nn.Sequential(
            nn.Linear(n * gnn_hidden_dim + n, head_hidden_dim),
            nn.Linear(head_hidden_dim, n + 1), # +1 skip
        )

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]
        adj = observations["adj"]
        selection = observations["selection"]

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

        out_degrees = adj.sum(dim=-1, keepdim=True)
        node_features = torch.cat([node_features, out_degrees], dim=-1)

        # fully connected pass
        h = self.gnn(node_features, full_edge_index)

        # calculate node scores for selection
        q_values = self.head(torch.cat([h.flatten(-2), selection], dim=-1)).squeeze(-1).reshape(batch_size, -1)

        # mask out self loops
        # print(f"q_values before: {q_values}")
        selected_mask = selection.bool().squeeze(-1)
        has_selected = selection.sum(dim=1) > 0
        has_selected = has_selected.unsqueeze(1).expand(-1, selected_mask.size(1))
        mask = selected_mask & has_selected   # (B, N)
        q_values[:, :-1] = q_values[:, :-1].masked_fill(mask, -1e9) # exclude the skip action
        # print(f"selected_mask: {selected_mask}")
        # print(f"has_selected: {has_selected}")
        # print(f"mask: {mask}")
        # print(f"q_values after: {q_values}")

        return q_values, {}

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# DDQN

# compatible with observation type "DictNodeFeaturesAndAdjAndSelection"
class DDQN_QNetwork_SelectNodesSequentially(TabularMixin, Model):
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
        self.gnn = GNNBackbone(
            node_feat_dim + 1, gnn_hidden_dim
        )  # output dim = hidden dim
        self.n = n

        # input all node features + selected
        self.head = nn.Sequential(
            nn.Linear(n * gnn_hidden_dim + n, head_hidden_dim),
            nn.Linear(head_hidden_dim, n + 1), # +1 skip
        )

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])

        node_features = observations["node_features"]
        adj = observations["adj"]
        selection = observations["selection"]

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

        out_degrees = adj.sum(dim=-1, keepdim=True)
        node_features = torch.cat([node_features, out_degrees], dim=-1)

        # fully connected pass
        h = self.gnn(node_features, full_edge_index)

        # calculate node scores for selection
        q_values = self.head(torch.cat([h.flatten(-2), selection], dim=-1)).squeeze(-1).reshape(batch_size, -1)

        # mask out self loops
        # print(f"q_values before: {q_values}")
        selected_mask = selection.bool().squeeze(-1)
        has_selected = selection.sum(dim=1) > 0
        has_selected = has_selected.unsqueeze(1).expand(-1, selected_mask.size(1))
        mask = selected_mask & has_selected   # (B, N)
        q_values[:, :-1] = q_values[:, :-1].masked_fill(mask, -1e9) # exclude the skip action
        # print(f"selected_mask: {selected_mask}")
        # print(f"has_selected: {has_selected}")
        # print(f"mask: {mask}")
        # print(f"q_values after: {q_values}")

        return q_values, {}
