import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import time
import signal
import json
from datetime import datetime
import os
import sys
from network import Network
from rigidity import *
from util import sample_gaussian
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from torch.utils.tensorboard import SummaryWriter
import torch

from visualizer import Visualizer
from scenario import load_scenario, random_scenario, randomize_scenario
from control import GradientBasedController


def random_edge_list(n, edge_count):
    """edge_count distinct directed edges, no self loops."""
    edge_set = set()
    while len(edge_set) < edge_count:
        i, j = np.random.choice(n, size=2, replace=False)
        edge_set.add((int(i), int(j)))
    return list(edge_set)


def define_action_space(type: str, env: "Environment"):
    network = env.network
    n = len(network.agents)

    action_space = None
    if type == "AllEdges":
        action_space = spaces.MultiBinary(n * n)
    elif type == "AddRemoveEdgeMultiDiscrete":
        # add/remove/skip, i_index, j_index
        action_space = spaces.MultiDiscrete([3, n, n])
    elif type == "AddRemoveEdgeDiscrete":
        # edge enumeration
        ec = n**2
        # [0, ec-1]: add, [ec, 2*ec-1]: remove, last: skip
        action_space = spaces.Discrete(2*ec + 1)
    elif type == "AddEdgeDiscrete":
        # edge enumeration
        ec = n**2
        # [0, ec-1]: add, last: skip
        action_space = spaces.Discrete(ec + 1)
    elif type == "AddEdgeDiscreteNoSkip":
        # edge enumeration
        ec = n**2
        action_space = spaces.Discrete(ec)
    elif type == "AddEdgeDiscreteNoSelfLoops":
        # edge enumeration
        ec = n**2
        action_space = spaces.Discrete(ec - n + 1)
    elif type == "AddEdgeDiscreteNoSkipNoSelfLoops":
        # edge enumeration
        ec = n**2
        action_space = spaces.Discrete(ec - n)
    elif type == "AddRemoveEdgeDiscreteNoSelfLoops":
        # edge enumeration
        ec = n**2
        action_space = spaces.Discrete(2*ec - 2*n + 1)
    elif type == "SelectNodesSequentially":
        action_space = spaces.Discrete(n+1)
    elif type == "DecideOnEdge":
        action_space = spaces.Discrete(3)

    return action_space

def action_AllEdges(action, env: "Environment", reward, action_info):
    action_info += f"(action={action}) "

    n = len(env.network.agents)
    action_adj = action.reshape((n, n))
    np.fill_diagonal(action_adj, 0) # ignore self-loops
    env.network.set_edges(action_adj)

    for i in range(n):
        for j in range(n):
            if action_adj[i, j]:
                action_info += f"{i}->{j}, "

    nr_edges = action.sum()
    reward -= nr_edges # measurement effort
    return reward, action_info

def action_AddRemoveEdgeMultiDiscrete(action, env: "Environment", reward, action_info):
    action_info += f"(action={action}) "

    # add
    if action[0] == 0:
        i_idx = action[1]
        j_idx = action[2]

        action_info += f"add {i_idx}-{j_idx}"
        if i_idx == j_idx:
            action_info += " (self loop)"

        if env.network.edge_exists(i_idx, j_idx):
            # reward -= 20 # unnecessary action
            action_info += " (existed)"

        if i_idx != j_idx:
            env.network.add_edge(i_idx, j_idx)
            # reward -= 1 # measurement effort
    # remove
    elif action[0] == 1:
        i_idx = action[1]
        j_idx = action[2]

        action_info += f"remove {i_idx}-{j_idx}"
        if i_idx == j_idx:
            action_info += " (self loop)"

        if not env.network.edge_exists(i_idx, j_idx):
            # reward -= 20 # unnecessary action
            action_info += " (didn't exist)"

        if i_idx != j_idx:
            env.network.remove_edge(i_idx, j_idx)
            # reward += 10 # measurement effort
    # skip
    elif action[0] == 2:
        action_info += "skip"
        if env.skip_is_stop:
            env.stop_action = True
        pass

    print(action_info)

    return reward, action_info

def action_AddRemoveEdgeDiscrete(action, env: "Environment", reward, action_info):
    action_info += f"(action={action}) "

    n = len(env.network.agents)
    ec = n**2
    if action == 2 * ec:
        # skip
        action_info += "skip"
        if env.skip_is_stop:
            env.stop_action = True
        pass
    elif action < ec:
        # add
        i_idx = action // n
        j_idx = action % n

        action_info += f"add {i_idx}-{j_idx}"
        if i_idx == j_idx:
            action_info += " (self loop)"

        if env.network.edge_exists(i_idx, j_idx):
            # reward -= 20 # unnecessary action
            action_info += " (existed)"

        if i_idx != j_idx:
            env.network.add_edge(i_idx, j_idx)
            reward -= 1 # measurement effort
    else:
        # remove
        i_idx = (action-ec) // n
        j_idx = (action-ec) % n

        action_info += f"remove {i_idx}-{j_idx}"
        if i_idx == j_idx:
            action_info += " (self loop)"

        if not env.network.edge_exists(i_idx, j_idx):
            # reward -= 20 # unnecessary action
            action_info += " (didn't exist)"

        if i_idx != j_idx:
            env.network.remove_edge(i_idx, j_idx)
            reward += 10 # measurement effort

    return reward, action_info

def action_AddEdgeDiscrete(action, env: "Environment", reward, action_info):
    action_info += f"(action={action}) "

    n = len(env.network.agents)
    ec = n**2
    if action == ec:
        # skip
        action_info += "skip"
        if env.skip_is_stop:
            env.stop_action = True
        pass
    else:
        # add
        i_idx = action // n
        j_idx = action % n

        action_info += f"add {i_idx}-{j_idx}"
        if i_idx == j_idx:
            action_info += " (self loop)"

        if env.network.edge_exists(i_idx, j_idx):
            # reward -= 20 # unnecessary action
            action_info += " (existed)"

        if i_idx != j_idx:
            env.network.add_edge(i_idx, j_idx)
            reward -= 1 # measurement effort

    return reward, action_info

def action_AddEdgeDiscreteNoSkip(action, env: "Environment", reward, action_info):
    action_info += f"(action={action}) "

    n = len(env.network.agents)

    # add
    i_idx = action // n
    j_idx = action % n

    action_info += f"add {i_idx}-{j_idx}"
    if i_idx == j_idx:
        action_info += " (self loop)"

    if env.network.edge_exists(i_idx, j_idx):
        # reward -= 20 # unnecessary action
        action_info += " (existed)"

    if i_idx != j_idx:
        env.network.add_edge(i_idx, j_idx)
        reward -= 1 # measurement effort

    return reward, action_info

def action_AddEdgeDiscreteNoSelfLoops(
    action, env: "Environment", reward, action_info
):
    action_info += f"(action={action}) "

    n = len(env.network.agents)

    # skip
    if action == n**2 - n:
        action_info += "skip"
        if env.skip_is_stop:
            env.stop_action = True
        pass
    # add
    else:
        i_idx = action // (n - 1)
        j_idx = action % (n - 1)
        if j_idx >= i_idx:
            j_idx += 1

        action_info += f"add {i_idx}-{j_idx}"
        if i_idx == j_idx:
            action_info += " (self loop)"

        if env.network.edge_exists(i_idx, j_idx):
            # reward -= 20 # unnecessary action
            action_info += " (existed)"

        env.network.add_edge(i_idx, j_idx)
        reward -= 1  # measurement effort

    print(action_info)

    return reward, action_info

def action_AddEdgeDiscreteNoSkipNoSelfLoops(
    action, env: "Environment", reward, action_info
):
    action_info += f"(action={action}) "

    n = len(env.network.agents)

    i_idx = action // (n - 1)
    j_idx = action % (n - 1)
    if j_idx >= i_idx:
        j_idx += 1

    action_info += f"add {i_idx}-{j_idx}"
    if i_idx == j_idx:
        action_info += " (self loop)"
        raise Exception("Shouldn't happen: " + action_info)

    if env.network.edge_exists(i_idx, j_idx):
        reward -= 2 # unnecessary action
        action_info += " (existed)"

    env.network.add_edge(i_idx, j_idx)
    # reward -= 1  # measurement effort

    return reward, action_info

def action_AddRemoveEdgeDiscreteNoSelfLoops(
        action, env: "Environment", reward, action_info
):
    action_info += f"(action={action}) "

    n = len(env.network.agents)
    ec = n**2
    action_space_len = 2*ec - 2*n + 1

    if action == action_space_len - 1:
        # skip
        action_info += "skip"
        if env.skip_is_stop:
            env.stop_action = True
        pass
    elif action < (action_space_len - 1) // 2:
        # add
        i_idx = action // (n - 1)
        j_idx = action % (n - 1)
        if j_idx >= i_idx:
            j_idx += 1

        action_info += f"add {i_idx}-{j_idx}"

        # shouldn't happen
        if i_idx == j_idx:
            action_info += " (self loop)"
            raise Exception("Shouldn't happen: " + action_info)

        if env.network.edge_exists(i_idx, j_idx):
            reward -= 2 # unnecessary action
            action_info += " (existed)"
        else:
            env.network.add_edge(i_idx, j_idx)
    else:
        # remove
        i_idx = (action - ((action_space_len - 1) // 2)) // (n - 1)
        j_idx = (action - ((action_space_len - 1) // 2)) % (n - 1)
        if j_idx >= i_idx:
            j_idx += 1

        action_info += f"remove {i_idx}-{j_idx}"

        # shouldn't happen
        if i_idx == j_idx:
            action_info += " (self loop)"
            raise Exception("Shouldn't happen: " + action_info)

        if not env.network.edge_exists(i_idx, j_idx):
            reward -= 2 # unnecessary action
            action_info += " (didn't exist)"
        else:
            env.network.remove_edge(i_idx, j_idx)

    return reward, action_info


def action_SelectNodesSequentially(action, env: "Environment", reward, action_info):
    action_info += f"(action={action}) "

    n = len(env.network.agents)

    if action == n:
        action_info += " skip"
        if env.skip_is_stop:
            env.stop_action = True
        env.selection = np.zeros(env.n, dtype=np.int64)
        return reward, action_info

    # select first node
    if not np.sum(env.selection):
        env.selection[action] = 1
        action_info += f"select node {action}"
    # select second node
    else:
        i = np.argwhere(env.selection).squeeze(-1).squeeze(-1)
        j = action

        if env.network.edge_exists(i, j):
            didnt_exist = not env.network.edge_exists(i, j)
            env.network.remove_edge(i, j)
            action_info += f"remove edge {i} -> {j}"
            if didnt_exist:
                action_info += " (didn't exist)"
        else:
            existed = env.network.edge_exists(i, j)
            env.network.add_edge(i, j)
            action_info += f"add edge {i} -> {j}"
            if existed:
                action_info += " (existed)"

        # reset
        env.selection = np.zeros(env.network.n, dtype=np.int64)

    return reward, action_info

def action_DecideOnEdge(action, env: "Environment", reward, action_info):
    action_info += f"(action={action}) proposal: {env.proposed_edge} "

    if action == 0:
        action_info += f"add "
        env.network.add_edge(env.proposed_edge[0], env.proposed_edge[1])
    elif action == 1:
        action_info += f"remove "
        env.network.remove_edge(int(env.proposed_edge[0]), int(env.proposed_edge[1]))
    elif action == 2:
        action_info += f"skip "
        if env.skip_is_stop:
            env.stop_action = True
    else:
        print(f"shouldn't happen: action_DecideOnEdge")
        quit()

    return reward, action_info


# obs types producing the EGNN feature dict (node/coord/edge/adj/selection);
# one set of models serves all of them
# The one graph observation, and the flag presets that reproduce each pre-merge
# Dict* layout exactly so old configs and checkpoints still load.
def build_dict_obs(env, define_type, node_set="graph", coords=True, edges=True,
                   selection=True, proposed_edge=None, candidate_bearings=None,
                   edge_exists=True, normalize_positions=True,
                   normalize_counts=True):
    network = env.network
    n = network.n

    if node_set == "graph":
        parts_n = [network.get_domain_features(),
                   network.get_degree_features_normalized(getattr(env, "m_req", None))
                   if normalize_counts
                   else network.get_degree_features()]
        if getattr(env, "graph_features", True):
            parts_n += [network.get_closeness_centrality_features(),
                        network.get_eigenvector_centrality_features(),
                        network.get_node_betweenness_features()]
        node_features = np.concat(parts_n, axis=-1)
    elif node_set == "domain_signbearing":
        node_features = np.concat([network.get_domain_features(),
                                   network.get_simplified_bearing_features().reshape(n, -1)], -1)
    elif node_set == "domain_bearing":
        node_features = np.concat([network.get_domain_features(),
                                   network.get_bearing_features().reshape(n, -1)], -1)
    elif node_set == "bearing":
        node_features = network.get_bearing_features().reshape((n, -1))
    else:
        raise ValueError(f"unknown node_set {node_set!r}")

    # tier-3 rigidity channels, when the ablation flags ask for them.
    rig = getattr(env, "last_rigidity", None)
    if rig:
        extra = []
        if env.rigidity_global:
            extra.append(np.tile(
                [rig["rank_deficit"], rig["m_ratio"], rig["is_IBR"]], (n, 1)))
        if getattr(env, "rigidity_quality", False):
            extra.append(np.tile([rig["quality"]], (n, 1)))
        if env.rigidity_flex:
            extra.append(rig["node_freedom"])
        if getattr(env, "rigidity_stiffness", False):
            extra.append(rig["node_slack"])
        if extra:
            node_features = np.concat([node_features] + extra, axis=-1)

    obs = {"node_features": node_features}
    spec = {"node_features": spaces.Box(-np.inf, np.inf, node_features.shape)}

    if coords:
        c = (network.get_normalized_position_features() if normalize_positions
             else network.get_position_features())
        obs["coord_features"] = c
        spec["coord_features"] = spaces.Box(-np.inf, np.inf, c.shape)

    if edges:
        want_candidates = (env.include_candidate_bearings if candidate_bearings is None
                           else candidate_bearings)
        parts = [network.get_all_pairs_bearings() if want_candidates
                 else network.get_bearing_features()]
        if edge_exists:
            parts.append(network.get_edge_exists_features())
        if getattr(env, "graph_features", True):
            parts.append(network.get_edge_betweenness_features())
        parts += [network.get_edge_reciprocity_features(),
                  network.get_common_neighbors_features_normalized(getattr(env, "m_req", None))
                  if normalize_counts
                  else network.get_common_neighbors_features()]
        if rig:
            if env.rigidity_flex:
                parts.append(rig["add_independence"])
            if env.rigidity_edge:
                parts += [rig["pair_max_rank"], rig["add_rank"]]
            if getattr(env, "rigidity_stiffness", False):
                parts.append(rig["add_stiffness"])
            if getattr(env, "rigidity_removal", False):
                parts += [rig["remove_rank"], rig["remove_stiffness"]]
        e = np.concat(parts, axis=-1)
        obs["edge_features"] = e
        spec["edge_features"] = spaces.Box(-np.inf, np.inf, e.shape)

    adj = network.edges.astype(np.float32)
    obs["adj"] = adj
    spec["adj"] = spaces.Box(0, 1, adj.shape)

    if selection:
        obs["selection"] = env.selection
        spec["selection"] = spaces.Box(0, 1, env.selection.shape, dtype=int)

    if proposed_edge is None:
        proposed_edge = env.action_space_type == "DecideOnEdge"
    if proposed_edge:
        # j != i: a self-loop proposal masks add AND remove, leaving no valid
        # action at all (the all-masked guard then has to unmask everything)
        i = np.random.randint(0, n)
        j = np.random.randint(0, n - 1)
        env.edge_proposal = np.array([i, j + 1 if j >= i else j])
        obs["proposed_edge"] = env.edge_proposal
        spec["proposed_edge"] = spaces.Box(0, n, [2])

    return obs, (spaces.Dict(spec) if define_type else None)


# "Dict" is current. The rest are pre-merge names kept working; each reproduces
# its old layout, including raw coordinates and edges-only bearings, because the
# checkpoints trained on them depend on both.
# width of q's sigmoid, in decades of lambda / stiffness_ref.
STIFFNESS_SIGMOID_DECADES = 0.75


OBS_PRESETS = {
    "Dict": dict(),
    "DictEquivariantNodeFeaturesAndAdjAndSelection": dict(
        edge_exists=False, normalize_positions=False, candidate_bearings=False, normalize_counts=False),
    "DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection": dict(
        coords=False, edge_exists=False, candidate_bearings=False, normalize_counts=False),
    "DictNodeFeaturesAndAdj": dict(
        node_set="domain_signbearing", coords=False, edges=False, selection=False),
    "DictNodeFeaturesAndAdjAndSelection": dict(
        node_set="domain_signbearing", coords=False, edges=False),
    "DictNodeFeaturesAndAdjAndEdgeProposal": dict(
        node_set="domain_bearing", coords=False, edges=False, selection=False,
        proposed_edge=True),
    "DictBearingNodeFeaturesAndAdjAndSelection": dict(
        node_set="bearing", coords=False, edges=False),
}

# a legacy obs_type implied which GNN consumed it; "Dict" does not, so the
# training scripts fall back to their BACKBONE constant
OBS_BACKBONE = {
    "DictEquivariantNodeFeaturesAndAdjAndSelection": "Equivariant",
    "DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection": "GINE",
    "DictNodeFeaturesAndAdj": "Default",
    "DictNodeFeaturesAndAdjAndSelection": "Default",
    "DictNodeFeaturesAndAdjAndEdgeProposal": "Default",
    "DictBearingNodeFeaturesAndAdjAndSelection": "Default",
}


def obs(type: str, env: "Environment", define_type=False):
    obs_space = None

    network = env.network
    n = network.n

    obs = None
    if type == "Complete":
        A = env.network.edges.astype(np.float32)
        positions = np.array(
            [agent.pose.position for agent in network.agents]
        ).flatten()
        orientations_euler = np.array(
            [agent.pose.euler_angles() for agent in network.agents]
        ).flatten()
        obs = np.hstack([positions, orientations_euler, A.flatten()])
        if define_type:
            obs_n = obs.shape[0]
            obs_space = spaces.Box(-np.inf, np.inf, (obs_n,))
    elif type == "CompleteAndEigenvalues":
        A = env.network.edges.astype(np.float32)
        # can't just take the nonzero ones since dimension might change
        eigenvalues = env.network.eigenvalues()
        positions = np.array(
            [agent.pose.position for agent in network.agents]
        ).flatten()
        orientations_euler = np.array(
            [agent.pose.euler_angles() for agent in network.agents]
        ).flatten()
        obs = np.hstack([positions, orientations_euler, eigenvalues, A.flatten()])
        if define_type:
            obs_n = obs.shape[0]
            obs_space = spaces.Box(-np.inf, np.inf, (obs_n,))
    elif type == "AdjFlatAndEigenvalues":
        A = env.network.edges.astype(np.float32)
        eigenvalues = env.network.eigenvalues()
        obs = np.hstack([A.flatten(), eigenvalues])
        if define_type:
            obs_n = obs.shape[0]
            obs_space = spaces.Box(-np.inf, np.inf, (obs_n,))
    elif type in OBS_PRESETS:
        obs, obs_space = build_dict_obs(env, define_type, **OBS_PRESETS[type])
    else:
        raise ValueError(
            f"unknown obs_type {type!r}; known: {sorted(OBS_PRESETS)}"
        )

    return obs, obs_space


class Environment(gym.Env):
    def __init__(self):
        super().__init__()
        print("environment constructed (not initialized)")

    def initialize(
        self,
        n,
        domains,
        action_space_type="AllEdges",
        obs_space_type="Complete",
        state_score_type="Rigid",
        termination_condition_type="MaxSteps",
        action_rewards_enable=False,
        skip_is_stop=True,
        random_graph_with_mean_min_edges=False,
        time_penalty_value=0.0,
        track_data_enable=False,
        max_steps=1e4,
        truncate_enable=True,
        truncate_max_steps=1e4,
        truncate_penalty_value=100,
        only_randomize_edges=False,
        include_candidate_bearings=True,
        graph_features=True,
        rigidity_global=False,
        rigidity_quality=False,
        rigidity_flex=False,
        rigidity_edge=False,
        rigidity_stiffness=False,
        rigidity_removal=False,
        rotation_augmentation=False,
        stiffness_kappa=0.0,
        stiffness_ref_samples=3,
        spectral_functional="eigenvalue",
        filepath=None,
    ):
        print("initializing environment")

        self.action_space_type = action_space_type
        self.obs_space_type = obs_space_type
        self.state_score_type = state_score_type
        self.termination_condition_type = termination_condition_type

        self.random_graph_with_mean_min_edges = random_graph_with_mean_min_edges

        # tier-2 information: an agent cannot know a bearing it has not measured.
        # False reverts to bearings on existing edges only, same obs shape.
        self.include_candidate_bearings = include_candidate_bearings

        # tier-3 rigidity information: an ablation arm, off by default.
               # closeness / eigenvector / betweenness: measured to carry less
        # rigidity-relevant signal than out-degree (which is free) while costing
        # ~60-70% of feature building.
        # bearings in R^d are global-frame, so the observation is not rotation
        # invariant even though the task is.
        self.rotation_augmentation = rotation_augmentation

        self.graph_features = graph_features
        self.rigidity_global = rigidity_global
        self.rigidity_quality = rigidity_quality
        self.rigidity_flex = rigidity_flex
        self.rigidity_edge = rigidity_edge
        self.rigidity_stiffness = rigidity_stiffness
        self.rigidity_removal = rigidity_removal
        self.last_rigidity = None

        # how many edges the whole stiffness range is worth; 0 disables it.
        self.stiffness_kappa = float(stiffness_kappa)
        self.stiffness_ref_samples = int(stiffness_ref_samples)
        if spectral_functional not in SPECTRAL_FUNCTIONALS:
            raise ValueError(f"unknown spectral_functional {spectral_functional!r}")
        self.spectral_functional = spectral_functional
        self.spectral_ref = None
        self.stiffness_ref = 0.0
        # private: the global stream is the one instances are drawn from. Reseeded
        # per episode, so stiffness_ref is a function of the poses and phi is a
        # function of the state.
        self.stiffness_seed = 0
        self.stiffness_rng = np.random.default_rng(self.stiffness_seed)

        self.filepath = filepath
        self.scenario_network = None
        if self.filepath is not None:
            self.network, self.goal_network = load_scenario(self.filepath)
            # cached so reset() does not re-read and re-parse the file every episode
            self.scenario_network = copy.deepcopy(self.network)
        else:
            if self.random_graph_with_mean_min_edges:
                self.network, self.goal_network = random_scenario(
                    n, domains, edge_count=self.sample_initial_edge_count(n, domains)
                )
            else:
                self.network, self.goal_network = random_scenario(n, domains)

        self.n = len(self.network.agents)
        # resolved per-agent domains; for a scenario these come from the file, and they
        # must survive every reset or a heterogeneous network silently homogenizes
        self.domains = [agent.domain for agent in self.network.agents]
        # carried for the same reason as domains: reset() rebuilds the network
        self.rotation_axes = [agent.rotation_axis for agent in self.network.agents]
        self.m = int(self.network.edges.sum())
        self.initial_m = self.m

        self.brm = self.network.extended_bearing_rigidity_matrix()

        self.compute_episode_constants()

        self.selection = np.zeros(self.n, dtype=np.int64)
        self.proposed_edge = np.zeros(2)

        # must run before the space is defined, or the declared space is missing
        # the rigidity channels that reset() then produces
        rank_i, _, lam_i = rigidity_decomposition(self.brm, self.rank_K)
        is_MBR_i, is_IBR_i, _ = self.network.is_MBR(
            rank_K=self.rank_K, brm=self.brm, rank_brm=rank_i)
        self.compute_rigidity_features(self.brm, rank_i, is_IBR_i, lam=lam_i)

        _, self.observation_space = obs(obs_space_type, self, define_type=True)
        self.action_space = define_action_space(action_space_type, self)
        self._get_obs = lambda: obs(obs_space_type, self, False)[0]

        self.nr_max_edges = self.n**2
        self.step_counter = 0
        self.max_steps = max_steps

        self.truncate_enable = truncate_enable
        self.truncate_max_steps = truncate_max_steps
        self.truncate_penalty_value = truncate_penalty_value

        self.only_randomize_edges = only_randomize_edges

        self.last_state_score = 0

        self.action_rewards_enable = action_rewards_enable
        self.track_data_enable = track_data_enable

        self.skip_is_stop = skip_is_stop

        self.time_penalty_value = time_penalty_value

        self.was_IBR = None
        self.was_MBR = None

        # when True, reset() keeps the current graph instead of drawing a new one
        self.freeze_network = False

        # compute the rigidity eigenvalue every step even without a writer, so a caller
        # can record it; costs one extra eigendecomposition per step
        self.trace_min_eig = False
        self.last_stats = None

        # metrics are written once per episode
        self.episode_counter = 0
        self.episode_accum = self.new_episode_accum()
        self.last_episode_stats = None
        # one scalar per episode says nothing about the sampler's spread, so the
        # initial edge counts are buffered and emitted as a histogram
        self.initial_edge_history = []
        self.initial_edge_hist_every = 25

        self.writer = None
        # self.initial_edges_writer = None

    def set_writer(self, experiment_name):
        # torch's writer rather than skrl's shim: the shim has add_scalar only, and
        # the decision-quality panel needs add_custom_scalars (one chart, four
        # series) and add_histogram. skrl's agent keeps its own writer in the same
        # directory; tensorboard merges them, as it already does for Loss/ vs Episode/.
        self.writer = SummaryWriter(log_dir=os.path.join("runs", experiment_name))
        # the single plot to watch: all four converge in a policy that is learning.
        self.writer.add_custom_scalars({
            "Decision": {"quality": ["Multiline", [
                "Decision/ useful", "Decision/ wasted",
                "Decision/ overshoot", "Decision/ converge",
            ]]},
            "Probe": {"argmax vs sample": ["Multiline", [
                "Probe/ argmax score", "Probe/ sample score",
            ]], "useful vs chance": ["Multiline", [
                "Probe/ useful (argmax)", "Probe/ useful (random)",
            ]]},
        })
        # self.initial_edges_writer = SummaryWriter(log_dir=os.path.join("runs", experiment_name))
        self.writer_counter = 0 # don't reset this
        self.episode_counter = 0

    def load(self, filepath):
        with open(filepath, "r") as f:
            config = json.load(f)

        n = config["n"]
        domains = config["domains"]
        ACTION_TYPE = config["action_type"]
        OBS_TYPE = config["obs_type"]
        # older configs named this "reward_type"
        STATE_SCORE_TYPE = config.get("state_score_type", config.get("reward_type"))
        TERMINATION_CONDITION_TYPE = config["termination_condition_type"]
        ACTION_REWARDS_ENABLE = config["action_rewards_enable"]
        # keys added after the older configs were generated
        SKIP_IS_STOP = config.get("skip_is_stop", False)
        RANDOM_GRAPH_WITH_MEAN_MIN_EDGES = config.get("random_graph_with_mean_min_edges", False)
        TIME_PENALTY_VALUE = config["time_penalty_value"]
        TRACK_DATA_ENABLE = config["track_data_enable"]
        MAX_STEPS = config["max_steps"]
        TRUNCATE_ENABLE = config["truncate_enable"]
        TRUNCATE_MAX_STEPS = config["truncate_max_steps"]
        TRUNCATE_PENALTY_VALUE = config["truncate_penalty_value"]
        ONLY_RANDOMIZE_EDGES = config["only_randomize_edges"]
        INCLUDE_CANDIDATE_BEARINGS = config.get("include_candidate_bearings", True)
        GRAPH_FEATURES = config.get("graph_features", True)
        RIGIDITY_GLOBAL = config.get("rigidity_global", False)
        RIGIDITY_QUALITY = config.get("rigidity_quality", False)
        RIGIDITY_FLEX = config.get("rigidity_flex", False)
        RIGIDITY_EDGE = config.get("rigidity_edge", False)
        for old_key, new_key in (("margin_kappa", "stiffness_kappa"),
                                 ("margin_ref_samples", "stiffness_ref_samples"),
                                 ("rigidity_margin", "rigidity_stiffness")):
            if old_key in config:
                raise KeyError(
                    f"{filepath}: '{old_key}' is now '{new_key}'. Regenerate the config "
                    f"with environment.py rather than renaming the key by hand.")
        RIGIDITY_STIFFNESS = config.get("rigidity_stiffness", False)
        RIGIDITY_REMOVAL = config.get("rigidity_removal", False)
        ROTATION_AUGMENTATION = config.get("rotation_augmentation", False)
        STIFFNESS_KAPPA = config.get("stiffness_kappa", 0.0)
        STIFFNESS_REF_SAMPLES = config.get("stiffness_ref_samples", 3)
        SPECTRAL_FUNCTIONAL = config.get("spectral_functional", "eigenvalue")
        scenario_name = config["scenario"]
        scenario_path = (
            "scenarios/" + scenario_name + ".json"
            if scenario_name is not None
            else None
        )

        self.initialize(
            n,
            domains,
            action_space_type=ACTION_TYPE,
            obs_space_type=OBS_TYPE,
            state_score_type=STATE_SCORE_TYPE,
            termination_condition_type=TERMINATION_CONDITION_TYPE,
            action_rewards_enable=ACTION_REWARDS_ENABLE,
            skip_is_stop=SKIP_IS_STOP,
            random_graph_with_mean_min_edges=RANDOM_GRAPH_WITH_MEAN_MIN_EDGES,
            time_penalty_value=TIME_PENALTY_VALUE,
            track_data_enable=TRACK_DATA_ENABLE,
            max_steps=MAX_STEPS,
            truncate_enable=TRUNCATE_ENABLE,
            truncate_max_steps=TRUNCATE_MAX_STEPS,
            truncate_penalty_value=TRUNCATE_PENALTY_VALUE,
            only_randomize_edges=ONLY_RANDOMIZE_EDGES,
            include_candidate_bearings=INCLUDE_CANDIDATE_BEARINGS,
            graph_features=GRAPH_FEATURES,
            rigidity_global=RIGIDITY_GLOBAL,
            rigidity_quality=RIGIDITY_QUALITY,
            rigidity_flex=RIGIDITY_FLEX,
            rigidity_edge=RIGIDITY_EDGE,
            rigidity_stiffness=RIGIDITY_STIFFNESS,
            rigidity_removal=RIGIDITY_REMOVAL,
            rotation_augmentation=ROTATION_AUGMENTATION,
            stiffness_kappa=STIFFNESS_KAPPA,
            stiffness_ref_samples=STIFFNESS_REF_SAMPLES,
            spectral_functional=SPECTRAL_FUNCTIONAL,
            filepath=scenario_path,
        )

    # -----------------------------------
    # Sample around the minimum requirement, not uniformly. Mean is only exact
    # for homogeneous R^d.
    def sample_initial_edge_count(self, n, domains):
        if isinstance(domains, str):
            domains = [domains]
        # required_edge_count is domain-correct; the R^d closed form is not. It said
        # 10 for SE(3) at n=8, which needs 21, so every SE(3) episode started
        # heavily under-connected (measured: 13.1 edges, 15% of them rigid).
        # m_req depends on n and the domain mix, not on the particular poses, so the
        # value cached from the previous episode is the right mean here. Only the
        # very first call (from initialize(), before a network exists) falls back.
        mean = getattr(self, "m_req", None)
        if mean is None:
            d = 2 if (("R^2" in domains) or ("R^2xS^1" in domains)) else 3
            mean = MBR_required_Rd(n, d)
        max_edges = n**2 - n
        mean = int(np.clip(mean, 1, max_edges))
        # spread proportional to the requirement, not to the gap up to the complete
        # graph. The old (max_edges - mean)^2/9 grows like n^4, so at n=16 the sd was
        # 72.7 against a mean of 22 and episodes actually started around 42 edges --
        # n=16 was a far harder pruning problem than n=8, not merely a bigger one,
        # which confounded every transfer comparison.
        sd = max(0.5 * mean, 1.0)
        edge_count = int(sample_gaussian(mean, sd**2, n).item())
        return int(np.clip(edge_count, 1, max_edges))

    # -----------------------------------
    # Pose-dependent, edge-independent; once per episode. rank_K and c_max are
    # exact, m_req is only a lower bound and must stay out of the reward.
    def compute_episode_constants(self):
        network_K = self.network.fully_connected()
        brmat_K = network_K.extended_bearing_rigidity_matrix()
        self.rank_K = np.linalg.matrix_rank(brmat_K)
        # position block only; equals rank_K unless a domain contributes orientation
        self.rank_K_pos = np.linalg.matrix_rank(brmat_K[:, :3 * self.network.n])
        self.c_max = max_edge_rank(self.network, brmat_K=brmat_K)
        # one pass of per-edge block ranks serves m_req, c_max and the block_rank
        # channel; they used to be recomputed independently
        blocks_K = (edge_block_ranks(brmat_K)
                    if self.rigidity_edge or len({a.domain for a in self.network.agents}) > 1
                    else None)
        self.m_req = required_edge_count(
            self.network, rank_K=self.rank_K, brmat_K=brmat_K, block_ranks=blocks_K
        )
        # edge-independent, so it stays an episode constant and the shaping
        # stays potential-based.
        self.stiffness_rng = np.random.default_rng(self.stiffness_seed)
        self.stiffness_ref = (reference_stiffness(self.network, self.rank_K,
                                                  self.stiffness_rng,
                                                  samples=self.stiffness_ref_samples)
                              if self.stiffness_kappa > 0 else 0.0)
        self.spectral_ref = None
        if (self.rigidity_quality
                or (self.stiffness_kappa > 0
                    and self.state_score_type == "WeightedNormalizedSpectral")):
            self.spectral_ref = reference_spectral(
                self.network, self.rank_K, np.random.default_rng(self.stiffness_seed),
                samples=self.stiffness_ref_samples,
                functional=self.spectral_functional)

        # pose-dependent, edge-independent, and needed every step by the rigidity
        # features: the trivial variation space and every pair's own block rank
        if self.rigidity_features_enabled():
            self.length_scale = characteristic_length(self.network)
            ZK = nullspace(brmat_K, int(self.rank_K))
            self.Z_K = nullspace_in_scaled_units(ZK, self.network.n, self.length_scale)
            self.block_rank_K = np.zeros((self.network.n, self.network.n))
            ii, jj = np.nonzero(network_K.edges)
            cm = max(int(self.c_max), 1)
            for k, r in enumerate(blocks_K if blocks_K is not None
                                  else edge_block_ranks(brmat_K)):
                self.block_rank_K[ii[k], jj[k]] = r / cm

    # -----------------------------------
    # Rigidity-derived observation features for the current graph, cached so obs()
    # does not recompute them. Only filled when some rigidity flag is on -- these
    # are tier-3 information, an ablation arm, not the default.
    def compute_rigidity_features(self, brm, rank_brm, is_IBR, Z=None, lam=None):
        if not self.rigidity_features_enabled():
            self.last_rigidity = None
            return

        n = self.network.n
        feats = {
            "rank_deficit": (self.rank_K - rank_brm) / max(int(self.rank_K), 1),
            "m_ratio": float(np.sum(self.network.edges)) / max(int(self.m_req), 1),
            "is_IBR": float(is_IBR),
        }

        if self.rigidity_quality:
            feats["quality"] = self.state_quality(brm, is_IBR, lam)

        v, eig_w, eig_V = None, None, None
        if Z is None:
            if self.rigidity_stiffness or self.rigidity_removal:
                Z, v, eig_w, eig_V = nullspace_and_softest(brm, int(rank_brm))
            else:
                Z = nullspace(brm, int(rank_brm))

        # lengths in units of the formation's own size, so the null space does not
        # move under a uniform scaling.
        L = self.length_scale
        Zs = nullspace_in_scaled_units(Z, n, L)
        cand = candidate_gain(self.network, Zs, length_scale=L)

        if self.rigidity_flex:
            # ker(B_K) is exactly the trivial variation set (Michieletto Thm 1),
            # so nothing has to be enumerated by hand.
            F = flex_space(Zs, self.Z_K)
            mag = node_flex_magnitude(F, n)
            # normalized to unit mean square, so it says which nodes are free
            # rather than how free in absolute terms, which is what transfers
            rms = np.sqrt(max(float((mag ** 2).mean()), 1e-12))
            feats["node_freedom"] = mag / rms

            # already in [0, 1] per pair; see rigidity.candidate_gain
            feats["add_independence"] = cand[0][:, :, None]

        if self.rigidity_edge:
            # from the COMPLETE graph, so a candidate pair reads its own value
            # rather than 0, which is indistinguishable from "contributes nothing"
            feats["pair_max_rank"] = self.block_rank_K[:, :, None]
            _, rk = cand
            feats["add_rank"] = (rk / max(int(self.c_max), 1))[:, :, None]

        if self.rigidity_stiffness:
            # ||b_ij v||, v the mode at the rigidity eigenvalue: how much adding
            # i->j would stiffen the weakest direction. Unlike add_independence it
            # is nonzero on a rigid graph, the only regime where stiffness exists.
            add_stiffness = np.zeros((n, n))
            node_slack = np.zeros((n, 2))
            if is_IBR and v is not None and v.shape[1] == 1:
                vs = nullspace_in_scaled_units(v, n, L)
                add_stiffness = candidate_gain(self.network, vs, length_scale=L)[0]
                vp = vs[:3 * n].reshape(n, 3, -1)
                va = vs[3 * n:].reshape(n, 3, -1)
                node_slack = np.stack([np.sqrt((vp ** 2).sum(axis=(1, 2))),
                                       np.sqrt((va ** 2).sum(axis=(1, 2)))], axis=-1)
            # per channel, by its own mean: which pair/node is slack rather than
            # how slack in absolute terms, which is what transfers
            feats["add_stiffness"] = (add_stiffness
                                      / max(float(add_stiffness.mean()), 1e-12))[:, :, None]
            feats["node_slack"] = node_slack / np.maximum(node_slack.mean(axis=0,
                                                                          keepdims=True), 1e-12)

        if self.rigidity_removal:
            rank_lost, stiffness_lost = removal_costs(
                brm, self.network, int(self.rank_K), lam=float(lam or 0.0),
                w=eig_w, V=eig_V, c_max=self.c_max)
            feats["remove_rank"] = rank_lost[:, :, None]
            feats["remove_stiffness"] = stiffness_lost[:, :, None]

        self.last_rigidity = feats

    def rigidity_features_enabled(self):
        return (self.rigidity_global or self.rigidity_quality
                or self.rigidity_flex or self.rigidity_edge
                or self.rigidity_stiffness or self.rigidity_removal)

    # -----------------------------------
    # Sums and counts only, so episode length is free.
    def new_episode_accum(self):
        return {
            "steps": 0,
            "edits": 0,     # steps that actually changed the edge set
            "skips": 0,     # steps the action space treated as a skip / no-op
            "return": 0.0,
            "return_action": 0.0,
            "return_state": 0.0,
            "return_termination": 0.0,
            "sum_score": 0.0,
            "sum_m": 0.0,
            "sum_rank": 0.0,
            "sum_IBR": 0.0,
            "sum_MBR": 0.0,
            "sum_min_eig": 0.0,
            "n_min_eig": 0, # min eig is not always computed, so it needs its own count
            "sum_shape_err": 0.0,
            "n_shape_err": 0,   # infinite while flexible, so it needs its own count
            # decision quality: what each step actually accomplished.
            "useful": 0,        # steps where phi strictly increased
            "kinds": {"add": 0, "remove": 0, "noop": 0, "skip": 0, "select": 0},
            "actions": [],      # raw action indices, for the histogram
            "first_rigid": -1,  # step at which the graph first became IBR
            "first_minimal": -1,
        }

    # -----------------------------------
    # The whole episode as one flat, float-valued record: Final / Best / Mean.
    def episode_summary(self, state_score, rank_brm, is_IBR, is_MBR, min_eig,
                        terminated, truncated):
        acc = self.episode_accum
        steps = max(acc["steps"], 1)
        m_final = int(self.network.edges.sum())
        m_initial = int(self.initial_m)
        m_req = max(int(getattr(self, "m_req", 1) or 1), 1)
        n_eig = acc["n_min_eig"]
        n_err = acc["n_shape_err"]
        return {
            "Episode index": self.episode_counter,
            "Length": acc["steps"],
            # 1 = a termination condition fired, 0 = ran out of steps
            "Terminated": float(terminated),
            "Nr edits": acc["edits"],
            "Skip fraction": acc["skips"] / steps,

            "Return": acc["return"],
            "Return (action)": acc["return_action"],
            "Return (state)": acc["return_state"],
            "Return (termination)": acc["return_termination"],

            "Nr initial edges": m_initial,
            # 1.0 = the sampler starts exactly at the requirement. Dimensionless,
            # so it is comparable across n and domain unlike the raw count.
            "Initial edges over m_req": m_initial / m_req,
            "Final state score": float(state_score),
            "Final nr edges": m_final,
            "Final rank": int(rank_brm),
            # 0 means rigid; how far the final graph is from rank_K
            "Final rank deficit": int(self.rank_K) - int(rank_brm),
            "Final is rigid": float(is_IBR),
            "Final is min rigid": float(is_MBR),
            "Final min eig": None if min_eig is None else float(min_eig),
            "Final shape err": self.last_stats.get("shape_err") if self.last_stats else None,
            # negative = the episode removed edges, positive = it added them
            "Edge delta": m_final - m_initial,

            "Best state score": float(self.best_state_score),
            "Best nr edges": self.best_stats["m"],
            "Best rank": self.best_stats["rank"],
            "Best is rigid": float(self.best_stats["is_IBR"]),
            "Best is min rigid": float(self.best_stats["is_MBR"]),
            "Best min eig": self.best_stats["min_eig"],
            "Best shape err": self.best_stats.get("shape_err"),
            "Best step": self.best_step,
            # 0 means the episode ended on the best graph it found; positive means
            # it found something better and then moved off it -- the difference
            # between "can find a good topology" and "knows to stop on one"
            "Best-final score gap": float(self.best_state_score) - float(state_score),

            "Mean state score": acc["sum_score"] / steps,
            "Mean nr edges": acc["sum_m"] / steps,
            "Mean rank": acc["sum_rank"] / steps,
            "Rigid fraction": acc["sum_IBR"] / steps,
            "Min rigid fraction": acc["sum_MBR"] / steps,
            "Mean min eig": (acc["sum_min_eig"] / n_eig) if n_eig else None,
            "Mean shape err": (acc["sum_shape_err"] / n_err) if n_err else None,

            # Decision quality -- blind to best-state-visited, so a policy that only
            # searches cannot score well here.
            "Decision/ useful": acc["useful"] / steps,
            "Decision/ wasted": (acc["kinds"]["noop"] + acc["kinds"]["skip"]) / steps,
            "Decision/ overshoot": max(0.0, m_final - m_req) / m_req,
            "Decision/ converge": self.best_step / steps,

            # 1 = every edit moved the edge count the same way, 0 = pure oscillation
            "Edit efficiency": abs(m_final - m_initial) / max(acc["edits"], 1),
            # -1 when it never got there
            "Steps to first rigid": acc["first_rigid"],
            "Steps to first minimal": acc["first_minimal"],
            # how long the *pruning* phase took, which is where n=16 stalls
            "Steps rigid to minimal": (
                acc["first_minimal"] - acc["first_rigid"]
                if acc["first_minimal"] >= 0 and acc["first_rigid"] >= 0 else -1
            ),

            "Actions/ add fraction": acc["kinds"]["add"] / steps,
            "Actions/ remove fraction": acc["kinds"]["remove"] / steps,
            "Actions/ noop fraction": acc["kinds"]["noop"] / steps,
            "Actions/ skip fraction": acc["kinds"]["skip"] / steps,
            "Actions/ select fraction": acc["kinds"]["select"] / steps,
        }

    # -----------------------------------
    # Keeps the highest-scoring graph seen this episode, so a policy can be judged
    # on what it found rather than on where it happened to stop.
    # RMS state error per radian of bearing noise: position in formation radii,
    # attitude in radians, both dimensionless once B is length-normalised.
    def shape_error_now(self, brm=None, rank_brm=None):
        a_opt, _, _ = estimation_error_of(self.network, self.rank_K, brmat=brm)
        if not np.isfinite(a_opt):
            return None
        return float(np.sqrt(a_opt / max(self.network.n, 1)))

    # How good this state is on the axis rank and edge count do not cover: where its
    # conditioning sits against a typical greedy graph on the same poses. 0 while
    # flexible, 0.5 for a typical answer, and bounded either side.
    def state_quality(self, brm=None, is_IBR=False, lam=None):
        if not is_IBR or self.spectral_ref is None:
            return 0.0
        a_opt = d_opt = None
        if self.spectral_functional != "eigenvalue":
            a_opt, _, d_opt = estimation_error_of(self.network, self.rank_K, brmat=brm)
        g = spectral_value(self.spectral_functional, lam, a_opt, d_opt)
        if g is None:
            return 0.0
        width = SPECTRAL_SIGMOID_WIDTH[self.spectral_functional]
        return float(1.0 / (1.0 + np.exp(-(g - self.spectral_ref) / width)))

    def update_best_state(self, state_score, is_IBR, is_MBR, rank_brm, min_eig=None,
                          shape_err=None, reset=False):
        if (not reset) and state_score <= self.best_state_score:
            return
        # only computed when this state is actually the new best, and reused from the
        # caller when it already had to compute it for logging (it rebuilds the
        # rigidity matrix, so it is one of the expensive per-step operations)
        if min_eig is None:
            min_eig = rigidity_eigenvalue(self.network, rank_K=self.rank_K)
        self.best_state_score = state_score
        self.best_edges = self.network.edges.copy()
        # how many steps it took to get here; a policy that converges fast and one that
        # stumbles onto the same graph late are otherwise indistinguishable
        self.best_step = self.step_counter
        self.best_stats = {
            "m": int(self.network.edges.sum()),
            "is_IBR": bool(is_IBR),
            "is_MBR": bool(is_MBR),
            "rank": int(rank_brm),
            "min_eig": float(min_eig),
            "shape_err": None if shape_err is None else float(shape_err),
        }

    # -----------------------------------
    # How good is the current graph. Callable outside step(): the reward is this
    # value's improvement, so reset() needs a baseline.
    def compute_state_score(self, brm, is_IBR, is_MBR, rank_brm, lam=None):
        state_score = 0
        if self.state_score_type == "Rigid":
            if is_IBR:
                state_score += 100
        elif self.state_score_type == "RigidAndMinEigenvalue":
            punish = 10
            if not is_IBR:
                state_score -= punish
            else:
                min_eig = rigidity_eigenvalue(self.network, rank_K=self.rank_K)
                state_score += min_eig
        elif self.state_score_type == "RigidAndMinRigid":
            if is_IBR:
                state_score += 10
            if is_MBR:
                state_score += 10
        elif self.state_score_type == "RigidAndLogMinEigenvalueAndEdges":
            bonus = 100
            if is_IBR:
                state_score += bonus

            min_eig = rigidity_eigenvalue(self.network, rank_K=self.rank_K)
            if min_eig != 0.0:
                state_score += np.log10(min_eig * 1e5)

            state_score -= np.sum(self.network.edges)

        elif self.state_score_type == "MinRigid":
            if is_MBR:
                state_score += 10
            else:
                state_score -= 10
        elif self.state_score_type == "MinRigidAndMinEigenvalue":
            punish = 10
            if not is_MBR:
                state_score -= punish
            else:
                min_eig = rigidity_eigenvalue(self.network, rank_K=self.rank_K)
                state_score += min_eig
        elif self.state_score_type == "MinEigenvalue":
            min_eig = rigidity_eigenvalue(self.network, rank_K=self.rank_K)
            state_score += min_eig
        elif self.state_score_type == "Eigenvalues":
            eigs = self.network.eigenvalues()
            state_score += 1e4 * np.sum(eigs)
        elif self.state_score_type == "EdgeCount":
            edge_count = self.network.edges.sum()
            state_score -= edge_count
        elif self.state_score_type == "LogMinEigenvalue":
            min_eig = rigidity_eigenvalue(self.network, rank_K=self.rank_K)
            state_score += np.log10(min_eig) if np.abs(min_eig) > 10e-10 else 0.0
        elif self.state_score_type == "RigidityMatrixRank":
            if np.sum(brm):
                state_score += rank_brm
        elif self.state_score_type == "RigidityMatrixRankAndEdges":
            if np.sum(brm):
                state_score += rank_brm
            state_score -= np.sum(self.network.edges)
        elif self.state_score_type == "Weighted":
            # TODO: tune hyperparameters somehow
            w_rank = 20.0
            w_edge = 10.0

            w_ibr = 0

            w_eig = 0 # 5
            w_eig1 = 0 # 1e5

            s_rank = 0
            s_ibr = 0
            s_eig = 0
            s_edge = 0

            if w_rank != 0:
                s_rank = w_rank * rank_brm
            if w_ibr != 0:
                s_ibr = w_ibr * np.float32(is_IBR)
            if w_eig != 0 and w_eig1 != 0:
                s_eig = w_eig * np.float32(is_IBR) * np.log1p(w_eig1 * rigidity_eigenvalue(self.network, rank_K=self.rank_K))
            if w_edge != 0:
                s_edge = -w_edge * np.sum(self.network.edges)

            state_score += s_rank + s_ibr + s_eig + s_edge
            # print(f"\nr_rank: {r_rank}: {w_rank}*...\nr_ibr: {r_ibr}: {w_ibr}*...\nr_eig: {r_eig}: {w_eig}*...\nr_edge: {r_edge}: {w_edge}*...")
            # print(f"\ntotal: {state_score}")

        elif self.state_score_type == "WeightedNormalized":
            w_rank = 100.0
            w_edge = 25.0

            m = np.sum(self.network.edges)
            rank_K = max(int(self.rank_K), 1)
            c_max = max(int(self.c_max), 1)

            state_score += (w_rank * rank_brm - w_edge * m * c_max) / rank_K

            if self.stiffness_kappa > 0 and is_IBR and lam and self.stiffness_ref > 0:
                one_edge = w_edge * c_max / rank_K
                q = 1.0 / (1.0 + np.exp(-np.log10(lam / self.stiffness_ref)
                                        / STIFFNESS_SIGMOID_DECADES))
                state_score += self.stiffness_kappa * one_edge * q

        elif self.state_score_type == "WeightedNormalizedSpectral":
            # WeightedNormalized with the spectral bonus read off whichever
            # functional spectral_functional names. At "eigenvalue" the two
            # branches agree exactly.
            w_rank = 100.0
            w_edge = 25.0

            m = np.sum(self.network.edges)
            rank_K = max(int(self.rank_K), 1)
            c_max = max(int(self.c_max), 1)

            state_score += (w_rank * rank_brm - w_edge * m * c_max) / rank_K

            if self.stiffness_kappa > 0 and is_IBR and self.spectral_ref is not None:
                a_opt = d_opt = None
                if self.spectral_functional != "eigenvalue":
                    a_opt, _, d_opt = estimation_error_of(
                        self.network, self.rank_K, brmat=brm)
                g = spectral_value(self.spectral_functional, lam, a_opt, d_opt)
                if g is not None:
                    one_edge = w_edge * c_max / rank_K
                    width = SPECTRAL_SIGMOID_WIDTH[self.spectral_functional]
                    q = 1.0 / (1.0 + np.exp(-(g - self.spectral_ref) / width))
                    state_score += self.stiffness_kappa * one_edge * q

        elif self.state_score_type == "None" or None:
            pass

        return state_score

    # -----------------------------------
    def step(self, action):
        reward = 0.0
        reward -= self.time_penalty_value # time taken
        time_penalty_reward = reward
        n = len(self.network.agents)
        # an action toggles at most one edge, so the edge count changing is exactly
        # "this step modified the graph"
        m_before = int(self.network.edges.sum())

        action_info = ""

        # action and reward based on action
        action_return = None
        if self.action_space_type == "AllEdges":
            action_return = action_AllEdges(action, self, reward, action_info)
        elif self.action_space_type == "AddRemoveEdgeMultiDiscrete":
            action_return = action_AddRemoveEdgeMultiDiscrete(action, self, reward, action_info)
        elif self.action_space_type == "AddRemoveEdgeDiscrete":
            action_return = action_AddRemoveEdgeDiscrete(action, self, reward, action_info)
        elif self.action_space_type == "AddEdgeDiscrete":
            action_return = action_AddEdgeDiscrete(action, self, reward, action_info)
        elif self.action_space_type == "AddEdgeDiscreteNoSkip":
            action_return = action_AddEdgeDiscreteNoSkip(action, self, reward, action_info)
        elif self.action_space_type == "AddEdgeDiscreteNoSelfLoops":
            action_return = action_AddEdgeDiscreteNoSelfLoops(action, self, reward, action_info)
        elif self.action_space_type == "AddEdgeDiscreteNoSkipNoSelfLoops":
            action_return = action_AddEdgeDiscreteNoSkipNoSelfLoops(
                action, self, reward, action_info
            )
        elif self.action_space_type == "AddRemoveEdgeDiscreteNoSelfLoops":
            action_return = action_AddRemoveEdgeDiscreteNoSelfLoops(
                action, self, reward, action_info
            )
        elif self.action_space_type == "SelectNodesSequentially":
            action_return = action_SelectNodesSequentially(
                action, self, reward, action_info
            )
        elif self.action_space_type == "DecideOnEdge":
            action_return = action_DecideOnEdge(
                action, self, reward, action_info
            )
        else:
            print(f"faulty action space definition?")
            quit()

        if self.action_rewards_enable:
            reward, action_info = action_return
        else:
            _, action_info = action_return

        action_reward = reward - time_penalty_reward

        # BRM
        brm = self.network.extended_bearing_rigidity_matrix()

        # counted before the best-state update so best_step is the number of steps
        # actually taken to reach that graph
        self.step_counter += 1

        # one SVD serves rank, null space and stiffness
        rank_brm, _, lam = rigidity_decomposition(brm, self.rank_K)
        is_MBR, is_IBR, _ = self.network.is_MBR(
            rank_K=self.rank_K, brm=brm, rank_brm=rank_brm)
        state_score = self.compute_state_score(brm, is_IBR, is_MBR, rank_brm, lam=lam)

        # obs comes after the rigidity computation, not before: the rigidity
        # features have to describe the graph this step produced
        self.compute_rigidity_features(brm, rank_brm, is_IBR, lam=lam)
        obs = self._get_obs()

        # computed once and shared
        tracking = self.track_data_enable and self.writer is not None
        min_eig = lam if (tracking or self.trace_min_eig) else None
        shape_err = self.shape_error_now(brm, rank_brm) if min_eig is not None else None
        self.update_best_state(state_score, is_IBR, is_MBR, rank_brm, min_eig=min_eig,
                               shape_err=shape_err)

        # everything an outside observer needs about this step, so nothing has to be
        # recomputed to record a trajectory
        self.last_stats = {
            "score": float(state_score),
            "m": int(self.network.edges.sum()),
            "rank": int(rank_brm),
            "rank_K": int(self.rank_K),
            "is_IBR": bool(is_IBR),
            "is_MBR": bool(is_MBR),
            "min_eig": float(min_eig) if min_eig is not None else None,
            "shape_err": float(shape_err) if shape_err is not None else None,
        }

        # (incremental) reward from state score
        reward_from_state_score = state_score - self.last_state_score
        reward += reward_from_state_score
        self.last_state_score = state_score

        # termination conditions
        truncated = False
        terminated = False
        if self.termination_condition_type == "MaxSteps":
            if self.step_counter >= self.max_steps:
                truncated = True # this is crucial since we do not want skrl to treat the final state having value=0
                # "truncated" combined with "time_limit_bootstrap" makes it so that the final state's value is also estimated
        elif self.termination_condition_type == "MaxStepsRankBonus":
            if self.step_counter >= self.max_steps:
                if np.sum(brm):
                    reward += rank_brm
                terminated = True
        elif self.termination_condition_type == "Rigid":
            if is_IBR:
                # reward += self.network.nr_max_edges * 10
                reward += 10
                terminated = True
        elif self.termination_condition_type == "RigidMinEigBonus":
            if is_IBR:
                w_eig = 5
                w_eig1 = 1e5
                reward += w_eig * np.float32(is_IBR) * np.log1p(w_eig1 * rigidity_eigenvalue(self.network, rank_K=self.rank_K))
                terminated = True
        elif self.termination_condition_type == "MinimallyRigid":
            if is_MBR:
                # reward += self.network.nr_max_edges * 10
                reward += self.network.n * 100
                terminated = True
        elif self.termination_condition_type == "RigidMinEigAndEdgesBonus":
            if is_IBR:
                min_eig = rigidity_eigenvalue(self.network, rank_K=self.rank_K)
                if min_eig != 0.0:
                    reward += np.log10(min_eig * 1e5)
                reward -= np.sum(self.network.edges)
                terminated = True
        elif self.termination_condition_type == "Bandit":
            if self.step_counter >= 1:
                terminated = True

        # custom truncate logic we can use for certain situations. this should be off if using "MaxSteps"
        if self.truncate_enable and self.termination_condition_type != "MaxSteps":
            if self.step_counter >= self.truncate_max_steps:
                reward -= self.truncate_penalty_value
                truncated = True

        termination_reward = reward - reward_from_state_score - action_reward - time_penalty_reward

        if self.stop_action:
            terminated = True

        # fold this step into the episode totals. Nothing is written yet -- the
        # environment logs once, at the end of the episode
        acc = self.episode_accum
        m_now = int(self.network.edges.sum())
        acc["steps"] += 1
        acc["edits"] += int(m_now != m_before)
        acc["skips"] += int("skip" in action_info)

        # What did this step accomplish? Derived centrally rather than in each of the
        # ten action_* functions. The action_info strings are built in this same file,
        # so the substring checks are a local convention (acc["skips"] already relies
        # on it). "select" is the pointer's first pick: protocol, not waste.
        if m_now > m_before:
            kind = "add"
        elif m_now < m_before:
            kind = "remove"
        elif "skip" in action_info:
            kind = "skip"
        elif "select" in action_info:
            kind = "select"
        else:
            kind = "noop"
        self.last_action_kind = kind
        acc["kinds"][kind] += 1
        acc["useful"] += int(reward_from_state_score > 0)
        try:
            acc["actions"].append(int(np.asarray(action).reshape(-1)[0]))
        except (TypeError, ValueError):
            pass
        if is_IBR and acc["first_rigid"] < 0:
            acc["first_rigid"] = self.step_counter
        if is_MBR and acc["first_minimal"] < 0:
            acc["first_minimal"] = self.step_counter
        acc["return"] += float(reward)
        acc["return_action"] += float(action_reward)
        acc["return_state"] += float(reward_from_state_score)
        acc["return_termination"] += float(termination_reward)
        acc["sum_score"] += float(state_score)
        acc["sum_m"] += m_now
        acc["sum_rank"] += int(rank_brm)
        acc["sum_IBR"] += float(is_IBR)
        acc["sum_MBR"] += float(is_MBR)
        if min_eig is not None:
            acc["sum_min_eig"] += float(min_eig)
            acc["n_min_eig"] += 1
        if shape_err is not None and np.isfinite(shape_err):
            acc["sum_shape_err"] += float(shape_err)
            acc["n_shape_err"] += 1

        # per-step detail for whoever is watching a single rollout (inference.py
        # renders this); it is not logged, so keep it free of extra computation
        info = {}
        if tracking:
            info = {
                "step": f"{self.step_counter}",
                "action (raw)": action,
                "action": action_info,
                "reward": reward,
                "reward (action)": action_reward,
                "reward (state)": reward_from_state_score,
                "reward (termination)": termination_reward,
                "state score": state_score,
                "is rigid": is_IBR,
                "was rigid": self.was_IBR,
                "is min rigid": is_MBR,
                "was min rigid": self.was_MBR,
                "nr edges": int(self.network.edges.sum()),
                "nr initial edges": int(self.initial_m),
                "terminated": terminated,
                "truncated": truncated,
                "min eigenvalue": min_eig,
                "best state score": self.best_state_score,
                "best nr edges": self.best_stats["m"],
                "best is rigid": self.best_stats["is_IBR"],
                "best is min rigid": self.best_stats["is_MBR"],
                "best min eigenvalue": self.best_stats["min_eig"],
                "best step": self.best_step,
            }
            # # print #######################
            # width = max(len(k) for k in info)
            # print("\n" + "=" * 60)
            # for k, v in info.items():
            #     print(f"{k:<{width}} : {v}")
            # print("=" * 60 + "\n")
            # # print #######################
            self.info = info

        # the episode-level scalars are written against the global env step, so they
        # line up with skrl's own curves; the counter therefore has to advance every
        # step even though nothing is written on most of them
        if tracking:
            self.writer_counter += 1

        if terminated or truncated:
            self.last_episode_stats = self.episode_summary(
                state_score, rank_brm, is_IBR, is_MBR, min_eig, terminated, truncated
            )
            if tracking:
                self.write_episode()
            self.episode_counter += 1

        self.was_IBR = is_IBR
        self.was_MBR = is_MBR

        return obs, reward, terminated, truncated, info

    # -----------------------------------
    # One data point per tag per episode, against writer_counter (the global env
    # step) so curves share skrl's x-axis.
    def write_episode(self):
        stats = self.last_episode_stats
        if self.writer is None or stats is None:
            return
        for key, value in stats.items():
            if value is None:
                continue
            # keys that already name their own group (Decision/, Actions/) keep it;
            # everything else lands under Episode/
            tag = key if "/" in key else f"Episode/ {key}"
            self.writer.add_scalar(tag, float(value), self.writer_counter)

        actions = self.episode_accum.get("actions")
        if actions:
            # a collapsed policy puts all its mass on one index
            self.writer.add_histogram(
                "Actions/ index", np.asarray(actions), self.writer_counter
            )

        # the distribution of starting graphs the sampler actually produces.
        # A single scalar per episode cannot show whether it is centred on m_req
        # or merely averages there.
        self.initial_edge_history.append(int(self.initial_m))
        if len(self.initial_edge_history) >= self.initial_edge_hist_every:
            hist = np.asarray(self.initial_edge_history)
            self.writer.add_histogram("Episode/ Initial edges", hist, self.writer_counter)
            self.writer.add_histogram(
                "Episode/ Initial edges over m_req",
                hist / max(int(getattr(self, "m_req", 1) or 1), 1),
                self.writer_counter,
            )
            self.initial_edge_history.clear()

    # Custom scalars from outside the environment; the environment's own metrics
    # go through write_episode()
    def write(self, value=None, tag=None):
        if self.writer is None or value is None or tag is None:
            return
        self.writer.add_scalar(tag, value, self.writer_counter)

    # -----------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # outputs.py sets this to run several methods from the *same* random
        # instance: the episode bookkeeping is redone, the graph is left alone
        if self.freeze_network:
            return self.begin_episode()

        n = self.n
        domains = self.domains

        edge_count = None
        # start with empty graph on addition action types
        if self.action_space_type == "AddEdgeDiscreteNoSkipNoSelfLoops":
            edge_count = 0
        # TODO add other addition action types
        elif self.random_graph_with_mean_min_edges:
            edge_count = self.sample_initial_edge_count(n, domains)

        if self.only_randomize_edges:
            # keep the poses, resample only the edges. With a scenario that means the
            # scenario's own geometry, which is what you want for a case study figure.
            if self.scenario_network is not None:
                self.network = copy.deepcopy(self.scenario_network)
            if edge_count is None:
                edge_count = np.random.randint(0, n**2 - n + 1)
            self.network.set_edges_list(random_edge_list(n, edge_count))
        else:
            # poses and edges are both redrawn; domains are carried over, so a
            # scenario contributes its domain mix rather than its geometry
            self.network, self.goal_network = random_scenario(
                n, domains, edge_count=edge_count,
                rotation_axes=getattr(self, "rotation_axes", None))
            self.randomly_rotate()

        return self.begin_episode()

    # Free augmentation: the task is rotation invariant, the R^d observation is not.
    # Planar agents restrict the admissible axis to z.
    def randomly_rotate(self):
        if not self.rotation_augmentation:
            return
        if any(a.domain in ("R^2", "R^2xS^1") for a in self.network.agents):
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = np.random.normal(size=3)
            norm = np.linalg.norm(axis)
            if norm < 1e-9:
                return
            axis = axis / norm
        self.network.rotate_network(axis, np.random.uniform(0.0, 2.0 * np.pi))

    # Per-episode bookkeeping for whatever graph self.network currently holds.
    def begin_episode(self):
        self.n = len(self.network.agents)
        self.m = int(self.network.edges.sum())
        self.initial_m = self.m

        self.compute_episode_constants()

        self.brm = self.network.extended_bearing_rigidity_matrix()

        self.selection = np.zeros(self.n, dtype=np.int64)
        self.proposed_edge = np.zeros(2)

        self.nr_max_edges = self.n**2
        self.step_counter = 0

        # The reward is the improvement in state score, so the baseline has to be
        # the initial graph's score. Leaving it at 0 would make the first step's
        # reward the *absolute* score of the graph after one action.
        rank_brm_0, _, lam0 = rigidity_decomposition(self.brm, self.rank_K)
        is_MBR_0, is_IBR_0, _ = self.network.is_MBR(
            rank_K=self.rank_K, brm=self.brm, rank_brm=rank_brm_0)
        self.last_state_score = self.compute_state_score(
            self.brm, is_IBR_0, is_MBR_0, rank_brm_0, lam=lam0
        )

        # Best graph seen during the episode. Scoring an episode on the final state
        # conflates "found a good topology" with "learned to stop on it"; this keeps
        # the two separate. Metric only, the reward does not use it.
        self.update_best_state(self.last_state_score, is_IBR_0, is_MBR_0, rank_brm_0, reset=True)
        # step 0 of a trajectory: the graph as the sampler produced it
        self.last_stats = {
            "score": float(self.last_state_score),
            "m": int(self.m),
            "rank": int(rank_brm_0),
            "rank_K": int(self.rank_K),
            "is_IBR": bool(is_IBR_0),
            "is_MBR": bool(is_MBR_0),
            "min_eig": float(self.best_stats["min_eig"]),
            "shape_err": self.best_stats.get("shape_err"),
        }

        self.stop_action = False

        # running totals for this episode; written out once, when it ends
        self.episode_accum = self.new_episode_accum()

        self.info = None

        self.was_IBR = None
        self.was_MBR = None

        self.compute_rigidity_features(self.brm, rank_brm_0, is_IBR_0, lam=lam0)
        return self._get_obs(), {}


if __name__ == "__main__":
    #############################################
    # ACTION_TYPE = "AllEdges"
    # ACTION_TYPE = "AddRemoveEdgeMultiDiscrete"
    # ACTION_TYPE = "AddRemoveEdgeDiscrete"
    # ACTION_TYPE = "AddEdgeDiscrete"
    # ACTION_TYPE = "AddEdgeDiscreteNoSkip"
    # ACTION_TYPE = "AddEdgeDiscreteNoSelfLoops"
    # ACTION_TYPE = "AddEdgeDiscreteNoSkipNoSelfLoops"
    # ACTION_TYPE = "AddRemoveEdgeDiscreteNoSelfLoops"
    ACTION_TYPE = "SelectNodesSequentially"
    # ACTION_TYPE = "DecideOnEdge"

    # ACTION_REWARDS_ENABLE = True
    ACTION_REWARDS_ENABLE = False

    TIME_PENALTY_VALUE = 0.0

    SKIP_ENABLED = False
    SKIP_IS_STOP = False
    RANDOM_GRAPH_WITH_MEAN_MIN_EDGES = True

    TRACK_DATA_ENABLE = True
    # TRACK_DATA_ENABLE = False

    # OBS_TYPE = "Complete"
    # OBS_TYPE = "CompleteAndEigenvalues"
    # OBS_TYPE = "AdjFlatAndEigenvalues"
    OBS_TYPE = "Dict"

    # STATE_SCORE_TYPE = "Rigid"
    # STATE_SCORE_TYPE = "RigidAndMinEigenvalue"
    # STATE_SCORE_TYPE = "RigidAndMinRigid"
    # STATE_SCORE_TYPE = "RigidAndLogMinEigenvalueAndEdges"
    # STATE_SCORE_TYPE = "MinRigid"
    # STATE_SCORE_TYPE = "MinRigidAndMinEigenvalue"
    # STATE_SCORE_TYPE = "MinEigenvalue"
    # STATE_SCORE_TYPE = "Eigenvalues"
    # STATE_SCORE_TYPE = "EdgeCount"
    # STATE_SCORE_TYPE = "LogMinEigenvalue"
    # STATE_SCORE_TYPE = "RigidityMatrixRank"
    # STATE_SCORE_TYPE = "RigidityMatrixRankAndEdges"
    # STATE_SCORE_TYPE = "Weighted"
    # STATE_SCORE_TYPE = "WeightedNormalized"
    STATE_SCORE_TYPE = "WeightedNormalizedSpectral"
    # STATE_SCORE_TYPE = "None"

    TERMINATION_CONDITION_TYPE = "MaxSteps"
    # TERMINATION_CONDITION_TYPE = "MaxStepsRankBonus"
    # TERMINATION_CONDITION_TYPE = "Rigid"
    # TERMINATION_CONDITION_TYPE = "RigidMinEigBonus"
    # TERMINATION_CONDITION_TYPE = "MinimallyRigid"
    # TERMINATION_CONDITION_TYPE = "RigidMinEigAndEdgesBonus"
    # TERMINATION_CONDITION_TYPE = "Bandit"

    MAX_STEPS = None  # set below, once n is known

    TRUNCATE_ENABLE = False
    TRUNCATE_MAX_STEPS = 100
    TRUNCATE_PENALTY_VALUE = 100

    ONLY_RANDOMIZE_EDGES = False

    ROTATION_AUGMENTATION = True

    STIFFNESS_KAPPA = 2.0
    STIFFNESS_REF_SAMPLES = 3
    SPECTRAL_FUNCTIONAL = "trace" # eigenvalue | trace | logdet

    INCLUDE_CANDIDATE_BEARINGS = True

    GRAPH_FEATURES = False
    # which observation channels each flag adds (node = per node, pair = per ordered pair)
    RIGIDITY_GLOBAL = True     # node: rank_deficit, m_ratio, is_IBR (tiled, identical on every node)
    RIGIDITY_QUALITY = True    # node: quality (tiled, identical on every node)
    RIGIDITY_FLEX = True       # node: node_freedom | pair: add_independence
    RIGIDITY_EDGE = True       # pair: pair_max_rank, add_rank
    RIGIDITY_STIFFNESS = True  # node: node_slack (position, attitude) | pair: add_stiffness
    RIGIDITY_REMOVAL = True    # pair: remove_rank, remove_stiffness (zero on non-edges)
    #############################################

    if len(sys.argv) < 3:
        print("Usage: python3 environment.py [n] [domains] [optional_suffix] or python3 environment.py file [scenario_name] [optional_suffix]")
        print(f"Note: Only homogeneous networks for now")
        quit()

    n = 0
    domains = "SE(3)"
    filepath = None
    scenario_name = None
    if sys.argv[1] != "file":
        n = int(sys.argv[1])
        domains = sys.argv[2:]
        if len(domains) != 1:
            print(f"domain list not implemented yet")
            quit()
            # if len(domains_input) != n:
            #     print(f"Number of domain entries ({len(domains_input)}) must match n ({n})")
            #     quit()
            # domains_list = domains_input
        domains = domains[0]
    else:
        filepath = sys.argv[2]
        scenario_name = sys.argv[2]
        filepath = "./scenarios/" + filepath + ".json"
        if not os.path.exists(filepath):
            print(f"file scenarios/{filepath}.json does not exists")
            quit()

        # the full per-agent list: taking domains[0] wrote a homogeneous label
        # into every mixed config, and MAX_STEPS below needs the real mix
        with open(filepath, "r") as f:
            config = json.load(f)
            n = len(config["positions"])
            domains = config["domains"]

    if isinstance(domains, str):
        domains_str = domains.replace("^", "").replace("(", "").replace(")", "")
    else:
        domains_str = scenario_name or "mixed"

    now = datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H_%M_%S")

    # ~4 edits per required edge. The old 4*n*(n-1) was 20-30x the measured
    # best@, and the horizon sets how many distinct instances a run and its
    # replay buffer ever see. m_req depends only on (n, domain mix), not on the
    # poses, so one draw settles it.
    _probe_net, _ = random_scenario(n, domains, edge_count=n)
    MAX_STEPS = 4 * int(required_edge_count(_probe_net)) + 10

    n_domains = f"n{n}_{domains_str}"
    # the rigidity arms differ only by these flags, so the name has to carry them
    rig_tag = "".join(t for t, on in
                      (("G", RIGIDITY_GLOBAL), ("F", RIGIDITY_FLEX), ("E", RIGIDITY_EDGE)) if on)
    rig_tag = f"_rig{rig_tag}" if rig_tag else ""
    rig_tag += "" if GRAPH_FEATURES else "_lean"
    model_name = f"action{ACTION_TYPE}_reward{STATE_SCORE_TYPE}_term{TERMINATION_CONDITION_TYPE}{rig_tag}_{scenario_name if scenario_name is not None else n_domains}"

    if len(sys.argv) > 3 and sys.argv[3] is not None:
        model_name += f"_{sys.argv[3]}"

    print(f"MODEL NAME: {model_name}")

    log_dir = "./tboard_logs/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("./models/", exist_ok=True)

    #########################

    if filepath is not None:
        print(f"loading environment from scenario {filepath}")
    else:
        print(f"creating environment with n={n}, domains={domains}")

    env_config = {
        "action_type": ACTION_TYPE,
        "obs_type": OBS_TYPE,
        "state_score_type": STATE_SCORE_TYPE,
        "termination_condition_type": TERMINATION_CONDITION_TYPE,
        "n": n,
        "domains": domains,
        "action_rewards_enable": ACTION_REWARDS_ENABLE,
        "skip_enabled": SKIP_ENABLED,
        "skip_is_stop": SKIP_IS_STOP,
        "random_graph_with_mean_min_edges": RANDOM_GRAPH_WITH_MEAN_MIN_EDGES,
        "time_penalty_value": TIME_PENALTY_VALUE,
        "track_data_enable": TRACK_DATA_ENABLE,
        "max_steps": MAX_STEPS,
        "truncate_enable": TRUNCATE_ENABLE,
        "truncate_max_steps": TRUNCATE_MAX_STEPS,
        "truncate_penalty_value": TRUNCATE_PENALTY_VALUE,
        "only_randomize_edges": ONLY_RANDOMIZE_EDGES,
        "include_candidate_bearings": INCLUDE_CANDIDATE_BEARINGS,
        "graph_features": GRAPH_FEATURES,
        "rigidity_global": RIGIDITY_GLOBAL,
        "rigidity_quality": RIGIDITY_QUALITY,
        "rigidity_flex": RIGIDITY_FLEX,
        "rigidity_edge": RIGIDITY_EDGE,
        "rigidity_stiffness": RIGIDITY_STIFFNESS,
        "rigidity_removal": RIGIDITY_REMOVAL,
        "rotation_augmentation": ROTATION_AUGMENTATION,
        "stiffness_kappa": STIFFNESS_KAPPA,
        "stiffness_ref_samples": STIFFNESS_REF_SAMPLES,
        "spectral_functional": SPECTRAL_FUNCTIONAL,
        "scenario": scenario_name,
    }
    env_filename = f"env_{model_name}.json"
    env_path = os.path.join("./environments/", env_filename)
    # before the open(), not inside it -- environments/ is gitignored, so on a fresh
    # clone the directory does not exist yet and open("w") is what fails
    os.makedirs("./environments/", exist_ok=True)
    with open(env_path, "w") as f:
        json.dump(env_config, f, indent=2)
        print(f"SAVED: {env_path}")
        print(f"env: env_{model_name}")
