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
from skrl.utils.tensorboard import SummaryWriter
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
    elif type == "DictNodeFeaturesAndAdj":
        node_features = np.concat([network.get_domain_features(), network.get_simplified_bearing_features().reshape(network.n, -1)], -1)
        adj = network.edges.astype(np.float32)
        obs = {
            "node_features": node_features,
            "adj": adj
        }
        if define_type:
            obs_space = spaces.Dict({
                "node_features": spaces.Box(-np.inf, np.inf, node_features.shape), # N agents, 10 features
                "adj": spaces.Box(0, 1, (n, n))
            })
    elif type == "DictNodeFeaturesAndAdjAndSelection":
        node_features = np.concat([network.get_domain_features(), network.get_simplified_bearing_features().reshape(network.n, -1)], -1)
        adj = network.edges.astype(np.float32)
        obs = {
            "node_features": node_features,
            "adj": adj,
            "selection": env.selection,
        }
        if define_type:
            obs_space = spaces.Dict(
                {
                    "node_features": spaces.Box(
                        -np.inf, np.inf, node_features.shape
                    ),  # N agents, 10 features
                    "adj": spaces.Box(0, 1, (n, n)),
                    "selection": spaces.Box(0, 1, [n], dtype=int),
                }
            )
    elif type == "DictNodeFeaturesAndAdjAndEdgeProposal":
        env.edge_proposal = np.array([np.random.randint(0, env.network.n),
                                      np.random.randint(0, env.network.n)])
        node_features = np.concat([network.get_domain_features(), network.get_bearing_features().reshape(network.n, -1)], -1)
        adj = network.edges.astype(np.float32)
        obs = {
            "node_features": node_features,
            "adj": adj,
            "proposed_edge": env.edge_proposal,
        }
        if define_type:
            obs_space = spaces.Dict(
                {
                    "node_features": spaces.Box(
                        -np.inf, np.inf, node_features.shape
                    ),  # N agents, 10 features
                    "adj": spaces.Box(0, 1, (n, n)),
                    "proposed_edge": spaces.Box(0, n, [2]),
                }
            )
    elif type == "DictEquivariantNodeFeaturesAndAdjAndSelection":
        node_features = np.concat([network.get_domain_features(),
                                   network.get_degree_features(),
                                   network.get_closeness_centrality_features(),
                                   network.get_eigenvector_centrality_features(),
                                   network.get_node_betweenness_features(),
                                   ], axis=-1)
        coord_features = network.get_position_features()
        edge_features = np.concat([network.get_bearing_features(),
                                   network.get_edge_betweenness_features(),
                                   network.get_edge_reciprocity_features(),
                                   network.get_common_neighbors_features(),], axis=-1)

        adj = network.edges.astype(np.float32)
        obs = {
            "node_features": node_features,
            "coord_features": coord_features,
            "edge_features": edge_features,
            "adj": adj,
            "selection": env.selection,
        }
        if define_type:
            obs_space = spaces.Dict(
                {
                    "node_features": spaces.Box(
                        -np.inf, np.inf, node_features.shape
                    ),  # N agents
                    "coord_features": spaces.Box(
                        -np.inf, np.inf, coord_features.shape
                    ),
                    "edge_features": spaces.Box(
                        -np.inf, np.inf, edge_features.shape
                    ),
                    "adj": spaces.Box(0, 1, adj.shape),
                    "selection": spaces.Box(0, 1, env.selection.shape, dtype=int),
                }
            )
    elif type == "DictBearingNodeFeaturesAndAdjAndSelection":
        node_features = network.get_bearing_features().reshape((network.n, -1))
        adj = network.edges.astype(np.float32)
        obs = {
            "node_features": node_features,
            "adj": adj,
            "selection": env.selection,
        }
        if define_type:
            obs_space = spaces.Dict(
                {
                    "node_features": spaces.Box(
                        -np.inf, np.inf, node_features.shape
                    ),  # N agents, 10 features
                    "adj": spaces.Box(0, 1, (n, n)),
                    "selection": spaces.Box(0, 1, [n], dtype=int),
                }
            )
    elif type == "DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection":
        node_features = np.concat([network.get_domain_features(),
                                   network.get_degree_features(),
                                   network.get_closeness_centrality_features(),
                                   network.get_eigenvector_centrality_features(),
                                   network.get_node_betweenness_features(),
                                   ], axis=-1)
        edge_features = np.concat([network.get_bearing_features(),
                                   network.get_edge_betweenness_features(),
                                   network.get_edge_reciprocity_features(),
                                   network.get_common_neighbors_features(),], axis=-1)
        adj = network.edges.astype(np.float32)
        obs = {
            "node_features": node_features,
            "edge_features": edge_features,
            "adj": adj,
            "selection": env.selection,
        }
        if define_type:
            obs_space = spaces.Dict(
                {
                    "node_features": spaces.Box(
                        -np.inf, np.inf, node_features.shape
                    ),
                    "edge_features": spaces.Box(
                        -np.inf, np.inf, edge_features.shape
                    ),
                    "adj": spaces.Box(0, 1, (n, n)),
                    "selection": spaces.Box(0, 1, [n], dtype=int),
                }
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
        filepath=None,
    ):
        print("initializing environment")

        self.action_space_type = action_space_type
        self.obs_space_type = obs_space_type
        self.state_score_type = state_score_type
        self.termination_condition_type = termination_condition_type

        self.random_graph_with_mean_min_edges = random_graph_with_mean_min_edges

        self.filepath = filepath
        self.scenario_network = None
        if self.filepath is not None:
            self.network, self.goal_network = load_scenario(self.filepath)
            # cached so reset() does not re-read and re-parse the file every episode
            self.scenario_network = copy.deepcopy(self.network)
        else:
            # Since learning with complete random number of edges gets basically impossible
            # as the number of nodes grow, we'll sample the number of edges to
            # be used centered around the minimum required amount for IBR.
            # Important: We'll sample a value centered around the edge count for R^d.
            # It should be less than the number for other domains but it should be okay I think.
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
        self.m = int(self.network.edges.sum())
        self.initial_m = self.m

        self.brm = self.network.extended_bearing_rigidity_matrix()

        network_K = self.network.fully_connected()
        self.rank_K = np.linalg.matrix_rank(network_K.extended_bearing_rigidity_matrix())

        self.selection = np.zeros(self.n, dtype=np.int64)
        self.proposed_edge = np.zeros(2)

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

        self.writer = None
        # self.initial_edges_writer = None

    def set_writer(self, experiment_name):
        self.writer = SummaryWriter(log_dir=os.path.join("runs", experiment_name))
        # self.initial_edges_writer = SummaryWriter(log_dir=os.path.join("runs", experiment_name))
        self.writer_counter = 0 # don't reset this

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
            filepath=scenario_path,
        )

    # -----------------------------------
    # Uniformly random edge counts are almost always far above the number of
    # edges rigidity actually needs (the requirement grows ~linearly in n while
    # n^2-n grows quadratically), so the agent would only ever see graphs that
    # need edges removed. Sample around the minimum requirement instead.
    # NOTE: the mean is only exact for homogeneous R^d networks.
    def sample_initial_edge_count(self, n, domains):
        if isinstance(domains, str):
            domains = [domains]
        d = 2 if (("R^2" in domains) or ("R^2xS^1" in domains)) else 3
        mean = MBR_required_Rd(n, d)
        max_edges = n**2 - n
        edge_count = int(sample_gaussian(mean, (max_edges - mean)**2 / 9, n).item())
        return int(np.clip(edge_count, 1, max_edges))

    # -----------------------------------
    # Keeps the highest-scoring graph seen this episode, so a policy can be judged
    # on what it found rather than on where it happened to stop.
    def update_best_state(self, state_score, is_IBR, is_MBR, rank_brm, min_eig=None, reset=False):
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
        }

    # -----------------------------------
    # How good is the current graph. The reward uses the *improvement* of this
    # value between steps, so it must be computable outside step() too (reset()
    # needs the initial graph's score as the baseline).
    def compute_state_score(self, brm, is_IBR, is_MBR, rank_brm):
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

        elif self.state_score_type == "None" or None:
            pass

        return state_score

    # -----------------------------------
    def step(self, action):
        reward = 0.0
        reward -= self.time_penalty_value # time taken
        time_penalty_reward = reward
        n = len(self.network.agents)

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

        # obs
        obs = self._get_obs()

        # BRM
        brm = self.network.extended_bearing_rigidity_matrix()

        # counted before the best-state update so best_step is the number of steps
        # actually taken to reach that graph
        self.step_counter += 1

        # state score, how good is the current state
        is_MBR, is_IBR, rank_brm = self.network.is_MBR(rank_K=self.rank_K, brm=brm)
        state_score = self.compute_state_score(brm, is_IBR, is_MBR, rank_brm)

        # when tracking, this is needed for logging anyway, so compute it once here and
        # hand it to the best-state tracker instead of letting it redo the work.
        # trace_min_eig asks for it without a writer attached (baselines.py records the
        # rigidity eigenvalue over time)
        tracking = self.track_data_enable and self.writer is not None
        min_eig = (rigidity_eigenvalue(self.network, rank_K=self.rank_K)
                   if (tracking or self.trace_min_eig) else None)
        self.update_best_state(state_score, is_IBR, is_MBR, rank_brm, min_eig=min_eig)

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
                terminated = True
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

        if self.truncate_enable:
            if self.step_counter >= self.truncate_max_steps:
                reward -= self.truncate_penalty_value
                truncated = True

        termination_reward = reward - reward_from_state_score - action_reward - time_penalty_reward

        if self.stop_action:
            terminated = True

        # the metrics below rebuild the rigidity matrix, so only pay for them
        # when we're actually logging (min_eig was already computed above)
        info = {}
        if tracking:
            eigs = self.network.eigenvalues()
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
                "eigenvalues": eigs,
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
            self.writer_counter += 1
            self.write()

        self.was_IBR = is_IBR
        self.was_MBR = is_MBR

        return obs, reward, terminated, truncated, info

    # TODO: this is expensive if logged every step, we should log mean values over episodes maybe or idk how to handle "is rigid"
    def write(self, value=None, tag=None):
        log_period = 1
        if value is None:
            if self.info is not None and (self.writer_counter % log_period == 0):
                # # 1. Log current edges to the main writer
                # self.writer.add_scalar(tag="Environment/Edges",
                #                        value=self.info["nr edges"],
                #                        timestep=self.writer_counter)
                # # 2. Log initial edges to the secondary writer using the EXACT SAME TAG
                # if hasattr(self, 'initial_edges_writer'):
                #     self.initial_edges_writer.add_scalar(tag="Environment/Edges",
                #                                          value=self.info["nr initial edges"],
                #                                          timestep=self.writer_counter)

                self.writer.add_scalar(tag="Environment/ Nr initial edges", value=self.info["nr initial edges"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Nr edges", value=self.info["nr edges"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Is rigid", value=self.info["is rigid"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Is min rigid", value=self.info["is min rigid"],timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Reward step", value=self.info["reward"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Reward action", value=self.info["reward (action)"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Reward state", value=self.info["reward (state)"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ State score", value=self.info["state score"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Reward termination", value=self.info["reward (termination)"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Min eig", value=self.info["min eigenvalue"], timestep=self.writer_counter)

                # best graph seen this episode (independent of where the policy stopped)
                self.writer.add_scalar(tag="Environment/ Best state score", value=self.info["best state score"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Best nr edges", value=self.info["best nr edges"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Best is rigid", value=self.info["best is rigid"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Best is min rigid", value=self.info["best is min rigid"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Best min eig", value=self.info["best min eigenvalue"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Best step", value=self.info["best step"], timestep=self.writer_counter)

                # Safely extract the action value, regardless of its data type
                action_val = self.info["action (raw)"]
                if isinstance(action_val, torch.Tensor):
                    action_val = action_val.item() if action_val.numel() == 1 else action_val[0].item()
                elif isinstance(action_val, np.ndarray):
                    action_val = action_val.item() if action_val.size == 1 else action_val[0]
                elif isinstance(action_val, list):
                    action_val = action_val[0]

                self.writer.add_scalar(tag="Environment/ Action", value=action_val, timestep=self.writer_counter)

                # if type(self.info["action (raw)"]) not in [list, np.ndarray, torch.Tensor]:
                #     self.writer.add_scalar(tag="Environment/ Action", value=self.info["action (raw)"], timestep=self.writer_counter)
        else:
            self.writer.add_scalar(tag=tag, value=value, timestep=self.writer_counter)

    # -----------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # baselines.py sets this to run several methods from the *same* random
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
            self.network, self.goal_network = random_scenario(n, domains, edge_count=edge_count)

        return self.begin_episode()

    # Per-episode bookkeeping for whatever graph self.network currently holds.
    def begin_episode(self):
        self.n = len(self.network.agents)
        self.m = int(self.network.edges.sum())
        self.initial_m = self.m

        network_K = self.network.fully_connected()
        self.rank_K = np.linalg.matrix_rank(network_K.extended_bearing_rigidity_matrix())

        self.brm = self.network.extended_bearing_rigidity_matrix()

        self.selection = np.zeros(self.n, dtype=np.int64)
        self.proposed_edge = np.zeros(2)

        self.nr_max_edges = self.n**2
        self.step_counter = 0

        # The reward is the improvement in state score, so the baseline has to be
        # the initial graph's score. Leaving it at 0 would make the first step's
        # reward the *absolute* score of the graph after one action.
        is_MBR_0, is_IBR_0, rank_brm_0 = self.network.is_MBR(rank_K=self.rank_K, brm=self.brm)
        self.last_state_score = self.compute_state_score(
            self.brm, is_IBR_0, is_MBR_0, rank_brm_0
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
        }

        self.stop_action = False

        self.info = None

        self.was_IBR = None
        self.was_MBR = None

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

    # False masks the skip action out of the policy entirely. Recommended with
    # MaxSteps: skip is a zero-reward action the agent can loop on forever
    # (select -> skip changes nothing), which on-policy methods collapse onto.
    # Score such runs with the best-state-visited metric instead.
    SKIP_ENABLED = False
    SKIP_IS_STOP = False
    RANDOM_GRAPH_WITH_MEAN_MIN_EDGES = True

    TRACK_DATA_ENABLE = True
    # TRACK_DATA_ENABLE = False

    # OBS_TYPE = "Complete"
    # OBS_TYPE = "CompleteAndEigenvalues"
    # OBS_TYPE = "AdjFlatAndEigenvalues"
    # OBS_TYPE = "DictNodeFeaturesAndAdj"
    # OBS_TYPE = "DictNodeFeaturesAndAdjAndSelection"
    # OBS_TYPE = "DictNodeFeaturesAndAdjAndEdgeProposal"
    OBS_TYPE = "DictEquivariantNodeFeaturesAndAdjAndSelection" ## Equivariant
    # OBS_TYPE = "DictBearingNodeFeaturesAndAdjAndSelection"
    # OBS_TYPE = "DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection" ## GINE

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
    STATE_SCORE_TYPE = "Weighted"
    # STATE_SCORE_TYPE = "None"

    TERMINATION_CONDITION_TYPE = "MaxSteps"
    # TERMINATION_CONDITION_TYPE = "MaxStepsRankBonus"
    # TERMINATION_CONDITION_TYPE = "Rigid"
    # TERMINATION_CONDITION_TYPE = "RigidMinEigBonus"
    # TERMINATION_CONDITION_TYPE = "MinimallyRigid"
    # TERMINATION_CONDITION_TYPE = "RigidMinEigAndEdgesBonus"
    # TERMINATION_CONDITION_TYPE = "Bandit"

    MAX_STEPS = 100

    TRUNCATE_ENABLE = False
    TRUNCATE_MAX_STEPS = 100
    TRUNCATE_PENALTY_VALUE = 100

    ONLY_RANDOMIZE_EDGES = False
    #############################################

    if len(sys.argv) < 3:
        print("Usage: python3 environment.py [n] [domains] or python3 environment.py file [scenario_name]")
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

        # get n and domains from scenario
        with open(filepath, "r") as f:
            config = json.load(f)
            n = len(config["positions"])
            domains = config["domains"][0]

    domains_str = domains
    domains_str = domains_str.replace("^", "").replace("(", "").replace(")", "")

    now = datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H_%M_%S")

    n_domains = f"n{n}_{domains_str}"
    model_name = f"action{ACTION_TYPE}_obs{OBS_TYPE}_reward{STATE_SCORE_TYPE}_term{TERMINATION_CONDITION_TYPE}_{scenario_name if scenario_name is not None else n_domains}"
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
        "scenario": scenario_name,
    }
    env_filename = f"env_{model_name}.json"
    env_path = os.path.join("./environments/", env_filename)
    with open(env_path, "w") as f:
        os.makedirs("./environments/", exist_ok=True)
        json.dump(env_config, f, indent=2)
        print(f"SAVED: {env_path}")
        print(f"env: env_{model_name}")
