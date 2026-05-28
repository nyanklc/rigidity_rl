import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import signal
import json
from datetime import datetime
import os
import sys
from network import Network
from rigidity import is_IBR
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from skrl.utils.tensorboard import SummaryWriter
import torch

from visualizer import Visualizer
from scenario import load_scenario, random_scenario
from control import GradientBasedController


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
        action_space = spaces.Discrete(n)
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

    if env.network.edge_exists(i_idx, j_idx):
        # reward -= 20 # unnecessary action
        action_info += " (existed)"

    env.network.add_edge(i_idx, j_idx)
    reward -= 1  # measurement effort

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
        pass
    elif action < (action_space_len - 1) // 2:
        # add
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
        else:
            env.network.add_edge(i_idx, j_idx)
            # TODO: i don't know if this is good
            if env.network.is_IBR():
                reward -= 1 # measurement effort
            else:
                reward += 1 # need for rigidity
    else:
        # remove
        i_idx = (action - ((action_space_len - 1) // 2)) // (n - 1)
        j_idx = (action - ((action_space_len - 1) // 2)) % (n - 1)
        if j_idx >= i_idx:
            j_idx += 1

        action_info += f"remove {i_idx}-{j_idx}"
        if i_idx == j_idx:
            action_info += " (self loop)"

        if not env.network.edge_exists(i_idx, j_idx):
            # reward -= 20 # unnecessary action
            action_info += " (didn't exist)"
        else:
            env.network.remove_edge(i_idx, j_idx)
            reward += 1 # measurement effort

    return reward, action_info


def action_SelectNodesSequentially(action, env: "Environment", reward, action_info):
    action_info += f"(action={action}) "

    n = len(env.network.agents)

    reward -= np.sum(env.network.edges)

    if action == n:
        action_info += " skip"
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
        env.selection = np.zeros(env.network.n)

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
        node_features = network.get_node_features()
        adj = network.edges.astype(np.float32)
        obs = {
            "node_features": node_features,
            "adj": adj
        }
        if define_type:
            obs_space = spaces.Dict({
                "node_features": spaces.Box(-np.inf, np.inf, (n, node_features.shape[1])), # N agents, 10 features
                "adj": spaces.Box(0, 1, (n, n))
            })
    elif type == "DictNodeFeaturesAndAdjAndSelection":
        node_features = network.get_node_features()
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
                        -np.inf, np.inf, (n, node_features.shape[1])
                    ),  # N agents, 10 features
                    "adj": spaces.Box(0, 1, (n, n)),
                    "selection": spaces.Box(0, 1, [n], dtype=int),
                }
            )
    elif type == "DictNodeFeaturesAndAdjAndEdgeProposal":
        env.edge_proposal = np.array([np.random.randint(0, env.network.n),
                                      np.random.randint(0, env.network.n)])
        node_features = network.get_node_features()
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
                        -np.inf, np.inf, (n, node_features.shape[1])
                    ),  # N agents, 10 features
                    "adj": spaces.Box(0, 1, (n, n)),
                    "proposed_edge": spaces.Box(0, n, [2]),
                }
            )
    elif type == "DictEquivariantNodeFeaturesAndAdjAndSelection":
        node_features = network.get_node_features_equivariant()
        coord_features = network.get_coords_equivariant()
        edge_features = network.get_edge_features_equivariant()

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
        reward_type="Rigid",
        termination_condition_type="MaxSteps",
        action_rewards_enable=False,
        time_penalty_value=0.0,
        incremental_rewards_enable=False,
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
        self.reward_type = reward_type
        self.termination_condition_type = termination_condition_type

        self.filepath = filepath
        if self.filepath is not None:
            self.network, self.goal_network = load_scenario(self.filepath)
        else:
            self.network, self.goal_network = random_scenario(n, domains)

        self.n = len(self.network.agents)
        self.m = int(self.network.edges.sum())

        self.brm = self.network.extended_bearing_rigidity_matrix()

        self.selection = np.zeros(self.n)
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

        self.last_reward = 0

        self.action_rewards_enable = action_rewards_enable
        self.incremental_rewards_enable = incremental_rewards_enable
        self.track_data_enable = track_data_enable

        self.time_penalty_value = time_penalty_value

        self.was_IBR = None
        self.was_MBR = None

        self.writer = None

    def set_writer(self, experiment_name):
        self.writer = SummaryWriter(log_dir=os.path.join("runs", experiment_name))
        self.writer_counter = 0 # don't reset this

    def load(self, filepath):
        with open(filepath, "r") as f:
            config = json.load(f)

        n = config["n"]
        domains = config["domains"]
        ACTION_TYPE = config["action_type"]
        OBS_TYPE = config["obs_type"]
        REWARD_TYPE = config["reward_type"]
        TERMINATION_CONDITION_TYPE = config["termination_condition_type"]
        ACTION_REWARDS_ENABLE = config["action_rewards_enable"]
        TIME_PENALTY_VALUE = config["time_penalty_value"]
        INCREMENTAL_REWARDS_ENABLE = config["incremental_rewards_enable"]
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
            reward_type=REWARD_TYPE,
            termination_condition_type=TERMINATION_CONDITION_TYPE,
            action_rewards_enable=ACTION_REWARDS_ENABLE,
            time_penalty_value=TIME_PENALTY_VALUE,
            incremental_rewards_enable=INCREMENTAL_REWARDS_ENABLE,
            track_data_enable=TRACK_DATA_ENABLE,
            max_steps=MAX_STEPS,
            truncate_enable=TRUNCATE_ENABLE,
            truncate_max_steps=TRUNCATE_MAX_STEPS,
            truncate_penalty_value=TRUNCATE_PENALTY_VALUE,
            only_randomize_edges=ONLY_RANDOMIZE_EDGES,
            filepath=scenario_path,
        )

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

        # reward based on state
        is_MBR, is_IBR = self.network.is_MBR()
        if self.reward_type == "Rigid":
            if is_IBR:
                reward += 100
        elif self.reward_type == "RigidAndMinEigenvalue":
            punish = 10
            if not is_IBR:
                reward -= punish
            else:
                eigs = self.network.eigenvalues()
                nonzero = eigs[eigs > 0.0]
                min_eig = nonzero[0] if len(nonzero) else 0
                reward += min_eig
        elif self.reward_type == "RigidAndMinRigid":
            if is_IBR:
                reward += 10
            if is_MBR:
                reward += 10
        elif self.reward_type == "RigidAndLogMinEigenvalueAndEdges":
            bonus = 100
            if is_IBR:
                reward += bonus

            eigs = self.network.eigenvalues()
            nonzero = eigs[eigs > 0.0]
            min_eig = nonzero[0] if len(nonzero) else 0
            if min_eig != 0.0:
                reward += np.log10(min_eig * 1e5)

            reward -= np.sum(self.network.edges)

        elif self.reward_type == "MinRigid":
            if is_MBR:
                reward += 10
            else:
                reward -= 10
        elif self.reward_type == "MinRigidAndMinEigenvalue":
            punish = 10
            if not is_MBR:
                reward -= punish
            else:
                eigs = self.network.eigenvalues()
                nonzero = eigs[eigs > 0.0]
                min_eig = nonzero[0] if len(nonzero) else 0
                reward += min_eig
        elif self.reward_type == "MinEigenvalue":
            eigs = self.network.eigenvalues()
            nonzero = eigs[eigs > 0.0]
            min_eig = nonzero[0] if len(nonzero) else 0
            reward += min_eig
        elif self.reward_type == "Eigenvalues":
            eigs = self.network.eigenvalues()
            nonzero = eigs[eigs > 0.0]
            min_eig = nonzero[0] if len(nonzero) else 0
            second_min_eig = nonzero[1] if len(nonzero)>=2 else 0
            reward += 1e4 * (min_eig + second_min_eig)
        elif self.reward_type == "EdgeCount":
            edge_count = self.network.edges.sum()
            reward -= edge_count
        elif self.reward_type == "LogMinEigenvalue":
            eigs = self.network.eigenvalues()
            nonzero = eigs[eigs > 0.0]
            min_eig = nonzero[0] if len(nonzero) else 0
            reward += np.log10(min_eig) if np.abs(min_eig) > 10e-10 else 0.0
        elif self.reward_type == "RigidityMatrixRank":
            brm = self.network.extended_bearing_rigidity_matrix()
            if np.sum(brm):
                reward += np.linalg.matrix_rank(brm)
        elif self.reward_type == "None" or None:
            pass

        state_reward = reward - action_reward

        self.step_counter += 1

        # termination conditions
        truncated = False
        terminated = False
        if self.termination_condition_type == "MaxSteps":
            if self.step_counter >= self.max_steps:
                terminated = True
        elif self.termination_condition_type == "MaxStepsRankBonus":
            if self.step_counter >= self.max_steps:
                brm = self.network.extended_bearing_rigidity_matrix()
                if np.sum(brm):
                    reward += np.linalg.matrix_rank(brm)
                terminated = True
        elif self.termination_condition_type == "Rigid":
            if is_IBR:
                # reward += self.network.nr_max_edges * 10
                reward += 10
                terminated = True
        elif self.termination_condition_type == "RigidMinEigBonus":
            if is_IBR:
                eigs = self.network.eigenvalues
                nonzero_eigs = eigs[eigs != 0.0]
                reward += min(nonzero_eigs) # TODO: the value of this is pretty small
                terminated = True
        elif self.termination_condition_type == "MinimallyRigid":
            if is_MBR:
                # reward += self.network.nr_max_edges * 10
                reward += 100
                terminated = True
        elif self.termination_condition_type == "RigidMinEigAndEdgesBonus":
            if not is_IBR:
                reward -= 10
            else:
                eigs = self.network.eigenvalues()
                nonzero = eigs[eigs > 0.0]
                min_eig = nonzero[0] if len(nonzero) else 0
                if min_eig != 0.0:
                    reward += np.log10(min_eig * 1e5)
                reward -= np.sum(self.network.edges)
        elif self.termination_condition_type == "Bandit":
            if self.step_counter >= 1:
                terminated = True

        if self.truncate_enable:
            if self.step_counter >= self.truncate_max_steps:
                reward -= self.truncate_penalty_value
                truncated = True

        termination_reward = reward - state_reward - action_reward

        # (incremental) reward
        last_reward_copy = self.last_reward
        self.last_reward = reward
        if self.incremental_rewards_enable:
            reward = reward - last_reward_copy

        # debug
        eigs = self.network.eigenvalues()
        nonzero = eigs[eigs != 0.0]
        min_eig = nonzero[0] if len(nonzero) else 0
        info = {
            "step": f"{self.step_counter}",
            "action (raw)": action,
            "action": action_info,
            "reward (raw)": self.last_reward,
            "reward (step)": reward,
            "reward (action)": action_reward,
            "reward (state)": state_reward,
            "reward (termination)": termination_reward,
            "last reward": last_reward_copy,
            "is rigid": is_IBR,
            "was rigid": self.was_IBR,
            "is min rigid": is_MBR,
            "was min rigid": self.was_MBR,
            "nr edges": int(self.network.edges.sum()),
            "terminated": terminated,
            "truncated": truncated,
            "eigenvalues": eigs,
            "nonzero_eigenvalues": nonzero,
            "min eigenvalue": min_eig,
            "second min eigenvalue": nonzero[1] if len(nonzero) >= 2 else 0.0,
        }
        # print(info)
        self.info = info
        if self.track_data_enable and self.writer is not None:
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
                self.writer.add_scalar(tag="Environment/ Nr edges", value=self.info["nr edges"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Is rigid", value=self.info["is rigid"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Is min rigid", value=self.info["is min rigid"],timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Reward raw", value=self.info["reward (raw)"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Reward step", value=self.info["reward (step)"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Reward action", value=self.info["reward (action)"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Reward state", value=self.info["reward (state)"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Reward termination", value=self.info["reward (termination)"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Min eig", value=self.info["min eigenvalue"], timestep=self.writer_counter)
                self.writer.add_scalar(tag="Environment/ Second min eig", value=self.info["second min eigenvalue"], timestep=self.writer_counter)
                if type(self.info["action (raw)"]) not in [list, np.ndarray, torch.Tensor]:
                    self.writer.add_scalar(tag="Environment/ Action", value=self.info["action (raw)"], timestep=self.writer_counter)
        else:
            self.writer.add_scalar(tag=tag, value=value, timestep=self.writer_counter)

    # -----------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.filepath:
            self.network, self.goal_network = load_scenario(self.filepath)
        else:
            # TODO: with "Complete" observations, this doesn't make sense since the pos/orient stay the same
            # just randomize the edges
            # TODO: create flags to handle the network reset.
            # depending on how we want to train, we may want to randomize only the
            # poses and remove all edges for instance (e.g. empty scenario with AllEdges actions).

            n = self.network.n
            domains = self.network.agents[0].domain # only homogeneous is supported
            if self.only_randomize_edges:
                edge_set = set()
                max_possible_edges = n**2 - n # no self loops
                m = np.random.randint(0, max_possible_edges + 1)
                while len(edge_set) < m:
                    i, j = np.random.choice(n, size=2, replace=False)
                    if ((i, j) not in edge_set):
                        edge_set.add((i, j))
                edges = np.array(list(edge_set))
                if len(edge_set) == 0:
                    self.network.set_edges(None)
                else:
                    self.network.set_edges_indices(edges[:, 0], edges[:, 1])
            else:
                self.network, self.goal_network = random_scenario(n, domains)

        self.n = len(self.network.agents)
        self.m = int(self.network.edges.sum())

        self.brm = self.network.extended_bearing_rigidity_matrix()

        self.selection = np.zeros(self.n)
        self.proposed_edge = np.zeros(2)

        self.nr_max_edges = self.n**2
        self.step_counter = 0

        self.last_reward = 0

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

    ACTION_REWARDS_ENABLE = True
    # ACTION_REWARDS_ENABLE = False

    TIME_PENALTY_VALUE = 1.0

    INCREMENTAL_REWARDS_ENABLE = False
    # INCREMENTAL_REWARDS_ENABLE = True

    TRACK_DATA_ENABLE = True
    # TRACK_DATA_ENABLE = False

    # OBS_TYPE = "Complete"
    # OBS_TYPE = "CompleteAndEigenvalues"
    # OBS_TYPE = "AdjFlatAndEigenvalues"
    # OBS_TYPE = "DictNodeFeaturesAndAdj"
    # OBS_TYPE = "DictNodeFeaturesAndAdjAndSelection"
    # OBS_TYPE = "DictNodeFeaturesAndAdjAndEdgeProposal"
    OBS_TYPE = "DictEquivariantNodeFeaturesAndAdjAndSelection"

    # REWARD_TYPE = "Rigid"
    # REWARD_TYPE = "RigidAndMinEigenvalue"
    # REWARD_TYPE = "RigidAndMinRigid"
    # REWARD_TYPE = "RigidAndLogMinEigenvalueAndEdges"
    # REWARD_TYPE = "MinRigid"
    # REWARD_TYPE = "MinRigidAndMinEigenvalue"
    # REWARD_TYPE = "MinEigenvalue"
    # REWARD_TYPE = "Eigenvalues"
    # REWARD_TYPE = "EdgeCount"
    # REWARD_TYPE = "LogMinEigenvalue"
    # REWARD_TYPE = "RigidityMatrixRank"
    REWARD_TYPE = "None"

    # TERMINATION_CONDITION_TYPE = "MaxSteps"
    # TERMINATION_CONDITION_TYPE = "MaxStepsRankBonus"
    # TERMINATION_CONDITION_TYPE = "Rigid"
    # TERMINATION_CONDITION_TYPE = "RigidMinEigBonus"
    # TERMINATION_CONDITION_TYPE = "MinimallyRigid"
    TERMINATION_CONDITION_TYPE = "RigidMinEigAndEdgesBonus"
    # TERMINATION_CONDITION_TYPE = "Bandit"

    MAX_STEPS = 200

    TRUNCATE_ENABLE = True
    TRUNCATE_MAX_STEPS = 200
    TRUNCATE_PENALTY_VALUE = 5

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
    model_name = f"action{ACTION_TYPE}_obs{OBS_TYPE}_reward{REWARD_TYPE}_term{TERMINATION_CONDITION_TYPE}_{scenario_name if scenario_name is not None else n_domains}"
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
        "reward_type": REWARD_TYPE,
        "termination_condition_type": TERMINATION_CONDITION_TYPE,
        "n": n,
        "domains": domains,
        "action_rewards_enable": ACTION_REWARDS_ENABLE,
        "time_penalty_value": TIME_PENALTY_VALUE,
        "incremental_rewards_enable": INCREMENTAL_REWARDS_ENABLE,
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
