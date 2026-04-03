import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import signal
from network import Network
from rigidity import is_IBR

from visualizer import Visualizer
from scenario import load_scenario
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
        ec = (n**2) // 2
        # [0, ec-1]: add, [ec, 2*ec-1]: remove, last: skip
        action_space = spaces.Discrete(2*ec + 1)

    return action_space

def define_obs_space(type: str, env: "Environment"):
    network = env.network

    obs_space = None
    if type == "Complete":
        brm = network.extended_bearing_rigidity_matrix()
        information_mat = brm.T @ brm
        u, singular_values, v = np.linalg.svd(information_mat)
        positions = np.array(
            [agent.pose.position for agent in network.agents]
        ).flatten()
        orientations_euler = np.array(
            [agent.pose.euler_angles() for agent in network.agents]
        ).flatten()

        obs_n = (
            singular_values.shape[0] + positions.shape[0] + orientations_euler.shape[0]
        )
        obs_space = spaces.Box(-np.inf, np.inf, (obs_n,))

    return obs_space

def obs(type: str, env: "Environment"):
    network = env.network

    obs = None
    if type == "Complete":
        brm = network.extended_bearing_rigidity_matrix()
        information_mat = brm.T @ brm
        # symmetric
        eigenvalues = np.linalg.eigvalsh(information_mat)
        positions = np.array(
            [agent.pose.position for agent in network.agents]
        ).flatten()
        orientations_euler = np.array(
            [agent.pose.euler_angles() for agent in network.agents]
        ).flatten()
        obs = np.hstack([eigenvalues, positions, orientations_euler])

    return obs

def reward(type: str, env: "Environment", action):
    network = env.network

    reward = 0.0
    if type == "Rigid":
        if network.is_IBR():
            reward += 10
        else:
            reward -= 10
    elif type == "RigidAndMinSingularValue":
        brm = network.extended_bearing_rigidity_matrix()
        _, singular_values, _ = np.linalg.svd(brm)
        nonzeros = singular_values[np.nonzero(singular_values)]
        min_singular = 0.0
        if len(nonzeros):
            min_singular = min(singular_values[np.nonzero(singular_values)])
        reward += min_singular

        is_rigid = network.is_IBR()
        if not is_rigid:
            reward = -1.0

    return reward


class Environment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        scenario_file,
        action_space_type="AllEdges",
        obs_space_type="Complete",
        reward_type="Rigid",
    ):
        super().__init__()

        self.filename = scenario_file
        self.action_space_type = action_space_type
        self.obs_space_type = obs_space_type
        self.reward_type = reward_type

        self.network, self.goal_network = load_scenario(self.filename)
        self.n = len(self.network.agents)
        self.m = len(self.network.edges)

        self.brm = self.network.extended_bearing_rigidity_matrix()

        self.observation_space = define_obs_space(obs_space_type, self)
        self.action_space = define_action_space(action_space_type, self)
        self._get_obs = lambda: obs(obs_space_type, self)
        self._compute_reward = lambda action: reward(reward_type, self, action)
        self.last_reward = 0.0

    # -----------------------------------
    def step(self, action):
        # take action
        if self.action_space_type == "AllEdges":
            n = len(self.network.agents)
            action_adj = action.reshape((n, n))
            i_indices = []
            j_indices = []
            for i in range(n):
                for j in range(n):
                    if action_adj[i, j]:
                        if i != j:
                            i_indices.append(int(i))
                            j_indices.append(int(j))
            self.network.set_edges(i_indices, j_indices)
        elif self.action_space_type == "AddRemoveEdgeMultiDiscrete":
            # TODO: punish unnecessary additions/removals
            if action[0] == 0:
            # add
                i_idx = action[1]
                j_idx = action[2]
                self.network.add_edge(i_idx, j_idx)
            elif action[0] == 1:
            # remove
                i_idx = action[1]
                j_idx = action[2]
                self.network.remove_edge(i_idx, j_idx)
            elif action[0] == 2:
            # skip
                pass
        elif self.action_space_type == "AddRemoveEdgeDiscrete":
            ec = (n**2) // 2
            if action == 2 * ec:
                # skip
                pass
            if action < ec:
                # add
                edge_idx = action
                i_idx, j_idx = np.triu_indices(n, k=1)
                i_idx = i_idx[edge_idx]
                j_idx = j_idx[edge_idx]
                self.network.add_edge(i_idx, j_idx)
            else:
                # remove
                edge_idx = action - ec
                i_idx, j_idx = np.triu_indices(n, k=1)
                i_idx = i_idx[edge_idx]
                j_idx = j_idx[edge_idx]
                self.network.remove_edge(i_idx, j_idx)

        # obs/reward
        obs = self._get_obs()

        reward = self._compute_reward(action)
        tmp = self.last_reward
        self.last_reward = reward
        reward = reward - tmp

        info = {
            "action": action,
            "singular_values": obs[:min(3*self.m, 6*self.n)],
            "is_rigid": self.network.is_IBR(),
            "nr_edges": len(self.network.edges),
            "reward": reward,
        }
        truncated = False
        terminated = True

        return obs, reward, terminated, truncated, info

    # -----------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.network, self.goal_network = load_scenario(self.filename)

        return self._get_obs(), {}
