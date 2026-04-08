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

    return action_space

def action_AllEdges(action, env: "Environment", reward, action_info):
    # TODO: handle reward
    n = len(env.network.agents)
    action_adj = action.reshape((n, n))
    i_indices = []
    j_indices = []
    for i in range(n):
        for j in range(n):
            if action_adj[i, j]:
                if i != j:
                    i_indices.append(int(i))
                    j_indices.append(int(j))
    env.network.set_edges(i_indices, j_indices)

    return reward, action_info

def action_AddRemoveEdgeMultiDiscrete(action, env: "Environment", reward, action_info):
    # add
    if action[0] == 0:
        i_idx = action[1]
        j_idx = action[2]
        action_info += f"add {i_idx}-{j_idx}"
        if i_idx == j_idx:
            reward -= 5 # adding memory
            action_info += " (self loop)"
        if env.network.edge_exists(i_idx, j_idx):
            reward -= 20 # unnecessary action
            action_info += " (existed)"
        env.network.add_edge(i_idx, j_idx)
        reward -= 1 # measurement effort
    # remove
    elif action[0] == 1:
        i_idx = action[1]
        j_idx = action[2]
        action_info += f"remove {i_idx}-{j_idx}"
        if i_idx == j_idx:
            reward += 5 # removing memory
            action_info += " (self loop)"
        if not env.network.edge_exists(i_idx, j_idx):
            reward -= 20 # unnecessary action
            action_info += " (didn't exist)"
        env.network.remove_edge(i_idx, j_idx)
        reward += 1 # measurement effort
    # skip
    elif action[0] == 2:
        action_info += "skip"
        pass

    reward -= 5 # time taken

    return reward, action_info

def action_AddRemoveEdgeDiscrete(action, env: "Environment", reward, action_info):
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
            reward -= 5 # adding memory
            action_info += " (self loop)"
        if env.network.edge_exists(i_idx, j_idx):
            reward -= 20 # unnecessary action
            action_info += " (existed)"
        env.network.add_edge(i_idx, j_idx)
        reward -= 1 # measurement effort
    else:
        # remove
        i_idx = (action-ec) // n
        j_idx = (action-ec) % n
        action_info += f"remove {i_idx}-{j_idx}"
        if i_idx == j_idx:
            reward += 5 # removing memory
            action_info += " (self loop)"
        if not env.network.edge_exists(i_idx, j_idx):
            reward -= 20 # unnecessary action
            action_info += " (didn't exist)"
        env.network.remove_edge(i_idx, j_idx)
        reward += 1 # measurement effort

    reward -= 5 # time taken

    return reward, action_info

def action_AddEdgeDiscrete(action, env: "Environment", reward, action_info):
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
            reward -= 5 # adding memory
            action_info += " (self loop)"
        if env.network.edge_exists(i_idx, j_idx):
            reward -= 20 # unnecessary action
            action_info += " (existed)"
        env.network.add_edge(i_idx, j_idx)
        reward -= 1 # measurement effort

    reward -= 5 # time taken

    return reward, action_info

def obs(type: str, env: "Environment", define_type=False):
    obs_space = None

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
        if define_type:
            obs_n = obs.shape[0]
            obs_space = spaces.Box(-np.inf, np.inf, (obs_n,))
    elif type == "AdjFlatAndEigenvalues":
        n = len(env.network.agents)
        A = env.network.edges.astype(np.float32)
        brm = network.extended_bearing_rigidity_matrix()
        information_mat = brm.T @ brm
        eigenvalues = np.linalg.eigvalsh(information_mat)
        obs = np.hstack([A.flatten(), eigenvalues])
        if define_type:
            obs_n = obs.shape[0]
            obs_space = spaces.Box(-np.inf, np.inf, (obs_n,))

    return obs, obs_space


class Environment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        n,
        domains,
        action_space_type="AllEdges",
        obs_space_type="Complete",
        reward_type="Rigid",
        termination_condition_type="MaxSteps",
        max_steps=1e4,
        filepath=None
    ):
        super().__init__()

        ###############################
        # to use in reset because i'm lazy
        self.arg_n = n
        self.arg_domains = domains
        self.arg_action_space_type = action_space_type
        self.arg_obs_space_type = obs_space_type
        self.arg_reward_type = reward_type
        self.arg_termination_condition_type = termination_condition_type
        self.arg_max_steps = max_steps
        ###############################


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

        _, self.observation_space = obs(obs_space_type, self, define_type=True)
        self.action_space = define_action_space(action_space_type, self)
        self._get_obs = lambda: obs(obs_space_type, self, False)[0]

        self.nr_max_edges = self.n**2
        self.step_counter = 0
        self.max_steps = max_steps

        self.last_reward = 0

    # -----------------------------------
    def step(self, action):
        reward = 0.0
        n = len(self.network.agents)

        was_IBR = self.network.is_IBR()
        was_MBR = self.network.is_MBR()
        action_info = ""

        # action and reward based on action
        if self.action_space_type == "AllEdges":
            reward, action_info = action_AllEdges(action, self, reward, action_info)
        elif self.action_space_type == "AddRemoveEdgeMultiDiscrete":
            reward, action_info = action_AddRemoveEdgeMultiDiscrete(action, self, reward, action_info)
        elif self.action_space_type == "AddRemoveEdgeDiscrete":
            reward, action_info = action_AddRemoveEdgeDiscrete(action, self, reward, action_info)
        elif self.action_space_type == "AddEdgeDiscrete":
            reward, action_info = action_AddEdgeDiscrete(action, self, reward, action_info)

        action_reward = reward

        # obs
        obs = self._get_obs()

        # reward based on state
        is_IBR = self.network.is_IBR()
        is_MBR = self.network.is_MBR()
        if self.reward_type == "Rigid":
            if is_IBR:
                reward += 10
            else:
                reward -= 10
        elif self.reward_type == "RigidAndMinEigenvalue":
            brm = self.network.extended_bearing_rigidity_matrix()
            information_mat = brm.T @ brm
            # symmetric
            eigenvalues = np.linalg.eigvalsh(information_mat)
            nonzeros = eigenvalues[np.nonzero(eigenvalues)]
            min_eig = 0.0
            if len(nonzeros):
                min_eig = min(eigenvalues[np.nonzero(eigenvalues)])
            reward += 10 + min_eig
            if not is_IBR:
                reward += -10
        elif self.reward_type == "RigidAndMinRigid":
            if is_IBR:
                reward += 10
                if is_MBR:
                    reward += 10
                else:
                    reward -= 10
            else:
                reward -= 20
        elif self.reward_type == "MinEigenvalue":
            brm = self.network.extended_bearing_rigidity_matrix()
            information_mat = brm.T @ brm
            # symmetric
            eigenvalues = np.linalg.eigvalsh(information_mat)
            nonzeros = eigenvalues[np.nonzero(eigenvalues)]
            min_eig = 0.0
            if len(nonzeros):
                min_eig = min(eigenvalues[np.nonzero(eigenvalues)])
            reward += min_eig

        state_reward = reward - action_reward

        self.step_counter += 1

        # termination conditions
        truncated = False
        terminated = False
        if self.termination_condition_type == "MaxSteps":
            if self.step_counter >= self.max_steps:
                terminated = True
        elif self.termination_condition_type == "Rigid":
            if is_IBR:
                reward += self.network.nr_max_edges * 10
                # reward += 100
                terminated = True
        elif self.termination_condition_type == "MinimallyRigid":
            if is_MBR:
                self.network.nr_max_edges * 10
                # reward += 100
                terminated = True

        termination_reward = reward - state_reward - action_reward

        # (incremental) reward
        last_reward_copy = self.last_reward
        self.last_reward = reward
        # TODO: do we want this? or perhaps add a flag
        # reward = reward - last_reward_copy

        # debug
        info = {
            "step": f"{self.step_counter}",
            "action": action_info,
            "reward (step)": reward,
            "reward (raw)": self.last_reward,
            "reward (action)": action_reward,
            "reward (state)": state_reward,
            "reward (termination)": termination_reward,
            "last reward": last_reward_copy,
            "is rigid": is_IBR,
            "was rigid": was_IBR,
            "is min rigid": is_MBR,
            "was min rigid": was_MBR,
            "nr edges": int(self.network.edges.sum()),
            "terminated": terminated,
            "truncated": truncated,
        }
        # print(info)
        return obs, reward, terminated, truncated, info

    # -----------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.action_space_type = self.arg_action_space_type
        self.obs_space_type = self.arg_obs_space_type
        self.reward_type = self.arg_reward_type

        if self.filepath:
            self.network, self.goal_network = load_scenario(self.filepath)
        else:
            self.network, self.goal_network = random_scenario(self.arg_n, self.arg_domains)

        self.n = len(self.network.agents)
        self.m = int(self.network.edges.sum())

        self.brm = self.network.extended_bearing_rigidity_matrix()

        self.nr_max_edges = self.n**2
        self.step_counter = 0
        self.max_steps = self.arg_max_steps

        self.last_reward = 0

        return self._get_obs(), {}


if __name__ == "__main__":
    #############################################
    # ACTION_TYPE = "AllEdges"
    # ACTION_TYPE = "AddRemoveEdgeMultiDiscrete"
    ACTION_TYPE = "AddRemoveEdgeDiscrete"
    # ACTION_TYPE = "AddEdgeDiscrete"

    # OBS_TYPE = "Complete"
    OBS_TYPE = "AdjFlatAndEigenvalues"

    # REWARD_TYPE = "Rigid"
    # REWARD_TYPE = "RigidAndMinEigenvalue"
    REWARD_TYPE = "RigidAndMinRigid"
    # REWARD_TYPE = "MinEigenvalue"

    # TERMINATION_CONDITION_TYPE = "MaxSteps"
    # TERMINATION_CONDITION_TYPE = "Rigid"
    TERMINATION_CONDITION_TYPE = "MinimallyRigid"

    MAX_STEPS = 1e4
    #############################################

    if len(sys.argv) < 3:
        print("Usage: python3 environment.py [n] [domains] or python3 environment.py file [scenario_name]")
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

    domains_str = domains
    domains_str = domains_str.replace("^", "").replace("(", "").replace(")", "")

    now = datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H_%M_%S")

    n_domains = f"n{n}_{domains_str}"
    model_name = f"{now_str}_action{ACTION_TYPE}_obs{OBS_TYPE}_reward{REWARD_TYPE}_term{TERMINATION_CONDITION_TYPE}_{scenario_name if scenario_name is not None else n_domains}"
    print(f"MODEL NAME: {model_name}")

    log_dir = "./tboard_logs/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("./models/", exist_ok=True)

    #########################

    if filepath is not None:
        print(f"loading environment from scenario {filepath}")
    else:
        print(f"creating environment with n={n}, domains={domains}")

    # just to make sure everything works as intended
    _ = Environment(
        n,
        domains,
        action_space_type=ACTION_TYPE,
        obs_space_type=OBS_TYPE,
        reward_type=REWARD_TYPE,
        termination_condition_type=TERMINATION_CONDITION_TYPE,
        max_steps=MAX_STEPS,
        filepath=filepath,
    )

    env_config = {
        "action_type": ACTION_TYPE,
        "obs_type": OBS_TYPE,
        "reward_type": REWARD_TYPE,
        "termination_condition_type": TERMINATION_CONDITION_TYPE,
        "n": n,
        "domains": domains,
        "max_steps": MAX_STEPS,
        "scenario": scenario_name,
    }
    env_filename = f"env_{model_name}.json"
    env_path = os.path.join("./environments/", env_filename)
    with open(env_path, "w") as f:
        os.makedirs("./environments/", exist_ok=True)
        json.dump(env_config, f, indent=2)
        print(f"SAVED: {env_path}")
        print(f"env: env_{model_name}")
