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
    # add
    if action[0] == 0:
        i_idx = action[1]
        j_idx = action[2]

        action_info += f"add {i_idx}-{j_idx}"
        if i_idx == j_idx:
            action_info += " (self loop)"

        if env.network.edge_exists(i_idx, j_idx):
            reward -= 20 # unnecessary action
            action_info += " (existed)"

        if i_idx != j_idx:
            env.network.add_edge(i_idx, j_idx)
            reward -= 1 # measurement effort
    # remove
    elif action[0] == 1:
        i_idx = action[1]
        j_idx = action[2]

        action_info += f"remove {i_idx}-{j_idx}"
        if i_idx == j_idx:
            action_info += " (self loop)"

        if not env.network.edge_exists(i_idx, j_idx):
            reward -= 20 # unnecessary action
            action_info += " (didn't exist)"

        if i_idx != j_idx:
            env.network.remove_edge(i_idx, j_idx)
            reward += 10 # measurement effort
    # skip
    elif action[0] == 2:
        action_info += "skip"
        pass

    # reward -= 5 # time taken

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
            action_info += " (self loop)"

        if env.network.edge_exists(i_idx, j_idx):
            reward -= 20 # unnecessary action
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
            reward -= 20 # unnecessary action
            action_info += " (didn't exist)"

        if i_idx != j_idx:
            env.network.remove_edge(i_idx, j_idx)
            reward += 10 # measurement effort

    # reward -= 5 # time taken

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
            action_info += " (self loop)"

        if env.network.edge_exists(i_idx, j_idx):
            reward -= 20 # unnecessary action
            action_info += " (existed)"

        if i_idx != j_idx:
            env.network.add_edge(i_idx, j_idx)
            reward -= 1 # measurement effort

    reward -= 5 # time taken

    return reward, action_info

def obs(type: str, env: "Environment", define_type=False):
    obs_space = None

    network = env.network

    obs = None
    if type == "Complete":
        A = env.network.edges.astype(np.float32)
        eigenvalues = env.network.eigenvalues()
        positions = np.array(
            [agent.pose.position for agent in network.agents]
        ).flatten()
        orientations_euler = np.array(
            [agent.pose.euler_angles() for agent in network.agents]
        ).flatten()
        obs = np.hstack([A.flatten(), eigenvalues, positions, orientations_euler])
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
        action_rewards_enable=False,
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
        self.arg_action_rewards_enable = action_rewards_enable
        self.arg_max_steps = max_steps
        self.arg_filepath = filepath
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

        self.action_rewards_enable = action_rewards_enable

        self.was_IBR = None
        self.was_MBR = None

    # -----------------------------------
    def step(self, action):
        reward = 0.0
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

        if self.action_rewards_enable:
            reward, action_info = action_return
        else:
            _, action_info = action_return

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
            min_eig = min(self.network.eigenvalues())
            reward += min_eig
            punish = 100
            if not is_IBR:
                reward -= punish
        elif self.reward_type == "RigidAndMinRigid":
            if is_IBR:
                reward += 10
                if is_MBR:
                    reward += 10
                else:
                    reward -= 10
            else:
                reward -= 10
        elif self.reward_type == "RigidAndMinRigidAndMinEigenvalue":
            min_eig = min(self.network.eigenvalues())
            reward += min_eig
            punish = 100
            if not is_IBR:
                reward -= punish
            if not is_MBR:
                reward -= punish
        elif self.reward_type == "MinEigenvalue":
            min_eig = min(self.network.eigenvalues())
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
                reward += self.network.nr_max_edges * 10
                # reward += 100
                terminated = True
        elif self.termination_condition_type == "Bandit":
            if self.step_counter >= 1:
                terminated = True

        termination_reward = reward - state_reward - action_reward

        # (incremental) reward
        last_reward_copy = self.last_reward
        self.last_reward = reward
        # TODO: do we want this? or perhaps add a flag
        # reward = reward - last_reward_copy

        # debug
        eigs = self.network.eigenvalues()
        info = {
            "step": f"{self.step_counter}",
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
            "min eigenvalue": eigs[0],
        }
        # print(info)

        self.was_IBR = is_IBR
        self.was_MBR = is_MBR

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
            # self.network, self.goal_network = random_scenario(self.arg_n, self.arg_domains)

            # TODO: with "Complete" observations, this doesn't make sense since the pos/orient stay the same
            # just randomize the edges
            # TODO: create flags to handle the network reset.
            # depending on how we want to train, we may want to randomize only the
            # poses and remove all edges for instance (e.g. empty scenario with AllEdges actions).
            redgs = np.random.choice(a=[False, True], size=(self.network.n, self.network.n), p=[0.5, 0.5])
            self.network.set_edges(redgs)

        self.n = len(self.network.agents)
        self.m = int(self.network.edges.sum())

        self.brm = self.network.extended_bearing_rigidity_matrix()

        self.nr_max_edges = self.n**2
        self.step_counter = 0
        self.max_steps = self.arg_max_steps

        self.last_reward = 0

        self.was_IBR = None
        self.was_MBR = None

        return self._get_obs(), {}


if __name__ == "__main__":
    #############################################
    ACTION_TYPE = "AllEdges"
    # ACTION_TYPE = "AddRemoveEdgeMultiDiscrete"
    # ACTION_TYPE = "AddRemoveEdgeDiscrete"
    # ACTION_TYPE = "AddEdgeDiscrete"

    ACTION_REWARDS_ENABLE = True
    # ACTION_REWARDS_ENABLE = False

    OBS_TYPE = "Complete"
    # OBS_TYPE = "AdjFlatAndEigenvalues"

    # REWARD_TYPE = "Rigid"
    # REWARD_TYPE = "RigidAndMinEigenvalue"
    REWARD_TYPE = "RigidAndMinRigid"
    # REWARD_TYPE = "RigidAndMinRigidAndMinEigenvalue"
    # REWARD_TYPE = "MinEigenvalue"

    TERMINATION_CONDITION_TYPE = "MaxSteps"
    # TERMINATION_CONDITION_TYPE = "Rigid"
    # TERMINATION_CONDITION_TYPE = "MinimallyRigid"
    # TERMINATION_CONDITION_TYPE = "Bandit"

    MAX_STEPS = 10
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

    # just to make sure everything works as intended
    _ = Environment(
        n,
        domains,
        action_space_type=ACTION_TYPE,
        obs_space_type=OBS_TYPE,
        reward_type=REWARD_TYPE,
        termination_condition_type=TERMINATION_CONDITION_TYPE,
        action_rewards_enable=ACTION_REWARDS_ENABLE,
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
        "action_rewards_enable": ACTION_REWARDS_ENABLE,
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
