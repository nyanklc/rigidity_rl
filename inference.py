from stable_baselines3.common.env_util import make_vec_env
import copy
import numpy as np
from environment import Environment
from rigidity import rigidity_eigenvalue
import sys
import os
import time
import json
from visualizer import Visualizer
import textwrap
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from environment import Environment
import json
from datetime import datetime
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.agents.torch.dqn import DQN, DQN_CFG
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.resources.preprocessors.torch import RunningStandardScaler
from policy import *
import torch


#############################################
BRUTE_FORCE_BEST = True
#############################################
DEVICE = "cpu"
NR_ENVS = 1
#############################################

def MBR_required_edges(network):
    n = len(network.agents)
    d = 2 if network.agents[0].domain in ["R^2", "R^2xS^1"] else 3

    if d < 2 or n < 3:
        return False

    k = (n - 2) // (d - 1)
    r = (n - 2) % (d - 1)
    sgn = 1 if r > 0 else 0

    m_required = 1 + k * d + r + sgn
    return m_required


if len(sys.argv) < 3:
    print(f"usage: python3 inference.py [model_name] [environment_name]")
    quit()

model_name = sys.argv[1]
env_name = sys.argv[2]

train_json_path = f"./train/{model_name}.json"
if not os.path.exists(train_json_path):
    print(f"file {train_json_path} does not exist. Cannot determine model architecture automatically.")
    quit()

with open(train_json_path, "r") as f:
    train_info = json.load(f)

MODEL_TYPE = train_info.get("algorithm", "PPO")
MEM_SIZE = train_info.get("mem_size", 2048 * 4)

if MODEL_TYPE == "PPO":
    modelpath = "./models/complete/PPO/" + model_name + ".pt"
elif MODEL_TYPE == "DQN":
    modelpath = "./models/complete/DQN/" + model_name + ".pt"
elif MODEL_TYPE == "DDQN":
    modelpath = "./models/complete/DDQN/" + model_name + ".pt"
else:
    print(f"Unknown algorithm {MODEL_TYPE}")
    quit()

if not os.path.exists(modelpath):
    print(f"file {modelpath} does not exist")
    quit()

filepath = "./environments/" + env_name + ".json"
if not os.path.exists(filepath):
    print(f"file environments/{env_name}.json does not exist")
    quit()

with open(filepath, "r") as f:
    config = json.load(f)
    scenario_name = config.get("scenario")
    action_type = config.get("action_type")
    obs_type = config.get("obs_type")
    n = config.get("n")
    domains_str = config.get("domains", "domain").replace("^", "").replace("(", "").replace(")", "")
    n_domains = f"n{n}_{domains_str}"

torch.set_printoptions(threshold=10000)

device = "cpu"

raw_env = Environment()
raw_env.load(filepath)
raw_env.device = device
env = wrap_env(raw_env)
env.reset()

n = len(raw_env.network.agents)
node_features_dim = raw_env.observation_space["node_features"].shape[1]
edge_features_dim = raw_env.observation_space["edge_features"].shape[-1]
################################################################################
import inspect

def get_class_name(architecture_lines):
    for line in architecture_lines:
        line = line.strip()
        if line.startswith("class "):
            return line.split()[1].split("(")[0].rstrip(":")
    return None

def instantiate_model(class_name, all_kwargs):
    cls = globals()[class_name]
    sig = inspect.signature(cls.__init__)
    valid_kwargs = {k: v for k, v in all_kwargs.items() if k in sig.parameters}
    return cls(**valid_kwargs)

all_kwargs = {
    "n": n,
    "node_feat_dim": node_features_dim,
    "edge_feat_dim": edge_features_dim,
    "gnn_hidden_dim": train_info.get("gnn_hidden_dim", 32),
    "observation_space": env.observation_space,
    "action_space": env.action_space,
    "device": device
}

models = {}

if MODEL_TYPE == "PPO":
    actor_class_name = get_class_name(train_info["actor_architecture"])
    critic_class_name = get_class_name(train_info["critic_architecture"])

    actor_kwargs = all_kwargs.copy()
    actor_kwargs["head_hidden_dim"] = train_info.get("head_hidden_dim", 32)
    models["policy"] = instantiate_model(actor_class_name, actor_kwargs)

    critic_kwargs = all_kwargs.copy()
    critic_kwargs["head_hidden_dim"] = train_info.get("critic_head_hidden_dim", 32)
    models["value"] = instantiate_model(critic_class_name, critic_kwargs)

elif MODEL_TYPE == "DQN":
    q_class_name = get_class_name(train_info["q_network_architecture"])

    q_kwargs = all_kwargs.copy()
    q_kwargs["head_hidden_dim"] = train_info.get("head_hidden_dim", 32)

    models["q_network"] = instantiate_model(q_class_name, q_kwargs)
    models["target_q_network"] = copy.deepcopy(models["q_network"])



################################################################################
memory = RandomMemory(memory_size=MEM_SIZE, num_envs=NR_ENVS, device=DEVICE)

if MODEL_TYPE == "DQN":
    cfg = DQN_CFG()
    cfg.experiment.directory = "runs_inference"
    cfg.experiment.experiment_name = model_name
    cfg.batch_size = 128
    cfg.target_update_interval = 1000
    cfg.update_interval = 4
    cfg.learning_starts = MEM_SIZE + 1
    cfg.discount_factor = 0.99
    cfg.random_timesteps = MEM_SIZE

    agent = DQN(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=DEVICE,
    )

if MODEL_TYPE == "PPO":
    cfg = PPO_CFG()
    cfg.rollouts = MEM_SIZE # to ensure we don't get garbage data from memory
    cfg.experiment.directory = "runs_inference"
    cfg.experiment.experiment_name = model_name
    # incentivize exploration more
    cfg.entropy_loss_scale = 0.01

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

agent.load(modelpath)


vis = Visualizer()
button_step = vis.server.gui.add_button("step")
step_command = vis.server.gui.add_command("step_command", hotkey="space")
def wait_for_step():
    while not button_step.value:
        vis.server.flush()
        time.sleep(0.05)
    button_step.value = False
step_command.on_trigger(lambda event: setattr(button_step, 'value', True))

obs, _ = env.reset()

homogeneous_domain = raw_env.network.agents[0].domain
for ag in raw_env.network.agents:
    if (ag.domain not in ["R^2", "R^3"]) or (ag.domain != homogeneous_domain):
        print("MBR is only for homogeneous R^d network.")
        BRUTE_FORCE_BEST = False
    homogeneous_domain = ag.domain
if raw_env.network.n >= 6:
    print("Brute force with more than 5 nodes is not a good idea.")
    BRUTE_FORCE_BEST = False

vis2 = None
if BRUTE_FORCE_BEST:
    netw = copy.deepcopy(raw_env.network)

    # i != j since self loop is not needed
    n = len(netw.agents)
    all_edges = [[i, j] for i in range(n) for j in range(n) if i != j]
    print(f"ALL POSSIBLE EDGES COUNT: {len(all_edges)}")

    k = MBR_required_edges(netw)
    print(f"MBR REQUIRED EDGE COUNT: {k}")

    subsets = list(itertools.combinations(all_edges, k))

    best_min_eig = -np.inf
    best_eigs = None
    best_edges = None

    for subset in tqdm(subsets):
        edgs = list(subset)
        netw.set_edges_list(edgs)

        is_MBR, is_IBR, _ = netw.is_MBR()

        if not is_MBR:
            continue

        if not is_IBR:
            continue

        min_eig = rigidity_eigenvalue(netw)

        if min_eig > best_min_eig:
            best_min_eig = min_eig
            best_eigs = netw.eigenvalues()
            best_edges = edgs

    if best_edges is not None:
        netw.set_edges_list(best_edges)
        netw.print()
        print(f"MBR, IBR, rank: {netw.is_MBR()}")

        vis2 = Visualizer(port="6767")
        # vis2.wait_for_start()
        vis2.reset()
        vis2.draw_viser(netw)
        vis2.draw_info(
            f"BEST POSSIBLE CONFIGURATION\n"
            f"min: {best_min_eig}, eigs: {best_eigs}, edges: {best_edges}"
        )
        vis2.server.flush()
    else:
        print("No valid MBR configuration found.")


done = False
truncated = False
step_idx = 1

vis.reset()
vis.draw_viser(raw_env.network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Env")
raw_env.network.print()
while not (done or truncated):
    wait_for_step()

    action_tensor, act_outputs = agent.act(obs, states=env.state(), timestep=step_idx, timesteps=1)
    obs, reward, terminated, truncated, info = env.step(action_tensor)

    done = terminated.any().item() if torch.is_tensor(terminated) else terminated
    is_truncated = truncated.any().item() if torch.is_tensor(truncated) else truncated

    reward_val = reward.item() if torch.is_tensor(reward) else reward

    # show info
    vis.reset()
    vis.draw_viser(raw_env.network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Env")
    info_str = "".join([f"{k}: {v}\n" for k, v in info.items()]) + "\n"
    info_str += str(env.network)
    vis.draw_info(info_str)
    vis.server.flush()

    step_idx += 1

info_str = "FINISHED.\n" + info_str
vis.draw_info(info_str)
vis.server.flush()
print("Episode finished. Press Enter to close the server and exit...")
try:
    input()
except KeyboardInterrupt:
    pass

vis.stop()
if vis2 is not None:
    vis2.stop()

print("Finished.")
