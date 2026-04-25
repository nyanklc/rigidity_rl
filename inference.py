from stable_baselines3.common.env_util import make_vec_env
import copy
import numpy as np
from environment import Environment
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
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.resources.preprocessors.torch import RunningStandardScaler
from policy import *


#############################################
BRUTE_FORCE_BEST = True

# these should be the same as the training
NR_ENVS = 1
MEM_SIZE = 2048
GNN_HIDDEN_DIM = 32
ACTOR_HEAD_HIDDEN_DIM = 128
CRITIC_HEAD_HIDDEN_DIM = 128
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
modelpath = "./models/complete/" + model_name + ".pt"
if not os.path.exists(modelpath):
    print(f"file models/complete/{model_name}.pt does not exist")
    quit()

env_name = sys.argv[2]
filepath = "./environments/" + env_name + ".json"
if not os.path.exists(filepath):
    print(f"file environments/{env_name}.json does not exist")
    quit()

torch.set_printoptions(threshold=10000)

device = "cpu"

raw_env = Environment()
raw_env.load(filepath)
raw_env.device = device
env = wrap_env(raw_env)
env.reset()

n = len(raw_env.network.agents)
node_features_dim = raw_env.observation_space["node_features"].shape[1]

models = {}
# actor
if raw_env.action_space_type == "AddEdgeDiscreteNoSkipNoSelfLoops":
    models["policy"] = ActorModel_AddEdgeDiscreteNoSkipNoSelfLoops(
        n,
        node_feat_dim=node_features_dim,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
elif raw_env.action_space_type == "AddRemoveEdgeDiscreteNoSelfLoops":
    models["policy"] = ActorModel_AddRemoveEdgeDiscreteNoSelfLoops_FC(
        n,
        node_feat_dim=node_features_dim,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
else:
    print(f"Actor for action {raw_env.action_space_type} is not implemented.")
    quit()

# critic
models["value"] = CriticModel(
    n,
    node_feat_dim=node_features_dim,
    gnn_hidden_dim=GNN_HIDDEN_DIM,
    head_hidden_dim=CRITIC_HEAD_HIDDEN_DIM,

    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

memory = RandomMemory(memory_size=MEM_SIZE, num_envs=NR_ENVS, device=device)
cfg = PPO_CFG()
cfg.rollouts = MEM_SIZE # to ensure we don't get garbage data from memory
cfg.experiment.directory = "runs_inference"
cfg.experiment.experiment_name = model_name


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
def wait_for_step():
    while not button_step.value:
        vis.server.flush()
        time.sleep(0.05)
    button_step.value = False

if raw_env.network.agents[0].domain not in ["R^2", "R^3"]:
    print("MBR is only for homogeneous R^d network.")
    BRUTE_FORCE_BEST = False
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

        is_MBR, is_IBR = netw.is_MBR()
        if not is_MBR:
            continue

        if not is_IBR:
            continue

        brm = netw.extended_bearing_rigidity_matrix()
        information_mat = brm.T @ brm
        eigenvalues = np.linalg.eigvalsh(information_mat)

        min_eig = eigenvalues.min()

        if min_eig > best_min_eig:
            best_min_eig = min_eig
            best_eigs = eigenvalues
            best_edges = edgs

    if best_edges is not None:
        netw.set_edges_list(best_edges)
        netw.print()
        print(f"IBR: {netw.is_IBR()}")
        print(f"MBR: {netw.is_MBR()[0]}")

        vis2 = Visualizer(port="6767")
        # vis2.wait_for_start()
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

vis.draw_viser(raw_env.network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Env")
raw_env.network.print()
obs, _ = env.reset()
while not (done or truncated):
    wait_for_step()

    action_tensor, act_outputs = agent.act(obs, states=env.state(), timestep=step_idx, timesteps=1)
    obs, reward, terminated, truncated, info = env.step(action_tensor)

    done = terminated.any().item() if torch.is_tensor(terminated) else terminated
    is_truncated = truncated.any().item() if torch.is_tensor(truncated) else truncated

    reward_val = reward.item() if torch.is_tensor(reward) else reward

    # show info
    vis.draw_viser(raw_env.network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Env")
    vis.draw_info("".join([f"{k}: {v}\n" for k, v in info.items()]))
    vis.server.flush()

    step_idx += 1

time.sleep(2)
vis.server.flush()

vis.stop()
if vis2 is not None:
    vis2.stop()

print("Finished.")
