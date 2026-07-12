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
from skrl.agents.torch.dqn import DQN, DQN_CFG
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.resources.preprocessors.torch import RunningStandardScaler
from policy import *
import torch


#############################################
MODEL_TYPE = "DQN"
BRUTE_FORCE_BEST = True
#############################################
DEVICE = "cpu"
#############################################

# these should be the same as the training

# PPO
if MODEL_TYPE == "PPO":
    NR_ENVS = 1
    MEM_SIZE = 2048 * 4
    GNN_HIDDEN_DIM = 32
    ACTOR_HEAD_HIDDEN_DIM = 32
    CRITIC_HEAD_HIDDEN_DIM = 32

# DQN
if MODEL_TYPE == "DQN":
    NR_ENVS = 1
    MEM_SIZE = 20000
    GNN_HIDDEN_DIM = 128
    QNETWORK_HEAD_HIDDEN_DIM = 256

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
if MODEL_TYPE == "PPO":
    modelpath = "./models/complete/PPO/" + model_name + ".pt"
    if not os.path.exists(modelpath):
        print(f"file models/complete/PPO/{model_name}.pt does not exist")
        quit()
if MODEL_TYPE == "DQN":
    modelpath = "./models/complete/DQN/" + model_name + ".pt"
    if not os.path.exists(modelpath):
        print(f"file models/complete/DQN/{model_name}.pt does not exist")
        quit()
if MODEL_TYPE == "DDQN":
    modelpath = "./models/complete/DDQN/" + model_name + ".pt"
    if not os.path.exists(modelpath):
        print(f"file models/complete/DDQN/{model_name}.pt does not exist")
        quit()

env_name = sys.argv[2]
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


# PPO
if MODEL_TYPE == "PPO":
    models = {}
    # actor
    if action_type == "AddEdgeDiscreteNoSkipNoSelfLoops":
        models["policy"] = PPO_ActorModel_AddEdgeDiscreteNoSkipNoSelfLoops(
            n,
            node_feat_dim=node_features_dim,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    elif action_type == "AddRemoveEdgeDiscreteNoSelfLoops":
        # models["policy"] = PPO_ActorModel_AddRemoveEdgeDiscreteNoSelfLoops_FC(
        #     n,
        #     node_feat_dim=node_features_dim,
        #     gnn_hidden_dim=GNN_HIDDEN_DIM,
        #     head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
        #     observation_space=env.observation_space,
        #     action_space=env.action_space,
        #     device=device,
        # )
        models["policy"] = PPO_ActorModel_AddRemoveEdgeDiscreteNoSelfLoops(
            n,
            node_feat_dim=node_features_dim,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    elif action_type == "AddRemoveEdgeMultiDiscrete":
        models["policy"] = PPO_ActorModel_AddRemoveEdgeMultiDiscrete(
            n,
            node_feat_dim=node_features_dim,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    elif action_type == "SelectNodesSequentially":
        if obs_type == "DictEquivariantNodeFeaturesAndAdjAndSelection":
            models["policy"] = PPO_ActorModel_Equivariant_SelectNodesSequentially(
                n,
                node_feat_dim=node_features_dim,
                gnn_hidden_dim=GNN_HIDDEN_DIM,
                head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
            )
        elif obs_type == "DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection":
            models["policy"] = PPO_ActorModel_GINE_SelectNodesSequentially(
                n,
                node_feat_dim=node_features_dim,
                edge_feat_dim=edge_features_dim,
                gnn_hidden_dim=GNN_HIDDEN_DIM,
                head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
            )
        else:
            models["policy"] = PPO_ActorModel_SelectNodesSequentially(
                n,
                node_feat_dim=node_features_dim,
                gnn_hidden_dim=GNN_HIDDEN_DIM,
                head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
            )
    elif action_type == "DecideOnEdge":
        models["policy"] = PPO_ActorModel_DecideOnEdge(
            n,
            node_feat_dim=node_features_dim,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    else:
        print(f"Actor for action {action_type} is not implemented.")
        quit()

    # critic
    if obs_type == "DictNodeFeaturesAndAdjAndSelection":
        models["value"] = PPO_CriticModel_Selection(
            n,
            node_feat_dim=node_features_dim,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            head_hidden_dim=CRITIC_HEAD_HIDDEN_DIM,

            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    elif obs_type == "DictEquivariantNodeFeaturesAndAdjAndSelection":
        models["value"] = PPO_CriticModel_Equivariant_Selection(
            n,
            node_feat_dim=node_features_dim,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            head_hidden_dim=CRITIC_HEAD_HIDDEN_DIM,

            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    elif obs_type == "DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection":
        models["value"] = PPO_CriticModel_GINE_Selection(
            n,
            node_feat_dim=node_features_dim,
            edge_feat_dim=edge_features_dim,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            head_hidden_dim=CRITIC_HEAD_HIDDEN_DIM,

            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    else:
        models["value"] = PPO_CriticModel_Default(
            n,
            node_feat_dim=node_features_dim,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            head_hidden_dim=CRITIC_HEAD_HIDDEN_DIM,

            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )


# DQN
if MODEL_TYPE == "DQN":
    models = {}
    # q network
    if action_type == "AddRemoveEdgeDiscreteNoSelfLoops":
        if obs_type == "DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection":
            models["q_network"] = DQN_QNetwork_GINE_AddRemoveEdgeDiscreteNoSelfLoops(
                n,
                node_feat_dim=node_features_dim,
                edge_feat_dim=edge_features_dim,
                gnn_hidden_dim=GNN_HIDDEN_DIM,
                head_hidden_dim=QNETWORK_HEAD_HIDDEN_DIM,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=DEVICE,
            )
        elif obs_type == "DictEquivariantNodeFeaturesAndAdjAndSelection":
            models["q_network"] = DQN_QNetwork_Equivariant_AddRemoveEdgeDiscreteNoSelfLoops(
                n,
                node_feat_dim=node_features_dim,
                edge_feat_dim=edge_features_dim,
                gnn_hidden_dim=GNN_HIDDEN_DIM,
                head_hidden_dim=QNETWORK_HEAD_HIDDEN_DIM,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=DEVICE,
            )
        else:
            models["q_network"] = DQN_QNetwork_AddRemoveEdgeDiscreteNoSelfLoops(
                n,
                node_feat_dim=node_features_dim,
                gnn_hidden_dim=GNN_HIDDEN_DIM,
                head_hidden_dim=QNETWORK_HEAD_HIDDEN_DIM,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=DEVICE,
            )
    elif action_type == "AddEdgeDiscreteNoSelfLoops":
        models["q_network"] = DQN_QNetwork_AddEdgeDiscreteNoSelfLoops(
            n,
            node_feat_dim=node_features_dim,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            head_hidden_dim=QNETWORK_HEAD_HIDDEN_DIM,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=DEVICE,
        )
    elif action_type == "SelectNodesSequentially":
        if obs_type == "DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection":
            models["q_network"] = DQN_QNetwork_GINE_SelectNodesSequentially(
                n,
                node_feat_dim=node_features_dim,
                edge_feat_dim=edge_features_dim,
                gnn_hidden_dim=GNN_HIDDEN_DIM,
                head_hidden_dim=QNETWORK_HEAD_HIDDEN_DIM,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=DEVICE,
            )
        elif obs_type == "DictEquivariantNodeFeaturesAndAdjAndSelection":
            models["q_network"] = DQN_QNetwork_Equivariant_SelectNodesSequentially(
                n,
                node_feat_dim=node_features_dim,
                gnn_hidden_dim=GNN_HIDDEN_DIM,
                edge_feat_dim=edge_features_dim,
                head_hidden_dim=QNETWORK_HEAD_HIDDEN_DIM,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=DEVICE,
            )
        else:
            models["q_network"] = DQN_QNetwork_SelectNodesSequentially(
                n,
                node_feat_dim=node_features_dim,
                gnn_hidden_dim=GNN_HIDDEN_DIM,
                head_hidden_dim=QNETWORK_HEAD_HIDDEN_DIM,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=DEVICE,
            )
    elif action_type == "AddEdgeDiscreteNoSkipNoSelfLoops":
        if obs_type == "DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection":
            models["q_network"] = DQN_QNetwork_GINE_AddEdgeDiscreteNoSkipNoSelfLoops(
                    n,
                    node_feat_dim=node_features_dim,
                    edge_feat_dim=edge_features_dim,
                    gnn_hidden_dim=GNN_HIDDEN_DIM,
                    head_hidden_dim=QNETWORK_HEAD_HIDDEN_DIM,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=DEVICE,
                )
        elif obs_type == "DictEquivariantNodeFeaturesAndAdjAndSelection":
            models["q_network"] = DQN_QNetwork_Equivariant_AddEdgeDiscreteNoSkipNoSelfLoops(
                n,
                node_feat_dim=node_features_dim,
                edge_feat_dim=edge_features_dim,
                gnn_hidden_dim=GNN_HIDDEN_DIM,
                head_hidden_dim=QNETWORK_HEAD_HIDDEN_DIM,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=DEVICE,
            )
        else:
            raise Exception(f"Not implemented {action_type} {obs_type}")
    else:
        print(f"Q network for {raw_env.action_space_type} is not implemented.")
        quit()
    # target
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

obs, _ = env.reset()
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

time.sleep(2)
vis.server.flush()

vis.stop()
if vis2 is not None:
    vis2.stop()

print("Finished.")
