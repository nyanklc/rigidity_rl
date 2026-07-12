from environment import Environment
import os
import sys
import json
import torch
import numpy as np
import copy
import gymnasium as gym
from datetime import datetime
import skrl

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.dqn import DQN, DQN_CFG
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from policy import *

######################################
TOTAL_TIMESTEPS = int(3e5)
NR_ENVS = 1
MEM_SIZE = 20000
EGREEDY_STEPS = 100000

GNN_HIDDEN_DIM = 128
QNETWORK_HEAD_HIDDEN_DIM = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
##################

if len(sys.argv) < 3:
    print(f"usage: python3 train.py [environment_name] [model_name]")
    quit()

model_name_prefix = sys.argv[2]
filename = sys.argv[1]
filepath = "./environments/" + filename + ".json"

if not os.path.exists(filepath):
    print(f"file environments/{filename}.json does not exist")
    quit()

with open(filepath, "r") as f:
    config = json.load(f)
    scenario_name = config.get("scenario")
    action_type = config.get("action_type")
    obs_type = config.get("obs_type")
    n = config.get("n")
    domains_str = config.get("domains", "domain").replace("^", "").replace("(", "").replace(")", "")
    n_domains = f"n{n}_{domains_str}"

model_name = (
    model_name_prefix
    + f"_action{action_type}_obs{obs_type}_{scenario_name if scenario_name is not None else n_domains}"
)

def make_env(i):
    e = Environment()
    e.load(filepath)
    writer_name = model_name if i == 0 else f"{model_name}-{i}"
    e.set_writer(writer_name)
    e.device = DEVICE
    return e

# FIX: Replaced AsyncVectorEnv with SyncVectorEnv to eliminate RNG duplication
raw_env = gym.vector.SyncVectorEnv([lambda idx=i: make_env(idx) for i in range(NR_ENVS)])
env = wrap_env(raw_env)

node_features_dim = raw_env.single_observation_space["node_features"].shape[1]
edge_features_dim = raw_env.single_observation_space["edge_features"].shape[-1]

models = {}

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

models["target_q_network"] = copy.deepcopy(models["q_network"])


env.action_space.seed(int(datetime.now().timestamp()))
env.observation_space.seed(int(datetime.now().timestamp()))
# env.state_space.seed(int(datetime.now().timestamp()))

memory = RandomMemory(memory_size=MEM_SIZE, num_envs=env.num_envs, device=DEVICE)

cfg = DQN_CFG()
cfg.experiment.directory = "runs"
cfg.experiment.experiment_name = model_name
cfg.batch_size = 128
cfg.target_update_interval = 1000
cfg.update_interval = 4
cfg.learning_starts = MEM_SIZE + 1
cfg.discount_factor = 0.99
cfg.random_timesteps = MEM_SIZE

def epsilon_schedule(timestep, timesteps):
    start = 0.8
    end = 0.05
    decay_steps = min(EGREEDY_STEPS, timesteps)
    eps = start - (start - end) * min(1.0, timestep / decay_steps)
    return eps
cfg.exploration_scheduler = epsilon_schedule

os.makedirs("./models", exist_ok=True)
os.makedirs("./models/complete/DQN", exist_ok=True)

agent = DQN(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=DEVICE,
)

trainer_cfg = SequentialTrainerCfg()
trainer_cfg.timesteps = TOTAL_TIMESTEPS
trainer_cfg.headless = True
trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

import dataclasses
import pprint
print("##########################################")
print(" TRAINING ")
print("="*40)
print(f"obs space: {trainer.env.observation_space}")
print(f"action space: {trainer.env.action_space}")
print(f"model: {models["q_network"].__class__.__name__}")
print(f"TOTAL_TIMESTEPS: {TOTAL_TIMESTEPS}")
print(f"NR_ENVS: {NR_ENVS}")
print(f"MEM_SIZE: {MEM_SIZE}")
print(f"EGREEDY_STEPS: {EGREEDY_STEPS}")
print(f"GNN_HIDDEN_DIM: {GNN_HIDDEN_DIM}")
print(f"QNETWORK_HEAD_HIDDEN_DIM: {QNETWORK_HEAD_HIDDEN_DIM}")
print("\n" + "="*40)
print(" CONFIG ")
print("="*40)
pprint.pprint(dataclasses.asdict(cfg), width=80, sort_dicts=False)
print("="*40 + "\n")
print("##########################################")

print(f"Training on {DEVICE}...")
print(f"Logging: {model_name}")
trainer.train()

agent.save(f"./models/complete/DQN/{model_name}.pt")

print(f"Completed.")
print(f"Model saved: models/complete/{model_name}.pt")
print(f"Model name: {model_name}")