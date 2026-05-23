from environment import Environment
import os
import sys
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
import json
import torch
from datetime import datetime
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddqn import DDQN, DDQN_CFG
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.resources.preprocessors.torch import RunningStandardScaler
import skrl
from policy import *
import copy
import numpy as np
# from util import RandomActionWrapper

######################################
TOTAL_TIMESTEPS = int(6e5)
NR_ENVS = 1
MEM_SIZE = 20000
EGREEDY_STEPS = 200000

GNN_HIDDEN_DIM = 32
QNETWORK_HEAD_HIDDEN_DIM = 32

DEVICE = "cuda"
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

raw_env = Environment()
raw_env.load(filepath)

n = len(raw_env.network.agents)
domains_str = raw_env.network.agents[0].domain if n > 0 else "domain"
domains_str = domains_str.replace("^", "").replace("(", "").replace(")", "")
n_domains = f"n{n}_{domains_str}"

# yeah i can't be bothered
with open(filepath, "r") as f:
    config = json.load(f)
    scenario_name = config["scenario"]

model_name = (
    model_name_prefix
    + f"_action{raw_env.action_space_type}_obs{raw_env.obs_space_type}_reward{raw_env.reward_type}_term{raw_env.termination_condition_type}_{scenario_name if scenario_name is not None else n_domains}"
)

device = DEVICE

raw_env.device = device
raw_env.set_writer(model_name) # initializes summary writer for env
# raw_env.action_space.seed(42) # doesn't work
env = wrap_env(raw_env)

node_features_dim = raw_env.observation_space["node_features"].shape[1]

models = {}
# q network
if raw_env.action_space_type == "SelectNodesSequentially":
    models["q_network"] = DDQN_QNetwork_SelectNodesSequentially(
        n,
        node_feat_dim=node_features_dim,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        head_hidden_dim=QNETWORK_HEAD_HIDDEN_DIM,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
else:
    print(f"Q network for {raw_env.action_space_type} is not implemented.")
    quit()

# target
models["target_q_network"] = copy.deepcopy(models["q_network"])

# for rollouts
# TODO: env.num_envs??
memory = RandomMemory(memory_size=MEM_SIZE, num_envs=NR_ENVS, device=device)

cfg = DDQN_CFG()
cfg.experiment.directory = "runs"
cfg.experiment.experiment_name = model_name
cfg.learning_rate = 1e-4
cfg.batch_size = 64
cfg.target_update_interval = 500
cfg.update_interval = 1
cfg.learning_starts = MEM_SIZE + 1
# cfg.random_timesteps = 200000
# cfg.discount_factor = 0.5

## TODO: we cannot use epsilon greedy because stupid gymnasium doesn't sample random actions
## see gymnasium/utils/seeding.py
## idk I added a custom change for now. gymnasium/vector/utils/space_utils.py:89

def epsilon_schedule(timestep, timesteps):
    start = 1.0
    end = 0.05
    decay_steps = min(EGREEDY_STEPS, timesteps)
    eps = start - (start - end) * min(1.0, timestep / decay_steps)
    return eps
cfg.exploration_scheduler = epsilon_schedule

os.makedirs("./models", exist_ok=True)
os.makedirs("./models/complete", exist_ok=True)
os.makedirs("./models/complete/DDQN", exist_ok=True)
os.makedirs("./models/experiment", exist_ok=True)

torch.set_printoptions(threshold=10000)


agent = DDQN(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

trainer_cfg = SequentialTrainerCfg()
trainer_cfg.timesteps = TOTAL_TIMESTEPS
trainer_cfg.headless = True # we don't have env.render()
trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

print("##########################################")
print(f"obs space: {trainer.env.observation_space}")
print(f"action space: {trainer.env.observation_space}")
print("##########################################")

print(f"Training on {device}...")
trainer.train()

agent.save(f"./models/complete/DDQN/{model_name}.pt")

print(f"Completed.")
print(f"Model saved: models/complete/{model_name}.pt")
print(f"Model name: {model_name}")
