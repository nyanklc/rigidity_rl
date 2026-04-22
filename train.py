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
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.resources.preprocessors.torch import RunningStandardScaler
from policy import ActorModel, CriticModel

######################################
TOTAL_TIMESTEPS = int(4e5)
NR_ENVS = 1
MEM_SIZE = 2048
USE_CHECKPOINTS = False

GNN_HIDDEN_DIM = 32
ACTOR_HEAD_HIDDEN_DIM = 128
CRITIC_HEAD_HIDDEN_DIM = 128
######################################


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

n = config["n"]
domains = config["domains"]
ACTION_TYPE = config["action_type"]
OBS_TYPE = config["obs_type"]
REWARD_TYPE = config["reward_type"]
TERMINATION_CONDITION_TYPE = config["termination_condition_type"]
ACTION_REWARDS_ENABLE = config["action_rewards_enable"]
INCREMENTAL_REWARDS_ENABLE = config["incremental_rewards_enable"]
TRACK_DATA_ENABLE = config["track_data_enable"]
MAX_STEPS = config["max_steps"]
ONLY_RANDOMIZE_EDGES = config["only_randomize_edges"]
scenario_name = config["scenario"]
scenario_path = (
    "scenarios/" + scenario_name + ".json" if scenario_name is not None else None
)

domains_str = domains
domains_str = domains_str.replace("^", "").replace("(", "").replace(")", "")
n_domains = f"n{n}_{domains_str}"

model_name = (
    model_name_prefix
    + f"_action{ACTION_TYPE}_obs{OBS_TYPE}_reward{REWARD_TYPE}_term{TERMINATION_CONDITION_TYPE}_{scenario_name if scenario_name is not None else n_domains}"
)
log_dir = "./tboard_logs/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("./models/", exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

raw_env = Environment(
    n,
    domains,
    action_space_type=ACTION_TYPE,
    obs_space_type=OBS_TYPE,
    reward_type=REWARD_TYPE,
    termination_condition_type=TERMINATION_CONDITION_TYPE,
    action_rewards_enable=ACTION_REWARDS_ENABLE,
    incremental_rewards_enable=INCREMENTAL_REWARDS_ENABLE,
    track_data_enable=TRACK_DATA_ENABLE,
    max_steps=MAX_STEPS,
    only_randomize_edges=ONLY_RANDOMIZE_EDGES,
    filepath=scenario_path,

    experiment_name=model_name, # hack to log values
)

raw_env.device = device
env = wrap_env(raw_env)

node_features_dim = raw_env.observation_space["node_features"].shape[1]

models = {}
# actor
models["policy"] = ActorModel(
    n,
    node_feat_dim=node_features_dim,
    gnn_hidden_dim=GNN_HIDDEN_DIM,
    head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,

    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)
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

# for rollouts
# TODO: env.num_envs??
memory = RandomMemory(memory_size=MEM_SIZE, num_envs=NR_ENVS, device=device)

cfg = PPO_CFG()
cfg.rollouts = MEM_SIZE # to ensure we don't get garbage data from memory
cfg.experiment.directory = "runs"
cfg.experiment.experiment_name = model_name

os.makedirs("./models", exist_ok=True)
os.makedirs("./models/complete", exist_ok=True)
os.makedirs("./models/experiment", exist_ok=True)

torch.set_printoptions(threshold=10000)


agent = PPO(
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

os.makedirs("./models/complete", exist_ok=True)
agent.save(f"./models/complete/{model_name}.pt")
