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
from policy import *

######################################
TOTAL_TIMESTEPS = int(2e5)
NR_ENVS = 1
MEM_SIZE = 2048

GNN_HIDDEN_DIM = 32
ACTOR_HEAD_HIDDEN_DIM = 128
CRITIC_HEAD_HIDDEN_DIM = 128

DEVICE = "cuda"
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
env = wrap_env(raw_env)

node_features_dim = raw_env.observation_space["node_features"].shape[1]

models = {}
# actor
if raw_env.action_space_type == "AddEdgeDiscreteNoSkipNoSelfLoops":
    models["policy"] = PPO_ActorModel_AddEdgeDiscreteNoSkipNoSelfLoops(
        n,
        node_feat_dim=node_features_dim,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
elif raw_env.action_space_type == "AddRemoveEdgeDiscreteNoSelfLoops":
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
elif raw_env.action_space_type == "AddRemoveEdgeMultiDiscrete":
    print(f"for action {raw_env.action_space_type} the actor is not fully implemented yet")
    quit()
    # models["policy"] = PPO_ActorModel_AddRemoveEdgeMultiDiscrete(
    #     n,
    #     node_feat_dim=node_features_dim,
    #     gnn_hidden_dim=GNN_HIDDEN_DIM,
    #     head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     device=device,
    # )
else:
    print(f"Actor for action {raw_env.action_space_type} is not implemented.")
    quit()

# critic
models["value"] = PPO_CriticModel(
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
# incentivize exploration more
cfg.entropy_loss_scale = 0.1

os.makedirs("./models", exist_ok=True)
os.makedirs("./models/complete", exist_ok=True)
os.makedirs("./models/complete/PPO", exist_ok=True)
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

agent.save(f"./models/complete/PPO/{model_name}.pt")

print(f"Completed.")
print(f"Model saved: models/complete/{model_name}.pt")
print(f"Model name: {model_name}")
