from environment import Environment
import os
import sys
import json
import torch
import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from policy import *

######################################
TOTAL_TIMESTEPS = int(1e6)
NR_ENVS = 8 # 1
MEM_SIZE = 1024 # 2048 * 4

GNN_HIDDEN_DIM = 128
ACTOR_HEAD_HIDDEN_DIM = 128
CRITIC_HEAD_HIDDEN_DIM = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

device = DEVICE

def make_env(i):
    e = Environment()
    e.load(filepath)
    # Give each env its own writer string, or none to prevent spam
    writer_name = model_name if i == 0 else f"{model_name}-{i}"
    e.set_writer(writer_name)
    e.device = device
    return e

# Gym Vector Envs expect a list of callables, so we use a lambda
raw_env = gym.vector.SyncVectorEnv([lambda idx=i: make_env(idx) for i in range(NR_ENVS)])
env = wrap_env(raw_env)

# Use single_observation_space since raw_env is now batched
node_features_dim = raw_env.single_observation_space["node_features"].shape[1]
edge_features_dim = raw_env.single_observation_space["edge_features"].shape[-1]

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

# for rollouts
# TODO: env.num_envs??
memory = RandomMemory(memory_size=MEM_SIZE, num_envs=env.num_envs, device=device)

cfg = PPO_CFG()
cfg.rollouts = MEM_SIZE # to ensure we don't get garbage data from memory
cfg.experiment.directory = "runs"
cfg.experiment.experiment_name = model_name
# incentivize exploration more
cfg.entropy_loss_scale = 0.01

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
print(" TRAINING ")
print("="*40)
print(f"obs space: {trainer.env.observation_space}")
print(f"action space: {trainer.env.action_space}")
print(f"actor: {models["policy"].__class__.__name__}")
print(f"critic: {models["value"].__class__.__name__}")
print(f"TOTAL_TIMESTEPS: {TOTAL_TIMESTEPS}")
print(f"NR_ENVS: {NR_ENVS}")
print(f"MEM_SIZE: {MEM_SIZE}")
print(f"GNN_HIDDEN_DIM: {GNN_HIDDEN_DIM}")
print(f"ACTOR_HEAD_HIDDEN_DIM: {ACTOR_HEAD_HIDDEN_DIM}")
print(f"CRITIC_HEAD_HIDDEN_DIM: {CRITIC_HEAD_HIDDEN_DIM}")
print("\n" + "="*40)
print(" CONFIG ")
print("="*40)
pprint.pprint(dataclasses.asdict(cfg), width=80, sort_dicts=False)
print("="*40 + "\n")
print("##########################################")

print(f"Training on {device}...")
print(f"Logging: {model_name}")
trainer.train()

agent.save(f"./models/complete/PPO/{model_name}.pt")

print(f"Completed.")
print(f"Model saved: models/complete/{model_name}.pt")
print(f"Model name: {model_name}")
