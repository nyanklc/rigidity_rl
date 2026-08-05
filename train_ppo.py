from environment import Environment
import os
import sys
import json
import torch
import inspect
from datetime import datetime
import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.resources.preprocessors.torch import RunningStandardScaler
import policy.gnn_backbone
from policy import *
import numpy as np
import manifest

######################################
TOTAL_TIMESTEPS = int(6e5)
NR_ENVS = 1
MEM_SIZE = 2048
SEED = 0  # recorded in the manifest; training was unseeded before this

GNN_HIDDEN_DIM = 64
ACTOR_HEAD_HIDDEN_DIM = 128
CRITIC_HEAD_HIDDEN_DIM = 128

cfg = PPO_CFG()
cfg.rollouts = MEM_SIZE # to ensure we don't get garbage data from memory
cfg.experiment.directory = "runs"
# incentivize exploration more
cfg.entropy_loss_scale = 0.01
cfg.learning_rate = 3e-4
cfg.learning_epochs = 4
cfg.mini_batches = 8
cfg.kl_threshold = 0.015
cfg.value_preprocessor = RunningStandardScaler
cfg.value_preprocessor_kwargs = {"size": 1, "device": "cuda"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
######################################


if len(sys.argv) < 3:
    print(f"usage: python3 train.py [environment_name] [model_name]\nuse 'prefix=...' for the model name to append action/obs type")
    quit()

model_name = sys.argv[2]

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
    # skip is opt-out: see policy/*/SelectNodesSequentially.py for why
    allow_skip = config.get("skip_enabled", True)
    n = config.get("n")
    domains_str = config.get("domains", "domain").replace("^", "").replace("(", "").replace(")", "")
    n_domains = f"n{n}_{domains_str}"

if "prefix=" in sys.argv[2]:
    model_name = model_name[7:] + f"_action{action_type}_obs{obs_type}_{scenario_name if scenario_name is not None else n_domains}"

train_dir = "./train"
os.makedirs(train_dir, exist_ok=True)
descriptor_path = os.path.join(train_dir, f"{model_name}.json")
model_save_path = f"./models/complete/PPO/{model_name}.pt"

resume = False
if os.path.exists(descriptor_path) or os.path.exists(model_save_path):
    print(f"\nA training run for '{model_name}' already exists.")
    choice = input("Do you want to [c]ontinue training, start [f]resh, or [a]bort? ").strip().lower()
    if choice == 'a':
        quit()
    elif choice == 'f':
        pass
    elif choice == 'c':
        resume = True
    else:
        print("Invalid choice. Aborting.")
        quit()

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

# seed everything so a run is reproducible from the manifest's recorded seed
np.random.seed(SEED)
torch.manual_seed(SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

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
    if obs_type == "DictEquivariantNodeFeaturesAndAdjAndSelection":
        models["policy"] = PPO_ActorModel_Equivariant_AddRemoveEdgeDiscreteNoSelfLoops(
            n,
            node_feat_dim=node_features_dim,
            edge_feat_dim=edge_features_dim,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    elif obs_type == "DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection":
        models["policy"] = PPO_ActorModel_GINE_AddRemoveEdgeDiscreteNoSelfLoops(
            n,
            node_feat_dim=node_features_dim,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            edge_feat_dim=edge_features_dim,
            head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    else:
        models["policy"] = PPO_ActorModel_AddRemoveEdgeDiscreteNoSelfLoops(
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
            edge_feat_dim=edge_features_dim,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            head_hidden_dim=ACTOR_HEAD_HIDDEN_DIM,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            allow_skip=allow_skip,
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
            allow_skip=allow_skip,
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
            allow_skip=allow_skip,
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
    if action_type == "SelectNodesSequentially":
        models["value"] = PPO_CriticModel_Selection(
            n,
            node_feat_dim=node_features_dim,
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
elif obs_type == "DictEquivariantNodeFeaturesAndAdjAndSelection":
    if action_type == "SelectNodesSequentially":
        models["value"] = PPO_CriticModel_Equivariant_Selection(
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
        models["value"] = PPO_CriticModel_Equivariant(
            n,
            node_feat_dim=node_features_dim,
            edge_feat_dim=edge_features_dim,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            head_hidden_dim=CRITIC_HEAD_HIDDEN_DIM,

            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
elif obs_type == "DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection":
    if action_type == "SelectNodesSequentially":
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
        models["value"] = PPO_CriticModel_GINE(
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

cfg.experiment.experiment_name = model_name

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

if resume and os.path.exists(model_save_path):
    print(f"Loading existing model from {model_save_path}...")
    agent.load(model_save_path)

trainer_cfg = SequentialTrainerCfg()
trainer_cfg.timesteps = TOTAL_TIMESTEPS
trainer_cfg.headless = True # we don't have env.render()
trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

import dataclasses
import pprint
import json
import inspect
from datetime import datetime

print("##########################################")
print(" TRAINING ")
print("="*40)
print(f"obs space: {trainer.env.observation_space}")
print(f"action space: {trainer.env.action_space}")
print(f"actor: {models['policy'].__class__.__name__}")
print(f"critic: {models['value'].__class__.__name__}")
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

def make_serializable(obj):
    if callable(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj

with open("environments/"+filename+".json", "r") as env_file:
    env_config_data = json.load(env_file)

descriptor = {
    "algorithm": "PPO",
    "model_name": model_name,
    "environment_config": filename,
    "timestamp_started": datetime.now().isoformat(),
    "total_timesteps_configured": TOTAL_TIMESTEPS,
    "nr_envs": NR_ENVS,
    "mem_size": MEM_SIZE,
    "gnn_hidden_dim": GNN_HIDDEN_DIM,
    "head_hidden_dim": ACTOR_HEAD_HIDDEN_DIM,
    "critic_head_hidden_dim": CRITIC_HEAD_HIDDEN_DIM,
    "hyperparameters": make_serializable(dataclasses.asdict(cfg)),
    "status": "training",
    "timesteps_completed": 0,
    # the model classes only *reference* the backbone, so archive it too or a checkpoint
    # stops loading the moment gnn_backbone.py changes
    "backbone_source": inspect.getsource(policy.gnn_backbone).split("\n"),
    "actor_architecture": inspect.getsource(models["policy"].__class__).split("\n"),
    "critic_architecture": inspect.getsource(models["value"].__class__).split("\n"),
    "environment_config_raw": env_config_data
}

# archive every file that determines this run, plus versions/seed/git state, so the
# checkpoint stays reproducible after the code moves on (see manifest.py)
descriptor = manifest.build_manifest(descriptor, env_config_data, seed=SEED, device=device)

_original_post_interaction = agent.post_interaction
def custom_post_interaction(*args, timestep, timesteps, **kwargs):
    descriptor["timesteps_completed"] = timestep
    return _original_post_interaction(*args, timestep=timestep, timesteps=timesteps, **kwargs)
agent.post_interaction = custom_post_interaction

with open(descriptor_path, "w") as f:
    json.dump(descriptor, f, indent=4)

print(f"Training on {device}...")
print(f"Logging: {model_name}")

try:
    trainer.train()
    descriptor["status"] = "completed"
except KeyboardInterrupt:
    print("\nStopping training gracefully (Ctrl+C)...")
    descriptor["status"] = "interrupted"

agent.save(model_save_path)

with open(descriptor_path, "w") as f:
    json.dump(descriptor, f, indent=4)

print(f"Completed.")
print(f"Model saved: {model_save_path}")
print(f"Model name: {model_name}")
