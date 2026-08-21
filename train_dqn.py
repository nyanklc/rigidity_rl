from environment import Environment, OBS_BACKBONE
import os
import sys
import json
import torch
import inspect
import numpy as np
import copy
import gymnasium as gym
from datetime import datetime
import skrl

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.dqn import DQN, DQN_CFG
from skrl.agents.torch.ddqn import DDQN
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
import policy.gnn_backbone
from policy import *
import manifest
from probe import Probe

######################################
# env-var overridable so an A/B or a 3-seed sweep does not need a source edit;
# both land in the manifest
TOTAL_TIMESTEPS = int(float(os.environ.get("TOTAL_TIMESTEPS", 2.5e5)))
NR_ENVS = 1
MEM_SIZE = 10000
SEED = int(os.environ.get("SEED", 0))
EGREEDY_STEPS = TOTAL_TIMESTEPS * 0.5
ALGORITHM = os.environ.get("ALGORITHM", "DQN")

# which GNN serves the model; the observation is one type now, so the backbone
# is a model choice. One of policy.BACKBONES.
BACKBONE = "Equivariant"
GNN_HIDDEN_DIM = 128
QNETWORK_HEAD_HIDDEN_DIM = 256

# Periodic deterministic evaluation during training: is the policy a decision
# rule or only a sampler? See DESIGN_NOTES.md#training-metrics
PROBE_INTERVAL = 25_000
PROBE_EPISODES = 3

cfg = DQN_CFG()
cfg.experiment.directory = "runs"
cfg.batch_size = 256
cfg.polyak = 0.005
cfg.target_update_interval = 1
cfg.update_interval = 4
cfg.learning_rate = 3e-4
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
##################

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
    # a pre-merge obs_type implied its backbone; honour that over the constant
    backbone = OBS_BACKBONE.get(obs_type, BACKBONE)
    n = config.get("n")
    # scenario configs carry the full per-agent list, homogeneous ones a bare string
    domains = config.get("domains", "domain")
    if isinstance(domains, list):
        domains = "-".join(sorted(set(domains)))
    domains_str = domains.replace("^", "").replace("(", "").replace(")", "")
    n_domains = f"n{n}_{domains_str}"

if "prefix=" in sys.argv[2]:
    model_name = model_name[7:] + f"_action{action_type}_{backbone}_{scenario_name if scenario_name is not None else n_domains}"

train_dir = "./train"
os.makedirs(train_dir, exist_ok=True)
descriptor_path = os.path.join(train_dir, f"{model_name}.json")
model_save_path = f"./models/complete/{ALGORITHM}/{model_name}.pt"

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

# seed everything so a run is reproducible from the manifest's recorded seed
np.random.seed(SEED)
torch.manual_seed(SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

node_features_dim = raw_env.single_observation_space["node_features"].shape[1]
edge_features_dim = raw_env.single_observation_space["edge_features"].shape[-1]

models = build_models(
    "DQN",
    backbone=backbone,
    action_type=action_type,
    n=n,
    node_feat_dim=node_features_dim,
    edge_feat_dim=edge_features_dim,
    gnn_hidden_dim=GNN_HIDDEN_DIM,
    head_hidden_dim=QNETWORK_HEAD_HIDDEN_DIM,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=DEVICE,
    allow_skip=allow_skip,
)

models["target_q_network"] = copy.deepcopy(models["q_network"])


# (the spaces are seeded from SEED above; re-seeding from the clock here would make the
# run irreproducible)

memory = RandomMemory(memory_size=MEM_SIZE, num_envs=env.num_envs, device=DEVICE)

cfg.experiment.experiment_name = model_name

os.makedirs("./models", exist_ok=True)
os.makedirs(f"./models/complete/{ALGORITHM}", exist_ok=True)

agent_class = {"DQN": DQN, "DDQN": DDQN}[ALGORITHM]
agent = agent_class(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=DEVICE,
)

if resume and os.path.exists(model_save_path):
    print(f"Loading existing model from {model_save_path}...")
    agent.load(model_save_path)

trainer_cfg = SequentialTrainerCfg()
trainer_cfg.timesteps = TOTAL_TIMESTEPS
trainer_cfg.headless = True
trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

import dataclasses
import pprint
import json
import inspect
print("##########################################")
print(" TRAINING ")
print("="*40)
print(f"obs space: {trainer.env.observation_space}")
print(f"action space: {trainer.env.action_space}")
print(f"model: {models['q_network'].__class__.__name__}")
print(f"ALGORITHM: {ALGORITHM}")
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
    "algorithm": ALGORITHM,
    "model_name": model_name,
    "environment_config": filename,
    "timestamp_started": datetime.now().isoformat(),
    "total_timesteps_configured": TOTAL_TIMESTEPS,
    "nr_envs": NR_ENVS,
    "mem_size": MEM_SIZE,
    "egreedy_steps": EGREEDY_STEPS,
    "backbone": backbone,
    "gnn_hidden_dim": GNN_HIDDEN_DIM,
    "head_hidden_dim": QNETWORK_HEAD_HIDDEN_DIM,
    "hyperparameters": make_serializable(dataclasses.asdict(cfg)),
    "status": "training",
    "timesteps_completed": 0,
    # the model classes only *reference* the backbone, so archive it too or a checkpoint
    # stops loading the moment gnn_backbone.py changes
    "backbone_source": inspect.getsource(policy.gnn_backbone).split("\n"),
    "q_network_architecture": inspect.getsource(models["q_network"].__class__).split("\n"),
    "environment_config_raw": env_config_data
}

# archive every file that determines this run, plus versions/seed/git state, so the
# checkpoint stays reproducible after the code moves on (see manifest.py)
descriptor = manifest.build_manifest(descriptor, env_config_data, seed=SEED, device=DEVICE)

probe = Probe(filepath, device=DEVICE,
              interval=PROBE_INTERVAL, episodes=PROBE_EPISODES)

_original_post_interaction = agent.post_interaction
def custom_post_interaction(*args, timestep, timesteps, **kwargs):
    descriptor["timesteps_completed"] = timestep
    probe.maybe_run(agent, timestep, raw_env.envs[0].writer)
    return _original_post_interaction(*args, timestep=timestep, timesteps=timesteps, **kwargs)
agent.post_interaction = custom_post_interaction

with open(descriptor_path, "w") as f:
    json.dump(descriptor, f, indent=4)

print(f"Training on {DEVICE}...")
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