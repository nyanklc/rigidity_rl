import sys

with open("train_ppo.py", "r") as f:
    content = f.read()

# Replace action and obs space type references
content = content.replace("raw_env.action_space_type", "action_type")
content = content.replace("raw_env.obs_space_type", "obs_type")

# Fix the memory num_envs
content = content.replace(
    "memory = RandomMemory(memory_size=MEM_SIZE, num_envs=NR_ENVS, device=device)",
    "memory = RandomMemory(memory_size=MEM_SIZE, num_envs=env.num_envs, device=device)"
)

# Replace the environment setup block
old_block = """def make_env(i):
    global filepath
    global model_name

    e = Environment()
    e.load(filepath)
    e.set_writer(model_name + f"-{i}")
    e.device = device
    return e

raw_env = gym.vector.SyncVectorEnv([make_env(i) for i in range(NR_ENVS)])
env = wrap_env(raw_env)

# raw_env = Environment()
# raw_env.load(filepath)

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
    + f"_action{action_type}_obs{obs_type}_{scenario_name if scenario_name is not None else n_domains}"
)

device = DEVICE

raw_env.device = device
raw_env.set_writer(model_name) # initializes summary writer for env
env = wrap_env(raw_env)

node_features_dim = raw_env.observation_space["node_features"].shape[1]
edge_features_dim = raw_env.observation_space["edge_features"].shape[-1]"""

# Note: The old content has action_type/obs_type because I replaced them above
# We need to use the exact old block AFTER the replacement

# Let's just do a regex or explicit split
import re

start_marker = "def make_env(i):"
end_marker = 'edge_features_dim = raw_env.observation_space["edge_features"].shape[-1]'

new_block = """with open(filepath, "r") as f:
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
edge_features_dim = raw_env.single_observation_space["edge_features"].shape[-1]"""

start_idx = content.find(start_marker)
end_idx = content.find(end_marker) + len(end_marker)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_block + content[end_idx:]

with open("train_ppo.py", "w") as f:
    f.write(content)

print("Fixes applied successfully!")
