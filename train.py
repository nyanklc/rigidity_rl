from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from environment import Environment
from stable_baselines3 import PPO, SAC
import os
import sys
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
import json

######################################
TOTAL_TIMESTEPS = 2e4
NR_ENVS = 1
USE_CHECKPOINTS = False
######################################


class InfoLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for i in range(len(infos)):
            info = infos[i]
            self.logger.record(f"env{i}/nr_edges", info.get("nr edges", 0))
            self.logger.record(f"env{i}/is_rigid", int(info.get("is rigid", False)))
            self.logger.record(f"env{i}/reward_step", info.get("reward (step)", 0))
            self.logger.record(f"env{i}/reward_raw", info.get("reward (raw)", 0))

        return True


if len(sys.argv) < 2:
    print(f"usage: python3 train.py [environment_name]")
    quit()

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
MAX_STEPS = config["max_steps"]
scenario_name = config["scenario"]
scenario_path = "scenarios/" + scenario_name + ".json" if scenario_name is not None else None

now = datetime.now()
now_str = now.strftime("%Y_%m_%d_%H_%M_%S")
domains_str = domains
domains_str = domains_str.replace("^", "").replace("(", "").replace(")", "")
n_domains = f"n{n}_{domains_str}"

model_name = f"{now_str}_{ACTION_TYPE}_{OBS_TYPE}_{REWARD_TYPE}_{scenario_name if scenario_name is not None else n_domains}"
log_dir = "./tboard_logs/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("./models/", exist_ok=True)

env = make_vec_env(
    lambda: Environment(
        n,
        domains,
        action_space_type=ACTION_TYPE,
        obs_space_type=OBS_TYPE,
        reward_type=REWARD_TYPE,
        termination_condition_type=TERMINATION_CONDITION_TYPE,
        max_steps=MAX_STEPS,
        filepath=scenario_path,
    ),
    n_envs=NR_ENVS,
)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    # learning_rate=3e-4,
    # n_steps=2048,
    # batch_size=64,
    # gamma=0.99,
    tensorboard_log=log_dir,
    device="cpu",
)

callbacks = []
if USE_CHECKPOINTS:
    os.makedirs("./models/checkpoints/", exist_ok=True)
    callbacks.append(
        CheckpointCallback(
            save_freq=10000, save_path="./models/checkpoints/", name_prefix=model_name
        )
    )
callbacks.append(InfoLoggingCallback())

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, tb_log_name=model_name)

os.makedirs("./models/complete/", exist_ok=True)
model.save("./models/complete/" + model_name)
print(f"MODEL SAVED: {"./models/complete/" + model_name + ".zip"}")
print(f"(ENVIRONMENT: {filepath})")
