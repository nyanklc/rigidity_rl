from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from environment import Environment
from stable_baselines3 import PPO, SAC
import os
import sys
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.ppo_mask import MaskablePPO
import json
from policy_sb3 import GNNBackbone

######################################
TOTAL_TIMESTEPS = 4e5
NR_ENVS = 1
USE_CHECKPOINTS = False
CUSTOM_MODEL = True
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
            self.logger.record(f"env{i}/is_min_rigid", int(info.get("is min rigid", False)))
            self.logger.record(f"env{i}/reward_raw", info.get("reward (raw)", 0))
            self.logger.record(f"env{i}/reward_step", info.get("reward (step)", 0))
            self.logger.record(f"env{i}/reward_action", info.get("reward (action)", 0))
            self.logger.record(f"env{i}/reward_state", info.get("reward (state)", 0))
            self.logger.record(f"env{i}/reward_termination", info.get("reward (termination)", 0))
            self.logger.record(f"env{i}/min_eig", info.get("min eigenvalue", 0.0))
            self.logger.record(f"env{i}/second_min_eig", info.get("second min eigenvalue", 0.0))
            # eigs = info.get("nonzero_eigenvalues", [])
            # for j, eig in enumerate(eigs):
            #     self.logger.record(f"env{i}/nonzero_eigs/eig_{j}", eig)

        return True


if len(sys.argv) < 3:
    print(f"usage: python3 train.py [model_name] [environment_name]")
    quit()

model_name_prefix = sys.argv[1]

filename = sys.argv[2]
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
STATE_SCORE_TYPE = config["state_score_type"]
TERMINATION_CONDITION_TYPE = config["termination_condition_type"]
ACTION_REWARDS_ENABLE = config["action_rewards_enable"]
MAX_STEPS = config["max_steps"]
ONLY_RANDOMIZE_EDGES = config["only_randomize_edges"]
scenario_name = config["scenario"]
scenario_path = "scenarios/" + scenario_name + ".json" if scenario_name is not None else None

now = datetime.now()
now_str = now.strftime("%Y_%m_%d_%H_%M_%S")
domains_str = domains
domains_str = domains_str.replace("^", "").replace("(", "").replace(")", "")
n_domains = f"n{n}_{domains_str}"

model_name = model_name_prefix + f"_action{ACTION_TYPE}_obs{OBS_TYPE}_reward{STATE_SCORE_TYPE}_term{TERMINATION_CONDITION_TYPE}_{scenario_name if scenario_name is not None else n_domains}"
log_dir = "./tboard_logs/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("./models/", exist_ok=True)

env = make_vec_env(
    lambda: Environment(
        n,
        domains,
        action_space_type=ACTION_TYPE,
        obs_space_type=OBS_TYPE,
        state_score_type=STATE_SCORE_TYPE,
        termination_condition_type=TERMINATION_CONDITION_TYPE,
        action_rewards_enable=ACTION_REWARDS_ENABLE,
        max_steps=MAX_STEPS,
        only_randomize_edges=ONLY_RANDOMIZE_EDGES,
        filepath=scenario_path,
    ),
    n_envs=NR_ENVS,
)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

callbacks = []
if USE_CHECKPOINTS:
    os.makedirs("./models/checkpoints/", exist_ok=True)
    callbacks.append(
        CheckpointCallback(
            save_freq=10000, save_path="./models/checkpoints/", name_prefix=model_name
        )
    )
callbacks.append(InfoLoggingCallback())

model = None
if not CUSTOM_MODEL:
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
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, tb_log_name=model_name)
else:
    policy_kwargs = dict(
        features_extractor_class=GNNBackbone,
        features_extractor_kwargs=dict(features_dim=128),
    )
    model = MaskablePPO()
    # TODO: not implemented
    quit()

os.makedirs("./models/complete/", exist_ok=True)
model.save("./models/complete/" + model_name)
print(f"MODEL SAVED: {"./models/complete/" + model_name + ".zip"}")
print(f"(ENVIRONMENT: {filepath})")
print(f"model: {model_name}")
print(f"env: {filename}")
