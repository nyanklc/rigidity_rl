from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from environment import Environment
from stable_baselines3 import PPO, SAC
import os
import sys
from datetime import datetime

#############################################
TOTAL_TIMESTEPS = 2e5
NR_ENVS = 8
ACTION_TYPE = "AddRemoveEdgeMultiDiscrete"
OBS_TYPE = "Complete"
REWARD_TYPE = "RigidAndMinSingularValue"
#############################################

if len(sys.argv) < 2:
    print(f"input scenario filename as argument")
filename = sys.argv[1]
filedir = "./scenarios/" + filename + ".json"
if not os.path.exists(filedir):
    print(f"file scenarios/{filename}.json does not exists")
    quit()

now = datetime.now()
now_str = now.strftime("%Y_%m_%d_%H_%M_%S")
model_name = now_str + "_" + filename + "_" + ACTION_TYPE + "_" + OBS_TYPE + "_" + REWARD_TYPE
print(f"MODEL NAME: {model_name}")

log_dir = "./tboard_logs/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("./models/", exist_ok=True)

env = make_vec_env(
    lambda: Environment(
        filedir,
        action_space_type=ACTION_TYPE,
        obs_space_type=OBS_TYPE,
        reward_type=REWARD_TYPE,
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
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix=model_name)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, tb_log_name=model_name)

model.save(model_name)
