from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from environment import Environment
from stable_baselines3 import PPO, SAC
import os

log_dir = "./tboard_logs/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("./models/", exist_ok=True)

env = make_vec_env(lambda: Environment("scenarios/hetero_6dof.json", visualize=False), n_envs=8)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = SAC(
    "MultiInputPolicy",
    env,
    verbose=1,
    # learning_rate=3e-4,
    # n_steps=2048,
    # batch_size=64,
    # gamma=0.99,
    tensorboard_log=log_dir,
)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='formation_model')

model.learn(total_timesteps=200_000, callback=checkpoint_callback)

model.save("formation_policy_final")
