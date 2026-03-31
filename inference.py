from stable_baselines3.common.env_util import make_vec_env
from environment import Environment
from stable_baselines3 import PPO, SAC

env = Environment("scenarios/hetero_6dof.json", visualize=True)

obs, _ = env.reset()
model = SAC.load("formation_policy_final")

for _ in range(2000000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    if done or truncated:
        break

env.close()