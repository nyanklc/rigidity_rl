from stable_baselines3.common.env_util import make_vec_env
from environment import Environment
from stable_baselines3 import PPO, SAC
import sys
import os


if len(sys.argv) < 2:
    print(f"input scenario filename as argument")
filename = sys.argv[1]
filedir = "./scenarios/" + filename + ".json"
if not os.path.exists(filedir):
    print(f"file scenarios/{filename}.json does not exists")
    quit()

env = Environment(filedir)
obs, _ = env.reset()

model = PPO.load("formation_policy_final", device="cpu")

action, _ = model.predict(obs, deterministic=True)
obs, reward, done, truncated, info = env.step(action)

print(info)
