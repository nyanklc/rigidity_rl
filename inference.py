from stable_baselines3.common.env_util import make_vec_env
from environment import Environment
from stable_baselines3 import PPO
import sys
import os
import time
import json
from visualizer import Visualizer
import textwrap


if len(sys.argv) < 3:
    print(f"usage: python3 inference.py [model_name] [environment_name]")
    quit()

modelpath = "./models/complete/" + sys.argv[1] + ".zip"
if not os.path.exists(modelpath):
    print(f"file {modelpath} does not exist")
    quit()

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
REWARD_TYPE = config["reward_type"]
TERMINATION_CONDITION_TYPE = config["termination_condition_type"]
MAX_STEPS = config["max_steps"]
scenario_name = config["scenario"]
scenario_path = "scenarios/" + scenario_name + ".json" if scenario_name is not None else None

env = Environment(
    n,
    domains,
    action_space_type=ACTION_TYPE,
    obs_space_type=OBS_TYPE,
    reward_type=REWARD_TYPE,
    termination_condition_type=TERMINATION_CONDITION_TYPE,
    max_steps=MAX_STEPS,
    filepath=scenario_path,
)

obs, _ = env.reset()

model = PPO.load(modelpath, device="cpu")

vis = Visualizer()
button_step = vis.server.gui.add_button("step")
def wait_for_step():
    while not button_step.value:
        time.sleep(0.05)
    button_step.value = False

done = False
truncated = False
step_idx = 0

vis.draw_viser(env.network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Env")
while not (done or truncated):
    wait_for_step()

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    info_str = textwrap.dedent(f"""Step {step_idx}\n
        Action: {action}\n
        Reward: {reward}\n
        ------------------\n
        {info}
        """)
    print(info_str)

    # show info
    vis.draw_viser(env.network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Env")
    vis.draw_info(info_str)
    vis.server.flush()

    step_idx += 1

time.sleep(2)
vis.server.flush()

vis.stop()

print("Finished.")
