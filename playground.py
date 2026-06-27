import sys
import os
import time
from visualizer import Visualizer
from environment import Environment
from skrl.envs.wrappers.torch import wrap_env
import torch
import numpy as np

if len(sys.argv) < 2:
    print(f"usage: python3 playground.py [environment_name]")
    quit()

env_name = sys.argv[1]
filepath = "./environments/" + env_name + ".json"
if not os.path.exists(filepath):
    print(f"file environments/{env_name}.json does not exist")
    quit()

torch.set_printoptions(threshold=10000)

device = "cpu"

env = Environment()
env.load(filepath)
env.device = device

n = len(env.network.agents)

################################################################################

# def decode_action_number(action_number, action_shape):
#     action = np.empty(action_shape, dtype=np.float32)
#     for i in range(action.size):
#         action.flat[-i-1] = (action_number >> i) & 1

#     print(f"DECODE: {action_number} -> {action}")
#     return action

vis = Visualizer()
action_gui = vis.server.gui.add_number(
    f"Action ({env.action_space_type} -> {env.action_space.n})",
    initial_value=0,
    step=1,
)
aslkdj = vis.server.gui.add_vector3
button_step = vis.server.gui.add_button("step")
step_command = vis.server.gui.add_command("step_command", hotkey="space")
def wait_for_step():
    while not button_step.value:
        vis.server.flush()
        time.sleep(0.05)
    button_step.value = False
step_command.on_trigger(lambda event: setattr(button_step, 'value', True))

done = False
truncated = False
step_idx = 1

obs, _ = env.reset()
vis.reset()
vis.draw_viser(env.network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Env")
env.network.print()
while not (done or truncated):
    wait_for_step()

    action_number = action_gui.value
    # action = decode_action_number(action_number, env.action_space.n)
    action = action_number

    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated.any().item() if torch.is_tensor(terminated) else terminated
    is_truncated = truncated.any().item() if torch.is_tensor(truncated) else truncated

    reward_val = reward.item() if torch.is_tensor(reward) else reward

    # show info
    vis.reset()
    vis.draw_viser(env.network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Env")
    info_str = "".join([f"{k}: {v}\n" for k, v in info.items()]) + "\n"
    info_str += str(env.network)
    vis.draw_info(info_str)
    vis.server.flush()

    step_idx += 1

time.sleep(2)
vis.server.flush()

vis.stop()

print("Finished.")
