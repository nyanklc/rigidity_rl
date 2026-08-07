from stable_baselines3.common.env_util import make_vec_env
import copy
import numpy as np
from environment import Environment
from rigidity import rigidity_eigenvalue
import sys
import os
import time
import json
from visualizer import Visualizer
import textwrap
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from environment import Environment
import json
from datetime import datetime
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.agents.torch.dqn import DQN, DQN_CFG
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.resources.preprocessors.torch import RunningStandardScaler
from policy import *
from agent_loader import load_agent, load_run, list_checkpoints, manifest_path
import torch


#############################################
BRUTE_FORCE_BEST = True
#############################################
DEVICE = "cpu"
NR_ENVS = 1
#############################################

def MBR_required_edges(network):
    n = len(network.agents)
    d = 2 if network.agents[0].domain in ["R^2", "R^2xS^1"] else 3

    if d < 2 or n < 3:
        return False

    k = (n - 2) // (d - 1)
    r = (n - 2) % (d - 1)
    sgn = 1 if r > 0 else 0

    m_required = 1 + k * d + r + sgn
    return m_required


if len(sys.argv) < 3:
    print(f"usage: python3 inference.py [model_name] [environment_name]")
    quit()

model_name = sys.argv[1]
env_name = sys.argv[2]

filepath = "./environments/" + env_name + ".json"
if not os.path.exists(filepath):
    print(f"file environments/{env_name}.json does not exist")
    quit()

with open(filepath, "r") as f:
    config = json.load(f)
    scenario_name = config.get("scenario")
    action_type = config.get("action_type")
    obs_type = config.get("obs_type")
    n = config.get("n")
    domains_str = config.get("domains", "domain").replace("^", "").replace("(", "").replace(")", "")
    n_domains = f"n{n}_{domains_str}"

torch.set_printoptions(threshold=10000)

device = "cpu"

print(f"env: {env_name} | action={action_type} obs={obs_type}")

# load_run replays the environment this model was trained against whenever the archived
# sources differ from the working tree, so a checkpoint keeps running after the observation
# format or action semantics change. Models predating the manifests fall back to recovery
# from the checkpoint's parameter shapes.
try:
    agent, env, raw_env, train_info = load_run(model_name, env_name=env_name, device=device, prefer_archived_env=False)
    MODEL_TYPE = (train_info or {}).get("algorithm", "PPO")
    n = len(raw_env.network.agents)
except (FileNotFoundError, ValueError) as e:
    print(f"\n{e}\n")
    available = list_checkpoints()
    if available:
        print("available checkpoints:")
        for algo, names in available.items():
            for name in names:
                mark = " " if os.path.exists(manifest_path(name)) else "*"
                print(f"  {mark} [{algo}] {name}")
        print("\n  * = no train/<name>.json manifest, will be recovered interactively")
    quit()


vis = Visualizer()
button_step = vis.server.gui.add_button("step")
step_command = vis.server.gui.add_command("step_command", hotkey="space")
def wait_for_step():
    while not button_step.value:
        vis.server.flush()
        time.sleep(0.05)
    button_step.value = False
step_command.on_trigger(lambda event: setattr(button_step, 'value', True))

obs, _ = env.reset()

homogeneous_domain = raw_env.network.agents[0].domain
for ag in raw_env.network.agents:
    if (ag.domain not in ["R^2", "R^3"]) or (ag.domain != homogeneous_domain):
        print("MBR is only for homogeneous R^d network.")
        BRUTE_FORCE_BEST = False
    homogeneous_domain = ag.domain
if raw_env.network.n >= 6:
    print("Brute force with more than 5 nodes is not a good idea.")
    BRUTE_FORCE_BEST = False

vis2 = None
if BRUTE_FORCE_BEST:
    netw = copy.deepcopy(raw_env.network)

    # i != j since self loop is not needed
    n = len(netw.agents)
    all_edges = [[i, j] for i in range(n) for j in range(n) if i != j]
    print(f"ALL POSSIBLE EDGES COUNT: {len(all_edges)}")

    k = MBR_required_edges(netw)
    print(f"MBR REQUIRED EDGE COUNT: {k}")

    subsets = list(itertools.combinations(all_edges, k))

    best_min_eig = -np.inf
    best_eigs = None
    best_edges = None

    for subset in tqdm(subsets):
        edgs = list(subset)
        netw.set_edges_list(edgs)

        is_MBR, is_IBR, _ = netw.is_MBR()

        if not is_MBR:
            continue

        if not is_IBR:
            continue

        min_eig = rigidity_eigenvalue(netw)

        if min_eig > best_min_eig:
            best_min_eig = min_eig
            best_eigs = netw.eigenvalues()
            best_edges = edgs

    if best_edges is not None:
        netw.set_edges_list(best_edges)
        netw.print()
        print(f"MBR, IBR, rank: {netw.is_MBR()}")

        vis2 = Visualizer(port="6767")
        # vis2.wait_for_start()
        vis2.reset()
        vis2.draw_viser(netw)
        vis2.draw_info(
            f"BEST POSSIBLE CONFIGURATION\n"
            f"min: {best_min_eig}, eigs: {best_eigs}, edges: {best_edges}"
        )
        vis2.server.flush()
    else:
        print("No valid MBR configuration found.")


done = False
truncated = False
step_idx = 1

vis.reset()
vis.draw_viser(raw_env.network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Env")
raw_env.network.print()
while not (done or truncated):
    wait_for_step()

    action_tensor, act_outputs = agent.act(obs, states=env.state(), timestep=step_idx, timesteps=1)
    obs, reward, terminated, truncated, info = env.step(action_tensor)

    done = terminated.any().item() if torch.is_tensor(terminated) else terminated
    is_truncated = truncated.any().item() if torch.is_tensor(truncated) else truncated

    reward_val = reward.item() if torch.is_tensor(reward) else reward

    # show info
    vis.reset()
    vis.draw_viser(raw_env.network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Env")
    info_str = "".join([f"{k}: {v}\n" for k, v in info.items()]) + "\n"
    info_str += str(env.network)
    vis.draw_info(info_str)
    vis.server.flush()

    step_idx += 1

info_str = "FINISHED.\n" + info_str
vis.draw_info(info_str)
vis.server.flush()
print("Episode finished. Press Enter to close the server and exit...")
try:
    input()
except KeyboardInterrupt:
    pass

vis.stop()
if vis2 is not None:
    vis2.stop()

print("Finished.")
