from stable_baselines3.common.env_util import make_vec_env
import copy
import numpy as np
from environment import Environment
from stable_baselines3 import PPO
import sys
import os
import time
import json
from visualizer import Visualizer
import textwrap
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm


#############################################
BRUTE_FORCE_BEST = True
DETERMINISTIC = False
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

filename = sys.argv[2]
filepath = "./environments/" + filename + ".json"
if not os.path.exists(filepath):
    print(f"file environments/{filename}.json does not exist")
    quit()

env = Environment()
env.load(filepath)
obs, _ = env.reset()

if env.network.agents[0].domain not in ["R^2", "R^3"]:
    print("MBR is only for homogeneous R^d network.")
    BRUTE_FORCE_BEST = False
if env.network.n >= 6:
    print("Brute force with more than 5 nodes is not a good idea.")
    BRUTE_FORCE_BEST = False

vis2 = None
if BRUTE_FORCE_BEST:
    netw = copy.deepcopy(env.network)

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

        if not netw.is_MBR():
            continue

        if not netw.is_IBR():
            continue

        brm = netw.extended_bearing_rigidity_matrix()
        information_mat = brm.T @ brm
        eigenvalues = np.linalg.eigvalsh(information_mat)

        min_eig = eigenvalues.min()

        if min_eig > best_min_eig:
            best_min_eig = min_eig
            best_eigs = eigenvalues
            best_edges = edgs

    if best_edges is not None:
        netw.set_edges_list(best_edges)
        netw.print()
        print(f"IBR: {netw.is_IBR()}")
        print(f"MBR: {netw.is_MBR()}")

        vis2 = Visualizer(port="6767")
        # vis2.wait_for_start()
        vis2.draw_viser(netw)
        vis2.draw_info(
            f"BEST POSSIBLE CONFIGURATION\n"
            f"min: {best_min_eig}, eigs: {best_eigs}, edges: {best_edges}"
        )
        vis2.server.flush()
    else:
        print("No valid MBR configuration found.")


modelpath = "./models/complete/" + sys.argv[1] + ".zip"
if not os.path.exists(modelpath):
    print(f"file {modelpath} does not exist")
    quit()

model = PPO.load(modelpath, device="cpu")

vis = Visualizer()
button_step = vis.server.gui.add_button("step")
def wait_for_step():
    while not button_step.value:
        vis.server.flush()
        time.sleep(0.05)
    button_step.value = False

done = False
truncated = False
step_idx = 0

vis.draw_viser(env.network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Env")
env.network.print()
while not (done or truncated):
    wait_for_step()

    action, _ = model.predict(obs, deterministic=DETERMINISTIC)
    obs, reward, done, truncated, info = env.step(action)

    info_str = f"""Step {step_idx}\n
        Action: {action}\n
        Reward: {reward}\n
        ------------------\n
        {info}
        """
    print(info_str)

    # show info
    vis.draw_viser(env.network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Env")
    vis.draw_info(info_str)
    vis.server.flush()

    step_idx += 1

time.sleep(2)
vis.server.flush()

vis.stop()
if vis2 is not None:
    vis2.stop()

print("Finished.")
