import copy
import numpy as np
import time
import sys
import os

from environment import Environment
from visualizer import Visualizer


if len(sys.argv) < 2:
    print("usage: python3 manual.py [environment_name]")
    quit()

env_name = sys.argv[1]
filepath = "./environments/" + env_name + ".json"

if not os.path.exists(filepath):
    print(f"{filepath} not found")
    quit()

raw_env = Environment()
raw_env.load(filepath)

netw = raw_env.network

vis = Visualizer()

# UI
button_add = vis.server.gui.add_button("Add Edge")
button_remove = vis.server.gui.add_button("Remove Edge")
button_reset = vis.server.gui.add_button("Reset")
button_remove_all = vis.server.gui.add_button("Remove All Edges")

slider_i = vis.server.gui.add_slider("i", min=0, max=len(netw.agents)-1, step=1, initial_value=0)
slider_j = vis.server.gui.add_slider("j", min=0, max=len(netw.agents)-1, step=1, initial_value=1)


def redraw(raw_env):
    vis.reset()
    vis.draw_viser(netw)

    info_str = f"""
    edges: {netw.get_edge_list()}\n
    eigs: {netw.eigenvalues()}\n
    eigs sum: {np.sum(netw.eigenvalues())}\n
    IBR: {netw.is_IBR()}\n
    MBR: {netw.is_MBR()[0]}
    """

    vis.draw_info(info_str)
    vis.server.flush()


# initial draw
redraw(raw_env)


while True:
    vis.server.flush()
    time.sleep(0.05)

    i = slider_i.value
    j = slider_j.value

    if button_add.value:
        button_add.value = False

        netw.add_edge(i, j)
        redraw(raw_env)

    if button_remove.value:
        button_remove.value = False

        netw.remove_edge(i, j)
        redraw(raw_env)

    if button_reset.value:
        button_reset.value = False

        raw_env.reset()
        redraw(raw_env)

    if button_remove_all.value:
        button_remove_all.value = False

        for i, j in netw.get_edge_list():
            netw.remove_edge(i, j)
        redraw(raw_env)
