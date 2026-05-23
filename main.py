import signal
import numpy as np
import time
import sys
import os
from visualizer import Visualizer
from network import Network
from control import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import Pose
from scenario import load_scenario, save_scenario, random_scenario
from tqdm import tqdm
import textwrap

####################################################
sim_step = 0.001  # seconds
tolerance = 1e-2  # bearing difference norm squared

ACCUMULATE = False
VISUALIZE = True

RENDER_FPS = 10
####################################################


def step(network: Network, controller: Controller, tolerance, converged, sim_step, sim_time):
    error = None

    # control
    if not converged:
        velocities = controller.control(network)
        error = controller.error(network)

        if sum(error) < tolerance:
            converged = True

        if converged:
            network.set_inputs(np.zeros(6 * len(network.agents)))
        else:
            network.set_inputs(velocities)

    # sim
    network.step(sim_step)
    sim_time += sim_step

    return sim_time, converged, error, velocities


if len(sys.argv) < 2:
    print(f"usage: python3 main.py [scenario_name]")
    quit()
filename = sys.argv[1]
filedir = "./scenarios/" + filename + ".json"
if not os.path.exists(filedir):
    print(f"file scenarios/{filename}.json does not exists")
    quit()

# setup
vis = Visualizer()
signal.signal(signal.SIGINT, vis.handle_sigint)

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(formatter={"all": lambda x: "{:.4g}".format(x)})

# graph/network
network, goal_network = load_scenario(filedir)
# network, goal_network = random_scenario(4, "R^2")
bearings = network.get_bearings()
goal_bearings = goal_network.get_bearings()

print(f"----------------network----------------")
network.print()
print(f"bearings: {bearings}")
print(f"rigid: {network.is_IBR()}")
print(f"----------------goal network----------------")
goal_network.print()
print(f"goal bearings: {goal_bearings}")
print(f"goal rigid: {goal_network.is_IBR()}")

print("#####################################")
print("#####################################")
print("#####################################")

# controller
leader_idx = 0
controller = GradientBasedController(
    np.asarray(goal_bearings), lin_velocity_gain=100, ang_velocity_gain=1
)
# controller = GradientBasedControllerWithLeader(
#     np.asarray(goal_bearings),
#     lin_velocity_gain=1000,
#     ang_velocity_gain=1,
#     leader_idx=leader_idx,
#     leader_goal=goal_network.agents[leader_idx].pose,
#     leader_vel_gain=0.05,
#     leader_ang_vel_gain=1,
# )
# controller = GradientBasedControllerWithUnicycleLeader(
#     np.asarray(goal_bearings), lin_velocity_gain=1000, ang_velocity_gain=1,
#     leader_idx=leader_idx, leader_goal=goal_network.agents[leader_idx].pose,
#     leader_vel_gain=0.05, leader_ang_vel_gain=1
# )

# sim
sim_time = 0.0
accumulator = 0.0

if VISUALIZE:
    vis.reset()
    vis.draw_viser(
        goal_network,
        node_color=(0, 255, 0),
        edge_color=(0, 128, 0),
        label_prefix="Goal",
    )
    vis.draw_viser(
        network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Current"
    )
    vis.draw_info("ready to start")
    vis.wait_for_start()

render_interval = 1.0 / RENDER_FPS
last_render_time = time.time()

start_wall_time = time.time()
curr_time = time.time()
prev_time = time.time()

converged = False
error = []
pbar = tqdm(total=1, bar_format="{desc}", position=0, leave=True)
while True:
    curr_time = time.time()

    # update
    velocities = np.zeros(6 * len(network.agents))
    if ACCUMULATE:
        dt = curr_time - prev_time
        prev_time = curr_time
        accumulator += dt
        while accumulator >= sim_step:
            sim_time, converged, error, velocities = step(
                network, controller, tolerance, converged, sim_step, sim_time
            )
            accumulator -= sim_step
            if converged:
                break
    else:
        sim_time, converged, error, velocities = step(
            network, controller, tolerance, converged, sim_step, sim_time
        )

    pbar.set_description(f"Sim time: {sim_time:.3f} s | Error(sum): {sum(error)}")
    pbar.refresh()

    # visualize\
    if VISUALIZE:
        if curr_time - last_render_time >= render_interval:
            last_render_time = curr_time

            vis.reset()
            vis.draw_viser(goal_network, node_color=(0, 255, 0), edge_color=(0, 128, 0), label_prefix="Goal")
            vis.draw_viser(network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Current")

            vels_info = "\n".join(
                f"velocities ({i}): {velocities[3*i:3*i+3]}-{velocities[3*len(network.agents)+3*i:3*len(network.agents)+3*i+3]}"
                for i in range(len(network.agents))
            )
            print(vels_info)
            vis.draw_info(
                f"""sim time: {sim_time}\n
                real time: {curr_time - start_wall_time}\n
                converged: {converged}\n
                error: {error}\n
                network is rigid: {network.is_IBR()}\n
                goal network is rigid: {goal_network.is_IBR()}\n
                {vels_info}
                """
            )

            vis.server.flush()

    # terminate
    if converged:
        print("Converged.")
        print(f"time: {curr_time - start_wall_time}, sim_time: {sim_time}, error: {error}")
        vels_info = "\n".join(
            f"velocities ({i}): {velocities[3*i:3*i+3]}-{velocities[3*len(network.agents)+3*i:3*len(network.agents)+3*i+3]}"
            for i in range(len(network.agents))
        )
        info = textwrap.dedent(f"""\
                sim time: {sim_time}
                real time: {curr_time - start_wall_time:.2f}s
                converged: {converged}
                error: {error}
                network is rigid: {network.is_IBR()}
                goal network is rigid: {goal_network.is_IBR()}
                {vels_info}
            """)
        print(info)
        break

vis.stop()

print("Finished.")
