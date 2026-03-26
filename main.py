import signal
import numpy as np
import time
from visualizer import Visualizer
from network import Network
from control import Controller
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import Pose

####################################################
sim_step = 0.001  # seconds
tolerance = 1e-2  # bearing difference norm squared

REALTIME = True
VISUALIZE = True
####################################################


def step(network: Network, controller: Controller, tolerance, converged, sim_step, sim_time):
    error = None

    # control
    if not converged:
        velocities = controller.control(network)
        error = controller.error(network.get_bearings())

        # print(f"current bearings: {network.get_bearings()}")
        # print(f"goal bearings: {controller.goal}")
        # print(f"error: {error}")

        if error < tolerance:
            converged = True

        # print(f"network is rigid: {network.is_IBR()}")
        # print(f"velocities: {velocities}")

        if converged:
            network.set_inputs(np.zeros(6 * n))
        else:
            network.set_inputs(velocities)

    # sim
    network.step(sim_step)
    sim_time += sim_step

    return sim_time, converged, error


# setup
vis = Visualizer()
signal.signal(signal.SIGINT, vis.handle_sigint)

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(formatter={"all": lambda x: "{:.4g}".format(x)})

# graph/network
positions = (
    np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0.5, 0.5, 2],
        ],
        dtype=float,
    )
    * 50
)
n = len(positions)
orientations_euler = np.zeros((n, 3))
# orientations_euler = np.random.rand(n, 3)
# fully connected (no self loops)
edges = np.asarray([(i, j) for i in range(n) for j in range(n) if i != j])
# edges = np.asarray(
#     [
#         (0, 1),
#         (1, 2),
#         (2, 3),
#         (3, 0),
#         (0, 2),
#     ]
# )
print(f"----------------network----------------")
network = Network(positions, orientations_euler, edges)
network.set_agents_domain_homogeneous("R^2xS^1")
network.agents[4].set_domain("R^3xS^1")
bearings = network.get_bearings()
network.print()
print(f"bearings: {bearings}")
print(f"rigid: {network.is_IBR()}")

# goal
goal_positions = (
    np.array(
        [
            [2, 0, 0],
            [3, 0, 0],
            [3, 1, 0],
            [2, 1, 0],
            [2.5, 3.5, 2],
        ],
        dtype=float,
    )
    * 50
)
# center = np.mean(goal_positions, axis=0)
# rotate = Pose(orientation_euler=(0, 0, np.pi/4)).rotation_mat()
# goal_positions = (goal_positions - center) @ rotate.T + center
goal_network = Network(goal_positions, orientations_euler, edges)
goal_network.set_agents_domain_homogeneous("R^2xS^1")
goal_network.agents[4].set_domain("R^3xS^1")
goal_bearings = goal_network.get_bearings()
print(f"----------------goal network----------------")
goal_network.print()
print(f"goal bearings: {goal_bearings}")
print(f"goal rigid: {goal_network.is_IBR()}")

print("#####################################")
print("#####################################")
print("#####################################")

# controller
controller = Controller(
    np.asarray(goal_bearings), lin_velocity_gain=100, ang_velocity_gain=100
)

# sim
sim_time = 0.0
accumulator = 0.0

if VISUALIZE:
    vis.draw_viser(
        goal_network,
        node_color=(0, 255, 0),
        edge_color=(0, 128, 0),
        label_prefix="Goal",
    )
    vis.draw_viser(
        network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Current"
    )
    vis.draw_info(sim_time, 0.0)
input("ready to start")

start_wall_time = time.time()
curr_time = time.time()
prev_time = time.time()

running = True
converged = False
while running:
    curr_time = time.time()

    # update
    if REALTIME:
        dt = curr_time - prev_time
        prev_time = curr_time
        accumulator += dt
        while accumulator >= sim_step:
            sim_time, converged, error = step(network,
                                              controller,
                                              tolerance,
                                              converged,
                                              sim_step,
                                              sim_time)
            accumulator -= sim_step
            if converged:
                break
    else:
        sim_time, converged, error = step(network,
                                          controller,
                                          tolerance,
                                          converged,
                                          sim_step,
                                          sim_time)

    # visualize\
    if VISUALIZE:
        vis.draw_viser(goal_network, node_color=(0, 255, 0), edge_color=(0, 128, 0), label_prefix="Goal")
        vis.draw_viser(network, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Current")
        vis.draw_info(sim_time, curr_time - start_wall_time)

    # terminate
    if converged:
        print("Converged.")
        print(f"time: {curr_time - start_wall_time}, sim_time: {sim_time}, error: {error}")
        break

vis.stop()

print("Finished.")
