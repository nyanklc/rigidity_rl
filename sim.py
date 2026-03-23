import numpy as np
import pygame
import time
from sim_window import SimWindow
from network import Network
from control import Controller
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import viser

####################################################
step = 0.001
tolerance = 1e-3

WINDOW_W = 640
WINDOW_H = 480
####################################################


def plot_bearings(bearing_list, goal_bearings):
    bearings = np.array(bearing_list)

    t = range(len(bearing_list))

    # Plot
    plt.figure()
    plt.plot(t, bearings[:, 0], label='first bearing x')
    plt.plot(t, np.repeat(goal_bearings[0], len(t)), label='goal first bearing x')
    plt.xlabel("Time step")
    plt.ylabel("Bearing component")
    plt.title("Bearing component vs. Time")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(t, bearings[:, 1], label='first bearing y')
    plt.plot(t, np.repeat(goal_bearings[1], len(t)), label='goal first bearing y')
    plt.xlabel("Time step")
    plt.ylabel("Bearing component")
    plt.title("Bearing component vs. Time")
    plt.legend()
    plt.grid()
    plt.show()

    # plt.plot(t, bearings[:, 1], label='first bearing y')
    # plt.plot(t, bearings[:, 2], label='first bearing z')
    # plt.plot(t, np.repeat(goal_bearings[1], len(t)), label='goal first bearing y')
    # plt.plot(t, np.repeat(goal_bearings[2], len(t)), label='goal first bearing z')

# setup
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(formatter={"all": lambda x: "{:.4g}".format(x)})

# graph/network
n = 4
d = 6
positions = (
    np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ],
        dtype=float,
    )
    * 50
)
# orientations_euler = np.zeros((n, 3))
orientations_euler = np.random.rand(n, 3)
# fully connected (no self loops)
edges = np.asarray([(i, j) for i in range(n) for j in range(n) if i != j])
print(f"----------------network----------------")
network = Network(positions, orientations_euler, edges)
bearings = network.get_bearings()
network.print()
print(f"bearings: {bearings}")

# goal
goal_positions = (
    np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [3, 1, 0],
        ],
        dtype=float,
    )
    * 100
)
goal_network = Network(goal_positions, orientations_euler, edges)
goal_bearings = goal_network.get_bearings()
print(f"----------------goal network----------------")
goal_network.print()
print(f"goal bearings: {goal_bearings}")

print("#####################################")
print("#####################################")
print("#####################################")

# controller
controller = Controller(
    np.asarray(goal_bearings), lin_velocity_gain=1000, ang_velocity_gain=200
)

running = True
converged = False
curr_step = 0
iteration = 0

bearing_arr = []

while running:
    # control
    velocities = controller.control(network)
    error = controller.error(network.get_bearings())
    current_bearings = network.get_bearings()
    bearing_arr.append(current_bearings)

    print(f"current bearings: {current_bearings}")
    print(f"goal bearings: {goal_bearings}")
    print(f"error: {error}")

    if error <= tolerance:
        print("Converged.")
        network.set_inputs(np.zeros(6 * n))
        break

    print(f"network is rigid: {network.is_IBR()}")
    print(f"velocities: {velocities}")

    network.set_inputs(velocities)

    # sim
    network.step(step)

    curr_step += step
    iteration += 1
    print(f"curr step, iteration: {curr_step, iteration}")

    # if iteration == 1000:
    #     plot_bearings(bearing_arr, goal_bearings)

# visualize
server = viser.ViserServer()
goal_network.plot_network_3d_viser(server, node_color=(0, 255, 0), edge_color=(0, 128, 0), label_prefix="Goal")
network.plot_network_3d_viser(server, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Current")
input("Press Enter to exit...")
