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
sim_step = 0.001  # seconds

WINDOW_W = 640
WINDOW_H = 480
####################################################

# setup
clock = pygame.time.Clock()
curr_time = time.time()
prev_time = curr_time
window = SimWindow((WINDOW_W, WINDOW_H))

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
orientations_euler = np.zeros((n, 3))
# orientations_euler = np.random.rand(n, 3)
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

# sim
sim_time = 0.0
accumulator = 0.0

start_wall_time = time.time()
prev_time = time.time()

control_timer = 0.0

running = True
converged = False
while running:
    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time

    accumulator += dt
    control_timer += dt

    # events
    events = window.get_events()
    terminate, event_ret = window.handle_events(events)
    if terminate:
        break

    # update
    while accumulator >= sim_step:

        # control
        if not converged:
            velocities = controller.control(network)
            error = controller.error(network.get_bearings())
            print(f"current bearings: {network.get_bearings()}")
            print(f"goal bearings: {goal_bearings}")
            print(f"error: {error}")
            if error < 1e-8:
                print("Converged.")
                converged = True
            print(f"network is rigid: {network.is_IBR()}")
            print(f"velocities: {velocities}")
            if converged:
                network.set_inputs(np.zeros(6 * n))
            else:
                network.set_inputs(velocities)

        # sim
        network.step(sim_step)
        sim_time += sim_step
        accumulator -= sim_step

    # render
    wall_time = time.time() - start_wall_time
    ratio = sim_time / wall_time if wall_time > 0 else 0.0
    info = f"sim: {sim_time:.2f}s | real: {wall_time:.2f}s | x{ratio:.2f}"
    window.set_info_text(info, (0, 0, 0))

    window.clear()
    window.draw(network)
    window.draw(goal_network, color_dummy=(255, 0, 0))
    window.flip()

if event_ret == "visualize":
    # visualize
    server = viser.ViserServer()
    goal_network.plot_network_3d_viser(server, node_color=(0, 255, 0), edge_color=(0, 128, 0), label_prefix="Goal")
    network.plot_network_3d_viser(server, node_color=(255, 0, 0), edge_color=(128, 0, 0), label_prefix="Current")
    input("Press Enter to exit...")

window.quit()
