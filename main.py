import numpy as np
import pygame
import time
from sim_window import SimWindow
from network import Network

####################################################
sim_step = 0.00001  # seconds
controller_frequency = 1  # Hz

WINDOW_W = 640
WINDOW_H = 480
####################################################

# setup
clock = pygame.time.Clock()
curr_time = time.time()
prev_time = curr_time
window = SimWindow((WINDOW_W, WINDOW_H))

# graph/network
n = 4
m = 5
d = 2
p = np.random.uniform(
    low=-min(WINDOW_W, WINDOW_H) / 2, high=min(WINDOW_W, WINDOW_H) / 2, size=(n, d)
)
edges = np.zeros((m, 2), dtype=np.int32)
edges[0][0] = 0
edges[0][1] = 1
edges[1][0] = 1
edges[1][1] = 2
edges[2][0] = 2
edges[2][1] = 3
edges[3][0] = 3
edges[3][1] = 0
edges[4][0] = 0
edges[4][1] = 2
network = Network(p, edges)

# sim
sim_time = 0.0
accumulator = 0.0

start_wall_time = time.time()
prev_time = time.time()

control_period = 1.0 / controller_frequency
control_timer = 0.0

running = True
while running:
    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time

    accumulator += dt
    control_timer += dt

    # events
    events = window.get_events()
    if window.handle_events(events):
        break

    # controller
    if control_timer >= control_period:
        control_timer -= control_period
        network.set_velocities((np.random.rand(n, p.shape[1]) * 2 - np.ones((n, p.shape[1]))) * 100)

    # sim update
    while accumulator >= sim_step:
        network.step(sim_step)
        sim_time += sim_step
        accumulator -= sim_step

    # render
    wall_time = time.time() - start_wall_time
    ratio = sim_time / wall_time if wall_time > 0 else 0.0
    info = f"sim: {sim_time:.2f}s | real: {wall_time:.2f}s | x{ratio:.2f}"
    window.set_info_text(info, (0, 0, 0))
    window.draw(network)
    # clock.tick(60)

window.quit()
