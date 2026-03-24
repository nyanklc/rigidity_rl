import numpy as np
from util import circle_polygon, move_polygon, Pose, invert_color
import quaternion
import rigidity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import viser


class Agent:
    def __init__(self, pose=None):
        self.pose = pose if pose is not None else Pose()
        self.velocity = np.zeros(len(self.pose.position))
        self.angular_velocity = np.zeros(3)

    def step(self, dt):
        self.pose.step(self.velocity, self.angular_velocity, dt)

    def set_velocity(self, vel):
        self.velocity = vel

    def set_angular_velocity(self, ang_vel):
        self.angular_velocity = ang_vel

    def get_footprint(self):
        x, y = self.pose.position[:2]
        yaw = quaternion.as_euler_angles(self.pose.orientation)[2]
        polygon = circle_polygon()
        footprint = move_polygon(polygon, x, y, yaw)
        return footprint

    def get_bearing(self, other: "Agent"):
        p = other.pose.position - self.pose.position
        p = p / np.linalg.vector_norm(p)
        # bearing in body frame
        bearing = self.pose.rotation_mat().T @ p
        return bearing


class Network:
    def __init__(self, positions, orientations_euler, edges):
        self.edges = edges
        self.agents: list[Agent] = []
        for i in range(len(positions)):
            self.agents.append(Agent(Pose(positions[i], orientations_euler[i])))

    def step(self, dt):
        for agent in self.agents:
            agent.step(dt)

    def set_inputs(self, u):
        for i in range(len(self.agents)):
            self.agents[i].set_velocity(u[6*i:6*i+3])
            self.agents[i].set_angular_velocity(u[6*i+3:6*i+6])

    def bearing_rigidity_matrix(self):
        return rigidity.extended_bearing_rigidity_matrix(
            [agent.pose.position for agent in self.agents],
            [agent.pose.rotation_mat() for agent in self.agents],
            self.edges)

    # TODO: brm is called twice in main
    def is_IBR(self):
        brmat = self.bearing_rigidity_matrix()
        return rigidity.is_IBR(brmat, 6)

    def get_bearings(self):
        bearings = np.zeros(len(self.edges)*3)
        for k, (i, j) in enumerate(self.edges):
            bearings[3*k:3*k+3] = self.agents[i].get_bearing(self.agents[j])
        return bearings

    def print(self):
        print(f"NETWORK")
        for i, agent in enumerate(self.agents):
            print(f"agent {i}")
            agent.pose.print()
        for k, (i, j) in enumerate(self.edges):
            print(f"edge {k}: ({i}, {j})")

            import matplotlib.pyplot as plt
