import numpy as np
from util import circle_polygon, move_polygon, Pose
import quaternion
from util import angular_velocity_to_quaternion


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
        x, y = self.pose.position
        yaw = quaternion.as_euler_angles(self.pose.orientation)[2]
        polygon = circle_polygon()
        footprint = move_polygon(polygon, x, y, yaw)
        return footprint

    def measure(self, other: "Agent"):
        p = other.pose.position - self.pose.position
        p = p / np.linalg.vector_norm(p)
        # bearing in body frame
        bearing = self.pose.rotation_mat().T @ p
        return bearing


class Network:
    def __init__(self, p, edges):
        self.edges = edges
        self.agents: list[Agent] = []
        for i in range(len(p)):
            self.agents.append(Agent(Pose(p[i])))

    def step(self, dt):
        for agent in self.agents:
            agent.step(dt)

    def set_velocities(self, v=None, w=None):
        for i in range(len(self.agents)):
            if v is not None:
                self.agents[i].set_velocity(v[i])
            if w is not None:
                self.agents[i].set_angular_velocity(w[i])
