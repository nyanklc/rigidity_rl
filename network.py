import numpy as np
from util import circle_polygon, move_polygon, Pose, invert_color
import quaternion
import rigidity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import viser
from util import skew_symmetric


class Agent:
    def __init__(self, pose=None):
        self.pose = pose if pose is not None else Pose()
        self.velocity = np.zeros(len(self.pose.position))
        self.angular_velocity = np.zeros(3)
        self.domain = "SE(3)"
        self.rotation_axis = None

    def step(self, dt):
        self.pose.step(self.velocity, self.angular_velocity, dt)

    def set_domain(self, domain, rotation_axis=None):
        self.domain = domain
        if domain == "R^3xS^1":
            # TODO: is this in the world frame?
            rax = (
                np.array([0, 0, 1])
                if rotation_axis is None
                else (rotation_axis / np.linalg.norm(rotation_axis))
            )
            self.rotation_axis = rax
        else:
            self.rotation_axis = None

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
        bearing = np.zeros(3)
        if self.domain not in ["R^3", "R^2"]:
            # bearing in body frame
            bearing = self.pose.rotation_mat().T @ p
        else:
            bearing = p
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
        n = len(self.agents)
        for i in range(n):
            self.agents[i].set_velocity(u[3 * i : 3 * i + 3])
            self.agents[i].set_angular_velocity(u[3 * n + 3 * i : 3 * n + 3 * i + 3])

    def translate_network(self, dp):
        for agent in self.agents:
            agent.pose.position += np.array(dp)

    def rotate_network(self, axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

        positions = np.array([agent.pose.position for agent in self.agents])
        center = np.mean(positions, axis=0)

        K = skew_symmetric(axis)
        # rodrigues
        R = (
            np.eye(3)
            + np.sin(angle) * K
            + (1 - np.cos(angle)) * (K @ K)
        )
        # rotate around center
        positions = (positions - center) @ R.T + center

        for i, agent in enumerate(self.agents):
            agent.pose.position = positions[i]

            if agent.domain not in ["R^3", "R^2"]:
                # rotate the orientations also
                R_agent = agent.pose.rotation_mat()
                agent.pose.set_rotation_mat(R @ R_agent)

    def scale_network(self, scale):
        for agent in self.agents:
            agent.pose.position.x *= scale[0]
            agent.pose.position.y *= scale[1]
            agent.pose.position.z *= scale[2]

    def set_agents_domain_homogeneous(self, domain: str, rotation_axis=None):
        print(f"agents' domain: {domain}")
        # default values are for SE(3)
        if domain == "R^3xSO(3)" or domain == "" or domain == None:
            domain = "SE(3)"

        if (
            domain != "SE(3)"
            and domain != "R^3xS^1"
            and domain != "R^2xS^1"
            and domain != "R^3"
            and domain != "R^2"
        ):
            print(f"given agents' domain {domain} is not valid.")
            quit()

        for agent in self.agents:
            agent.set_domain(domain, rotation_axis)

    def extended_bearing_rigidity_matrix(self):
        # return rigidity.old_extended_bearing_rigidity_matrix(self)
        return rigidity.extended_bearing_rigidity_matrix(self)

    def is_IBR(self):
        # return rigidity.old_is_IBR(self)
        return rigidity.is_IBR(self)

    def get_bearings(self):
        bearings = np.zeros(len(self.edges)*3)
        for k, (i, j) in enumerate(self.edges):
            bearings[3*k:3*k+3] = self.agents[i].get_bearing(self.agents[j])
        return bearings

    def print(self):
        print(f"NETWORK")
        for i, agent in enumerate(self.agents):
            print(
                f"agent {i} in {agent.domain} with rotation axis {agent.rotation_axis}"
            )
            agent.pose.print()
        print("edges: [")
        for k, (i, j) in enumerate(self.edges):
            print(f"({i}, {j}),")
        print("]")
