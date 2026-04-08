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

    def randomize_position(self, low=[-100, -100, -100], high=[100, 100, 100]):
        if self.domain in ["R^2", "R^2xS^1"]:
            pos = np.random.uniform(low[:2], high[:2], size=2)
            self.pose.position = np.hstack([pos, [0.0]])
        else:
            self.pose.position = np.random.uniform(low[:3], high[:3], size=3)

    def randomize_orientation(self):
        if self.domain == "SE(3)":
            euler = np.random.uniform(0, 2*np.pi, size=3)
            self.pose.orientation = quaternion.from_euler_angles(euler)
        elif self.domain in ["R^3xS^1", "R^2xS^1"]:
            axis = self.rotation_axis if self.rotation_axis is not None else np.array([0,0,1])
            angle = np.random.uniform(0, 2*np.pi)
            self.pose.orientation = quaternion.from_rotation_vector(axis * angle)
        else:
            self.pose.orientation = quaternion.one


class Network:
    def __init__(self, positions, orientations_euler, edges):
        n = len(positions)
        self.edges = np.zeros((n, n), dtype=bool)
        if edges.shape[1] == n: # adj matrix provided
            self.edges = edges
        else: # edge list provided
            for k, (i, j) in enumerate(edges):
                self.edges[i, j] = True
        self.agents: list[Agent] = []
        for i in range(len(positions)):
            self.agents.append(Agent(Pose(positions[i], orientations_euler[i])))
        self.nr_max_edges = n**2

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

    def randomize_positions(self, low, high):
        for agent in self.agents:
            agent.randomize_position(low, high)

    def randomize_orientations(self):
        for agent in self.agents:
            agent.randomize_orientation()

    def set_edges(self, i_indices, j_indices):
        n = len(self.agents)
        self.edges = np.zeros((n, n), dtype=bool)
        m = len(i_indices)
        if m == 0:
            return
        self.edges[i_indices, j_indices] = True

    def add_edge(self, i_idx, j_idx):
        self.edges[i_idx, j_idx] = True

    def remove_edge(self, i_idx, j_idx):
        self.edges[i_idx, j_idx] = False

    def edge_exists(self, i_idx, j_idx):
        return self.edges[i_idx, j_idx]

    def set_agents_domain_homogeneous(self, domain: str, rotation_axis=None):
        # print(f"agents' domain: {domain}")
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

    def is_MBR(self):
        # for agent in self.agents:
        #     if agent.domain not in ["R^2", "R^3"]:
        #         raise Exception("Minimally Bearing Rigidity is not defined for domains other than R^d.")

        return rigidity.is_MBR(self)

    def get_bearings(self):
        i_indices, j_indices = np.nonzero(self.edges)
        m = len(i_indices)
        bearings = np.zeros(3 * m)
        for k, (i, j) in enumerate(zip(i_indices, j_indices)):
            bearings[3*k:3*k+3] = self.agents[i].get_bearing(self.agents[j])
        return bearings

    def print(self):
        print(f"NETWORK")
        for i, agent in enumerate(self.agents):
            print(
                f"agent {i} in {agent.domain} with rotation axis {agent.rotation_axis}"
            )
            agent.pose.print()
        print("edges:")
        n = len(self.agents)
        for i in range(n):
            for j in range(n):
                if self.edges[i, j]:
                    print(f"{i} -> {j}")
