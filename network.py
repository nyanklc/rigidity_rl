import numpy as np
from util import circle_polygon, move_polygon, Pose
import quaternion
import rigidity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    def plot_network_3d(
        self,
        ax=None,
        title="3D Formation",
        node_color='gray',
        edge_color='gray',
        node_alpha=1.0,
        edge_alpha=0.5,
        label_prefix=""
    ):
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        positions = np.array([agent.pose.position for agent in self.agents])

        for i, j in self.edges:
            p1, p2 = positions[i], positions[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=edge_color, linestyle='--', alpha=edge_alpha)

        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                color=node_color, s=50, alpha=node_alpha)

        offset = np.ptp(positions) * 0.02 if len(positions) > 0 else 0.1
        for i, p in enumerate(positions):
            ax.text(p[0] + offset, p[1] + offset, p[2] + offset, f"{label_prefix}{i}",
                    alpha=node_alpha)

        # scale = 10
        # for agent in self.agents:
        #     p = agent.pose.position
        #     R = agent.pose.rotation_mat()
        #     for idx, color in enumerate(['r', 'g', 'b']):
        #         direction = R[:, idx]
        #         ax.plot(
        #             np.array([p[0], p[0]+direction[0]]) / np.linalg.norm(np.array([p[0], p[0]+direction[0]]))*scale,
        #             np.array([p[1], p[1]+direction[1]]) / np.linalg.norm(np.array([p[1], p[1]+direction[1]]))*scale,
        #             np.array([p[2], p[2]+direction[2]]) / np.linalg.norm(np.array([p[2], p[2]+direction[2]]))*scale,
        #             color=color, alpha=node_alpha
        #         )

        # axis_len = 100
        # ax.quiver(0, 0, 0, axis_len, 0, 0, color='red', arrow_length_ratio=0.1)
        # ax.quiver(0, 0, 0, 0, axis_len, 0, color='green', arrow_length_ratio=0.1)
        # ax.quiver(0, 0, 0, 0, 0, axis_len, color='blue', arrow_length_ratio=0.1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.set_box_aspect([1, 1, 1])

        return ax
