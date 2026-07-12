import numpy as np
import quaternion
import matplotlib.pyplot as plt
import math
import torch


class Pose:
    # TODO: make sure trans and rot dofs are 3x3
    def __init__(self, position=None, orientation_euler=None):
        self.position = np.array(position if position is not None else [0.0, 0.0, 0.0],
                                 dtype=float)
        self.orientation = (
            quaternion.from_euler_angles(orientation_euler)
            if orientation_euler is not None
            else quaternion.quaternion(1, 0, 0, 0)
        )

    def homo_transform(self):
        T = np.eye(4)
        T[:3, :3] = quaternion.as_rotation_matrix(self.orientation)
        T[:3, 3] = self.position
        return T

    # w in world frame
    def step(self, v, w, dt):
        self.position += v * dt
        dq = angular_velocity_to_quaternion(w, dt)
        self.orientation = dq * self.orientation  # w_W * R_WB -> B2W
        self.orientation = self.orientation.normalized()

    def rotation_mat(self):
        return quaternion.as_rotation_matrix(self.orientation)

    def set_rotation_mat(self, R):
        q = quaternion.from_rotation_matrix(R)
        self.orientation = q

    def euler_angles(self):
        return quaternion.as_euler_angles(self.orientation)

    def print(self):
        print(
            f"x: {self.position[0]}\ny: {self.position[1]}\nz: {self.position[2]}\nangles: {quaternion.as_euler_angles(self.orientation)}"
        )

    def __str__(self):
        return (
            f"x: {self.position[0]}\n"
            f"y: {self.position[1]}\n"
            f"z: {self.position[2]}\n"
            f"angles: {quaternion.as_euler_angles(self.orientation)}"
        )

# class RandomActionWrapper:
#     def __init__(self, env, random_steps):
#         self.env = env
#         self.random_steps = random_steps
#         self.t = 0

#     def step(self, action):
#         if self.t < self.random_steps:
#             action = self.env.action_space.sample()
#         self.t += 1
#         return self.env.step(action)

#     def reset(self, *args, **kwargs):
#         return self.env.reset(*args, **kwargs)

#     def __getattr__(self, name):
#         return getattr(self.env, name)

# ang vel in world frame
def angular_velocity_to_quaternion(w, dt):
    theta = np.linalg.norm(w) * dt

    # SVD convergence?
    if theta < 1e-8:
        return quaternion.quaternion(1, 0, 0, 0)

    axis = w / np.linalg.norm(w)

    half_theta = theta / 2.0
    w = np.cos(half_theta)
    xyz = axis * np.sin(half_theta)

    return quaternion.quaternion(w, *xyz)


def plot_graph(positions, edges):
    plt.figure()

    plt.scatter(positions[:, 0], positions[:, 1])

    for i, (x, y) in enumerate(positions):
        plt.text(x, y, str(i), fontsize=12, ha="right")

    for i, j in edges:
        x = [positions[i, 0], positions[j, 0]]
        y = [positions[i, 1], positions[j, 1]]
        plt.plot(x, y)

    plt.title("graph")
    plt.gca().set_aspect("equal")
    plt.show()


def skew_symmetric(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def orthogonal_projection_matrix(v):
    norm_sq = np.dot(v, v)
    return np.eye(3) - np.outer(v, v) / norm_sq


def move_polygon(polygon, x, y, yaw, rotation_axis: tuple = None):
    rotate_polygon(polygon, yaw, rotation_axis)
    translate_polygon(polygon, x, y)
    return polygon


def circle_polygon():
    radius = 10
    rot_count = 20
    polygon_points: list = []
    angle = math.pi * 2 / rot_count
    x = 0
    y = radius
    polygon_points.append([x, y])
    temp: float
    for i in range(rot_count):
        temp = x
        x = x * math.cos(angle) - y * math.sin(angle)
        y = temp * math.sin(angle) + y * math.cos(angle)
        polygon_points.append([x, y])
    return polygon_points


def rotate_polygon(polygon, angle, rotation_axis: tuple = None):
    if rotation_axis is not None:
        # translate to origin
        for point in polygon:
            point[0] -= rotation_axis[0]
            point[1] -= rotation_axis[1]
    # rotate
    for i in range(len(polygon)):
        temp = polygon[i][0]
        polygon[i][0] = polygon[i][0] * math.cos(angle) - polygon[i][1] * math.sin(
            angle
        )
        polygon[i][1] = temp * math.sin(angle) + polygon[i][1] * math.cos(angle)
    if rotation_axis is not None:
        # translate back
        for point in polygon:
            point[0] += rotation_axis[0]
            point[1] += rotation_axis[1]


def translate_polygon(polygon, x, y):
    for i in range(len(polygon)):
        polygon[i][0] += x
        polygon[i][1] += y


def invert_color(color):
    return (255 - color[0], 255 - color[1], 255 - color[2])


def adj_to_edge_index(adj):
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
    edge_index = adj.nonzero().t().contiguous()
    return edge_index

def batched_adj_to_edge_index(adj_batch):
    batch_size, n, _ = adj_batch.shape
    edge_indices = []
    for i in range(batch_size):
        # get edges for this specific graph
        edges = adj_batch[i].nonzero().t()
        # offset indices by the number of nodes already processed
        edges += i * n
        edge_indices.append(edges)
    return torch.cat(edge_indices, dim=1)

def discretize_array(vec):
    bins = np.linspace(0, 1, 0.05)
    return np.digitize(vec, bins)
