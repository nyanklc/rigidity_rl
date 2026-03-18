import numpy as np
import quaternion
import matplotlib.pyplot as plt
import math


class Pose:
    def __init__(self, position=None, orientation=None):
        self.position = np.array(position if position is not None else [0.0, 0.0, 0.0])

        self.orientation = (
            orientation
            if orientation is not None
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
        self.orientation = dq * self.orientation  # w_W * R_WB -> B2W in the end
        self.orientation = self.orientation.normalized()

    def rotation_mat(self):
        return quaternion.as_rotation_matrix(self.orientation)

    def print(self):
        print(
            f"x: {self.position[0]}\ny: {self.position[1]}\nz: {self.position[2]}\nq: {self.orientation}"
        )


# ang vel in world frame
def angular_velocity_to_quaternion(omega, dt):
    theta = np.linalg.norm(omega) * dt

    if theta < 1e-8:
        return quaternion.quaternion(1, 0, 0, 0)

    axis = omega / np.linalg.norm(omega)

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


def skew_symmetric(vec):
    skew = None
    if vec.shape[0] == 2:
        skew = np.asarray([[vec[0], -vec[1]], [vec[1], vec[0]]], dtype=vec.dtype)
    elif vec.shape[0] == 3:
        skew = np.asarray(
            [[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]],
            dtype=vec.dtype,
        )
    return skew


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
