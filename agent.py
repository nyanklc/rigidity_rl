import numpy as np
from util import circle_polygon, move_polygon, Pose
import quaternion
from util import angular_velocity_to_quaternion


class Agent:
    def __init__(self, pose=None):
        self.pose = pose if pose is not None else Pose()
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

    def step(self, dt):
        self.pose.step(self.velocity, self.angular_velocity, dt)

    def set_velocity(self, vel):
        self.velocity = vel

    def set_angular_velocity(self, ang_vel):
        self.angular_velocity = ang_vel

    def get_footprint(self):
        x, y, _ = self.pose.position
        yaw = quaternion.as_euler_angles(self.pose.orientation)[2]
        polygon = circle_polygon()
        footprint = move_polygon(polygon, x, y, yaw)
        return footprint
