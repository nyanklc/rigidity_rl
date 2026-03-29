from abc import ABC, abstractmethod
import numpy as np
from network import Network
from util import Pose


class Controller(ABC):
    def __init__(self, goal=None, lin_velocity_gain=100, ang_velocity_gain=1):
        self.goal = goal
        self.lin_gain = lin_velocity_gain
        self.ang_gain = ang_velocity_gain

    def set_goal(self, goal):
        self.goal = goal

    @abstractmethod
    def control(self, network: Network):
        pass

    def error(self, network: Network):
        d = self.goal - network.get_bearings()
        return [np.inner(d, d)]


class GradientBasedController(Controller):
    def control(self, network: Network):
        brm = network.extended_bearing_rigidity_matrix()
        gains = [
            self.lin_gain,
            self.lin_gain,
            self.lin_gain,
            self.ang_gain,
            self.ang_gain,
            self.ang_gain,
        ]
        gain_mask = np.tile(gains, len(network.agents))
        vels = np.diag(gain_mask) @ (brm.T @ self.goal)
        return vels

class GradientBasedControllerWithLeader(Controller):
    def __init__(
        self,
        goal=None,
        lin_velocity_gain=100,
        ang_velocity_gain=1,
        leader_idx=0,
        leader_goal: Pose = None,
        leader_vel_gain=1,
        leader_ang_vel_gain=1,
    ):
        super().__init__(goal, lin_velocity_gain, ang_velocity_gain)
        self.leader_idx = leader_idx
        self.leader_goal = leader_goal
        self.leader_vel_gain = leader_vel_gain
        self.leader_ang_vel_gain = leader_ang_vel_gain

    def control(self, network: Network):
        brm = network.extended_bearing_rigidity_matrix()
        gains = [self.lin_gain, self.lin_gain, self.lin_gain, self.ang_gain, self.ang_gain, self.ang_gain]
        gain_mask = np.tile(gains, len(network.agents))
        vels = np.diag(gain_mask) @ (brm.T @ self.goal)

        leader = network.agents[self.leader_idx]

        def angle_diff(a, b):
            return (a - b + np.pi) % (2 * np.pi) - np.pi

        angle_error = angle_diff(self.leader_goal.euler_angles()[0], leader.pose.euler_angles()[0])
        leader_ang_vel_0 = self.leader_ang_vel_gain * angle_error
        angle_error = angle_diff(self.leader_goal.euler_angles()[1], leader.pose.euler_angles()[1])
        leader_ang_vel_1 = self.leader_ang_vel_gain * angle_error
        angle_error = angle_diff(self.leader_goal.euler_angles()[2], leader.pose.euler_angles()[2])
        leader_ang_vel_2 = self.leader_ang_vel_gain * angle_error
        leader_vel = self.leader_vel_gain * (self.leader_goal.position - leader.pose.position)

        vels[3 * self.leader_idx : 3 * self.leader_idx + 3] = leader_vel
        vels[
            3 * len(network.agents)
            + 3 * self.leader_idx : 3 * len(network.agents)
            + 3 * self.leader_idx
            + 3
        ] = np.array([leader_ang_vel_0, leader_ang_vel_1, leader_ang_vel_2])

        return vels

    def error(self, network: Network):
       leader = network.agents[self.leader_idx]

       d_bearing = self.goal - network.get_bearings()
       d_bearing = np.dot(d_bearing, d_bearing)

       d_pos = self.leader_goal.position - leader.pose.position
       d_pos = np.dot(d_pos, d_pos)

       d_ori = leader.pose.euler_angles() - self.leader_goal.euler_angles()
       d_ori = sum([1 - np.cos(angle) for angle in d_ori])

       return [d_bearing, d_pos, d_ori]


class GradientBasedControllerWithUnicycleLeader(Controller):
    def __init__(
        self,
        goal=None,
        lin_velocity_gain=100,
        ang_velocity_gain=1,
        leader_idx=0,
        leader_goal: Pose = None,
        leader_vel_gain=1,
        leader_ang_vel_gain=1,
    ):
        super().__init__(goal, lin_velocity_gain, ang_velocity_gain)
        self.leader_idx = leader_idx
        self.leader_goal = leader_goal
        self.leader_vel_gain = leader_vel_gain
        self.leader_ang_vel_gain = leader_ang_vel_gain

    def control(self, network: Network):
        brm = network.extended_bearing_rigidity_matrix()
        gains = [self.lin_gain, self.lin_gain, self.lin_gain, self.ang_gain, self.ang_gain, self.ang_gain]
        gain_mask = np.tile(gains, len(network.agents))
        vels = np.diag(gain_mask) @ (brm.T @ self.goal)

        leader = network.agents[self.leader_idx]
        theta = leader.pose.euler_angles()[2]
        direction = leader.pose.rotation_mat() @ np.asarray([1, 0, 0])

        # unicycle posture controller
        ep = self.leader_goal.position - leader.pose.position
        dx = ep[0]
        dy = ep[1]
        dtheta = -(self.leader_goal.euler_angles()[2] - theta)

        rho = np.sqrt(dx**2 + dy**2)
        gamma = np.atan2(dy, dx) - theta
        delta = dtheta - gamma

        k1 = self.leader_vel_gain
        k2 = self.leader_ang_vel_gain
        k3 = 1

        v = k1 * rho * np.cos(gamma)
        w = k2 * gamma + k1 * np.sin(gamma) * np.cos(gamma) / gamma * (
            gamma + k3 * delta
        )

        leader_vel = v * direction
        leader_ang_vel = np.asarray([0, 0, w])

        vels[3 * self.leader_idx : 3 * self.leader_idx + 3] = leader_vel
        vels[
            3 * len(network.agents)
            + 3 * self.leader_idx : 3 * len(network.agents)
            + 3 * self.leader_idx
            + 3
        ] = leader_ang_vel

        return vels

    def error(self, network: Network):
       leader = network.agents[self.leader_idx]

       d_bearing = self.goal - network.get_bearings()
       d_bearing = np.dot(d_bearing, d_bearing)

       d_pos = self.leader_goal.position - leader.pose.position
       d_pos = np.dot(d_pos, d_pos)

       d_ori = leader.pose.euler_angles() - self.leader_goal.euler_angles()
       d_ori = sum([1 - np.cos(angle) for angle in d_ori])

       return [d_bearing, d_pos, d_ori]
