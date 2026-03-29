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

    def error(self, current):
        d = self.goal - current
        return np.inner(d, d)


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
        theta = leader.pose.euler_angles()[2]

        # unicycle posture controller
        goal_orientation = self.leader_goal.rotation_mat()
        alpha = self.leader_goal.euler_angles()[2]
        theta = theta - alpha

        error = self.leader_goal.position - leader.pose.position
        error = goal_orientation @ error

        ro = np.linalg.norm(error)
        gamma = np.atan2(error[1], error[0]) - theta
        delta = gamma + theta

        k_delta = 1
        direction = leader.pose.rotation_mat() @ np.asarray([1, 0, 0])

        leader_vel = self.leader_vel_gain * ro * np.cos(gamma) * direction
        leader_ang_vel = np.asarray(
            [
                0,
                0,
                self.leader_ang_vel_gain * gamma
                + self.leader_vel_gain
                * np.sinc(gamma) * np.cos(gamma)
                * (gamma + k_delta * delta),
            ]
        )

        vels[3 * self.leader_idx : 3 * self.leader_idx + 3] = leader_vel
        vels[
            3 * len(network.agents)
            + 3 * self.leader_idx : 3 * len(network.agents)
            + 3 * self.leader_idx
            + 3
        ] = leader_ang_vel

        return vels
