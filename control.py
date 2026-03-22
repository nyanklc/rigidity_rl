import numpy as np
from network import Network

class Controller:
    def __init__(self, goal=None, lin_velocity_gain=100, ang_velocity_gain=1):
        self.goal = goal
        self.lin_gain = lin_velocity_gain
        self.ang_gain = ang_velocity_gain

    def set_goal(self, goal):
        self.goal = goal

    def control(self, network: Network):
        brm = network.bearing_rigidity_matrix()
        # print(f"brm shape: {brm.shape} rank: {np.linalg.matrix_rank(brm)}")
        # print(f"goal shape: {self.goal.shape}")
        # print(f"brm: {brm}")
        # print(f"brm is rigid: {network.is_IBR()}")
        gains = [self.lin_gain, self.lin_gain, self.lin_gain, self.ang_gain, self.ang_gain, self.ang_gain]
        gain_mask = np.tile(gains, len(network.agents))
        return np.diag(gain_mask) @ (-brm.T @ self.goal)

    def error(self, current):
        return np.linalg.norm(self.goal - current)
