import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import signal

from visualizer import Visualizer
from scenario import load_scenario
from control import GradientBasedController


class Environment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, scenario_file, sim_step=0.001, max_time=10.0, visualize=True):
        super().__init__()

        self.sim_step = sim_step
        self.max_time = max_time
        self.filename = scenario_file
        self.visualize = visualize

        self.network, self.goal_network = load_scenario(self.filename)
        self.n = len(self.network.agents)
        self.m = len(self.network.edges)

        self.brm = self.network.extended_bearing_rigidity_matrix()

        self.observation_space = spaces.Dict({
            "brm": spaces.Box(-np.inf, np.inf, (min(self.brm.shape),), dtype=np.float32),
            "controller": spaces.Box(-np.inf, np.inf, (6*self.n,), dtype=np.float32),
            "bearings": spaces.Box(-np.inf, np.inf, (3*self.m,), dtype=np.float32),
            "goal_bearings": spaces.Box(-np.inf, np.inf, (3*self.m,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-10.0, 10.0, (6*self.n,), dtype=np.float32)

        self.controller = GradientBasedController(
            np.asarray(self.goal_network.get_bearings()),
            lin_velocity_gain=100,
            ang_velocity_gain=1,
        )

        self.sim_time = 0.0
        self.converged = False

        if self.visualize:
            self.vis = Visualizer()
            signal.signal(signal.SIGINT, self.vis.handle_sigint)

            self.render_interval = 1.0 / self.metadata["render_fps"]
            self.last_render_time = time.time()

    def _get_obs(self):
        self.brm = self.network.extended_bearing_rigidity_matrix()
        u, s, v = np.linalg.svd(self.brm)
        controller_velocities = self.controller.control(self.network)
        return {"brm": s,
                "controller": controller_velocities,
                "bearings": self.network.get_bearings(),
                "goal_bearings": self.goal_network.get_bearings()}

    def _compute_reward(self, action, error):
        reward = -np.sum(error) # error
        reward -= 0.01 * np.linalg.norm(action) # control effort
        reward -= 0.1 * self.sim_step # time penalty

        if self.converged:
            reward += 100.0

        return reward

    # -----------------------------------
    def step(self, action):
        if not self.converged:
            self.network.set_inputs(self.controller.control(self.network) + action)

        self.network.step(self.sim_step)
        self.sim_time += self.sim_step

        error = self.controller.error(self.network)

        if np.sum(error) < 1e-2:
            self.converged = True

        obs = self._get_obs()
        reward = self._compute_reward(action, error)

        terminated = self.converged
        truncated = self.sim_time >= self.max_time

        info = {
            "error": error,
            "sim_time": self.sim_time,
            "is_rigid": self.network.is_IBR(),
        }

        if self.visualize:
            self._render_frame(action, error)

        return obs, reward, terminated, truncated, info

    # -----------------------------------
    def _render_frame(self, velocities, error):
        curr_time = time.time()

        if curr_time - self.last_render_time < self.render_interval:
            return

        self.last_render_time = curr_time

        self.vis.draw_viser(
            self.goal_network,
            node_color=(0, 255, 0),
            edge_color=(0, 128, 0),
            label_prefix="Goal",
        )
        self.vis.draw_viser(
            self.network,
            node_color=(255, 0, 0),
            edge_color=(128, 0, 0),
            label_prefix="Current",
        )

        vels_info = "\n".join(
            f"vel ({i}): {velocities[3*i:3*i+3]} | "
            f"{velocities[3*self.n+3*i:3*self.n+3*i+3]}"
            for i in range(self.n)
        )
        self.vis.draw_info(
            f"""sim time: {self.sim_time}\n
            converged: {self.converged}\n
            error: {error}\n
            network is rigid: {self.network.is_IBR()}\n
            goal network is rigid: {self.goal_network.is_IBR()}\n
            {vels_info}
            """
        )

        self.vis.server.flush()

    # -----------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.network, self.goal_network = load_scenario(self.filename)

        self.sim_time = 0.0
        self.converged = False

        if self.visualize:
            self.vis.draw_viser(
                self.goal_network,
                node_color=(0, 255, 0),
                edge_color=(0, 128, 0),
                label_prefix="Goal",
            )
            self.vis.draw_viser(
                self.network,
                node_color=(255, 0, 0),
                edge_color=(128, 0, 0),
                label_prefix="Current",
            )
            self.vis.draw_info("reset")
            self.vis.server.flush()

        return self._get_obs(), {}

    # -----------------------------------
    def close(self):
        if self.visualize:
            self.vis.stop()