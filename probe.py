"""Periodic mid-training evaluation of the policy as a *decision rule*.

Training metrics are computed from the exploring policy on best-state-visited, which
cannot tell "found a good graph" from "searched until it stumbled on one". This probe
rolls the policy out deterministically on a fixed set of instances and reports the gap
against sampling and against uniform random.

"""

import numpy as np
import torch

from environment import Environment


def deterministic_action(agent, obs):
    """argmax over the model's own scores, for PPO and DQN alike.

    skrl's CategoricalMixin.act always *samples*, so a PPO agent asked to act behaves as
    it did during training. Going through compute() makes both algorithms deterministic
    in the same way and keeps the models' action masking intact.
    """
    role = "policy" if "policy" in agent.models else "q_network"
    with torch.no_grad():
        scores, _ = agent.models[role].compute({"observations": obs}, role=role)
    return scores, torch.argmax(scores, dim=-1, keepdim=True)


class Probe:
    """Holds one evaluation environment for the lifetime of a run.

    `interval` and `episodes` are deliberately small: the point is a trend line, not a
    publication number. Instances are re-seeded identically on every probe so the curve
    tracks the policy rather than instance noise.
    """

    def __init__(self, env_path, device="cpu", interval=25_000, episodes=3, seed=12345):
        self.interval = interval
        self.episodes = episodes
        self.seed = seed
        self.device = device
        self._next = interval
        self.raw = Environment()
        self.raw.load(env_path)
        self.raw.device = device
        # no writer: the probe must not pollute the training episode metrics
        self.raw.track_data_enable = False
        from skrl.envs.wrappers.torch import wrap_env
        self.env = wrap_env(self.raw)
        self.steps = int(self.raw.max_steps)

    # ------------------------------------------------------------------
    def _rollout(self, agent, mode):
        """One episode. Returns (best score, stats at best, useful-action rate, max|logit|)."""
        obs, _ = self.env.reset()
        best, best_stats, useful, max_logit = -np.inf, None, 0, 0.0
        prev = self.raw.last_stats["score"]
        seen = set()

        for t in range(self.steps):
            if mode == "random":
                action = torch.tensor([[self.raw.action_space.sample()]], device=self.device)
            else:
                scores, greedy = deterministic_action(agent, obs)
                finite = scores[torch.isfinite(scores)]
                if finite.numel():
                    max_logit = max(max_logit, finite.abs().max().item())
                if mode == "argmax":
                    action = greedy
                else:
                    with torch.no_grad():
                        action, _ = agent.act(obs, states=self.env.state(),
                                              timestep=t, timesteps=self.steps)

            obs, _, terminated, truncated, _ = self.env.step(action)
            s = self.raw.last_stats
            useful += int(s["score"] > prev)
            prev = s["score"]
            if s["score"] > best:
                best, best_stats = s["score"], dict(s)

            done = terminated.any().item() if torch.is_tensor(terminated) else terminated
            trunc = truncated.any().item() if torch.is_tensor(truncated) else truncated
            if done or trunc:
                break
            if mode == "argmax":
                # deterministic policy in a deterministic env: once a state repeats,
                # nothing new can be found
                key = (self.raw.network.edges.tobytes(), self.raw.selection.tobytes())
                if key in seen:
                    break
                seen.add(key)

        return best, best_stats, useful / max(t + 1, 1), max_logit

    # ------------------------------------------------------------------
    def maybe_run(self, agent, timestep, writer):
        if writer is None or timestep < self._next:
            return
        self._next = timestep + self.interval

        was_training = True
        try:
            agent.enable_models_training_mode(False)
        except Exception:
            was_training = False

        # DQN's models are deterministic already, so argmax and "sample" coincide;
        # reporting a fake gap would be misleading
        has_sampling = "policy" in agent.models
        out = {}
        try:
            for mode in (("argmax", "sample", "random") if has_sampling else ("argmax", "random")):
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
                rows = [self._rollout(agent, mode) for _ in range(self.episodes)]
                out[mode] = rows
        finally:
            if was_training:
                agent.enable_models_training_mode(True)

        def mean(mode, fn):
            return float(np.mean([fn(r) for r in out[mode]]))

        a = out["argmax"]
        writer.add_scalar("Probe/ argmax score", mean("argmax", lambda r: r[0]), timestep)
        writer.add_scalar("Probe/ argmax edges", mean("argmax", lambda r: r[1]["m"]), timestep)
        writer.add_scalar("Probe/ argmax rigid", mean("argmax", lambda r: float(r[1]["is_IBR"])), timestep)
        writer.add_scalar("Probe/ argmax minimal", mean("argmax", lambda r: float(r[1]["is_MBR"])), timestep)
        writer.add_scalar("Probe/ useful (argmax)", mean("argmax", lambda r: r[2]), timestep)
        writer.add_scalar("Probe/ useful (random)", mean("random", lambda r: r[2]), timestep)
        # the drift that silently killed two runs; -inf mask entries are excluded
        writer.add_scalar("Probe/ max abs logit", mean("argmax", lambda r: r[3]), timestep)

        if has_sampling:
            writer.add_scalar("Probe/ sample score", mean("sample", lambda r: r[0]), timestep)
            # ~0 means a genuine policy; strongly negative means a sampling search
            writer.add_scalar("Probe/ argmax-sample gap",
                              mean("argmax", lambda r: r[0]) - mean("sample", lambda r: r[0]),
                              timestep)
        else:
            writer.add_scalar("Probe/ argmax-sample gap", 0.0, timestep)
