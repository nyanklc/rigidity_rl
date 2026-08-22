"""probe.py -- the policy evaluated as a decision rule."""
import numpy as np
import pytest
import torch

from conftest import find_checkpoint, requires_artifacts
from probe import Probe, deterministic_action


class DummyAgent:
    """Minimal stand-in with the surface Probe touches."""
    def __init__(self, n_actions, role="policy"):
        torch.manual_seed(0)
        self.models = {role: self}
        self._role = role
        self._w = torch.randn(n_actions)

    def compute(self, inputs, role=""):
        b = inputs["observations"].shape[0]
        return self._w.unsqueeze(0).expand(b, -1).clone(), {}

    def act(self, obs, states=None, timestep=0, timesteps=1):
        b = obs.shape[0]
        return torch.randint(0, self._w.numel(), (b, 1)), {}

    def enable_models_training_mode(self, flag):
        pass


def test_probe_builds_from_a_config_file(env_config_file):
    p = Probe(env_config_file(n=5, max_steps=8), device="cpu", interval=1, episodes=1)
    assert p.raw.n == 5
    assert p.steps == 8
    assert p.raw.track_data_enable is False   # must not pollute training metrics


def test_deterministic_action_is_the_argmax():
    agent = DummyAgent(7)
    obs = torch.zeros(1, 3)
    scores, action = deterministic_action(agent, obs)
    assert int(action.reshape(-1)[0]) == int(scores.flatten().argmax())


def test_rollout_respects_the_step_budget(env_config_file):
    p = Probe(env_config_file(n=5, max_steps=6), device="cpu", interval=1, episodes=1)
    agent = DummyAgent(p.raw.action_space.n)
    best, stats, useful, max_logit = p._rollout(agent, "argmax")
    assert np.isfinite(best)
    assert set(stats) >= {"m", "rank", "is_IBR", "is_MBR"}
    assert 0.0 <= useful <= 1.0
    assert max_logit >= 0.0


def test_argmax_rollout_is_reproducible(env_config_file):
    p = Probe(env_config_file(n=5, max_steps=10), device="cpu", interval=1, episodes=1)
    agent = DummyAgent(p.raw.action_space.n)
    outs = []
    for _ in range(2):
        np.random.seed(p.seed)
        torch.manual_seed(p.seed)
        outs.append(p._rollout(agent, "argmax")[0])
    assert outs[0] == outs[1]


def test_random_mode_differs_from_argmax(env_config_file):
    p = Probe(env_config_file(n=6, max_steps=30), device="cpu", interval=1, episodes=1)
    agent = DummyAgent(p.raw.action_space.n)
    np.random.seed(0); torch.manual_seed(0)
    a = p._rollout(agent, "argmax")
    np.random.seed(0); torch.manual_seed(0)
    r = p._rollout(agent, "random")
    assert a[0] != r[0] or a[2] != r[2]


def test_max_abs_logit_ignores_the_infinite_mask(env_config_file):
    """-inf mask entries must not swamp the drift detector."""
    p = Probe(env_config_file(n=5, max_steps=5), device="cpu", interval=1, episodes=1)
    agent = DummyAgent(p.raw.action_space.n)
    agent._w[0] = float("-inf")
    _, _, _, max_logit = p._rollout(agent, "argmax")
    assert np.isfinite(max_logit)


def test_interval_gating(env_config_file):
    p = Probe(env_config_file(n=5, max_steps=4), device="cpu", interval=1000, episodes=1)
    agent = DummyAgent(p.raw.action_space.n)
    calls = []

    class W:
        def add_scalar(self, tag, value, step):
            calls.append(tag)

    p.maybe_run(agent, 10, W())
    assert calls == []                 # below the interval
    p.maybe_run(agent, 1000, W())
    assert any(t.startswith("Probe/") for t in calls)


def test_dqn_reports_a_zero_gap_rather_than_faking_a_sample(env_config_file):
    p = Probe(env_config_file(n=5, max_steps=6), device="cpu", interval=1, episodes=1)
    agent = DummyAgent(p.raw.action_space.n, role="q_network")
    logged = {}

    class W:
        def add_scalar(self, tag, value, step):
            logged[tag] = value

    p.maybe_run(agent, 1, W())
    assert logged["Probe/ argmax-sample gap"] == 0.0
    assert "Probe/ sample score" not in logged
    assert "Probe/ useful (random)" in logged


@pytest.mark.slow
@requires_artifacts
def test_calibration_against_known_checkpoints():
    """A good policy has gap ~0 and beats the random floor; a collapsed one does not."""
    import agent_loader
    found = find_checkpoint(prefer=("phase4_ppo_equi_n8_R3_at300k",))
    if not found:
        pytest.skip("no usable checkpoint")
    name, env_name, _ = found
    import os
    from conftest import ROOT
    cfg = os.path.join(ROOT, "environments", f"{env_name}.json")
    if not os.path.exists(cfg):
        pytest.skip("environment config for this checkpoint is not present")
    agent, _, _, _ = agent_loader.load_run(name, env_name, device="cpu")
    p = Probe(cfg, device="cpu", interval=1, episodes=2)
    res = {}
    for mode in ("argmax", "random"):
        np.random.seed(p.seed); torch.manual_seed(p.seed)
        res[mode] = [p._rollout(agent, mode) for _ in range(2)]
    useful_a = np.mean([r[2] for r in res["argmax"]])
    useful_r = np.mean([r[2] for r in res["random"]])
    max_logit = np.mean([r[3] for r in res["argmax"]])
    assert np.isfinite(useful_a) and np.isfinite(max_logit)
    # a healthy policy's logits stay in a sane range; the collapsed one hit 1e23
    if "at300k" in name:
        assert useful_a > useful_r
        assert max_logit < 1e12
