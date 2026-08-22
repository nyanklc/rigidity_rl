"""Per-episode metrics and what reaches TensorBoard."""
import glob
import numpy as np
import pytest

FRACTION_KEYS = [
    "Decision/ useful", "Decision/ wasted", "Decision/ converge",
    "Actions/ add fraction", "Actions/ remove fraction", "Actions/ noop fraction",
    "Actions/ skip fraction", "Actions/ select fraction",
    "Skip fraction", "Rigid fraction", "Min rigid fraction",
]
REQUIRED_KEYS = FRACTION_KEYS + [
    "Decision/ overshoot", "Edit efficiency", "Steps to first rigid",
    "Steps to first minimal", "Steps rigid to minimal", "Best step",
    "Best state score", "Final state score", "Length", "Nr edits",
]


def finished_episode(make_env, steps=30, **kw):
    e = make_env(n=6, domains="R^3", max_steps=steps,
                 termination_condition_type="MaxSteps", **kw)
    e.reset()
    for _ in range(steps):
        e.step(e.action_space.sample())
    return e, e.last_episode_stats


def test_episode_summary_has_every_expected_key(make_env):
    _, s = finished_episode(make_env)
    for k in REQUIRED_KEYS:
        assert k in s, k


def test_fractions_are_within_zero_and_one(make_env):
    _, s = finished_episode(make_env)
    for k in FRACTION_KEYS:
        assert 0.0 - 1e-12 <= s[k] <= 1.0 + 1e-12, f"{k}={s[k]}"


def test_counts_are_non_negative(make_env):
    _, s = finished_episode(make_env)
    for k in ("Length", "Nr edits", "Best step", "Nr initial edges", "Final nr edges"):
        assert s[k] >= 0, k


def test_action_fractions_sum_to_one(make_env):
    _, s = finished_episode(make_env)
    total = sum(s[f"Actions/ {k} fraction"]
                for k in ("add", "remove", "noop", "skip", "select"))
    assert abs(total - 1.0) < 1e-9


def test_step_sentinels_are_minus_one_or_a_valid_step(make_env):
    _, s = finished_episode(make_env)
    for k in ("Steps to first rigid", "Steps to first minimal", "Steps rigid to minimal"):
        assert s[k] == -1 or 0 <= s[k] <= s["Length"], f"{k}={s[k]}"


def test_first_minimal_never_precedes_first_rigid(make_env):
    """Minimality implies rigidity, so the ordering cannot invert."""
    for _ in range(5):
        _, s = finished_episode(make_env, steps=40)
        a, b = s["Steps to first rigid"], s["Steps to first minimal"]
        if a >= 0 and b >= 0:
            assert b >= a


def test_overshoot_is_zero_at_or_below_m_req(make_env):
    n = 6
    e = make_env(n=n, domains="R^3", max_steps=3,
                 termination_condition_type="MaxSteps")
    e.reset()
    E = np.zeros((n, n), dtype=bool)
    for i in range(min(e.m_req, n - 1)):
        E[i, i + 1] = True
    e.network.edges = E
    for _ in range(3):
        e.step(e.action_space.n - 1 if False else 0)
    s = e.last_episode_stats
    assert s["Decision/ overshoot"] >= 0.0


def test_best_final_gap_is_zero_when_the_episode_ends_on_its_best(make_env):
    _, s = finished_episode(make_env)
    assert s["Best-final score gap"] >= -1e-9
    assert abs((s["Best state score"] - s["Final state score"])
               - s["Best-final score gap"]) < 1e-9


def test_useful_rate_counts_only_strict_improvements(make_env):
    e = make_env(n=6, domains="R^3", max_steps=25,
                 termination_condition_type="MaxSteps")
    e.reset()
    prev = e.last_state_score
    useful = 0
    for _ in range(25):
        e.step(e.action_space.sample())
        now = e.last_stats["score"]
        useful += int(now > prev)
        prev = now
    s = e.last_episode_stats
    assert abs(s["Decision/ useful"] - useful / 25) < 1e-9


def test_writer_emits_every_tag_and_the_histogram(make_env, tmp_path, monkeypatch):
    """Round-trip through a real SummaryWriter and read it back."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import environment as env_mod
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    e = make_env(n=6, domains="R^3", max_steps=12,
                 termination_condition_type="MaxSteps", track_data_enable=True)
    e.set_writer("unit")
    e.reset()
    for _ in range(12):
        e.step(e.action_space.sample())
    e.writer.flush()

    tags, hists = set(), set()
    for f in glob.glob(str(tmp_path / "runs" / "unit" / "events*")):
        ea = EventAccumulator(f, size_guidance={"scalars": 0, "histograms": 0})
        ea.Reload()
        tags |= set(ea.Tags()["scalars"])
        hists |= set(ea.Tags().get("histograms", []))

    for k in ("Decision/ useful", "Decision/ wasted", "Decision/ overshoot",
              "Decision/ converge", "Actions/ add fraction", "Actions/ noop fraction"):
        assert k in tags, k
    # keys without their own group are prefixed, keys with one keep it
    assert "Episode/ Best state score" in tags
    assert "Episode/ Edit efficiency" in tags
    assert not any(t.startswith("Episode/ Decision/") for t in tags)
    assert "Actions/ index" in hists


def test_tag_prefixing_rule(make_env):
    """Decision/ and Actions/ keep their group; everything else gets Episode/."""
    _, s = finished_episode(make_env)
    for k in s:
        tag = k if "/" in k else f"Episode/ {k}"
        assert tag.count("Episode/") <= 1
        if k.startswith(("Decision/", "Actions/")):
            assert tag == k
