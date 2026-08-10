"""Cost budgets. Ceilings are generous; the point is to catch a regression in kind,
not to benchmark. Actual timings always print. DESIGN_NOTES.md#graph-features."""
import time
import numpy as np
import pytest


def step_ms(env, iters=40, warmup=5):
    for _ in range(warmup):
        env.step(env.action_space.sample())
    t = time.perf_counter()
    for _ in range(iters):
        env.step(env.action_space.sample())
    return (time.perf_counter() - t) / iters * 1e3


@pytest.mark.parametrize("n,domain,ceiling_ms", [
    (4, "R^2", 12.0),
    (8, "R^3", 25.0),
    (8, "SE(3)", 40.0),
    (16, "R^3", 120.0),
])
def test_step_time_is_within_budget(make_env, capsys, n, domain, ceiling_ms):
    e = make_env(n=n, domains=[domain] * n, graph_features=False)
    e.reset()
    ms = step_ms(e, iters=20 if n >= 16 else 40)
    with capsys.disabled():
        print(f"\n    n={n:<3d} {domain:8s} lean step {ms:7.2f} ms (ceiling {ceiling_ms})")
    assert ms < ceiling_ms


def test_dropping_graph_features_is_faster_at_n16(make_env, capsys):
    """Closeness and Brandes betweenness are O(n^3) pure Python."""
    times = {}
    for gf in (True, False):
        e = make_env(n=16, domains="R^3", graph_features=gf)
        e.reset()
        times[gf] = step_ms(e, iters=12)
    with capsys.disabled():
        print(f"\n    n=16 graph_features on {times[True]:7.2f} ms  "
              f"off {times[False]:7.2f} ms  ({times[True]/times[False]:.1f}x)")
    assert times[False] < times[True]


def test_rigidity_features_cost_is_bounded(make_env, capsys):
    base = make_env(n=8, domains="R^3", graph_features=False)
    base.reset()
    t0 = step_ms(base)
    rich = make_env(n=8, domains="R^3", graph_features=False,
                    rigidity_global=True, rigidity_flex=True, rigidity_edge=True)
    rich.reset()
    t1 = step_ms(rich)
    with capsys.disabled():
        print(f"\n    n=8 rigidity flags off {t0:6.2f} ms  on {t1:6.2f} ms")
    assert t1 < 4.0 * t0 + 5.0


def test_episode_bookkeeping_is_constant_per_step(make_env, capsys):
    """new_episode_accum keeps sums and counts, so long episodes must not slow down."""
    e = make_env(n=6, domains="R^3", graph_features=False, max_steps=10 ** 6)
    e.reset()
    early = step_ms(e, iters=40)
    for _ in range(400):
        e.step(e.action_space.sample())
    late = step_ms(e, iters=40)
    with capsys.disabled():
        print(f"\n    step 40 {early:6.2f} ms   step 440 {late:6.2f} ms")
    assert late < 2.0 * early + 2.0


@pytest.mark.slow
@pytest.mark.parametrize("n", [32])
def test_large_n_still_steps(make_env, capsys, n):
    e = make_env(n=n, domains="R^3", graph_features=False)
    e.reset()
    ms = step_ms(e, iters=5)
    with capsys.disabled():
        print(f"\n    n={n} lean step {ms:8.2f} ms")
    assert ms < 2000.0
