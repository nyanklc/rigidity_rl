"""Real training runs. Slow by construction -- these launch train_ppo/train_dqn."""
import glob
import json
import os
import re
import subprocess
import sys

import pytest

from conftest import ROOT, config_dict

pytestmark = pytest.mark.slow

COMBOS = [(algo, bb, act)
          for algo in ("ppo", "dqn")
          for bb in ("Equivariant", "GINE")
          for act in ("SelectNodesSequentially", "AddRemoveEdgeDiscreteNoSelfLoops")]


def write_env_config(name, **overrides):
    """Training scripts resolve ./environments/<name>.json, so it must live there."""
    d = os.path.join(ROOT, "environments")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.json")
    with open(path, "w") as f:
        # track_data_enable must be on or the env writes no Episode/ metrics
        json.dump(config_dict(n=5, domains="R^3", max_steps=12,
                              track_data_enable=True, **overrides), f)
    return path


def patch_script(script, backbone, timesteps=1500):
    """Set the backbone and shrink the run. Returns the original text to restore."""
    p = os.path.join(ROOT, script)
    src = open(p).read()
    out = re.sub(r'^BACKBONE = ".*"$', f'BACKBONE = "{backbone}"', src, count=1, flags=re.M)
    out = re.sub(r'^TOTAL_TIMESTEPS = .*$', f'TOTAL_TIMESTEPS = {timesteps}',
                 out, count=1, flags=re.M)
    out = re.sub(r'^PROBE_INTERVAL = .*$', 'PROBE_INTERVAL = 600', out, count=1, flags=re.M)
    open(p, "w").write(out)
    return src


@pytest.mark.parametrize("algo,backbone,action", COMBOS,
                         ids=[f"{a}-{b}-{c}" for a, b, c in COMBOS])
def test_training_runs_and_logs(tmp_path, temp_run_name, algo, backbone, action):
    script = f"train_{algo}.py"
    cfg_name = f"pytest_cfg_{temp_run_name}"
    cfg_path = write_env_config(cfg_name, action_type=action)
    original = patch_script(script, backbone)
    try:
        proc = subprocess.run(
            [sys.executable, script, cfg_name, temp_run_name],
            cwd=ROOT, capture_output=True, text=True, timeout=240,
            input="", env={**os.environ, "PYTEST_TRAINING_SMOKE": "1"})
        combined = proc.stdout + proc.stderr
        assert "Traceback" not in combined, combined[-2500:]
        assert "Training on" in combined
        assert os.path.exists(os.path.join(ROOT, "train", f"{temp_run_name}.json"))
    finally:
        open(os.path.join(ROOT, script), "w").write(original)
        if os.path.exists(cfg_path):
            os.remove(cfg_path)


def test_logged_tags_cover_the_decision_panel(temp_run_name):
    """One full run, checked for the metric groups we actually watch."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    cfg_name = f"pytest_cfg_{temp_run_name}"
    cfg_path = write_env_config(cfg_name, action_type="SelectNodesSequentially")
    original = patch_script("train_ppo.py", "Equivariant", timesteps=2000)
    try:
        proc = subprocess.run([sys.executable, "train_ppo.py", cfg_name, temp_run_name],
                              cwd=ROOT, capture_output=True, text=True, timeout=300, input="")
        assert "Traceback" not in proc.stdout + proc.stderr
        tags = set()
        for f in glob.glob(os.path.join(ROOT, "runs", temp_run_name, "events*")):
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
            tags |= set(ea.Tags()["scalars"])
        for k in ("Episode/ Best state score", "Decision/ useful", "Decision/ wasted",
                  "Actions/ add fraction", "Episode/ Edit efficiency",
                  "Probe/ argmax score", "Probe/ argmax-sample gap",
                  "Probe/ useful (random)", "Probe/ max abs logit"):
            assert k in tags, f"{k} missing from {sorted(tags)[:20]}"
    finally:
        open(os.path.join(ROOT, "train_ppo.py"), "w").write(original)
        if os.path.exists(cfg_path):
            os.remove(cfg_path)
