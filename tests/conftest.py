"""Shared fixtures.

Two rules everything here exists to enforce:

1. `environments/`, `scenarios/`, `models/` and `train/` are gitignored, so the fast
   suite must never read them -- environments are built programmatically and configs
   are written to tmp_path. Tests that genuinely need a trained checkpoint skip.
2. Nothing may leave artefacts behind in runs/ train/ models/.
"""

import json
import os
import shutil
import sys
import uuid

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from environment import Environment  # noqa: E402

# ---------------------------------------------------------------- sweep constants

PLANAR = {"R^2", "R^2xS^1"}
ALL_DOMAINS = ["R^2", "R^3", "R^2xS^1", "R^3xS^1", "SE(3)"]
ORIENTED_DOMAINS = ["R^2xS^1", "R^3xS^1", "SE(3)"]   # bearings are frame-relative here
RD_DOMAINS = ["R^2", "R^3"]                          # no frame: bearings are global
BACKBONES = ["Equivariant", "GINE", "Default"]

# every action space the environment dispatches
ACTION_SPACES = [
    "SelectNodesSequentially",
    "AddRemoveEdgeDiscreteNoSelfLoops",
    "AddRemoveEdgeDiscrete",
    "AddEdgeDiscrete",
    "AddEdgeDiscreteNoSkip",
    "AddEdgeDiscreteNoSelfLoops",
    "AddEdgeDiscreteNoSkipNoSelfLoops",
    "AddRemoveEdgeMultiDiscrete",
    "AllEdges",
    "DecideOnEdge",
]

STATE_SCORES = [
    "Weighted", "WeightedNormalized", "WeightedNormalizedSpectral", "Rigid", "RigidAndMinEigenvalue",
    "RigidAndMinRigid", "RigidAndLogMinEigenvalueAndEdges", "MinRigid",
    "MinRigidAndMinEigenvalue", "MinEigenvalue", "Eigenvalues", "EdgeCount",
    "LogMinEigenvalue", "RigidityMatrixRank", "RigidityMatrixRankAndEdges", "None",
]

TERMINATIONS = [
    "MaxSteps", "MaxStepsRankBonus", "Rigid", "RigidMinEigBonus",
    "MinimallyRigid", "RigidMinEigAndEdgesBonus", "Bandit",
]

# closed forms from (dof per agent)*n - trivial
RANK_K_FORMULA = {
    "R^2":     lambda n: 2 * n - 3,
    "R^3":     lambda n: 3 * n - 4,
    "R^2xS^1": lambda n: 3 * n - 4,
    "R^3xS^1": lambda n: 4 * n - 5,
    "SE(3)":   lambda n: 6 * n - 7,
}
# a bearing is one angle in the plane, two in 3-space
C_MAX = {"R^2": 1, "R^2xS^1": 1, "R^3": 2, "R^3xS^1": 2, "SE(3)": 2}

# dim D_i: how many coordinates agent i can actually vary
DOF_PER_AGENT = {"R^2": 2, "R^3": 3, "R^2xS^1": 3, "R^3xS^1": 4, "SE(3)": 6}

# No closed form for rank_K on a mix, so the tests assert the DOF budget instead.
MIXES = [
    ["R^2", "R^3", "SE(3)"],
    ["R^2"] * 3 + ["R^3"] * 3,
    ["R^2"] * 5 + ["R^3"],
    ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"],                      # mixed5
    ["R^2"] * 2 + ["R^2xS^1"] * 2 + ["R^3"] * 2 + ["R^3xS^1"] * 2 + ["SE(3)"] * 2,  # mixed
    ["R^2"] * 4 + ["SE(3)"] * 2,
    ["R^3"] * 3 + ["SE(3)"] * 3,
    ["R^2xS^1"] * 4 + ["R^3xS^1"] * 2,
]


def max_rank_K(domains):
    """Upper bound on rank(B_K): sum(DOF) minus the trivial motions.5."""
    dof = sum(DOF_PER_AGENT[d] for d in domains)
    trivial = 3 if any(d in PLANAR for d in domains) else 4
    return dof - trivial

TOL = 1e-9          # "exactly invariant" for float64 pipelines
LOOSE_TOL = 1e-6    # after a float32 round trip through a model


# ---------------------------------------------------------------- --slow plumbing

def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
                     help="also run the slow tests (training, brute force, large n)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: excluded unless --slow is passed")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="slow: pass --slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def seeded():
    """Every test starts from the same RNG state; the env uses global np.random."""
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def make_env():
    """Build an Environment in-process. Never touches environments/."""
    def _make(n=6, domains="R^3", **kw):
        opts = dict(
            action_space_type="SelectNodesSequentially",
            obs_space_type="Dict",
            state_score_type="WeightedNormalized",
            termination_condition_type="MaxSteps",
            max_steps=10 ** 6,
            track_data_enable=False,
            # every generated config sets this False; the initialize() default is
            # True, which would let a randomly sampled skip end the episode
            skip_is_stop=False,
        )
        opts.update(kw)
        e = Environment()
        if isinstance(domains, str) and domains not in ("SE(3)", "R^2", "R^3", "R^2xS^1", "R^3xS^1"):
            raise ValueError(domains)
        e.initialize(n, domains, **opts)
        return e
    return _make


def config_dict(n=6, domains="R^3", **overrides):
    """The env-config schema environment.py's __main__ writes."""
    cfg = {
        "action_type": "SelectNodesSequentially",
        "obs_type": "Dict",
        "state_score_type": "WeightedNormalized",
        "termination_condition_type": "MaxSteps",
        "n": n,
        "domains": domains,
        "action_rewards_enable": False,
        "skip_enabled": False,
        "skip_is_stop": False,
        "random_graph_with_mean_min_edges": True,
        "time_penalty_value": 0.0,
        "track_data_enable": False,
        "max_steps": 40,
        "truncate_enable": False,
        "truncate_max_steps": 100,
        "truncate_penalty_value": 100,
        "only_randomize_edges": False,
        "include_candidate_bearings": True,
        "graph_features": True,
        "rigidity_global": False,
        "rigidity_flex": False,
        "rigidity_edge": False,
        "rigidity_stiffness": False,
        "rigidity_removal": False,
        "stiffness_kappa": 0.0,
        "stiffness_ref_samples": 3,
        "scenario": None,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def env_config_file(tmp_path):
    """Write a config to tmp_path for the load() / Probe paths."""
    def _write(name="cfg", **overrides):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(config_dict(**overrides), indent=2))
        return str(p)
    return _write


@pytest.fixture
def temp_run_name():
    """A unique model name, with runs/ train/ models/ entries removed afterwards."""
    name = f"pytest_{uuid.uuid4().hex[:8]}"
    yield name
    for p in (f"runs/{name}", f"train/{name}.json", f"train/{name}.json.bak"):
        path = os.path.join(ROOT, p)
        shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else (
            os.path.exists(path) and os.remove(path))
    for algo in ("PPO", "DQN", "DDQN"):
        f = os.path.join(ROOT, f"models/complete/{algo}/{name}.pt")
        if os.path.exists(f):
            os.remove(f)
    for extra in os.listdir(os.path.join(ROOT, "runs")) if os.path.isdir(os.path.join(ROOT, "runs")) else []:
        if extra.startswith(name):
            shutil.rmtree(os.path.join(ROOT, "runs", extra), ignore_errors=True)


def has_artifacts():
    """True when this working copy has trained checkpoints to test against."""
    return (os.path.isdir(os.path.join(ROOT, "train"))
            and os.path.isdir(os.path.join(ROOT, "models", "complete"))
            and any(f.endswith(".json") and not f.endswith(".bak")
                    for f in os.listdir(os.path.join(ROOT, "train"))))


def find_checkpoint(prefer=()):
    """(model_name, env_config_name, algorithm) for a usable manifest+checkpoint pair."""
    import glob
    train_dir = os.path.join(ROOT, "train")
    if not os.path.isdir(train_dir):
        return None
    names = [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(train_dir, "*.json"))]
    ordered = [n for n in prefer if n in names] + [n for n in names if n not in prefer]
    for name in ordered:
        try:
            info = json.load(open(os.path.join(train_dir, f"{name}.json")))
        except Exception:
            continue
        algo = info.get("algorithm", "PPO")
        ckpt = os.path.join(ROOT, f"models/complete/{algo}/{name}.pt")
        if os.path.exists(ckpt) and info.get("environment_config"):
            return name, info["environment_config"], algo
    return None


requires_artifacts = pytest.mark.skipif(
    not has_artifacts(), reason="no trained checkpoints in this working copy (gitignored)")
