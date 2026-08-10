"""manifest.py -- the self-contained record of a run."""
import json
import numpy as np
import pytest

import manifest
from conftest import config_dict


def test_encode_decode_sources_round_trip():
    src = {"a.py": "print('x')\n", "policy/gnn_backbone.py": "# hello\n" * 200}
    assert manifest.decode_sources(manifest.encode_sources(src)) == src


def test_collect_sources_includes_the_files_a_run_depends_on():
    src = manifest.collect_sources()
    for f in ("environment.py", "network.py", "rigidity.py", "policy/gnn_backbone.py"):
        assert f in src and src[f].strip(), f


def test_encoded_sources_are_smaller_than_the_raw_text():
    src = manifest.collect_sources()
    raw = sum(len(v) for v in src.values())
    assert len(manifest.encode_sources(src)) < raw


def test_build_manifest_carries_the_required_blocks():
    base = {"algorithm": "PPO", "model_name": "unit"}
    m = manifest.build_manifest(dict(base), config_dict(), seed=7, device="cpu")
    assert m["manifest_version"] >= 2
    assert "sources_b64gz" in m
    assert m["provenance"]["seed"] == 7
    assert m["provenance"]["device"] == "cpu"
    assert m["provenance"]["captured_at_training"] is True
    assert "python" in m["provenance"]["packages"]


def test_sources_of_reads_the_bundle():
    m = manifest.build_manifest({"algorithm": "PPO"}, config_dict())
    got = manifest.sources_of(m)
    assert "policy/gnn_backbone.py" in got
    assert "class GNNBackboneEquivariant" in got["policy/gnn_backbone.py"]


def test_sources_of_falls_back_to_the_older_backbone_source_key():
    legacy = {"backbone_source": ["# archived", "x = 1"]}
    got = manifest.sources_of(legacy)
    assert "policy/gnn_backbone.py" in got
    assert "archived" in got["policy/gnn_backbone.py"]


def test_current_sources_differ_detects_an_edit():
    src = manifest.collect_sources()
    assert not manifest.current_sources_differ(src)
    edited = dict(src)
    edited["rigidity.py"] = src["rigidity.py"] + "\n# drift\n"
    drift = manifest.current_sources_differ(edited)
    assert "rigidity.py" in drift
    assert manifest.describe_drift(drift)


def test_collect_provenance_shape():
    p = manifest.collect_provenance(seed=3, device="cuda", captured_at_training=False)
    assert p["captured_at_training"] is False
    assert p["seed"] == 3 and p["device"] == "cuda"
    assert "git_commit" in p and "command" in p and "timestamp" in p


def test_scenario_raw_is_none_without_a_scenario():
    assert manifest.scenario_raw(config_dict(scenario=None)) is None
