"""The ablation tool must not lie about which channel it perturbed.

Every number ablation.py prints is attributed to a named channel, so a stale
slice table would silently blame the wrong feature -- worse than no tool. These
check the attribution, not the conclusions.
"""
import numpy as np
import pytest
import torch
from skrl.utils.spaces.torch import flatten_tensorized_space, unflatten_tensorized_space

import ablation
from conftest import ALL_DOMAINS


def obs_dict(env):
    return {k: torch.as_tensor(np.asarray(v)[None], dtype=torch.float32)
            for k, v in env._get_obs().items()}


@pytest.mark.parametrize("domain", ALL_DOMAINS)
def test_layout_covers_every_channel_exactly_once(make_env, domain):
    e = make_env(n=6, domains=[domain] * 6, rigidity_global=True,
                 rigidity_flex=True, rigidity_edge=True)
    e.reset()
    d = obs_dict(e)
    layout = ablation.resolve_layout(e, d)

    assert len({n for n, _, _ in layout}) == len(layout), "duplicate channel names"
    for key in ("node_features", "edge_features"):
        covered = sorted((s.start, s.stop) for n, k, s in layout if k == key)
        assert covered[0][0] == 0
        assert covered[-1][1] == d[key].shape[-1], f"{key} not fully covered: {covered}"
        for (_, prev_stop), (start, _) in zip(covered, covered[1:]):
            assert start == prev_stop, f"{key} slices overlap or gap: {covered}"


def test_layout_tracks_the_optional_flags(make_env):
    """The flags shift every later slice; the widths must follow."""
    def names(**kw):
        e = make_env(n=6, domains="R^3", **kw)
        e.reset()
        return {n for n, _, _ in ablation.resolve_layout(e, obs_dict(e))}

    lean = names(graph_features=False)
    assert "closeness" not in lean and "edge_between" not in lean
    rich = names(graph_features=True, rigidity_flex=True, rigidity_edge=True,
                 rigidity_global=True)
    assert {"closeness", "edge_between", "flex_mag", "flex_align",
            "block_rank", "rigidity_glob"} <= rich


def test_stale_layout_degrades_instead_of_mislabelling(capsys):
    """A width mismatch must never silently reassign names to the wrong slices."""
    flags = {k: True for k in ("graph_features", "edge_exists", "rigidity_global",
                               "rigidity_flex", "rigidity_edge")}
    out = ablation._blocks(ablation.EDGE_BLOCKS, flags, 999, "edge_features")
    assert out == [("edge_features (whole)", slice(0, 999))]
    assert "mislabel" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["shuffle", "zero", "noise"])
def test_perturb_touches_only_its_own_slice(mode):
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(1, 5, 5, 6)
    sl = slice(2, 4)
    y = ablation.perturb(x, sl, mode, rng)

    untouched = torch.cat([y[..., :2], y[..., 4:]], dim=-1)
    assert torch.equal(untouched, torch.cat([x[..., :2], x[..., 4:]], dim=-1))
    assert not torch.equal(y[..., sl], x[..., sl])


def test_shuffle_preserves_the_marginal_distribution():
    """That is the reason shuffle is the default: it removes the association
    with a node/pair without changing the channel's scale."""
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(1, 8, 8, 4)
    y = ablation.perturb(x, slice(1, 2), "shuffle", rng)
    assert torch.allclose(torch.sort(y[..., 1:2].flatten())[0],
                          torch.sort(x[..., 1:2].flatten())[0])


def test_constant_channel_reports_that_it_was_not_ablated(make_env):
    """A one-hot domain channel in a homogeneous network is constant across
    nodes, so shuffling it is a no-op -- and must not read as 0% dependence."""
    e = make_env(n=6, domains="R^3")
    e.reset()
    d = obs_dict(e)
    space = e.observation_space
    flat = flatten_tensorized_space(dict(d))
    rng = torch.Generator().manual_seed(0)

    dom = next(s for n, k, s in ablation.resolve_layout(e, d)
               if n == "domain" and k == "node_features")
    _, changed = ablation.ablate_obs(flat, space, "node_features", dom, "shuffle", rng)
    assert not changed, "homogeneous domain one-hot should be unshufflable"

    _, changed = ablation.ablate_obs(flat, space, "node_features", dom, "noise", rng)
    assert changed, "noise must still reach a constant channel"


def test_ablate_obs_leaves_other_keys_alone(make_env):
    e = make_env(n=6, domains="R^3")
    e.reset()
    d = obs_dict(e)
    space = e.observation_space
    flat = flatten_tensorized_space(dict(d))
    rng = torch.Generator().manual_seed(0)

    out, changed = ablation.ablate_obs(flat, space, "edge_features", slice(0, 3),
                                       "zero", rng)
    assert changed
    after = unflatten_tensorized_space(space, out)
    for key in d:
        if key != "edge_features":
            assert torch.equal(after[key], d[key]), f"{key} was disturbed"
    assert torch.count_nonzero(after["edge_features"][..., 0:3]) == 0


# --device cuda puts the observations on the gpu while the generator stays on the
# cpu; torch refuses that pairing outright, so every perturbation mode has to be
# exercised on a non-cpu tensor.
@pytest.mark.parametrize("mode", ["shuffle", "zero", "noise"])
@pytest.mark.parametrize("device", ["cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                  reason="no cuda"))])
def test_perturb_works_off_cpu(mode, device):
    rng = torch.Generator().manual_seed(0)          # deliberately a cpu generator
    x = torch.randn(1, 6, 6, 4, device=device)
    y = ablation.perturb(x, slice(1, 3), mode, rng)
    assert y.device == x.device and y.dtype == x.dtype
    assert not torch.equal(y[..., 1:3], x[..., 1:3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
@pytest.mark.parametrize("mode", ["shuffle", "noise"])
def test_perturbation_is_identical_on_cpu_and_cuda(mode):
    """Randomness is drawn on the cpu and moved, so --device must not change the
    experiment -- otherwise two runs of the same seed are not comparable."""
    x = torch.randn(1, 6, 6, 4)
    a = ablation.perturb(x, slice(0, 2), mode, torch.Generator().manual_seed(3))
    b = ablation.perturb(x.cuda(), slice(0, 2), mode, torch.Generator().manual_seed(3))
    assert torch.allclose(a, b.cpu(), atol=1e-6)


# --- the csv and the terminal table must not drift -------------------------

class _Args:
    mode, episodes, seed = "shuffle", 4, 0


def _rows():
    return [
        dict(channel="degree", flip=.9, shift=.8, mask_changed=0., perturbed=1.,
             d_phi=2.5, d_m=-1., d_rigid=0., d_minimal=.5),
        dict(channel="bearings", flip=.1, shift=.05, mask_changed=0., perturbed=1.,
             d_phi=0., d_m=0., d_rigid=0., d_minimal=0.),
        dict(channel="adj", flip=.5, shift=0., mask_changed=1., perturbed=1.,
             d_phi=1.0, d_m=-.5, d_rigid=0., d_minimal=.25),
        dict(channel="domain", flip=0., shift=0., mask_changed=0., perturbed=0.,
             d_phi=0., d_m=0., d_rigid=0., d_minimal=0.),
    ]


_REF = dict(phi=75.0, m=10.0, rigid=1.0, minimal=1.0)


def test_rows_are_ranked_by_dependence():
    ordered = [r["channel"] for r in ablation.order_rows(_rows())]
    assert ordered == ["degree", "adj", "bearings", "domain"]


def test_table_carries_the_reference_row_last_and_absolute():
    t = ablation.table_rows(_rows(), _REF)
    assert t[-1]["channel"] == "(reference)" and t[-1]["status"] == "reference"
    assert t[-1]["d_phi"] == 75.0, "the reference row is absolute, not a delta"


def test_unablated_channels_get_empty_cells_not_zeros():
    """A 0.0 would be averaged and plotted as evidence of independence."""
    row = next(r for r in ablation.table_rows(_rows(), _REF) if r["channel"] == "domain")
    assert row["status"] == "not_ablated"
    assert all(row[c] == "" for c in
               ("flip_pct", "abs_dscore", "d_phi", "d_edges", "d_rigid_pct", "d_minimal_pct"))


def test_csv_matches_the_terminal_table(tmp_path):
    import csv as _csv
    import io

    path = tmp_path / "a.csv"
    data, notes = ablation.write_csv(str(path), _rows(), _REF, _Args, "meta line")

    with io.StringIO() as buf:
        ablation.report(_rows(), _REF, _Args, "meta line", out=buf)
        printed = buf.getvalue()

    with open(data) as f:
        got = list(_csv.DictReader(f))
    expected = ablation.table_rows(_rows(), _REF)
    assert [r["channel"] for r in got] == [str(r["channel"]) for r in expected]
    assert [r["d_phi"] for r in got] == [str(r["d_phi"]) for r in expected]

    # the prose moved next door, and must be the same prose
    assert open(notes).read() == printed
    for line in ablation.legend(_rows(), _Args, "meta line"):
        assert line in printed


def test_csv_is_a_plain_rectangle(tmp_path):
    """No comment preamble: a legend commented into the head makes every
    spreadsheet and csv viewer show noise before the table."""
    data, notes = ablation.write_csv(str(tmp_path / "a.csv"), _rows(), _REF, _Args, "m")
    lines = open(data).read().splitlines()

    assert lines[0] == ",".join(ablation.COLUMNS)
    assert not any(l.startswith("#") for l in lines)
    assert len({l.count(",") for l in lines}) == 1, "ragged rows"
    assert len(lines) == len(_rows()) + 2          # header + channels + reference
    assert notes.endswith(".txt") and notes != data
