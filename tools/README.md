# tools

Small scripts worth keeping: verifications, ablations, one-off measurements that
turned out to be worth re-running, quick views of some part of the
configuration. Anything that answered a question once and might answer it again.

Actual tests belong in `tests/`, where the suite picks them up. Anything under
`docs/` belongs to a specific document and is versioned with it.

Conventions: one file per question, a docstring saying what it answers and how to
run it, and no dependency on the repository's state beyond what it imports.
Run from the repository root, e.g.

```bash
PYTHONPATH=. uv run tools/<name>.py
```

## Current contents

| script | answers |
|---|---|
| `constructive_greedy.py` | how good is the classical baseline, and is the problem a matroid at this domain and size? |
| `env_report.py` | what is in this environment config: switches, observation layout and channel statistics, episode constants, cost |
| `compare_runs.py` | how do two or more training runs differ on the metrics that separate a policy from a search? |
| `verify_results.py` | do the numbers quoted in the README still reproduce? |
| `submodularity.py` | which objectives have diminishing returns, and therefore which ones greedy is guaranteed on? |
| `checkpoint_fingerprint.py` | does an edit to `policy/` change what an already-trained checkpoint computes? |
| `backbone_capacity.py` | at these settings, is the EGNN-vs-GINE comparison matched on width, on parameters, or on neither? |
| `kappa_sweep.py` | what does raising `stiffness_kappa` buy in stiffness, and what does it cost in edges? |
| `crlb_validation.py` | does the predicted shape error match the measured one, and at what noise level does the prediction stop holding? |
| `spectral_criteria.py` | do A-, D- and E-optimality rank graphs differently, or are they the same statistic? |
| `functional_vs_error.py` | which spectral criterion orders topologies the way the measured error does? |
| `repair_bound.py` | is the repair bound sound, and is it the true minimum? |
| `repair_choice.py` | among equally-sized repairs, does it matter which one you pick? |
| `greedy_landscape.py` | is greedy's phi landscape the same thing the observation already computes? |
| `flag_cost.py` | what does each observation flag cost per step and per episode? |
| `rigidity_cost.py` | which rigidity primitive costs what, and which one blocks a larger n? |
| `policy_cost.py` | what do the flags cost on the policy side: observation width, forward time, parameters? |

`constructive_greedy.py` is the standalone version of the `constructive` baseline
now wired into `evaluation.py`, for difficulty sweeps that need no env config. It
also shows that the `c_max = 1` domains are a matroid where any greedy is already
optimal, which is why a "beats greedy" claim only means something in the spatial
domains.

`checkpoint_fingerprint.py` is the check to run either side of a change to a model
class or to `policy/gnn_backbone.py`. The manifest archives both, and the loader
replays the archived text, so an old checkpoint is supposed to be unaffected by
those edits -- this prints a digest that makes "supposed to" testable. It needs
`models/` and `train/`, which are gitignored, so it does not run on a fresh clone.

`backbone_capacity.py` exists because the two controls conflict. Equalizing the
EGNN's width against GINE puts it at 10.9x GINE's parameters, and matching
parameters instead would put it at a quarter of GINE's width -- so "we compared
the backbones" is not a complete statement. Run it for whatever `node_feat_dim`
and `gnn_hidden_dim` an experiment actually uses, and report which control it
ran. Widths are measured by forwarding rather than assumed, since assuming them
is what hid the original 11-vs-128 mismatch.

`verify_results.py` is the reproducibility check to run before quoting a number
anywhere. It builds the environment programmatically rather than reading the
gitignored `environments/`, so it works on a fresh clone: benchmark digests tie a
number to an instance set, and `greedy` / `constructive` need no checkpoint. The
`learned` rows cannot be checked from a clone, because `models/` and `train/` are
gitignored; it prints the `evaluation.py` commands instead of duplicating the
rollout, since a second rollout path would drift and the check would become the
thing that is wrong.

`env_report.py` cross-checks the observation layout against `build_dict_obs` and
warns if the two have drifted, which is the same table `ablation.py` mirrors by
hand.

`compare_runs.py` reads `runs/` directly so it works mid-run. Read its docstring
before drawing conclusions from training curves: epsilon exploration makes them
understate any arm with short episodes.

`submodularity.py` reproduces `THEORY.md` §14. It is slow (a few minutes) because
every triple costs three rank or eigenvalue computations; the conclusion is not
close to the noise floor, so the default trial count is enough.

`greedy_landscape.py` is the source of `DESIGN_NOTES.md#greedy-vs-policy`. Read
the caveat on its `stiffness_proxy_deltas`: that column scales `add_stiffness`
into phi units by hand, so a miss there is not evidence the information is
absent. It is the one number in the script that is not exact.

The three cost scripts are the source of every number in
`DESIGN_NOTES.md#observation-cost`. Run all three pinned to one BLAS thread
(`OMP_NUM_THREADS=1`, and `policy_cost.py` sets `torch.set_num_threads(1)`
itself): unpinned and on a loaded machine, the same `eigh` timed between 0.2 and
16 ms and a batch-1 forward read 160 ms against a true 1 ms, which is enough to
invert the ranking. `flag_cost.py` needs its default 120 steps per measurement;
at 40 the flag rows cross the baseline and the table says nothing.
`rigidity_cost.py` measures `removal_costs` at three densities on purpose, since
its cost is a function of how many edges are redundant rather than of `n`.

The three estimation scripts answer one question each and are meant to be read in
that order. `spectral_criteria.py` says the trace is a monotone restatement of the
min eigenvalue and that log-det is the only one decorrelated from it;
`crlb_validation.py` says the analytic prediction is right where it applies;
`functional_vs_error.py` says that log-det's decorrelation is not signal, which is
what stopped a training run being spent on it. `functional_vs_error.py` is the slow
one -- it solves for the shape once per topology per noise level per trial.
