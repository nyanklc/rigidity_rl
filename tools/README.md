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
| `verify_results.py` | do the numbers quoted in `ROADMAP.md` §1.0 and the README still reproduce? |
| `submodularity.py` | which objectives have diminishing returns, and therefore which ones greedy is guaranteed on? |

`constructive_greedy.py` is the standalone version of the `constructive` baseline
now wired into `baselines.py`, for difficulty sweeps that need no env config. It
also shows that the `c_max = 1` domains are a matroid where any greedy is already
optimal, which is why a "beats greedy" claim only means something in the spatial
domains.

`verify_results.py` is the reproducibility check to run before quoting a number
anywhere. It builds the environment programmatically rather than reading the
gitignored `environments/`, so it works on a fresh clone: benchmark digests tie a
number to an instance set, and `greedy` / `constructive` need no checkpoint. The
`learned` rows cannot be checked from a clone, because `models/` and `train/` are
gitignored; it prints the `baselines.py` commands instead of duplicating the
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
