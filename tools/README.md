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

`constructive_greedy.py` is the honest opponent for WP8: it beats the current
policy everywhere except its training configuration, and it shows that the
`c_max = 1` domains are a matroid where any greedy is already optimal.

`env_report.py` cross-checks the observation layout against `build_dict_obs` and
warns if the two have drifted, which is the same table `ablation.py` mirrors by
hand.

`compare_runs.py` reads `runs/` directly so it works mid-run. Read its docstring
before drawing conclusions from training curves: epsilon exploration makes them
understate any arm with short episodes.
