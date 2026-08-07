# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Master's thesis: **network topology optimization for bearing rigid multi-agent networks via deep RL and GNNs.**

A multi-agent network is a **directed graph**. Each node/agent lives in a specific domain (`R^2`, `R^3`, `R^2xS^1`, `R^3xS^1`, `SE(3)`), which determines its DOFs. A directed edge `i -> j` means **agent `i` measures the bearing to agent `j`**, expressed in `i`'s *local frame* (for orientation-less domains `R^2`/`R^3`, that is just the global frame). Edges are therefore not symmetric and not free — each one is a sensing/communication cost.

The RL task: given a set of agents at (random) poses, **choose the edge set** so the network is bearing rigid with as few edges as possible. A GNN encodes the graph into node embeddings that (implicitly) capture how rigidity-critical each node/edge is; an action head turns those embeddings into logits / Q-values shaped by the action space.

`thesis_skeleton.txt` has the intended chapter structure and terminology.

## What is live vs. obsolete

The repo carries a lot of history. **Currently in focus:**

| Axis | In use |
|---|---|
| Action spaces | `SelectNodesSequentially` (pointer-network style: pick a node per step; every 2nd pick toggles the edge between the two picks — add if absent, remove if present), `AddRemoveEdgeDiscreteNoSelfLoops` |
| GNN backbones | `GNNBackboneEquivariant` (EGNN) and `GNNBackboneGINE` (GINE) |
| Obs types | `DictEquivariantNodeFeaturesAndAdjAndSelection` → EGNN; `DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection` → GINE |
| State score | `Weighted` (essentially the only one now) |
| Algorithms | PPO and DQN, both via `skrl` |

**Obsolete / ignore unless asked:** `main.py`, `control.py` (the gradient-based formation controllers — the thesis originally aimed at control), everything `sb3` (`train_ppo_sb3.py`, `policy_sb3.py`, `models/sb3/`), `junk/`, `runs_old*/`, `fix_train.py`, `dummy*`, the GAT backbone, and most of the older action/obs/state-score variants still present in the dispatchers.

`gpu_environment.py` / `gpu_network.py` / `gpu_rigidity.py` are a **WIP** batched torch reimplementation of the env + rigidity math (tensors shaped `(num_envs, n, ...)`), not yet wired into training.

## Running things

`uv` manages the environment, but only as a plain `.venv` populated with `uv pip` (no `pyproject.toml`/lockfile). Run scripts with:

```bash
uv run <script>.py <args>
```

```bash
# Create/overwrite an environment config interactively (edit the constants in the __main__ block first)
uv run environment.py <n> <domain>          # e.g. uv run environment.py 4 "R^2"
uv run environment.py file <scenario_name>  # derive n/domains from scenarios/<name>.json

# Train (writes a run manifest to train/<model_name>.json, checkpoints under models/, TB logs under runs/)
uv run train_ppo.py <environment_name> <model_name>
uv run train_dqn.py <environment_name> <model_name>
#   <model_name> may be "prefix=foo" to auto-append action/obs type + n_domain
#   If the run already exists it prompts: [c]ontinue / start [f]resh / [a]bort

# Roll out a trained model (BRUTE_FORCE_BEST compares against exhaustive search on small graphs)
uv run inference.py <model_name> <environment_name>

# Reference points: initial / random / greedy / learned / optimal, all scored with the same phi
uv run baselines.py <environment_name> [--episodes N] [--model <name>] [--brute-force] [--methods a,b] [--replay-env]

# Inspect / verify / backfill training manifests (archived sources, provenance)
uv run manifest.py list | show <name> | diff <name> | verify <name> | backfill [--write]

# Interactive viser GUI to hand-edit a graph and watch rigidity metrics
uv run manual.py <environment_name>

tensorboard --logdir runs
```

Names are filenames without extension: `<environment_name>` → `environments/<name>.json`, `<scenario_name>` → `scenarios/<name>.json`.

There is no test suite, linter, or CI. `dummy/test_mbr.py` is a scratch file, not a test.

## Architecture

### Rigidity core (`rigidity.py`, `network.py`, `util.py`)

- `Agent` — pose (position + quaternion) + `domain`. `Network` — list of agents + `edges`, an `(n, n)` boolean adjacency matrix (row = measuring agent).
- `rigidity.bearing_DOFs(agent_i, agent_j)` returns the `(U_ij, V_ij)` projection matrices encoding **which DOFs the bearing `i -> j` actually constrains, as a function of both agents' domains**. This is the one place heterogeneity is handled, and everything else depends on it.
- `extended_bearing_rigidity_matrix(network)` → `B`, shape `(3m, 6n)`, built as `[D_p U E^T | D_a V E_o^T]`. Rows come in **3-row blocks, one block per directed edge**, in `np.nonzero(edges)` order — this per-edge block structure is what `is_MBR` exploits.
- `is_IBR` — Infinitesimally Bearing Rigid iff `rank(B) == rank(B_K)`, where `B_K` is the rigidity matrix of the fully-connected graph on the same poses. `rank_K` is cached on the env per episode; always pass it through rather than recomputing.
- `rigidity_eigenvalue` — the first nonzero eigenvalue of `B^T B` (index `6n - rank_K` into the ascending spectrum); the standard "how robustly rigid" scalar.
- `is_MBR(network, rank_K, brmat)` — the **minimality heuristic**. Per-edge block rank `c_k = rank(B[3k:3k+3, :])`, sorted descending, greedily accumulated until `Σ c ≥ rank_K`, giving `m_req`; minimal iff IBR and `m == m_req`. See "Known issues / open questions" below for its reliability.
- `MBR_required_Rd(n, d)` — closed-form minimum edge count, **valid only for homogeneous `R^d`**.

### Environment (`environment.py`)

One `gymnasium.Env` (`Environment`) for all experiments, configured entirely by a JSON file in `environments/` via `env.load(path)`. Nothing is subclassed — each axis is a string dispatched in a module-level function:

- `action_type` → `define_action_space()` + `action_<Type>(...)`
- `obs_type` → `obs()`
- `state_score_type` → inline `if/elif` chain inside `step()`
- `termination_condition_type` → inline `if/elif` chain inside `step()`

To add a variant, add an `elif` branch in the relevant dispatcher (and a matching model in `policy/`, registered in `train_ppo.py` / `train_dqn.py`).

**Reward structure** (`step()`): `reward = -time_penalty + [action_reward if action_rewards_enable] + (state_score(s') - state_score(s)) + [terminal bonus]`. The state-score term is **potential-based shaping** — the reward is how much *better* the graph got, not the absolute quality. `Weighted` is currently `20 * rank(B) - 10 * m` (the IBR and rigidity-eigenvalue weights are set to 0 in the code).

**Episode reset** re-randomizes poses *and* edges (a fresh `random_scenario`), so the policy must generalize across geometries, not memorize one. Setting `env.freeze_network = True` makes `reset()` redo only the per-episode bookkeeping (`begin_episode()`) and keep the current graph — that is how `baselines.py` runs several methods on one instance.

**`skip_enabled`** (env config, default `True`). When `False`, `train_ppo.py` / `train_dqn.py` pass `allow_skip=False` to the `SelectNodesSequentially` models, which mask the skip logit to `-1e9` in `compute()` *and* in the DQN `random_act()`. The action space stays `Discrete(n+1)`, so checkpoints and `agent_loader` stay compatible. Turn skip off with `MaxSteps`: `select -> skip` is a zero-reward 2-cycle that never touches the graph, and on-policy methods collapse onto it (observed: entropy → 0, all rewards exactly 0, graph unmodified for two thirds of training). Score skip-less runs with the best-state-visited metric below.

**Best-state-visited metric.** `Environment` tracks the highest-scoring graph seen during an episode (`best_state_score` / `best_edges` / `best_step` / `best_stats` with `m`/`is_IBR`/`is_MBR`/`rank`/`min_eig`, updated in `update_best_state()`), exposed in `info` and logged as `Episode/ Best *`. This is observational — the reward does not use it. It exists because scoring an episode on its *final* state conflates "found a good topology" with "learned to stop on it", which matters under `MaxSteps` where the agent is expected to converge and then hold with `skip`. `best_step` records how many steps it took to get there, which is the only way to tell a policy that converges fast from one that stumbles onto the same graph late.

`Best min eig` has no meaningful absolute scale: rigidity-matrix entries scale as `1/‖p_ij‖`, so it tracks `random_scenario`'s `pos_limits` (`scenario.py`, currently `[-1, 1]`; it was `[-100, 100]`, which put the eigenvalue at ~1e-5). Plot it on a log axis and don't compare across pose ranges. It frequently sits *below* `Min eig`, which is correct: `Weighted` has `w_eig = 0`, so φ trades rigidity margin away for fewer edges.

**Scenarios.** With `"scenario": "<name>"`, `initialize()` loads `scenarios/<name>.json` and caches it. What a scenario contributes on reset depends on `only_randomize_edges`: `false` carries over only the **domain mix** (poses and edges are redrawn each episode — use this for heterogeneous generalization experiments), `true` keeps the scenario's **actual geometry** and resamples only the edges (use this for a fixed case-study figure). Both paths honour `random_graph_with_mean_min_edges`.

**Config format changed** — current configs use `state_score_type` / `skip_is_stop` / `random_graph_with_mean_min_edges`; ~80 of the 82 files in `environments/` are the older `reward_type` / `incremental_rewards_enable` format and will `KeyError` in `load()`. Only the two `...rewardWeighted_termMinimallyRigid_{n4_R2,n8_R3}.json` (EGNN) files are current. Regenerate stale ones via `uv run environment.py` rather than hand-editing.

### Policies (`policy/`)

`policy/gnn_backbone.py` holds the backbones; `policy/{actor,critic,q_func}/<Name>.py` hold one model per (backbone × action-space) combination, all re-exported from `policy/__init__.py`. Naming: `Equivariant_*` = EGNN, `GINE_*` = GINE, bare name = old GAT/MLP.

Conventions that matter when writing a new model:
- `unflatten_tensorized_space(self.observation_space, inputs["observations"])` recovers the obs dict — skrl flattens `Dict` spaces.
- **Action masking is done in the model**, by writing `-1e9` into invalid logits/Q-values (e.g. masking the already-selected node in `SelectNodesSequentially`, masking add-existing / remove-nonexistent in `AddRemoveEdge*`). The env does not mask.
- DQN Q-networks additionally override `random_act()` so epsilon-greedy exploration also respects the mask.
- `GNNBackboneEquivariant` output width is `node_feat_dim` (EGNN preserves feature dim; `gnn_hidden_dim` only sets the internal message width `m_dim`), whereas `GNNBackboneGINE` outputs `gnn_hidden_dim`. Head input sizes differ accordingly — a common source of shape errors.
- GINE flips `edge_index` before message passing so a node aggregates its *outgoing* bearings ("I measure this bearing to that node"), which is the semantically right direction here.

### Baselines (`baselines.py`, `agent_loader.py`)

`baselines.py` scores every method through `Environment.compute_state_score`, so all rows of its
table are measured by the exact φ the agent trains on. `greedy` and `optimal` work in edit space
(action-space agnostic); `random` and `learned` go through `env.step()` and are therefore specific
to the configured action space, and are scored on best-state-visited. `optimal` scans edge count
ascending and stops at the first level admitting an IBR graph — unlike `MBR_required_Rd` this makes
no homogeneity assumption, but it is gated at `n ≤ 5`.

**`greedy` is the expensive baseline**, not brute force: it evaluates all `n(n-1)` candidate
toggles per single edit, so cost is `O(n^2)` φ-evaluations per improvement. `score_network()`
therefore skips `is_MBR` (which costs one rank computation *per edge* on top of the full-matrix
rank) unless the configured `state_score_type` actually reads the flag — `Weighted` does not. The
full stats are computed once for the reported row. Use `--methods` to drop `greedy` entirely.
Brute force is separate and already refuses above `MAX_BRUTE_FORCE_N = 5`.

`--policy-mode` selects how `learned` is rolled out: `sample` (default, the policy used as a
sampling search, reproducible under `--seed`) or `greedy` (argmax, what a deployed policy does,
terminates on a repeated state). A DQN q-network is argmax either way.

Pass `--device` if you want the `--model` rollout on GPU; it defaults to cpu and also sets
`env.device` so the skrl wrapper puts observations on the same device as the agent.
`--brute-force` prints per-level progress because the cost explodes with the required edge count:
1.6k subsets at `n=4` (needs 5 edges) but ~432k at `n=5` when 9 edges are needed.

**Output lives in one directory per run** (`report.py`), named
`runs_baselines/<timestamp>__<short-env>[__<model>][__<tag>]/`:

```
summary.txt        the printed table and its legend
results.csv        one row per (episode, method) -- the final outcome
trajectories.csv   one row per (episode, method, step) -- the time series
meta.json          args, env config, and manifest.collect_provenance()
plots/pdf/         table + trajectories + outcomes + summary + episode_NNN
plots/png/         the same figures again -- every figure is written in both formats,
                   filed by format (`PLOT_FORMATS` / `_save()`) rather than interleaved
```

The table is written to be read without the source: `work` counts graph modifications actually
applied and `best_at` is the step the best graph was reached at (the old single `steps` column
meant different things per method), episode count moved into the header, every column states its
direction, and a legend explains each method and column in plain language (`--brief` drops it).
Every column is a mean over the episodes carrying its own `+-` spread; the percentage columns
(`rigid`/`minimal`/`=best`) deliberately have none, because they are means of a 0/1 indicator
whose sd is `sqrt(p(1-p))` — fully determined by the value already shown. The margin appears
twice, arithmetic (`mean+-sd`) and geometric (`gmean x/gsd`, via `_gmean`/`_gsd`), because it
spans decades and an arithmetic `+-` implies a range crossing zero. `_fmt_geo` marks a row `*`
when zero-margin (non-rigid) networks had to be dropped, since a geometric mean cannot take them.

**The figures are built to survive being pasted into a slide with no caption.** Every one carries
a title block (what the figure is, then the *full* environment and model names, wrapped rather
than truncated) and a notes card along the bottom: method key on the left, how the figure is
computed on the right. Panel titles name the quantity and put the reading direction on a second,
muted line — `_panel_title()`. Layout mechanics that matter: the card is drawn straight onto the
figure in a reserved band (`_draw_card`), *not* as a gridspec row, because a row's height is
scaled down by whatever the panels spend on decorations and the card came out too short for its
own text; `_figure()`/`_finish()` size the figure as `2*panel_h + header + card` and hand
`tight_layout` the matching `rect`.

`plot_table()` (`plots/*/table.*`) renders the same table as `summary.txt` as a figure — driven by
the same `aggregate()`, so the two can't drift. Column direction moves into a subtitle under each
column name, `initial`/`optimal` are drawn in muted ink with their reference strokes, and the
column legend becomes the card. The card flows newspaper-style when the notes outrun the method
list (`_card_rows`): the left column continues under the method key, the right column picks up
from there, and splits only happen between notes.

`plot_outcomes()` (`plots/*/outcomes.*`) is the **final / best / mean** figure — the same three
views `Environment.write_episode()` logs per episode, so a baselines bar and a tensorboard curve
mean the same thing. `outcome_stats()` derives all three from `traces` (final = last step, best =
highest-scoring step, mean = over recorded steps), so nothing extra is computed. The gap between
a method's `final` and `best` bar is the "found it but did not stop on it" failure, visible per
method. Caveat stated on the card: greedy records one point per applied edit, so its `mean` is
over edits, not over a step budget.

Per-step tracing rides with the plots — `--no-plots` skips both. It costs one extra
eigendecomposition per step (+31% at `n=4`, +21% at `n=8`) and is served by
`Environment.last_stats`, which `step()` fills from values it already computes; the
`trace_min_eig` flag makes it also compute the rigidity eigenvalue without a TensorBoard writer.
Both are guarded with `getattr`/`hasattr` in `baselines.py` because `--replay-env` can hand back
an *archived* `Environment` from before those attributes existed.

Plot colours come from the data-viz reference palette, used unchanged: the three compared methods
take categorical slots 1–3 (certified for the all-pairs case), while `initial` and `optimal` are
reference points drawn in neutral ink with dashed strokes rather than competing for a hue.

`agent_loader.load_agent()` rebuilds a trained skrl agent from its `train/<name>.json` manifest.
Both `inference.py` and `baselines.py` use it.

**Manifests are self-contained records of a run** (`manifest.py`, `manifest_version: 2`). Besides
the hyperparameters and env config they archive the full text of every file that determined the
model — `util.py`, `rigidity.py`, `network.py`, `scenario.py`, `environment.py`,
`policy/gnn_backbone.py` — gzipped and base64'd into `sources_b64gz` (~25 KB rather than ~99 KB),
plus `scenario_raw` (`scenarios/` is gitignored, so a run naming one is otherwise unreplayable) and
a `provenance` block (git commit/dirty, command, package versions, device, seed). Training is now
seeded from a `SEED` constant in both scripts and records it.

```bash
uv run manifest.py list                    # every run: what it carries, whether it still verifies
uv run manifest.py show <name> [file]      # print archived source / provenance
uv run manifest.py diff <name> [file]      # archived vs working tree
uv run manifest.py verify <name>           # rebuild the model from the archive, check the weights
uv run manifest.py backfill [--write]      # add what older manifests are missing
```

`backfill` only archives today's sources for a run whose weights they demonstrably rebuild
(shape-checked against the `.pt`); it marks `provenance.captured_at_training: false` because those
sources were not captured at the time. Where the check fails it writes `reconstructible: false`
with the reason rather than a plausible lie. It also recovers `environment_config_raw` from
`environments/<name>.json` for manifests that only recorded the config's name.

**`load_run(model_name, env_name)`** is the entry point `inference.py` uses (and `baselines.py`
under `--replay-env`). When the archived sources differ from the working tree it says so and
rebuilds *the environment the run was trained against*: `archived_modules()` execs each archived
file into a fresh module registered under its real name in dependency order — so the archived
`environment.py`'s own `from network import Network` resolves to the archived `network.py` — and
restores `sys.modules` afterwards. `control`/`visualizer`/skrl/torch deliberately stay as
installed. This is what lets a checkpoint keep running after the observation format, action
semantics or rigidity maths change. With a clean tree it uses the live environment, so replay
costs nothing when nothing moved.

**A checkpoint survives edits to `policy/`.** The manifest archives the model class source *and*
`backbone_source` (the whole of `policy/gnn_backbone.py`) — the model classes only *reference*
`GNNBackbone*`, so without the latter a checkpoint would break the moment the backbone changed.
`resolve_model()` replays the archived source in a fresh namespace (`build_class_from_source()`,
with `MODEL_SOURCE_PREAMBLE` supplying the module-level imports `inspect.getsource` drops), and
falls back to the current class of the same name if the archived source fails to execute. Either
way the result is verified against the checkpoint's parameter shapes before use, and the loader
prints which one it used. When the archived architecture differs from what is in `policy/` today
it says so and keeps the archived one, so the weights stay valid — meaning you can refactor the
backbone freely without invalidating old runs. Manifests written before this carry no
`backbone_source` and are replayed against the current backbone, which is only correct while it
is unchanged; if it no longer fits, the loader reports a shape mismatch rather than guessing.

**Runs older than the manifests** fall back to `load_agent_legacy()`, which reads the architecture
back out of the checkpoint's parameter shapes (`infer_architecture()`: backbone, depth,
`gnn_hidden_dim`, `head_hidden_dim`, feature dims) and then finds the model class by *constructing
every candidate in `policy/` and comparing state-dict keys and shapes* — so it can never drift out
of sync with the training scripts. It runs unattended in the normal case: the algorithm comes from
which `models/complete/<ALGO>/` holds the file, cross-checked against the saved model roles
(`policy`+`value` ⇒ PPO, `q_network` ⇒ DQN/DDQN). It only asks when that is genuinely undecidable —
the same name under several algorithm directories, or roles matching no known agent — or when
several model classes fit the checkpoint equally well. The GNN
backbones therefore take a `num_layers` argument (older runs used 2 layers, current ones 3);
`rebuild_backbone()` swaps the depth before loading. Submodules are still named `conv1..convN`, so
3-layer checkpoints are unaffected.

Most old checkpoints are **not** recoverable, because the observation format changed: they expect
6–13 node features and 3 edge features where the current obs types produce 10 and 6. The loader
reports that explicitly rather than failing on a shape error. Of the 26 manifest-less checkpoints,
4 load (the EGNN ones from after the feature set settled).

### Training (`train_ppo.py`, `train_dqn.py`)

Both: load env config → build `SyncVectorEnv` → select actor/critic (or Q-net) by `(action_type, obs_type)` → `skrl` agent + `SequentialTrainer`. They write `train/<model_name>.json` containing hyperparameters, the env config, and **the actual source of the model classes** (`inspect.getsource`) — these manifests are the record of what was run, but `train/` is gitignored, so they live only on the machine that produced them. Env-side metrics go to `runs/<experiment_name>` via `Environment.write_episode()`, **once per episode** (see below).

**Episode-level logging.** All environment metrics are written at episode end, not per step — a step-resolution scalar costs one tensorboard event per step and is then downsampled and averaged for display, so the detail was paid for and never seen. `step()` folds each step into `episode_accum` (`new_episode_accum()`: sums and counts only, so episode length is free), and on `terminated or truncated` builds `episode_summary()` → `last_episode_stats` → `write_episode()`, which dumps every entry under `Episode/ <key>`. Three views per episode: `Final *` (where it ended), `Best *` (best graph visited, plus `Best-final score gap` — 0 iff the episode ended on its own best graph), and `Mean *` / `* fraction` (the episode average). Scalars are written against `writer_counter` (global env step), *not* the episode index, so they share an x-axis with skrl's loss/reward curves; `writer_counter` therefore still advances every step. `last_episode_stats` is a plain attribute rather than an `info` key because `SyncVectorEnv` aggregates sub-env `info` dicts into arrays. `Environment.write(value, tag)` remains for custom scalars.

### Gitignored (don't assume present)
`environments/`, `scenarios/`, `models/`, `runs/`, `runs_old*/`, `runs_baselines/`, `train/`, `tboard_logs/`, `junk/`.

**Nothing a run produces is tracked any more.** `runs_baselines/` and `train/` were both tracked until they were untracked wholesale — the first churned hundreds of binary files per run, the second is paired with `models/` which was already ignored. Consequence: a manifest now only exists on the machine that trained it, so `train/<name>.json` is no longer a shared record — back it up alongside the checkpoint it describes. Anything that has to survive (the README's figures) is copied into `resources/`.

Because every output directory is ignored, **a fresh clone has none of them**, and any code that writes output has to `os.makedirs(..., exist_ok=True)` *before* opening the file. Two places had the call inside the `with open(...)` block, which fails on exactly that fresh clone (`environment.py` writing `environments/`, `scenario.py` writing `scenarios/`); both are fixed. The library-managed paths take care of themselves: `SummaryWriter` creates its `log_dir`, and skrl creates its checkpoint directory (`skrl/agents/torch/base.py`).

## Known issues / open questions

Live research questions, not things to silently "fix":

1. **Termination condition.** There is no way to know the true optimal topology. The only sound stopping test is minimal bearing rigidity: exact via `MBR_required_Rd` for homogeneous `R^d`, otherwise the greedy `is_MBR` heuristic. The heuristic is a *sound lower bound* (rank subadditivity over edge blocks ⇒ no proper subset of the current edges can be rigid with fewer than `m_req` edges), and it reproduces the closed form exactly for homogeneous `R^2` and `R^3`. It can produce **false negatives** in heterogeneous networks, where the greedy sum over the highest-rank blocks may not be jointly realizable — a truly minimal graph is then never recognized and the episode never terminates.
2. **Random graph generation.** `random_scenario` samples `m ~ Uniform{0, …, n²-n}`, but the required edge count grows only ~linearly in `n`. Expected initial edges exceed the MBR requirement by 1.2× at `n=4` and ~2.8× at `n=8` (`R^3`), so most episodes start over-constrained and the agent mostly learns deletion. The `random_graph_with_mean_min_edges` Gaussian sampling was written to fix this but is currently inert (see below).
3. **Constructive vs. editing formulation.** Starting from the empty graph and only adding edges is under consideration; the open worry is whether a purely constructive agent can reach *optimal* topologies rather than merely feasible ones.
4. `Environment.load()` never reads `random_graph_with_mean_min_edges`, and `reset()` passes `edge_count` (`None`/`0`) to `random_scenario` instead of the sampled `random_edge_count` — the Gaussian sampling only ever runs in `initialize()`, i.e. once at construction, and is discarded on every reset.
5. `reset()` sets `last_state_score = 0` rather than the initial graph's score, so the first step of an episode is rewarded with the *absolute* state score instead of the improvement.
6. `step()` binds `info` only inside the `if track_data_enable and writer is not None` branch → `UnboundLocalError` when tracking is off.
7. `reset()` with no scenario file rebuilds the network from `agents[0].domain` only, so heterogeneous domains survive only via the `scenario` path (`randomize_scenario`).
8. Per-step cost is dominated by pure-Python graph features in `obs()` (Floyd–Warshall closeness, Brandes betweenness) plus repeated rigidity-matrix construction (`step()` builds `B` once, then `rigidity_eigenvalue`/`eigenvalues()` rebuild it). Fine at `n=4–8`, the bottleneck beyond.
