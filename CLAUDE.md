# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: how to work on this repository

These are explicit, standing instructions from the user. They override default behaviour and
apply to every task in this repository.

- **This is a research project, not a software product.** Judge work by whether it answers the
  research question, not by product-engineering standards.
- **Do not make assumptions. Ask.** If something is unclear or ambiguous, or you do not have
  enough knowledge about it, ask the user for input instead of guessing and proceeding.
- **Do not over-engineer.** The goal is a simple, clear implementation. Prefer the smallest thing
  that answers the question. If a built-in tool or three flags will do, do not write a framework.
- **Be impartial and honest in assessments.** Report what the numbers say, including when they
  are inconvenient, weak, or contradict earlier claims (yours or the user's).
- **Scientific accuracy is paramount.** Do not overstate a result. Distinguish evidence from
  conjecture, and a measurement from an anecdote. Say when a sample is too small to support a
  conclusion.
- **Keep the research questions in mind**, including the ones not explicitly stated in the task,
  and approach every task from an experienced researcher's point of view.
- **Push back on bad premises.** If the user makes a claim that does not make sense, or that
  rests on an incorrect premise or assumption, say so plainly and give reliable feedback rather
  than building on it.
- **Reason from first principles** where possible, rather than repeating talking points from
  sources or from this file.
- **State uncertainty explicitly** — both the degree of it and the reason for it — whenever you
  are not confident.
- **NEVER commit, and never create a branch's first commit for the user.** `git add` / `git commit`
  / `git push` are the user's job — they verify every change manually first. Leave work in the
  working tree, say what changed, and stop there. Creating a branch is fine; committing to it is not.
- **Keep the documentation current as you go, not at the end.** `ROADMAP.md` is the live plan and
  work log and must be updated whenever a work package's status changes; `CLAUDE.md`, `THEORY.md`
  and `DESIGN_NOTES.md` must be corrected in the same change that invalidates them. The test is
  whether a *fresh session with no conversation history* can read the repo and continue the work.
- **Keep useful throwaways.** If a script written to answer a question would be worth having again
  (a verification, a regression, an ablation, a table of every configurable switch, a plot of some
  invariant), say so and offer to keep it. Real tests go in `tests/`; anything else useful goes in
  `tools/`. Ask before adding, and default to keeping rather than discarding: a small library of
  these accumulates into something worth having, and a one-off answer that cannot be re-run is
  worth much less than one that can. This is not only about tests.
- **Write code to be read, and to match the maths.** The primary audience is a person checking the
  implementation against the derivation, so name things after the symbols they are: `B`, `Z`, `S_i`,
  `P_i`, `c_k`, `rank_K`, `m_req`. Where a line implements a numbered equation, say which one
  (`# (13.1)`). Prefer the form that visibly *is* the formula over the form that is clever. A
  straightforward loop that mirrors a sum over edges is better than a vectorized expression whose
  index bookkeeping has to be decoded, unless the loop is a measured bottleneck.
- **Performance is not the first priority.** This is an experiment setup, not a product. Optimize
  only against a *measured* bottleneck that actually blocks an experiment (the n≥32 step cost is the
  real one), and when you do, say in the code what the readable version was and keep it as the
  reference the tests check against. Do not trade clarity for a speedup nobody asked for or measured.
- **Keep code comments brief.** One or two lines, saying what is non-obvious, with a pointer to the
  document that explains it (`see THEORY.md §12`). Derivations, measurements, rejected alternatives
  and rationale belong in `THEORY.md` / `DESIGN_NOTES.md` / `ROADMAP.md`, not in a comment block
  above a function. The same goes for test docstrings: state what the test pins down, not why.

## Project

Master's thesis: **network topology optimization for bearing rigid multi-agent networks via deep RL and GNNs.**

A multi-agent network is a **directed graph**. Each node/agent lives in a specific domain (`R^2`, `R^3`, `R^2xS^1`, `R^3xS^1`, `SE(3)`), which determines its DOFs. A directed edge `i -> j` means **agent `i` measures the bearing to agent `j`**, expressed in `i`'s *local frame* (for orientation-less domains `R^2`/`R^3`, that is just the global frame). Edges are therefore not symmetric and not free — each one is a sensing/communication cost.

The RL task: given a set of agents at (random) poses, **choose the edge set** so the network is bearing rigid with as few edges as possible. A GNN encodes the graph into node embeddings that (implicitly) capture how rigidity-critical each node/edge is; an action head turns those embeddings into logits / Q-values shaped by the action space.

`thesis_skeleton.txt` has the intended chapter structure and terminology.

**Long-term motivation (not currently pursued, may never be):** the original aim was a
*distributed* method for maintaining rigid formations in swarms. The centralized formulation here
is a deliberate first step. Relevant when weighing design choices — see `ROADMAP.md` appendix A.
Two kinds of information a local decision maker would **not** have, and it is worth keeping them
straight:

- **Global algebra** — `rank(B)`, rank deficit, per-edge block rank `c_k`, `is_IBR`, and the graph
  centralities. Too expensive to compute and to communicate.
- **Bearings to nodes it is not measuring** — an agent does not know `p_hat_ij` *before* measuring
  it, so it cannot locally evaluate "would adding `i -> j` help?". Whether this is a real
  obstruction depends on an unmade modelling choice: if detection is cheap (omnidirectional vision
  within radius `R`) and only the *maintained* link costs anything, candidate bearings are locally
  available after all; if every measurement costs what an edge costs, they are not.

This matters for the observation design specifically: candidate-edge bearings were exactly what
the observations used to be missing, and are now included (see `#all-pairs-bearings`).

**`ROADMAP.md` is the live plan and diagnosis.** It records what is currently broken, why, and the
phased fix. Read it before changing the environment, the observations or the reward — several
things in this file that *look* like design decisions are recorded there as known errors.

## Results and where they live

**Measured results live in exactly one place: `ROADMAP.md` §1.** Nothing else in the repository
should carry policy numbers. When a number is superseded, delete it rather than archiving it; git
history is the archive, and a stale number in a file that loads into every session costs more than
it is worth.

This file therefore records *what the code is* and the structural facts that do not move (the
`rank_K` formulas, `c_max` per domain, cost scaling). For "how well does it currently work", read
`ROADMAP.md` §1. For a human-readable summary, `README.md`.

## What is live vs. obsolete

The repo carries a lot of history. **Currently in focus:**

| Axis | In use |
|---|---|
| Action spaces | `SelectNodesSequentially` (pointer-network style: pick a node per step; every 2nd pick toggles the edge between the two picks — add if absent, remove if present), `AddRemoveEdgeDiscreteNoSelfLoops` |
| GNN backbones | `GNNBackboneEquivariant` (EGNN) and `GNNBackboneGINE` (GINE) |
| Obs type | `Dict`. The six `Dict*` variants were merged into it and survive as flag presets that reproduce their old layouts byte-for-byte, so pre-merge configs and checkpoints still load. The backbone is a `BACKBONE` constant in the training script now, overridden by `OBS_BACKBONE` when a legacy obs type implies one. See `DESIGN_NOTES.md#dict-observation` |
| State score | `WeightedNormalized` (dimensionless, transfers across n and domain); `Weighted` kept so old runs replay |
| Algorithms | PPO and DQN, both via `skrl` |

**Obsolete / ignore unless asked:** `main.py`, `control.py` (the gradient-based formation controllers — the thesis originally aimed at control), everything `sb3` (`train_ppo_sb3.py`, `policy_sb3.py`, `models/sb3/`), `junk/`, `runs_old*/`, `fix_train.py`, `dummy*`, the GAT backbone, and most of the older action/obs/state-score variants still present in the dispatchers.

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
uv run baselines.py <environment_name> [--episodes N] [--model <name>] [--brute-force] [--methods a,b] [--restarts K] [--replay-env]
#   methods: initial, greedy, constructive, random, learned
#   --benchmark <name> evaluates on a frozen instance set instead of sampling

# Freeze evaluation instances so results stay comparable across config regenerations
uv run benchmark.py <environment_name> <benchmark_name> [--instances N] [--seed S]
uv run benchmark.py list

# Which observation channels does a trained policy actually depend on?
uv run ablation.py <model_name> [environment_name] [--episodes N] [--mode shuffle|zero|noise] [--channels a,b] [--csv out.csv]

# Reproduce every number in the heterogeneous rigidity-matrix note (docs/)
PYTHONPATH=. uv run docs/verify_dof_restriction.py [--quick]
PYTHONPATH=. uv run docs/verify_dof_restriction_2.py   # independent reimplementation

# Inspect / verify / backfill training manifests (archived sources, provenance)
uv run manifest.py list | show <name> | diff <name> | verify <name> | backfill [--write]

# Interactive viser GUI to hand-edit a graph and watch rigidity metrics
uv run manual.py <environment_name>

tensorboard --logdir runs
```

Names are filenames without extension: `<environment_name>` → `environments/<name>.json`, `<scenario_name>` → `scenarios/<name>.json`.

**Tests: `uv run tests/run_all.py`** (fast suite, ~45 s, 559 checks) or
`uv run tests/run_all.py --slow` (~3 min, adds training runs, brute force and large n).
Individual files run standalone: `uv run pytest tests/test_flex.py -v`.

The suite is written against the invariants this project keeps breaking -- the scale-free
mask sentinel, n-invariance of every observation channel and of both backbones' activations,
the flex tensor's frame and shape, the exact addition criterion against rebuilt-matrix
ground truth in every domain, per-domain `rank_K`/`c_max`/`m_req`,
similarity invariance per channel across all five domains, `allow_skip` over every model,
the legacy obs presets being byte-exact, and phi's closed form. Re-introducing any of the
six bugs found during the last stretch of work makes a *named* test fail; that is the
suite's acceptance criterion, not its pass rate.

`environments/`, `models/`, `train/` and `scenarios/` are gitignored, so tests build
environments programmatically and write configs to `tmp_path` -- **the fast suite passes on
a fresh clone**, with checkpoint-dependent tests skipping rather than failing. Anything that
writes to `runs/`/`train/`/`models/` uses the `temp_run_name` fixture and cleans up.

There is no linter or CI.

## Architecture

### Rigidity core (`rigidity.py`, `network.py`, `util.py`)

- `Agent` — pose (position + quaternion) + `domain`. `Network` — list of agents + `edges`, an `(n, n)` boolean adjacency matrix (row = measuring agent).
- `rigidity.node_dof_projectors(agent)` returns `(S_i, P_i)`, the **per-node** translational and rotational DOF projectors — `S = diag(1,1,0)` for a planar agent, `P = v vᵀ` for `R^dxS^1`, and so on. This is where heterogeneity is handled. **The restriction belongs to the node, not to the edge**: `bearing_DOFs`'s per-edge `U_ij` (Michieletto Table III) re-enables a planar agent's z DOF whenever it measures a spatial one, which made `rank_K` exceed the system's own DOF count on mixed networks. See `THEORY.md` §12 and `ROADMAP.md` §1.2. `bearing_DOFs` is retained, unused by the matrix, as the reference implementation of Table I that the homogeneous-equivalence test compares against — but it is faithful to Table I **only at `v = e₃`** (it uses the row form `e₃vᵀ` for `R^3xS^1` where the paper has the column form `[0_{3x2} v]`), so a Table I comparison off the default axis must not use it. See `THEORY.md` §12.4.
- `extended_bearing_rigidity_matrix(network)` → `B`, shape `(3m, 6n)`, built as `[D_p E^T S | D_a E_o^T P]` with `S`, `P` block-diagonal over nodes. Rows come in **3-row blocks, one block per directed edge**, in `np.nonzero(edges)` order — this per-edge block structure is what `is_MBR` exploits. Homogeneous output is bit-identical to the previous per-edge construction; a coordinate an agent cannot vary is now an exactly-zero column, which is what Michieletto's nullity accounting requires. Validated against its own definition by central differences (`tests/test_rigidity_matrix.py`).
- `is_IBR` — Infinitesimally Bearing Rigid iff `rank(B) == rank(B_K)`, where `B_K` is the rigidity matrix of the fully-connected graph on the same poses. `rank_K` is cached on the env per episode; always pass it through rather than recomputing.
- `rigidity_eigenvalue` — the first nonzero eigenvalue of `B^T B` (index `6n - rank_K` into the ascending spectrum); the standard "how robustly rigid" scalar.
- `is_MBR(network, rank_K, brmat)` — the **minimality heuristic**. Per-edge block rank `c_k = rank(B[3k:3k+3, :])`, sorted descending, greedily accumulated until `Σ c ≥ rank_K`, giving `m_req`; minimal iff IBR and `m == m_req`. See "Known issues / open questions" below for its reliability.
- `MBR_required_Rd(n, d)` — closed-form minimum edge count, **valid only for homogeneous `R^d`**.
- `max_edge_rank(network, brmat_K)` → `c_max`, the most rank a single edge can contribute. **Exact.** This is what `WeightedNormalized` normalizes by.
- **`S -> rank(B_S)` is monotone submodular**, so minimum-edge rigidity is minimum submodular cover
  and the `constructive` baseline is Wolsey's greedy with an `H(c_max)` guarantee: exact at
  `c_max = 1`, 1.5 at `c_max = 2`. Measured, greedy sits 0–5% above `m_req`, so the headroom for any
  method on edge count is small. **The rigidity margin is *not* submodular** (59% of tested triples
  violate diminishing returns), so greedy carries no guarantee there. That asymmetry is the
  structural argument for WP3. See `THEORY.md` §14.
- `required_edge_count(network, ...)` → `m_req`, fewest edges that could make these poses rigid: closed form for homogeneous `R^d`, greedy block-rank accumulation otherwise. **A lower bound, not a ground truth** — it stays out of the reward and is used for reporting and the MBR metric only. Brute force finds it tight on everything checkable (24/24 at n=4, 6/6 at n=5, all five domains), which is evidence, not proof.

### Environment (`environment.py`)

One `gymnasium.Env` (`Environment`) for all experiments, configured entirely by a JSON file in `environments/` via `env.load(path)`. Nothing is subclassed — each axis is a string dispatched in a module-level function:

- `action_type` → `define_action_space()` + `action_<Type>(...)`
- `obs_type` → `obs()`
- `state_score_type` → inline `if/elif` chain inside `step()`
- `termination_condition_type` → inline `if/elif` chain inside `step()`

To add a variant, add an `elif` branch in the relevant dispatcher, plus a matching model in `policy/` registered in `policy/registry.py`.

**Model selection is a registry, not an if/else chain.** `policy/registry.py` maps `(role, backbone, action_type)` → class, where role is skrl's model-dict key (`policy`/`value`/`q_network`). `build_models(algorithm, backbone, action_type, **kwargs)` returns the dict the agent takes directly; `instantiate()` filters the kwargs superset against each constructor's signature, so classes needing `edge_feat_dim` or `allow_skip` just declare them. `agent_loader` shares the same `instantiate`. A `(role, backbone, None)` entry is the per-backbone fallback, which is how the critics cover every non-selection action space. See `DESIGN_NOTES.md#model-registry`.

**Reward structure** (`step()`): `reward = -time_penalty + [action_reward if action_rewards_enable] + (state_score(s') - state_score(s)) + [terminal bonus]`. The state-score term is **potential-based shaping** — the reward is how much *better* the graph got, not the absolute quality. `WeightedNormalized` is `(w_rank·rank - w_edge·m·c_max) / rank_K` at `(100, 25)` — dimensionless, so its optimum is ~75 at any `n` and in any domain. The older `Weighted` is `20·rank(B) - 10·m` and does **not** transfer (see below); it is kept only so old runs replay.

**The discount factor is not a free hyperparameter here.** With a purely potential-based reward,
γ=1 and no stop action, the episode return telescopes to `φ(s_T) - φ(s_0)`, so the advantage is
`E[φ(s_T)|s'] - E[φ(s_T)|s]` — which is ≈0 under a near-uniform policy, because the random walk
over edge sets mixes and forgets `s`. **There is then no gradient to bootstrap from**, which is
exactly what killed the PPO run (entropy frozen at ~1.9 nats of a ~2.0 ceiling). With γ<1, Abel
summation turns the same reward into `-φ(s_0) + (1-γ)·Σ γ^(t-1) φ(s_t)`, i.e. *maximize the
discounted average of φ along the trajectory* — get good fast and stay good. DQN uses γ=0.99 and
works; PPO used γ=1.0 and does not. Do not set γ=1 to make the logged return match what is being
optimized; log the undiscounted return separately instead (`Episode/ Return` already does).

**Why `Weighted` does not transfer** (it is legacy, but every pre-2026-08 checkpoint uses it): an
edge's rigidity-matrix block has rank **2 in R^3** and **1 in R^2**, so at `w_rank=20, w_edge=10` a
rank-adding edge is worth +30 in R^3 but only +10 in R^2, against +10 for pruning in both — R^3 is
three times more eager to add than to prune, R^2 is neutral. Its optimum also moves with the
configuration (50 at n=4/R^2, 300 at n=8/R^3), shifting the critic's target range.
`WeightedNormalized` is the dimensionless replacement.

**Episode reset** re-randomizes poses *and* edges (a fresh `random_scenario`), so the policy must generalize across geometries, not memorize one. Setting `env.freeze_network = True` makes `reset()` redo only the per-episode bookkeeping (`begin_episode()`) and keep the current graph — that is how `baselines.py` runs several methods on one instance.

**`skip_enabled`** (env config). When `False`, `train_ppo.py` / `train_dqn.py` pass `allow_skip=False` to the models, which mask the skip logit to `MASK_VALUE` (`-inf`) in `compute()` *and* in the DQN `random_act()`. The action space keeps its width, so checkpoints and `agent_loader` stay compatible. **Skip must either be masked out or be a real stop** (`skip_is_stop: True`): as a free no-op it is an absorbing zero-reward cycle that on-policy methods collapse onto (observed: entropy → 0, all rewards exactly 0, graph unmodified for two thirds of training). Generated configs default to skip masked out, scored with the best-state-visited metric below; the stop arm is `skip_enabled` + `skip_is_stop` + a small `time_penalty_value`, and is measured but unresolved (`DESIGN_NOTES.md#horizon`).

**Best-state-visited metric.** `Environment` tracks the highest-scoring graph seen during an episode (`best_state_score` / `best_edges` / `best_step` / `best_stats` with `m`/`is_IBR`/`is_MBR`/`rank`/`min_eig`, updated in `update_best_state()`), exposed in `info` and logged as `Episode/ Best *`. This is observational — the reward does not use it. It exists because scoring an episode on its *final* state conflates "found a good topology" with "learned to stop on it", which matters under `MaxSteps` where the agent is expected to converge and then hold with `skip`. `best_step` records how many steps it took to get there, which is the only way to tell a policy that converges fast from one that stumbles onto the same graph late.

`Best min eig` has no meaningful absolute scale: rigidity-matrix entries scale as `1/‖p_ij‖`, so it tracks `random_scenario`'s `pos_limits` (`scenario.py`, currently `[-1, 1]`; it was `[-100, 100]`, which put the eigenvalue at ~1e-5). Plot it on a log axis and don't compare across pose ranges. It frequently sits *below* `Min eig`, which is correct: `Weighted` has `w_eig = 0`, so φ trades rigidity margin away for fewer edges.

**Scenarios.** With `"scenario": "<name>"`, `initialize()` loads `scenarios/<name>.json` and caches it. What a scenario contributes on reset depends on `only_randomize_edges`: `false` carries over only the **domain mix** (poses and edges are redrawn each episode — use this for heterogeneous generalization experiments), `true` keeps the scenario's **actual geometry** and resamples only the edges (use this for a fixed case-study figure). Both paths honour `random_graph_with_mean_min_edges`.

**Config format keeps moving — regenerate, never hand-edit.** Current keys: `state_score_type`, `skip_is_stop`, `random_graph_with_mean_min_edges`, `include_candidate_bearings`, `rotation_augmentation`, plus the `graph_features` / `rigidity_*` flags. `max_steps` is now `4*m_req + 10` (n=8/R^3 → 50, `mixed` → 78, n=16/R^3 → 98), not `4*n*(n-1)`. Two switchable arms, **both off in generated configs**: the stop action (`skip_enabled` + `skip_is_stop` + `time_penalty_value`) and `rotation_augmentation`. For a scenario the generator writes the **full per-agent domain list** rather than `domains[0]`, which used to label every mixed config with one domain. See `DESIGN_NOTES.md#horizon` and `#rotation-augmentation`. `environments/` is gitignored and accumulates files from older formats, which will either `KeyError` in `load()` or raise on a merged-away `obs_type`. **Regenerating is the user's call** — see the note under "Gitignored" below. The filename no longer carries the obs type, since there is only one.

### Domains and scaling (measured, all five domains, n up to 64)

`rank_K` follows `(DOF per agent)·n − (trivial motions)`, and the trivial count grows with how much
of the rotation group the frames absorb (`THEORY.md` §3). Verified at n=8/16/32/64:

| domain | DOF/agent | `rank_K` | trivial | `c_max` | `m_req` at n=16 |
|---|---|---|---|---|---|
| `R^2` | 2 | `2n − 3` | 2 transl + scale | 1 | 29 |
| `R^3` | 3 | `3n − 4` | 3 transl + scale | 2 | 22 |
| `R^2xS^1` | 3 | `3n − 4` | + 1 rotation | 1 | 44 |
| `R^3xS^1` | 4 | `4n − 5` | + 1 rotation | 2 | 30 |
| `SE(3)` | 6 | `6n − 7` | + 3 rotations | 2 | 45 |

`c_max = 1` in the planar domains and `2` in the spatial ones, because a bearing is **one** angle in
the plane and **two** in 3-space. Note the oriented domains need far denser graphs — `R^2xS^1` at
n=16 needs 44 edges where `R^2` needs 29 — since each agent's heading must also be pinned down.

**Step cost is the blocker for large n.** Roughly, per env step: ~3 ms at n=8, ~10 ms at n=16,
25–100 ms at n=32, and **0.1–6 s at n=64**. At 600k steps that is hours at n=32 and over a week at
n=64. Anything beyond n≈32 needs a batched reimplementation of the environment, not tuning. (These timings re-randomize the graph each measurement and cost scales with `m`, so they are
*not* a clean `graph_features` comparison — for that see the controlled measurement in
`DESIGN_NOTES.md#graph-features`: 43.4 → 9.2 ms at n=16 on a fixed graph.)

### Invariance (`THEORY.md` §3, §9; audited per channel and end-to-end)

The whole point of a bearing framework is that it is defined up to a similarity transform, so the
observation should be too. Audited by transforming the network and diffing every channel, across
**all five domains** and both backbones.

**Translation and uniform scaling: invariant everywhere.** Every channel, every domain, to 1e-14.
Bearings are unit vectors; `coord_features` are centred and RMS-normalized; rank, flex and the
graph statistics are all similarity invariants.

**Rotation: invariant exactly in the domains that carry a frame.**

| domain | `edge_features` under rotation | policy logits |
|---|---|---|
| `R^2`, `R^3` | **changes** (~0.7–0.9) | **rotation dependent** |
| `R^2xS^1`, `R^3xS^1`, `SE(3)` | 1e-14 | **invariant** |

The cause is `Agent.get_bearing`: for an oriented agent it returns `R_i^T p̂_ij`, which is genuinely
invariant because the frame rotates with the world; for `R^2`/`R^3` there is no frame and it returns
the global-frame vector. Any heterogeneous mix containing an `R^d` agent inherits the problem.
**This is a property of the R^d testbed, not of the target domains** — see "Known issues" for what
to do about it.

**`coord_features` rotate by design** and that is fine: EGNN consumes coordinates only through
`‖x_i − x_j‖²`, verified to give *identical* feats under a rotation of `coors` alone; GINE never
reads them.

**Two traps in measuring this.** First, a stale `env.network` reference after a second `reset()`
silently makes everything look invariant. Second, and more insidious: at `egnn_pytorch`'s
`init_eps = 1e-3` an untrained EGNN is numerically blind to edge features, so it reports
`0.000e+00` under rotation *even in R^3 where the inputs plainly changed*. Only at trained-scale
weights does the dependence appear:

| `init_eps` | R^3 Δlogit | SE(3) Δlogit |
|---|---|---|
| 1e-3 (`egnn_pytorch` default) | 0.0 — false negative | 0.0 |
| 1e-2 | 1.8e-07 | 0.0 |
| 1e-1 (trained scale) | **4.8e-03** | 6e-08 |

Any invariance test on an EGNN must be run at trained-scale weights or it proves nothing.

### Policies (`policy/`)

`policy/gnn_backbone.py` holds the backbones; `policy/{actor,critic,q_func}/<Name>.py` hold one model per (backbone × action-space) combination, all re-exported from `policy/__init__.py`. Naming: `Equivariant_*` = EGNN, `GINE_*` = GINE, bare name = old GAT/MLP.

Conventions that matter when writing a new model:
- `unflatten_tensorized_space(self.observation_space, inputs["observations"])` recovers the obs dict — skrl flattens `Dict` spaces.
- **Action masking is done in the model**, by writing `MASK_VALUE` (`-inf`, scale-free by necessity) into invalid logits/Q-values (e.g. masking the already-selected node in `SelectNodesSequentially`, masking add-existing / remove-nonexistent in `AddRemoveEdge*`). The env does not mask.
- DQN Q-networks additionally override `random_act()` so epsilon-greedy exploration also respects the mask.
- `GNNBackboneEquivariant` output width is `node_feat_dim` (EGNN preserves feature dim; `gnn_hidden_dim` only sets the internal message width `m_dim`), whereas `GNNBackboneGINE` outputs `gnn_hidden_dim`. Head input sizes differ accordingly — a common source of shape errors.
- GINE flips `edge_index` before message passing so a node aggregates its *outgoing* bearings ("I measure this bearing to that node"), which is the semantically right direction here.
- **Both backbones do dense all-pairs message passing.** `GNNBackboneGINE.forward(nodes, edges)` takes the dense `(B, N, N, E)` edge tensor and builds the complete digraph itself; it used to message-pass over `adj.nonzero()` only, which silently discarded the all-pairs bearings. They now differ only in *how* they mix, which is what makes a backbone comparison meaningful. See `DESIGN_NOTES.md#gine-dense-all-pairs`.
- **The EGNN starts nearly blind to edge features.** `egnn_pytorch` inits every Linear at `std=1e-3`; three layers deep against the node residual, the edge path begins at ~1e-10 of the output. Structural rather than absent (a trained run grows those weights to ~1e-1), but it is a slow start, and an asymmetry against GINE, which inits at the torch default (~1e-1). `init_eps` is a `GNNBackboneEquivariant` argument and **now defaults to 1e-2**, not 1e-3. It is also a measurement trap: at 1e-3 an untrained EGNN reports invariance it does not have and reports sum vs mean pooling as identical, so any sensitivity test must run at trained-scale weights. See `DESIGN_NOTES.md#egnn-init-eps`.

**Aggregation is `mean` and the EGNN coordinate update is off**, both because dense all-pairs
passing otherwise makes activations scale with `n` and kills transfer. These are not tuning knobs —
`m_pool="sum"` or `update_coors=True` reintroduces the drift, and `tests/test_scale_invariance.py`
asserts both directions (it checks the fixed config is flat *and* that the broken configs still blow
up, so the guard cannot go vacuous). `update_coors=False` is also semantically right: `coors` are
ground-truth poses, EGNN reads them only via `rel_dist`, and the backbone discards the output
`coors`. See `DESIGN_NOTES.md#aggregation-and-scale`.

**The EGNN runs dense all-pairs message passing, deliberately.** `adj_mat` was passed to
`GNNBackboneEquivariant` and silently ignored — `egnn_pytorch` reads it *only* in nearest-neighbour
mode (verified: `max abs diff 0.0` between an all-zeros and an all-ones adjacency). It is no longer
forwarded; the graph reaches the model through the edge features, where an explicit `edge_exists`
channel states adjacency. Dense all-pairs is right for this task — you want to reason about edges
you do not have. `EGNN` also accepts a `mask` the backbone never passes, which is what variable-`n`
batching will need. See `DESIGN_NOTES.md#egnn-dense-all-pairs`.

**Bearings now cover every ordered pair**, not just existing edges, so the policy can see the
geometry of an edge it might add — previously it could not, which was the first-order cause of the
generalization failure. `include_candidate_bearings=False` (env config) reverts to edges-only at
the same observation shape; that is a modelling switch, not a tuning knob, because a distributed
agent cannot know a bearing it has not measured. See `DESIGN_NOTES.md#all-pairs-bearings`.

**`graph_features` (default `True`) toggles the expensive centralities** (closeness, eigenvector,
node/edge betweenness). They measure *worse* than free out-degree against rigidity-relevant targets
and cost 4.7x the step time at n=16 (43.4 -> 9.2 ms). `_lean` configs turn them off. See
`DESIGN_NOTES.md#graph-features`.

**Rigidity features are an ablation arm, off by default** (`rigidity_global` / `rigidity_flex` /
`rigidity_edge` in the env config). They add rank deficit, `m/m_req` and `is_IBR` as node channels,
plus per-node flex magnitude and two per-pair channels answering "would this edge raise the rank".
These are tier-3 quantities no distributed agent could compute, so the *gap* between arms is the
result, not the informed arm's number. Note `c_k` (per-edge block rank) is constant in every
homogeneous domain and so carries nothing at n=4/8/16 — it has its own flag for that reason. See
`DESIGN_NOTES.md#rigidity-features`.

**The pair channels are exact, not heuristic.** With `Z` an orthonormal basis of `ker(B)`, adding
edge `i -> j` raises the rank by exactly `rank(b_ij Z)`, so `add_gain = ||b_ij Z||/||b_ij||` is zero
precisely on the pairs that would add nothing and `add_rank = rank(b_ij Z)/c_max` is the gain
itself. Measured AUC 1.000 with a clean split in all five domains and three heterogeneous mixes,
exact rank on 1,501 pairs. `candidate_gain_reference` is the readable loop form and
`candidate_gain` the fast expansion of it, held to the reference by test in every domain. They
replaced `flex_align`, a position-block-only construction that was
at chance in the oriented domains (AUC 0.634 in SE(3)) because it could not see the attitude
columns. Two things they depend on that are easy to break: `ker(B)` is **not** scale-invariant
(position columns carry `1/length`, attitude columns are dimensionless), so the length unit is fixed
to the formation's RMS radius; and `rotate_network` must rotate `agent.rotation_axis`, since
`P_i = v v^T` is in world coordinates. See `THEORY.md` §13.

**Positions are pose-normalized** in the observation (centred, unit RMS radius). Bearings are
already scale-invariant but EGNN's internal `rel_dist` is not, which is why `pos_limits` used to
matter. The rigidity maths still uses the true poses.

**On feeding raw bearing vectors as EGNN edge features.** The rationale was that a bearing is
measured in the *measuring agent's local frame*, so for an agent that has a frame (`R^2xS^1`,
`R^3xS^1`, `SE(3)`) `R_i^T p̂_ij` is genuinely invariant to a global rotation of the network. That
holds — but **not for `R^2` / `R^3`**, where `Agent.get_bearing` returns the global-frame vector
(see the `if self.domain not in ["R^3", "R^2"]` branch in `network.py`). Every current experiment
is homogeneous `R^d`, so in practice global-frame vectors are being consumed as invariant scalar
features: rotate the whole network and the policy output changes, which defeats the reason for
choosing an EGNN.

**The pointer head has no pairwise term.** `SelectNodesSequentially` models score target `j` given
selected `i` as `MLP([h_j, h_i])` — no `e_ij`, no `adj_ij`, no `p̂_ij`, and no flag separating the
first pick from the second. "Nothing selected" is encoded as `h_sel = 0`, ambiguous with a
genuinely-zero embedding, and one MLP serves both picks. ROADMAP phase 5.

### Feature ablation (`ablation.py`)

Answers "what is this policy actually reading?" by destroying one observation channel at a time.
Two independent readings, because they answer different questions: **sensitivity** perturbs the
channel at each state of an unperturbed reference trajectory and reports how often the argmax
action changes (`flip%`) and how far the scores move (`|dscore|`); **outcome** re-runs the whole
episode with the channel perturbed and reports what it cost in phi, edges, rigid% and minimal%.
A channel can be sensitive but not matter (reacts, then recovers), or matter without being
sensitive (a small nudge compounds). `d phi` is *reference minus ablated*, so positive means the
policy got worse without it. Every variant is scored on the same instances (`freeze_network` plus
a deep-copied restore between rollouts — without the restore each variant starts where the
previous one ended, which silently invalidates the pairing).

`--mode shuffle` (default) permutes the channel across nodes/pairs, so the marginal distribution
is preserved and only the association with a particular node or pair is destroyed; `zero` and
`noise` also change the input *scale*, which a net can react to for reasons unrelated to meaning.

**Two traps it guards against explicitly.** Shuffling a channel that is constant along the
shuffled axis is a no-op — true of the one-hot `domain` channel in a homogeneous network and of
the global rigidity channels, which are tiled identically across nodes — and would otherwise print
a confident `0.0%` that reads as "ignored" when nothing was ablated; those rows say so and need
`--mode noise`. And `adj` feeds the model's *action mask*, so perturbing it changes which actions
are legal rather than what the policy knows; it is marked `*` and its `flip%` is not comparable.

The channel slice table mirrors `build_dict_obs`'s concatenation order and is checked against the
real observation width; on a mismatch (an archived observation format, typically) it degrades to
one block per key rather than risk attributing a number to the wrong feature.

`--csv out.csv` writes the same table the terminal shows -- same ranking, same rounding, the same
`(reference)` row -- and puts the legend in `out.txt` beside it, the same split `report.py` uses
for `results.csv` / `summary.txt`. The csv is a plain rectangle with the header on line 1: a legend
commented into its head parses fine with `pandas.read_csv(comment='#')` but shows ~27 lines of
noise in every spreadsheet, csv viewer and `column -s,`. Both outputs render from one
`order_rows()` / `table_rows()` / `legend()` path, so they cannot drift. A channel that was never actually
perturbed gets **empty** cells rather than zeros -- a 0.0 there would be averaged and plotted as
evidence of independence, which is the one thing it is not -- and `status` / `feeds_action_mask`
carry the two caveats above as columns.

What it found, and why it is not the failure it looks like: **destroying any geometric channel costs
the policy nothing.** That holds in all three modes and is the robust result. It is the *correct*
behaviour under an objective with no geometry in it (`ROADMAP.md` §1.1), not shortcut learning, and
it becomes the acceptance test for WP3: once the margin enters the reward, the geometric channels
must start costing something.

**Run more than one mode before believing a positive.** Under `zero`, `degree`, `rigidity_glob` and
`add_rank` look enormously important (+14.09, +12.20, +8.48 phi). Under `shuffle` they collapse to
noise except `degree`. Zeroing a normalized degree channel asserts every node has degree 0, and
zeroing `rigidity_glob` asserts a contradictory state (deficit 0 *and* `is_IBR` 0) — inputs the
network never saw, so the reaction measures out-of-distribution surprise rather than dependence. The
negative results are the ones that survive mode changes, because `zero` is the *aggressive*
ablation: a channel that costs nothing even when zeroed is genuinely unused.

### Baselines (`baselines.py`, `agent_loader.py`)

`baselines.py` scores every method through `Environment.compute_state_score`, so all rows of its
table are measured by the exact φ the agent trains on. `greedy` and `optimal` work in edit space
(action-space agnostic); `random` and `learned` go through `env.step()` and are therefore specific
to the configured action space, and are scored on best-state-visited. `optimal` scans edge count
ascending and stops at the first level admitting an IBR graph — unlike `MBR_required_Rd` this makes
no homogeneity assumption, but it is gated at `n ≤ 5`.

**`constructive` is the classical opponent.** From the empty graph, keep any edge that raises
`rank(B)`, stop at `rank_K`, best of `--restarts` random orders. It is the only method that does
*not* start from the initial graph, since it is a construction rather than an edit, and it carries
its own RNG so enabling it does not change the instances the other methods are scored on. Measured
at n=8/`R^3` (`m_req` = 10): 11.50 edges at 1 restart, 11.00 at 5, 10.75 at 20. Note that in the
`c_max = 1` domains the independent sets form a matroid and greedy is optimal by construction, so
beating it is only meaningful in `R^3` / `R^3xS^1` / `SE(3)`. See
`DESIGN_NOTES.md#constructive-baseline`.

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

**A manifest is now required.** The shape-sniffing fallback that used to reconstruct a bare `.pt`
(`load_agent_legacy`, `infer_architecture`, `match_model_class`, interactive algorithm prompts —
~230 lines) is gone; `load_agent()` raises if `train/<name>.json` is missing. That made 26
manifest-less checkpoints unloadable, all of them pre-dating the current observation format and
mostly unrecoverable anyway. What survived the cut is `backbone_depth()` + `rebuild_backbone()`,
because manifest-bearing runs at **both 2 and 3 layers** exist and `num_layers` is not a constructor
argument of the model classes — a depth-2 checkpoint still reports
`(archived source, backbone rebuilt at 2 layers)` on load.

### Training (`train_ppo.py`, `train_dqn.py`)

Both: load env config → build `SyncVectorEnv` → select actor/critic (or Q-net) by `(action_type, obs_type)` → `skrl` agent + `SequentialTrainer`. They write `train/<model_name>.json` containing hyperparameters, the env config, and **the actual source of the model classes** (`inspect.getsource`) — these manifests are the record of what was run, but `train/` is gitignored, so they live only on the machine that produced them. Env-side metrics go to `runs/<experiment_name>` via `Environment.write_episode()`, **once per episode** (see below).

**PPO: `memory_size` must equal `cfg.rollouts`.** skrl's `PPO.update()` runs `compute_gae` over the
*whole* `memory.get_tensor_by_name("rewards")` ring and then `memory.sample(batch_size=len(memory))`.
If the memory is larger than one rollout, every update trains on stale off-policy data (7/8 of it at
`memory_size=8192, rollouts=1024`) with `last_values` bootstrapped at the ring's wrap point instead
of the trajectory end — the stale samples fall outside the ratio clip band and contribute no
gradient. This mismatch existed only at commit `809f13a` and is what broke
`bigPPOSelectEquivariant3e-4lrNormalizedPositions`; `train_ppo.py` now uses one `ROLLOUT_SIZE`
constant for both. Do not reintroduce a separate `MEM_SIZE` — the "to ensure we don't get garbage
data from memory" comment was aimed at this and got it backwards.

**DQN target updates are soft, on one convention.** `polyak=0.005` with
`target_update_interval=1` (counted in *updates*, which happen every `update_interval=4`
timesteps) gives a target-network time constant of `update_interval/polyak` = 800 timesteps.
It used to pair that polyak with `target_update_interval=200`, mixing the soft and hard
conventions into a ~160k-timestep constant — effectively a frozen target across a 400k run.
Every checkpoint up to and including `letsgo_dqn_gine` was trained that way. Do not raise the
interval without dropping polyak to 1: `tests/test_training_smoke.py` asserts the constant.

**`ALGORITHM` selects DQN or DDQN** (`train_dqn.py`, env-var overridable, default `DQN`). They
share a config, a models dict and an argmax rollout, and differ only in the target value, so a
checkpoint is loadable either way — but the manifest and `models/complete/<ALGORITHM>/` both
follow the switch. DDQN is an arm needing its own control run, not a default.

**Decision-quality metrics and the training probe.** `Best`/`Final` cannot distinguish a policy
from a search, which is how two runs failed undetected. `Decision/ {useful,wasted,overshoot,converge}`
share one `Decision/quality` multiline chart; `Actions/ *` give the action-kind mix plus a real
histogram over action indices; `Steps rigid to minimal` isolates the pruning phase; `Edit efficiency`
is 1 for monotone editing and 0 for oscillation. `probe.py` rolls the policy out deterministically on
fixed seeded instances every `PROBE_INTERVAL` steps and logs `Probe/ argmax-sample gap` (~0 = a real
policy, negative = a sampler), `Probe/ useful (argmax)` against a random floor, and
`Probe/ max abs logit` (the drift detector). Calibrated: 0.00 gap and 0.725 useful for the good
checkpoint, -16.00 gap for the known sampler, 0.080 useful and 2.5e23 logits for the collapsed one.
The environment writer is now `torch.utils.tensorboard.SummaryWriter`, not skrl's shim, because the
shim has `add_scalar` only. See `DESIGN_NOTES.md#training-metrics`.

**Episode-level logging.** All environment metrics are written at episode end, not per step — a step-resolution scalar costs one tensorboard event per step and is then downsampled and averaged for display, so the detail was paid for and never seen. `step()` folds each step into `episode_accum` (`new_episode_accum()`: sums and counts only, so episode length is free), and on `terminated or truncated` builds `episode_summary()` → `last_episode_stats` → `write_episode()`, which dumps every entry under `Episode/ <key>`. Three views per episode: `Final *` (where it ended), `Best *` (best graph visited, plus `Best-final score gap` — 0 iff the episode ended on its own best graph), and `Mean *` / `* fraction` (the episode average). Scalars are written against `writer_counter` (global env step), *not* the episode index, so they share an x-axis with skrl's loss/reward curves; `writer_counter` therefore still advances every step. `last_episode_stats` is a plain attribute rather than an `info` key because `SyncVectorEnv` aggregates sub-env `info` dicts into arrays. `Environment.write(value, tag)` remains for custom scalars.

**`environments/` and `scenarios/` are the user's to manage.** Do not regenerate or hand-edit
existing ones; generate a new config under a new name if an experiment needs different settings.

### Gitignored (don't assume present)
`environments/`, `scenarios/`, `models/`, `runs/`, `runs_old*/`, `runs_baselines/`, `train/`, `tboard_logs/`, `junk/`, `dummy/`.

**`tools/` is where reusable scripts accumulate** (see the standing instruction above). `tests/` is
for anything the suite should run; `tools/` is for everything else worth re-running.

**`docs/` holds the note on the heterogeneous rigidity matrix** (`dof_restriction_note.tex`,
compiled `.pdf`) and two verification scripts. `verify_dof_restriction.py` is keyed section by
section to the note and checks the repository's own implementation; `verify_dof_restriction_2.py`
re-derives both constructions from the paper's conventions and shares no code with it. Keeping them
separate is the point: two independent implementations agreeing is the evidence.

**`benchmarks/` is deliberately NOT ignored.** A frozen instance set is a fixture, not an output —
its whole purpose is that a number measured today is comparable next month, which an untracked file
cannot deliver. Three 20-instance sets cost 32 KB. See `DESIGN_NOTES.md#benchmarks`.

**Nothing a run produces is tracked any more.** `runs_baselines/` and `train/` were both tracked until they were untracked wholesale — the first churned hundreds of binary files per run, the second is paired with `models/` which was already ignored. Consequence: a manifest now only exists on the machine that trained it, so `train/<name>.json` is no longer a shared record — back it up alongside the checkpoint it describes. Anything that has to survive (the README's figures) is copied into `resources/`.

Because every output directory is ignored, **a fresh clone has none of them**, and any code that writes output has to `os.makedirs(..., exist_ok=True)` *before* opening the file. Two places had the call inside the `with open(...)` block, which fails on exactly that fresh clone (`environment.py` writing `environments/`, `scenario.py` writing `scenarios/`); both are fixed. The library-managed paths take care of themselves: `SummaryWriter` creates its `log_dir`, and skrl creates its checkpoint directory (`skrl/agents/torch/base.py`).

## Known issues / open questions

Live research questions, not things to silently "fix":

1. **Termination condition.** There is no way to know the true optimal topology. The only sound stopping test is minimal bearing rigidity: exact via `MBR_required_Rd` for homogeneous `R^d`, otherwise the greedy `is_MBR` heuristic. The heuristic is a *sound lower bound* (rank subadditivity over edge blocks ⇒ no proper subset of the current edges can be rigid with fewer than `m_req` edges), and it reproduces the closed form exactly for homogeneous `R^2` and `R^3`. It can produce **false negatives** in heterogeneous networks, where the greedy sum over the highest-rank blocks may not be jointly realizable — a truly minimal graph is then never recognized and the episode never terminates. **Note the two `m_req`s**: `required_edge_count` accumulates block ranks of the *complete* graph (the true lower bound, and what `env.m_req` holds), while `is_MBR` recomputes from the *current* graph's blocks — which on a heterogeneous network can also **false-positive**, reporting a non-minimal graph as minimal. They coincide in every homogeneous domain, and the false positive does not fire on the `mixed` scenario (0 in 122 rigid graphs), but `is_MBR` should take `env.m_req` instead. WP7 hygiene.
2. **Initial-graph difficulty.** With `random_graph_with_mean_min_edges` the initial edge count is drawn around `m_req` with `sd = 0.5·m_req`, so `m0/m_req` is centred on 1 at every `n` and domain. With the flag *off* it falls back to `m ~ Uniform{0, …, n²-n}`, which over-constrains badly (~2.8× the requirement at n=8/R^3) and teaches deletion only — so leave it on unless you specifically want that.
3. **Constructive vs. editing formulation.** Starting from the empty graph and only adding edges is under consideration; the open worry is whether a purely constructive agent can reach *optimal* topologies rather than merely feasible ones. This is also the natural bridge to a distributed protocol (Henneberg-style vertex addition attaches each new agent with `d` edges, which is exactly what `MBR_required_Rd` counts) — see `ROADMAP.md` appendix A.
4. **Generalization is the blocking problem, and it is partly resolved.** Three causes were
   identified, all in `ROADMAP.md` §1: the reward contains no geometry so nothing forces the policy
   to read it (§1.1); the domain one-hot columns for unseen domains never receive a gradient (§1.5);
   and until WP1 the heterogeneous physics was wrong anyway (§1.2). **WP7 phase A (training on the
   `mixed` scenario) has now run and fixed the second one:** the catastrophic cross-domain failure
   is gone (100% rigid on homogeneous `SE(3)`, against 5% for the R^3-specialist). What remains is
   graded rather than catastrophic — transfer now degrades with agent DOF, matching both classical
   baselines at 3 DOF per agent and failing at 4 and 6, with the policy accumulating edges instead
   of pruning. Whether that is composition coverage (phase B resamples the mix, so high-DOF agents
   can dominate) or a capacity limit is **open**, and phase B carries a pre-registered acceptance
   criterion so the answer cannot be argued after the fact. WP3 (margin in the reward) addresses
   §1.1 and is untouched. Neither is a tuning change. Current numbers: `ROADMAP.md` §1.0.
5. `reset()` with no scenario file rebuilds the network from `agents[0].domain` only, so heterogeneous domains survive only via the `scenario` path (`randomize_scenario`).
6. Per-step cost is dominated by pure-Python graph features in `obs()` (Floyd–Warshall closeness, Brandes betweenness) plus repeated rigidity-matrix construction: `step()` builds `B`, then calls `is_MBR` unconditionally (a full-matrix rank *plus* one rank per edge, ~25 SVDs at n=8) and `rigidity_eigenvalue` when tracking (which rebuilds `B` and does a 48×48 `eigvalsh`). **`Weighted` needs neither.** Env stepping is several times the network's own cost at n=8, spent on metrics that never enter the reward. `graph_features=False` removes the centralities (4.7x at n=16); the rigidity-matrix work remains. Fine at `n=4–8`, the bottleneck beyond.
7. **Bearings are not rotation-invariant in `R^d`** (see Invariance above). The task is invariant;
   the observation is not, so the policy must learn the invariance from data. It **disappears in
   `R^2xS^1` / `R^3xS^1` / `SE(3)`**, which are the eventual targets, so this is an artifact of the
   `R^d` testbed rather than a defect of the formulation. It still confounds any generalization
   conclusion drawn on `R^d`. Options, cheapest first: random global rotation at `reset()` (the task
   is unchanged, so it is free data augmentation); per-instance frame canonicalization via the
   position covariance (exact, but discontinuous when the covariance spectrum is near-degenerate —
   common in symmetric configurations); replacing raw bearings with invariant pair descriptors
   (mutual angles `p̂_ij·p̂_ik`, a redesign of the edge features); or an architecture that consumes
   directions equivariantly (GVP / e3nn / vector neurons), which is the principled fix. Note also
   that rigidity is invariant under **reflection**, so whether the target group is `SO(d)` or `O(d)`
   is an open modelling choice.
8. **Four model classes in the obsolete `Default` (GAT/MLP) backbone are broken**, and were before
   any recent work: `AllEdges` references a missing `fc_edge_index`, `AddRemoveEdgeMultiDiscrete`
   is missing a `global_mean_pool` import, `AddEdgeDiscreteNoSkipNoSelfLoops` uses an undefined `n`,
   and `Default`+`SelectNodesSequentially` has a mis-sized head. They are still registered, so
   selecting them fails with a Python error rather than a clean "not implemented".
9. Small correctness items, none load-bearing but all live: `action_SelectNodesSequentially` computes `didnt_exist`/`existed` *after* the branch condition, so both are always `False` (dead code); `Network.fully_connected()` sets `edges = np.ones((n,n))` **including the diagonal** and `rank_K` survives only because of the `if i == j: continue` guard in `extended_bearing_rigidity_matrix`; `rigidity.is_MBR_Rd` has an unreachable-but-broken `if brmat: is_IBR_explicit()` branch (no arguments, and `if` on an ndarray raises); `np.random.seed(SEED)` runs *after* the sub-envs are constructed and the envs use global `np.random` rather than `self.np_random`, so all sub-envs share one stream.

**Previously listed here and since fixed** (do not re-chase): `Environment.load()` not reading `random_graph_with_mean_min_edges` and `reset()` discarding the sampled edge count; `reset()` leaving `last_state_score = 0` instead of the initial graph's score (`begin_episode()` computes it now); `step()` binding `info` only inside the tracking branch (`info = {}` is bound before it).
