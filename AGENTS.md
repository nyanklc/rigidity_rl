# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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
- **State uncertainty explicitly** - both the degree of it and the reason for it - whenever you
  are not confident.
- **NEVER commit, and never create a branch's first commit for the user.** `git add` / `git commit`
  / `git push` are the user's job - they verify every change manually first. Leave work in the
  working tree, say what changed, and stop there. Creating a branch is fine; committing to it is not.
- **NEVER edit documentation unless explicitly asked.** `README.md`, `THEORY.md`,
  `DESIGN_NOTES.md` and this file are the user's to write. This is not a preference
  about style, it is about the user being able to see what changed: doc edits buried inside a code
  change are invisible in review, and wrong claims have reached `README.md` more than once that way.
  When a change makes something in them stale or false, **say so and stop** - name the file, quote
  the line, say what it should say instead, and wait for an answer. Do not fix it yourself, do not
  fix it "while you are in there", and do not treat "keep the docs current" as authorisation.
  Reporting staleness is required; editing is not yours to do.
- **What the documents are for**, when you are asked to write one. The test is whether a *fresh
  session with no conversation history* can read the repo and continue the work. Document the
  current state, not how it got there - no changelogs, no decision logs, no "this used to be X".
  Git history is the record of what changed; a stale narrative in a file that loads into every
  session costs more than it is worth.
- **Every number in a document must be re-derivable.** If a measurement is quoted anywhere, the
  script that produces it belongs in `tools/`. A number whose derivation exists only in a deleted
  section or in a past conversation cannot be checked and will eventually be wrong without anyone
  noticing.
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
- **Keep code comments brief, and do not cite the docs from code.** One or two lines saying what is
  non-obvious, and nothing else - no `see THEORY.md §12` pointers, no measurements, no history of
  what the code used to do, no ticket or work-item names. Derivations, measurements and rationale
  belong in `THEORY.md` / `DESIGN_NOTES.md`; a reader who wants them will search for the symbol.
  The same goes for test docstrings: state what the test pins down, not why.
- **Figures: plain titles, plain wording, and always two variants.** A figure or panel title names
  the quantity plotted and nothing else. No slogans, no rhetorical questions, no invented names,
  no puns. Any text on a figure states what the reader is looking at and how it was computed.
  Do not put the conclusion on the figure. Report findings in the conversation and let the user
  decide what they mean.
  Write the way a person writes. No em-dashes. No colon-led reveals, meaning a short lead-in that
  sets up a payoff after a colon or dash, of the kind `Here's the thing: ...`, `The catch: ...`,
  `Bottom line: ...`, `And the result? ...`. State the thing directly in one sentence instead.
  No summary flourishes and no marketing tone. A plain glossary that maps a column name to what
  it measures is not this and is fine to keep, since a table is unreadable without one.
  Every figure is written out twice under the same name, with `-plain` appended to the second.
  One carries whatever annotation is useful and one is the plot alone. Plain does not mean raw.
  Reference lines, legends, axis labels, units and the markers that identify a series stay in
  both, because without them the plot cannot be read.
- **American English, always.** `behavior`, `normalize`, `optimize`, `center`, `neighbor`,
  `analyze`, `modeling`, `labeled`, `while` rather than `whilst`, `among` rather than `amongst`.
  This is about the prose only: code identifiers, file names, LaTeX labels and quoted output keep
  whatever spelling they already have, so `SHAPE_ERR_CENTRE`, `experiments/generalisation.py` and
  `\label{sec:generalisation-results}` are not to be renamed for spelling.
- **Write so the user does not have to rewrite it.** The test of a paragraph is that it needs
  minimal editing, not that it is impressive. The figure rules above are the general rule, not a
  special case for figures: no em-dashes, no colon-led reveals, no summary flourishes, no
  marketing tone. Add to that no semicolons doing the work a full stop should do, and no lists of
  three for rhythm. State the thing directly. This applies to everything written for the user.

## Project

Master's thesis, completed: **network topology optimization for bearing rigid multi-agent systems
via deep reinforcement learning and graph neural networks.** The thesis PDF is
`resources/thesis.pdf`, and its LaTeX source is in `resources/thesis/tex/` (gitignored).

A multi-agent network is a **directed graph**. Each agent lives in a domain (`R^2`, `R^3`,
`R^2xS^1`, `R^3xS^1`, `SE(3)`) that fixes its DOFs. A directed edge `i -> j` means agent `i`
measures the bearing to agent `j` in its own frame (the global frame for `R^2`/`R^3`). Edges are
not symmetric and each one is a sensing cost.

The RL task: given agents at random poses, **choose the edge set** so the network is
infinitesimally bearing rigid with as few edges as possible and, through the conditioning term, as
little shape error under bearing noise as possible. A GNN encodes the graph into node embeddings and
an action head turns them into logits (PPO) or Q-values (DQN).

**Long-term motivation, not pursued here:** a *distributed* method for maintaining rigid formations.
The centralized formulation is a first step. A local decision maker would lack two kinds of
information: global algebra (`rank(B)`, rank deficit, `c_k`, `is_IBR`, centralities) and bearings to
agents it is not measuring. Whether the second is an obstruction depends on whether detection is
cheap and only a maintained link costs. See `DESIGN_NOTES.md#distributed-feasibility`.

## What is live vs. obsolete

| Axis | In use |
|---|---|
| Action spaces | `AddRemoveEdgeDiscreteNoSelfLoops` (toggle one directed edge), `SelectNodesSequentially` (pointer style, every second pick toggles the edge between the two picks) |
| GNN backbones | `GNNBackboneEquivariant` (EGNN) and `GNNBackboneGINE` (GINE) |
| Obs type | `Dict`. The six older `Dict*` names survive as flag presets that reproduce their old layouts, so old configs and checkpoints load. See `DESIGN_NOTES.md#dict-observation` |
| State score | `WeightedNormalizedShapeError`. `WeightedNormalized`, `WeightedNormalizedSpectral` and `Weighted` are kept so old runs replay. All agree at `stiffness_kappa = 0` |
| Algorithms | PPO and DQN, both via `skrl` |

**Obsolete, ignore unless asked:** `main.py`, `control.py` (gradient formation controllers from the
original control-oriented aim), everything `sb3` (`train_ppo_sb3.py`, `policy_sb3.py`,
`models/sb3/`), `junk/`, `runs_old*/`, `fix_train.py`, `dummy*`, the GAT (`Default`) backbone, and
most of the older action/obs/state-score variants still present in the dispatchers.

## Running things

`uv` manages a plain `.venv` populated with `uv pip` (no `pyproject.toml` or lockfile). `setup.sh`
creates it. Run scripts with `uv run <script>.py <args>`.

```bash
# Create an environment config. The generator reads its constants from the __main__ block,
# each overridable by an environment variable of the same name
uv run environment.py <n> <domain>          # e.g. uv run environment.py 8 "R^3"
uv run environment.py file <scenario_name>  # n and domains from scenarios/<name>.json

# Train (manifest to train/<model_name>.json, checkpoints under models/, TB logs under runs/)
uv run train_dqn.py <environment_name> <model_name>
uv run train_ppo.py <environment_name> <model_name>
#   <model_name> may be "prefix=foo" to auto-append action type and domain
#   if the run exists it prompts: [c]ontinue / start [f]resh / [a]bort

# Roll out a trained model
uv run inference.py <model_name> <environment_name>

# Freeze evaluation instances
uv run benchmark.py <environment_name> <benchmark_name> [--instances N] [--seed S]
uv run benchmark.py list

# Experiments: one script each, each writing its own directory under runs_outputs/
uv run experiments/comparison.py <environment_name> [--episodes N] [--benchmark NAME]
    [--model A,B=label] [--methods a,b|all] [--restarts K] [--noise-sweep 0.5,1,5]
#   methods: initial, random, degree, greedy, spectral, anneal, constructive, learned, optimal
uv run experiments/feature_ablation.py <environment_name> --model NAME [--mode shuffle|zero|noise|all]
uv run experiments/training.py <environment_name> [--model NAME] [--runs a,b]
uv run experiments/generalisation.py <environment_name> --model NAME [--prior GLOB]
uv run experiments/scale.py <environment_name> --model NAME [--methods ...] [--prior GLOB]
uv run experiments/kappa.py <environment_name> [--prior GLOB]
uv run experiments/budget.py <environment_name> --benchmark NAME --model NAME [--steps N]
uv run experiments/topology.py <environment_name> [--benchmark NAME] [--methods ...]
uv run experiments/repair.py <environment_name> --benchmark NAME [--events all] [--k 1,2]
uv run experiments/control.py <environment_name> --model NAME [--methods greedy,complete,learned]
#   each script's docstring lists its full arguments

# Which observation channels a trained policy reads
uv run ablation.py <model_name> [environment_name] [--mode shuffle|zero|noise] [--csv out.csv]

# Manifests: list, show, diff against the working tree, verify, backfill
uv run manifest.py list | show <name> | diff <name> | verify <name> | backfill [--write]

# Interactive viser GUI to hand-edit a graph and watch rigidity metrics
uv run manual.py <environment_name>

# Reproduce the heterogeneous rigidity-matrix note (docs/)
PYTHONPATH=. uv run docs/verify_dof_restriction.py [--quick]
PYTHONPATH=. uv run docs/verify_dof_restriction_2.py

tensorboard --logdir runs
```

`tools/` holds reusable verification and measurement scripts, listed in `tools/README.md`. Run them
from the root with `PYTHONPATH=.`.

Names are filenames without extension: `<environment_name>` is `environments/<name>.json`,
`<scenario_name>` is `scenarios/<name>.json`.

**Tests: `uv run tests/run_all.py`** (fast suite) or `uv run tests/run_all.py --slow` (adds training
runs, brute force and large `n`). Individual files run standalone, e.g.
`uv run pytest tests/test_flex.py -v`. The fast suite passes on a fresh clone, since tests build
environments programmatically and checkpoint-dependent tests skip.

`test_feature_scale_*` is unseeded and its verdict is a draw: it samples fresh graphs at each size,
so the ratio it thresholds is a random variable and `add_independence` / `add_rank` straddle the
gate. Do not read a change in its failure set as a regression without running
`tools/feature_scale_noise.py`.

The suite pins the invariants: the `-inf` mask sentinel, `n`-invariance of observation channels and
backbone activations, the exact addition criterion against rebuilt-matrix ground truth in every
domain, per-domain `rank_K`/`c_max`/`m_req`, similarity invariance per channel, the legacy obs
presets, the state score's closed form, the shape estimator against the Cramér-Rao bound, and a
nonlinearity between stacked `Linear`s in every live model. There is no linter or CI.

## Architecture

### Rigidity core (`rigidity.py`, `network.py`, `util.py`)

The maths is in `THEORY.md`; this is where it lives.

- `Agent` holds a pose (position + quaternion) and a `domain`. `Network` holds the agents and
  `edges`, an `(n, n)` boolean adjacency matrix (row = measuring agent).
- `node_dof_projectors(agent)` returns the per-node `(S_i, P_i)`. The DOF restriction belongs to
  the node, not the edge (`THEORY.md` §2.3). `bearing_DOFs` is the per-edge Table I construction,
  unused by the matrix and kept as a homogeneous reference, faithful only at `v = e₃`.
- `extended_bearing_rigidity_matrix(network)` returns `B`, `(3m, 6n)`, one 3-row block per directed
  edge in `np.nonzero(edges)` order.
- `is_IBR`: `rank(B) == rank_K`. `rank_K` is cached on the env per episode, so pass it through.
- `rigidity_decomposition(B, rank_K)` returns rank, singular values and `λ` from one thin SVD. `λ`
  is 0 unless rigid. It is the conditioning of shape recovery, not a degree of rigidity.
- `estimation_error(s, rank_K)` returns the A-, D- and E-criteria from the singular values.
  `estimation_error_of(network, rank_K)` runs it on `scaled_rigidity_matrix`, which makes it
  comparable across `n`, domain and pose range. `shape_err = sqrt(a_opt / n)`.
- `is_MBR` is the minimality heuristic, `max_edge_rank` gives `c_max` (exact),
  `required_edge_count` gives `m_req` (a lower bound), `repair_edge_count` the repair bound,
  `MBR_required_Rd` the closed form for homogeneous `R^d`.
- `candidate_gain` / `candidate_gain_reference`, `nullspace_and_softest`, `removal_costs` compute the
  rigidity observation channels (`THEORY.md` §8). The fast forms are held to the readable ones by
  test.
- `measurement_sensitivity` splits `tr((BᵀB)⁺)` exactly over measurements and agents.
- `flex_tensor`, `flex_constraint_power`, `trivial_modes` are an older position-only flex
  construction, tested but not used by the environment.

### Cost accounting (`cost.py`)

A `@counted` decorator on the linear-algebra primitives and a `Meter` context manager that reports
the deltas and wall time for one method. Counts are per call to a named primitive (`cost.OPERATION`
says what one call does). `cost.LEAVES` is the subset that calls no other counted primitive, and the
headline total sums only those. `@cost.measurement` / `cost.reporting()` move tracing and analysis
work into its own bucket. Under `--replay-env` against a pre-counter manifest, a method that
evaluated `φ` with zero counted work is reported as UNMEASURED. See `DESIGN_NOTES.md#cost-counters`.

### Estimation (`estimation.py`)

Evaluation only. `perturb_bearings` adds tangent-plane noise of per-axis `sigma` radians to every
bearing, `solve_shape` recovers the poses by damped Gauss-Newton with `B` as the Jacobian (steps
restricted by `node_dof_projectors`), `shape_error` projects the error off `ker(B_K)`,
`monte_carlo_error` drives the sweep and `predicted_error` is the analytic counterpart. The gauge
quotient is linear, so this is a small-error metric. Do not replace it with exact centering and
rescaling, which shifts `z` for planar agents. See `DESIGN_NOTES.md#estimation-monte-carlo`.

### Environment (`environment.py`)

One `gymnasium.Env` configured by a JSON file in `environments/` through `env.load(path)`. Each axis
is a string dispatched in a module-level function:

- `action_type` → `define_action_space()` + `action_<Type>(...)`
- `obs_type` → `obs()`
- `state_score_type` → `compute_state_score()`
- `termination_condition_type` → inline chain in `step()`

To add a variant, add a branch in the dispatcher and a matching model in `policy/` registered in
`policy/registry.py`.

**Reward:** `−time_penalty + [action_reward] + γφ(s') − φ(s) + [terminal bonus]`, potential-based
shaping on the state score (`THEORY.md` §7.3). `γ = 0.99` is a literal in `step()` and must match
the trainers' `discount_factor`. Keep `γ < 1`. `Episode/ Return` logs the
undiscounted return.

**Conditioning term:** `stiffness_kappa · one_edge · 1[IBR] · q`, with `q` a fixed-center sigmoid of
`log10(shape_err)` under `WeightedNormalizedShapeError`. `κ` bounds the edges traded for
conditioning and in practice selects among graphs of equal edge count, so minimality stays a valid
metric. `WeightedNormalized` / `WeightedNormalizedSpectral` instead build a per-episode `λ_ref` from
greedy constructions, which is expensive at reset.

**Episode reset** re-randomizes poses and edges. `env.freeze_network = True` makes `reset()` redo
only `begin_episode()` and keep the graph, which is how experiments run several methods on one
instance. The initial edge count is drawn around `m_req` when `random_graph_with_mean_min_edges` is
on.

**`skip_enabled`.** When `False` the models mask the skip logit (also in DQN `random_act()`), and the
action space keeps its width. Skip must be either masked or a real stop (`skip_is_stop: True`), since
a free no-op is an absorbing zero-reward cycle. Without a stop action an episode is scored on the
best state visited (`best_state_score`, `best_edges`, `best_step`, `best_stats`, logged as
`Episode/ Best *`). See `DESIGN_NOTES.md#horizon`.

**`random_domains`** draws each agent's domain every `reset()`, so the composition is part of the
instance distribution. Runs are named `randdom`.

**Scenarios.** With `"scenario": "<name>"`, `only_randomize_edges: false` carries over only the
domain mix, and `true` keeps the scenario's geometry and resamples edges.

**Config keys** include `state_score_type`, `skip_enabled`, `skip_is_stop`, `time_penalty_value`,
`random_graph_with_mean_min_edges`, `random_domains`, `include_candidate_bearings`,
`rotation_augmentation`, `stiffness_kappa`, `stiffness_ref_samples`, `spectral_functional`,
`graph_features` and the six `rigidity_*` flags. `max_steps = 4 · m_req + 10`. The generator's
`__main__` constants currently give `WeightedNormalizedShapeError`, `stiffness_kappa = 50`,
`random_domains = True`, `rotation_augmentation = True`, `graph_features = False`, every `rigidity_*`
flag on, and skip masked out. Old configs may `KeyError` in `load()`, and the renamed
`margin_*` keys raise. **Regenerate rather than hand-edit, and `environments/` and `scenarios/` are
the user's to manage**: generate a new config under a new name if an experiment needs different
settings.

### Domains

| domain | DOF/agent | `rank_K` | `c_max` |
|---|---|---|---|
| `R^2` | 2 | `2n − 3` | 1 |
| `R^3` | 3 | `3n − 4` | 2 |
| `R^2xS^1` | 3 | `3n − 4` | 1 |
| `R^3xS^1` | 4 | `4n − 5` | 2 |
| `SE(3)` | 6 | `6n − 7` | 2 |

Step cost grows steeply with `n`. The dense `(3m, 3m)` `Dp`/`Da` allocation on the complete graph
at reset is the ceiling, and anything well beyond `n ≈ 32` needs a batched reimplementation.

### Invariance

Every observation channel is invariant to translation and uniform scaling in every domain. Rotation
invariance is exact in `R^2xS^1`, `R^3xS^1`, `SE(3)` and fails for raw bearings in `R^2`/`R^3`, where
`Agent.get_bearing` returns a global-frame vector. `rotation_augmentation` is the mitigation. The
rigidity channels are rotation invariant everywhere. `coord_features` rotate by design, and EGNN
reads them only through distances. Two traps when measuring this: a stale `env.network` reference
after a second `reset()` makes everything look invariant, and an EGNN at small `init_eps` is
numerically blind to edge features, so invariance tests must use trained-scale weights.

### Policies (`policy/`)

`policy/gnn_backbone.py` holds the backbones and `policy/{actor,critic,q_func}/<Name>.py` one model
per backbone and action space, re-exported from `policy/__init__.py`. Naming: `Equivariant_*` is
EGNN, `GINE_*` is GINE, a bare name is the old GAT/MLP. `policy/registry.py` maps
`(role, backbone, action_type)` to a class, and `build_models()` returns the dict skrl takes. See
`DESIGN_NOTES.md#model-registry`.

Conventions:

- `unflatten_tensorized_space(self.observation_space, inputs["observations"])` recovers the obs
  dict, since skrl flattens `Dict` spaces.
- Action masking is done in the model with `MASK_VALUE = -inf`. DQN Q-networks also mask in
  `random_act()`.
- Both backbones output `gnn_hidden_dim` (EGNN through an input embedder), do dense all-pairs
  message passing over the `(B, N, N, E)` edge tensor, use mean aggregation, and the EGNN coordinate
  update is off. Adjacency reaches the model through the `edge_exists` channel. `m_pool="sum"` or
  `update_coors=True` reintroduces `n`-dependence, and `tests/test_scale_invariance.py` checks both
  directions. See `DESIGN_NOTES.md#aggregation-and-scale`.
- GINE flips `edge_index` so a node aggregates its outgoing bearings.
- `GNNBackboneEquivariant` defaults to `init_eps = 1e-2`.
- `AddRemoveEdge` heads take `[h_i, h_j, e_ij]`. The `SelectNodesSequentially` pointer head scores
  `MLP([h_j, h_i])` with no pairwise term and encodes "nothing selected" as `h_sel = 0`.

### Observation

Bearings cover every ordered pair, with an explicit `edge_exists` channel.
`include_candidate_bearings=False` reverts to existing-edge bearings at the same shape. Positions in
the observation are centered and RMS-normalized. `graph_features` toggles the expensive
centralities. The `rigidity_*` flags add rigidity-matrix channels (`THEORY.md` §8,
`DESIGN_NOTES.md#rigidity-features`), all read off the length-normalized matrix:

- `rigidity_global`: rank deficit, `m/m_req`, `is_IBR`, tiled across nodes.
- `rigidity_quality`: the state score's `q` for the current graph, tiled.
- `rigidity_flex`: `node_freedom`, `add_independence`.
- `rigidity_edge`: `c_k / c_max`, `add_rank`.
- `rigidity_stiffness`: `node_slack`, `add_stiffness`, nonzero only once rigid.
- `rigidity_removal`: `remove_rank`, `remove_stiffness`, nonzero only on existing edges.

### Feature ablation (`ablation.py`)

Destroys one observation channel at a time. **Sensitivity** perturbs the channel at each state of
an unperturbed reference trajectory (`flip%`, `|dscore|`). **Outcome** re-runs the episode with the
channel perturbed (`d phi` is reference minus ablated, so positive means worse without it). Every
variant runs on the same instances, restored by deep copy. `--mode shuffle` permutes across
nodes/pairs, while `zero` and `noise` also change the input scale. A channel constant along the
shuffled axis (the domain one-hot in a homogeneous network, the tiled global channels) is reported
as not ablated and needs `--mode noise`. `adj` feeds the action mask and is marked `*`. The reference
stops at its first repeated state and every variant is capped to that budget. `coord_features` is
the null control for GINE. `--csv` writes the table with the legend in a `.txt` beside it, and a
channel never perturbed gets empty cells. Run more than one mode before believing a positive. See
`DESIGN_NOTES.md#ablation-protocol`.

### Experiments (`experiments/`, `baselines.py`, `report.py`, `agent_loader.py`)

One standalone script per experiment under `experiments/`, each writing
`runs_outputs/<timestamp>__<env>[__<models>][__<tag>]/`. They share the methods (`baselines.py`),
figures and tables (`report.py`), harnesses (`repair.py`, `ablation.py`, `convergence.py`) and setup
(`experiments/common.py`). Each driver puts the repository root on `sys.path` in place of
`experiments/`.

| script | what it measures |
|---|---|
| `comparison.py` | every method on the same instances, scored by the same `φ` |
| `feature_ablation.py` | what a trained policy reads |
| `training.py` | learning curves from the tensorboard events |
| `generalisation.py` | earlier runs aggregated, one per instance set |
| `scale.py` | earlier runs aggregated across network size, with cost |
| `kappa.py` | edge count against shape error across `stiffness_kappa` |
| `budget.py` | answer quality as the number of allowed moves is cut |
| `topology.py` | the structure of the graphs each method builds |
| `repair.py` | how each method restores a damaged network |
| `control.py` | how fast a formation reaches a goal shape over each method's edge set |

**Methods** (`baselines.py`, dispatched by `run_method(name, ctx)` inside a `cost.Meter`):
`initial`, `random`, `degree` (degree-only, the distributed-plausible reference), `greedy` (hill
climbing on `φ`), `spectral` (closed-form landscape plus a verified shortlist), `anneal` (budget
matched to greedy's `φ` evaluations), `constructive` (Wolsey's greedy from the empty graph),
`learned` (each `--model`), `optimal` (exhaustive, `n ≤ 5`). `greedy`, `spectral`, `anneal`,
`degree`, `constructive` and `optimal` work in edit space; `random` and `learned` go through
`env.step()` and are scored on best state visited. `constructive`, `anneal` and `degree` have private
RNGs so they do not move the instance stream. `phi_landscape` raises on scores outside
`CLOSED_FORM_SCORES`. See the per-method entries in `DESIGN_NOTES.md`.

**`--model` takes a list.** `NAME[=LABEL]`, repeatable and comma-splittable. One model is labeled
`learned`; several are labeled by the name tokens that differ. `--replay-env` takes one model only.
`observation_mismatch` names the differing config key before a model that does not fit is loaded.
`--policy-mode sample|greedy` and `--device` are one setting per run.

**Outputs of `comparison.py`:** `summary.txt` (table, cost block, paired block, legends),
`results.csv`, `trajectories.csv`, `cost.csv`, `decisions.csv` (with `--model`), `meta.json`, and
`plots/{pdf,png}/`. Every figure is written twice, the second as `<name>-plain` with the header and
notes card removed (`report.plain()`). The table reports means with spreads; `rigid`/`minimal`/`=best`
have none since they are 0/1 means; stiffness and shape error are geometric. `shape err` is the
column to read, as RMS shape error per radian of bearing noise, lower is better. `at bound` (rigid
on exactly `m_req` edges, exact) and `minimal` (`is_MBR`, heuristic) are both reported. The paired
block (`report.paired_stats`) bootstraps per-network differences against `--paired-reference`.
`run_info` records each model's training setup and objective. Colors go to models once there is
more than one (`report.configure_methods`, `method_style`).

**Manifests** (`manifest.py`, `train/<name>.json`) archive hyperparameters, the env config, the
model and backbone source, the full text of `util.py`, `rigidity.py`, `network.py`, `scenario.py`,
`environment.py` and `policy/gnn_backbone.py` (gzipped, base64), the scenario file and provenance
(commit, command, versions, device, seed). `agent_loader.load_agent()` requires a manifest.
`load_run(model, env)` rebuilds the archived environment when the working tree differs
(`archived_modules()`), and `resolve_model()` replays archived model and backbone source, verified
against the checkpoint's parameter shapes. `backbone_depth()` / `rebuild_backbone()` handle
checkpoints at 2 or 3 layers. A checkpoint therefore survives edits to `policy/` and the environment.

### Training (`train_ppo.py`, `train_dqn.py`)

Load env config, build a `SyncVectorEnv`, build models from the registry, train with skrl's
`SequentialTrainer`. `BACKBONE`, `SEED` and the main hyperparameters are module constants
overridable by environment variable. Environment metrics go to `runs/<experiment_name>` once per
episode through `Environment.write_episode()` (`Final *`, `Best *`, `Mean *`, decision-quality and
action-kind metrics), and `probe.py` adds periodic deterministic rollouts. See
`DESIGN_NOTES.md#training-metrics`.

- **PPO `memory_size` must equal `cfg.rollouts`**, which is one `ROLLOUT_SIZE` constant. A larger
  memory trains on stale off-policy data.
- **DQN target updates are slow**: `polyak = 0.005` with `target_update_interval = 200` updates
  gives a time constant of roughly 160k timesteps.

### Gitignored (do not assume present)

`environments/`, `scenarios/`, `models/`, `runs/`, `runs_old*/`, `runs_outputs/`, `train/`,
`tboard_logs/`, `junk/`, `dummy/`, `resources/thesis/`, `resources/papers/`. A fresh clone has none
of them, so code that writes output must create its directory first. A manifest exists only on the
machine that trained it, so back it up with its checkpoint. `benchmarks/` is tracked on purpose,
since a frozen instance set is a fixture.

`docs/` holds the note on the heterogeneous rigidity matrix (`dof_restriction_note`) with two
independent verification scripts, and `observation_reward_note`, the reference for every reward and
observation component.

## Known issues

1. **Termination.** The only sound stopping test is minimal rigidity, exact via `MBR_required_Rd`
   for homogeneous `R^d` and otherwise the `is_MBR` heuristic. On heterogeneous networks `is_MBR`
   can false-negative (a minimal graph never recognized) and, because it uses the current graph's
   blocks rather than the complete graph's, false-positive. It should take `env.m_req`.
2. `reset()` with no scenario file rebuilds the network from `agents[0].domain` only, so
   heterogeneous domains survive only through the scenario path or `random_domains`.
3. Per-step cost is dominated by rigidity-matrix construction and decompositions, plus the
   pure-Python centralities when `graph_features` is on.
4. **Bearings are not rotation invariant in `R^d`** (see Invariance). This is a property of the
   `R^d` testbed, not of the oriented target domains. Principled fixes are invariant pair
   descriptors or an architecture that consumes directions equivariantly.
5. **Four model classes in the obsolete `Default` backbone are broken**: `AllEdges` (missing
   `fc_edge_index`), `AddRemoveEdgeMultiDiscrete` (missing `global_mean_pool` import),
   `AddEdgeDiscreteNoSkipNoSelfLoops` (undefined `n`), and `Default`+`SelectNodesSequentially` (head
   mis-sized). They are still registered.
6. Small items: `action_SelectNodesSequentially` computes `didnt_exist`/`existed` after the branch,
   so both are always `False`; `Network.fully_connected()` sets the diagonal, and `rank_K` survives
   only through the `i == j` guard in the matrix builder; `rigidity.is_MBR_Rd` has an unreachable
   broken `if brmat:` branch; `np.random.seed(SEED)` runs after the sub-envs are built and the envs
   use global `np.random`, so all sub-envs share one stream.
