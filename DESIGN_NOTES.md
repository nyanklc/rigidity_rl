# Design notes

Why the code is written the way it is. The source carries short comments; the reasoning behind them
is here, under a heading named after the symbol or flag. `CLAUDE.md` describes what the code is and
`THEORY.md` the mathematics.

---

## environment.py

### state-score

`compute_state_score()` sits outside `step()` because the reward is the shaped change of the score
between steps (`γφ(s') − φ(s)`, `THEORY.md` §7.3), so `reset()` needs the initial graph's score as a baseline (`begin_episode()`). Every
method in the experiments is scored through the same function, so all rows are measured by the
objective the agent trains on.

`WeightedNormalizedShapeError` is the score in use (`THEORY.md` §7.2). It is a separate
`state_score_type` rather than a change to an existing one, so archived runs replay, and at
`stiffness_kappa = 0` it equals `WeightedNormalized` exactly. `m_req` is not in the score because it
is a lower bound, and an understated denominator would over-penalize edges. `rank_K` and `c_max` are
exact.

One SVD per step serves three readers. `step()` computes `shape_err` once and passes it to
`compute_state_score`, to `compute_rigidity_features` (the `quality` channel uses the same scale, so
channel and reward agree) and to logging. `Environment.stiffness_term_active()` is the single gate
for whether the conditioning bonus is live, since `baselines.phi_landscape` needs the same answer.

`WeightedNormalized` and `WeightedNormalizedSpectral` build a per-episode reference `λ_ref` from
greedy constructions. That construction draws from the private `self.stiffness_rng`, reseeded per
episode, never from `np.random`, which is the stream instances are drawn from. Only the reference
the configured score reads is built.

### episode-constants

`compute_episode_constants()` depends on the poses and not on the edges, so it runs once per
episode and builds `B_K` once.

- `rank_K`, the rank a rigid graph must reach. Exact.
- `c_max`, the most rank one edge can contribute. Exact.
- `m_req`, the fewest edges that could make these poses rigid. A lower bound, used for reporting
  and the minimality metric, never in the reward.

### initial-edge-count

`sample_initial_edge_count()` draws around `m_req` with `sd = max(0.5 · m_req, 1)`, so `m0/m_req` is
centered on 1 at every size and domain. A uniform edge count would almost always be far above what
rigidity needs, and the agent would only ever learn to delete.

### random-domains

`random_domains` draws each agent's domain uniformly and independently at every `reset()`. The
observation width does not change, since the domain one-hot is always five wide and `rank_K`,
`c_max`, `m_req` are already per episode.

- The sampling order is mix, then poses, then `m_req`, then edge count, because the edge count reads
  this episode's `m_req`.
- It is refused together with `only_randomize_edges`, which would silently keep the poses.
- Benchmarks store domains and rotation axes per instance.
- Runs are named `randdom`.

### horizon

`max_steps = 4 · m_req + 10`. A run sees `total_timesteps / max_steps` distinct instances, so an
over-long horizon spends data on a policy cycling after it has already converged.

Skip is masked out by default. As a free no-op it is an absorbing zero-reward cycle that on-policy
methods collapse onto. The alternative is a real stop action (`skip_enabled` + `skip_is_stop` + a
small `time_penalty_value`), where `Q(s, stop) = −c`. The penalty has to stay well below one edge's
worth of `φ` or the agent stops early. Without a stop action an episode is scored on the best state
visited.

### rotation-augmentation

The task is invariant to a global rotation, but in `R^2`/`R^3` the observation is not (`THEORY.md`
§12). `rotation_augmentation` applies a random global rotation in `reset()`, which is free data
augmentation for frameless agents and a no-op for the rest. Only the `z` axis is admissible when a
planar agent is present. `rotate_network` rotates about the centroid and rotates each agent's
`rotation_axis`, and `random_scenario` carries `rotation_axes` alongside `domains`.

### dict-observation

One observation type, `"Dict"`, always emits:

| key | contents |
|---|---|
| `node_features` | domain one-hot (5), normalized in/out degree (2), optional graph centralities, optional rigidity node channels |
| `coord_features` | pose-normalized positions (3) |
| `edge_features` | bearing (3), `edge_exists`, reciprocity, common neighbors, optional edge betweenness, optional rigidity pair channels |
| `adj` | adjacency, used for the action mask |
| `selection` | pointer state for `SelectNodesSequentially` |

Models select the keys they need. The backbone is chosen by a `BACKBONE` constant in the training
script. The six older `Dict*` names map to flag presets that reproduce their old layouts exactly,
including raw coordinates, so older checkpoints are fed what they were trained on. `OBS_BACKBONE`
records the backbone each legacy name implied, and an unknown `obs_type` raises.

### all-pairs-bearings

`edge_features` carries the bearing for every ordered pair, edge or not, plus an explicit
`edge_exists` channel. The bearing of a candidate edge is what decides whether adding it raises the
rank, so the policy has to see it. `include_candidate_bearings=False` keeps only existing-edge
bearings at the same shape. That is a modelling switch for the distributed setting, where an agent
does not know a bearing it has not measured ([distributed-feasibility](#distributed-feasibility)).

### pose-normalization

`coord_features` are centered and scaled to unit RMS radius. Bearings are already scale invariant,
but EGNN's internal `rel_dist` is not. Normalization applies to the observation only, and the
rigidity maths uses the true poses.

### graph-features

`graph_features` toggles closeness, eigenvector centrality and node/edge betweenness. They are pure
Python `O(n³)` and dominate the step at moderate `n`, while carrying less rigidity-relevant signal
than out-degree, which is always present. The flag is off in generated configs.

### rigidity-features

Quantities derived from the rigidity matrix, which no local decision maker could compute. Six
flags, so information levels can be compared:

| flag | node channels | pair channels |
|---|---|---|
| `rigidity_global` | `(rank_K − rank)/rank_K`, `m/m_req`, `is_IBR` | |
| `rigidity_quality` | `quality` | |
| `rigidity_flex` | `node_freedom` | `add_independence` |
| `rigidity_edge` | | `c_k / c_max`, `add_rank` |
| `rigidity_stiffness` | `node_slack` (position, attitude) | `add_stiffness` |
| `rigidity_removal` | | `remove_rank`, `remove_stiffness` |

Derivations are in `THEORY.md` §8. Implementation points:

- **Computed on `scaled_rigidity_matrix`.** `ker(B)` moves under a uniform scaling because position
  and attitude columns carry different units. Taking `Z`, `v` and the removal downdate from the
  scaled matrix itself makes every channel scale invariant. Mapping a null space into scaled units
  afterwards would work for a subspace but not for the single eigenvector `v`.
- **One decomposition.** `nullspace_and_softest` returns the kernel, the softest mode (the column
  just past the kernel in ascending order) and `(w, V)` from one `eigh(BᵀB)`. The rank itself comes
  from the thin SVD in `rigidity_decomposition`, since squaring halves the precision of the
  eigenvalues.
- **No rebuilding for removal.** An existing edge's block is `B[3k:3k+3]` in `np.nonzero(edges)`
  order, a layout pinned by test.
- **Per-pair normalization** for `add_independence`, by the pair's own `||b_ij||`. Dividing by the
  spread over pairs would turn machine-zero gains on a rigid graph into an O(1) feature.
- **Stiffness channels are zero unless rigid**, and `λ` is passed as 0 unless rigid, which also lets
  `removal_costs` skip its per-edge downdate on flexible graphs and when the rank drops.
- **Ordering.** `_get_obs()` runs after the state score, so channels describe the current graph.
  `initialize()` fills the cache before declaring the observation space.
- **The two global flags are tiled** identically across nodes, which is why the ablation needs
  `--mode noise` for them.
- `c_k` is constant in every homogeneous domain and varies only in mixes, which is why it has its
  own flag.

Cost. `compute_rigidity_features()` computes the null space and `candidate_gain` before branching on
flags, so the first flag pays for both. `removal_costs` does one `eigvalsh(6n)` per redundant edge
and is the steepest per-step cost. `extended_bearing_rigidity_matrix` allocates `Dp` and `Da` densely
as `(3m, 3m)`, which on the complete graph at reset is `Θ(n⁴)` memory and is the real ceiling on `n`.
Applying `Dp` blockwise would remove it.

### episode-logging

Environment metrics are written once per episode, not per step. `step()` folds each step into
`episode_accum` (sums and counts only), and `episode_summary()` emits the episode as one flat record:
`Final *` (where it ended), `Best *` (best graph visited, plus the best-final gap) and `Mean *`.
Scalars are written against the global step `writer_counter`, so they share an x-axis with skrl's
curves. `last_episode_stats` is an attribute rather than an `info` key because `SyncVectorEnv`
aggregates `info` dicts.

### training-metrics

`Best` and `Final` cannot tell a decision rule from a search, so these metrics are blind to
best-state-visited.

| tag | definition |
|---|---|
| `Decision/ useful` | steps where `φ` strictly increased / steps |
| `Decision/ wasted` | (noop + skip) / steps |
| `Decision/ overshoot` | `max(0, m_final − m_req)/m_req` |
| `Decision/ converge` | `best_step / steps` |

All four share the `Decision/quality` chart. Action kinds are derived centrally in `step()` (add,
remove, skip, select, noop) and logged as fractions plus a histogram over action indices.
`Steps to first rigid`, `Steps to first minimal` and `Edit efficiency` (`|m_final − m_initial| /
edits`) describe the trajectory.

`probe.py` rolls the policy out on fixed seeded instances every `PROBE_INTERVAL` steps in argmax,
sample and random modes. `Probe/ argmax-sample gap` near zero means a decision rule and strongly
negative means a sampler. `Probe/ max abs logit` catches logit drift before it crosses the mask.

### benchmarks

`benchmark.py` freezes instances (poses, orientations, domains, rotation axes, initial edges) into
`benchmarks/<name>.npz`, and `--benchmark` evaluates on them. Regenerating a config resamples the
instance distribution, so frozen sets are what keep numbers comparable. They are tracked in git
because a fixture that is not tracked cannot do that.

---

## rigidity.py

### per-node-dof

`B = [Dp Ēᵀ S̄ | Da Ē_oᵀ P̄]` applies each agent's DOF projector on the column side (`THEORY.md`
§2.3). Homogeneous output equals the per-edge Table I construction, which `bearing_DOFs` is kept to
assert. `P_i` is the projector `v vᵀ`, correct for any rotation axis. The check is the definition
itself, a central-difference Jacobian test over all five domains and mixed networks with a
non-default axis.

### max-edge-rank

`max_edge_rank()` is the largest block rank over the complete graph. It is exact and turns "one
edge" into a unit comparable across domains without claiming anything about what is achievable.

### required-edge-count

Closed form for homogeneous `R^d`, greedy block-rank accumulation otherwise. A lower bound by rank
subadditivity, and the greedy sum may not be jointly realizable in a mix. Computed once per episode.

### repair-edge-count

The same bound from the graph in hand, using the exact per-pair marginals `rank(b_ij Z)` over absent
pairs (`THEORY.md` §10). Using the complete graph's block ranks instead would make it unsound.

---

## estimation.py

### estimation-monte-carlo

Evaluation only, never called from `step()`.

- **Full 2-DOF tangent noise in every domain.** The DOF restriction is on motion, not on the
  camera, and the component it makes unobservable is already a zero row of `B`.
- **The Jacobian is `B`, and `B` differentiates this module's own bearing map.** `true_bearings`
  builds `R_iᵀ p̂_ij` the way the matrix does, checked by central differences in all five domains.
- **Steps are restricted through `node_dof_projectors`**, so an agent never leaves its domain.
- **The gauge quotient is linear**, projecting off `ker(B_K)`. Centering and rescaling exactly
  would shift `z` for planar agents and delete real error. `max_scale_ratio` returns `inf` for a
  collapsed formation.
- **Gauss-Newton rather than an anchored linear solve.** The linear solve works in `R^d` but is
  bilinear in `(R, p)` in the oriented domains and needs domain-dependent anchoring. Gauss-Newton
  covers all five domains in one path and is the maximum-likelihood estimator the Cramér-Rao bound
  describes.

---

## Training (`train_ppo.py`, `train_dqn.py`)

### ppo-rollout-size

One constant feeds both the memory size and `cfg.rollouts`. skrl's `PPO.update()` runs GAE over the
whole memory ring and samples all of it, so a memory larger than one rollout trains on stale
off-policy data bootstrapped at the wrong point.

### discount-factor

`γ < 1` (0.99 in both trainers). The reward is `γφ(s') − φ(s)` with `γ = 0.99` written as a literal in
`step()`, so it must match the trainers' `discount_factor`. At `γ = 1` the advantage vanishes under a
near-uniform policy (`THEORY.md` §7.3). `Episode/ Return` logs the undiscounted return.

### dqn-target

`polyak = 0.005` with `target_update_interval = 200` updates (one update every 4 timesteps) gives a
target-network time constant of roughly 160k timesteps.

---

## Baselines (`baselines.py`)

### greedy

Hill climbing on the configured `φ`, scoring all `n(n−1)` single-edge toggles per improvement.
`score_network` takes rank and `λ` from one `rigidity_decomposition`, so greedy is
conditioning-aware at `κ > 0`, and skips `is_MBR` unless the score reads it.

### constructive

From the empty graph, keep any edge that raises `rank(B)`, stop at `rank_K`, best of `--restarts`
random orders by `φ`. This is Wolsey's greedy for submodular cover (`THEORY.md` §5), the classical
algorithm for the problem. It is the only method that does not start from the initial graph. It has
its own RNG, so enabling it does not change the instances other methods see.

### spectral

Ranks toggles by the closed-form landscape `phi_landscape` (`THEORY.md` §9) and rescores the
`--spectral-shortlist` best with the real `φ` before committing. At `κ = 0` the ranking is exact and
it reproduces greedy. At `κ > 0` the addition side uses `add_stiffness`, a prior normalized by its
own maximum that always shows some candidate as positive, so without the rescoring the hill climb
would never stop. The tie break is row-major stable to match greedy's, since in the `c_max = 2`
domains a rank-1 addition and a redundant removal tie exactly. `phi_landscape` raises on scores
outside `CLOSED_FORM_SCORES`, whose weights it hardcodes. `removal_costs(need_stiffness=False)` skips
the expensive half at `κ = 0`.

### anneal

Simulated annealing over single-edge toggles on `φ`, scored on best state visited. Temperature is in
units of `one_edge = w_e · c_max / rank_K`, so the schedule transfers across `n` and domain. The
default budget is the number of `φ` evaluations greedy spent on the same instance, read off the
`cost.py` counters. Own RNG. It is the opponent that assumes nothing about the objective, which
matters above `κ = 0` where greedy has no guarantee.

### degree

Add the absent pair minimizing `outdeg(i) + indeg(j)` until rigid, then drop the highest-degree edge
that keeps it rigid. It reads only degrees and the rigidity test, the reference for what a
distributed method could know. A test asserts it never calls `candidate_gain`, `removal_costs`,
`nullspace_and_softest` or `score_network`.

### cost-counters

`cost.py` counts calls to named primitives with a `@counted` decorator (one line each, no call site
changes, `functools.wraps` kept because manifests replay sources by name). A count is per call, so
`cost.OPERATION` says what one call is and wall time weighs them. `LEAVES` are primitives that call
no other counted primitive, and the headline total sums only those, which a test enforces.
`@cost.measurement` moves tracing and analysis work (`stats_now`, `edit_landscape`, `measure_noise`,
`repair_spread`) to its own bucket, so a method's cost does not depend on whether plots are on. The
meter wraps the method and not the instance restore.

---

## policy/

### model-registry

`policy/registry.py` maps `(role, backbone, action_type)` to a class, with roles being skrl's model
keys (`policy`, `value`, `q_network`). `instantiate()` filters a superset of kwargs against each
constructor's signature, so callers pass everything and each class takes what it declares.
`agent_loader` uses the same function. A `(role, backbone, None)` entry is the per-backbone
fallback, which covers the critics.

### action-masking

Invalid actions are masked in the model by writing `MASK_VALUE = -inf` into their logits or
Q-values. A finite sentinel stops being a sentinel once logits drift past it. A row with every
action masked would give NaN under softmax, and `unmask_if_all_masked` falls back to uniform. DQN
Q-networks also mask in `random_act`, so exploration respects the mask.

### pair-head

`AddRemoveEdgeDiscreteNoSelfLoops` heads score a pair from `[h_i, h_j, e_ij]`, so pair channels reach
the head directly rather than only through mean aggregation over `n − 1` pairs. Head width follows
`edge_feat_dim`. `SelectNodesSequentially` scores target `j` given selected `i` as `MLP([h_j, h_i])`
with no pair term.

---

## policy/gnn_backbone.py

### dense-all-pairs

Both backbones do dense all-pairs message passing, because the task is reasoning about edges the
graph does not have yet. `GNNBackboneGINE.forward(nodes, edges)` takes the dense `(B, N, N, E)` edge
tensor and builds a complete digraph internally (row-major, no self loops, cached per batch size,
`n` and device). `GNNBackboneEquivariant` does not forward `adj_mat` to `EGNN`, which reads it only in
nearest-neighbor mode. Adjacency reaches both models through `edge_exists`. The backbones then
differ only in how they mix.

### gine-edge-direction

GINE flips `edge_index` before message passing so a node aggregates its outgoing bearings ("I
measure this bearing to that node").

### egnn-input-embedder

`EGNN` preserves feature width, so `GNNBackboneEquivariant` embeds node features to `gnn_hidden_dim`
first (`Linear → LeakyReLU → Linear`). Both backbones then output the same width and the action heads
are shared. Embedding the invariant `feats` keeps equivariance, which is with respect to `coors`.
Equal width is not equal capacity, since the EGNN layers widen with `dim`, so a backbone comparison
should say which of the two it controls.

### egnn-init-eps

`egnn_pytorch` initializes every Linear at `std = init_eps`. At its default 1e-3, three layers deep
against the node residual, the edge path starts negligibly small and the model is initially blind
to edge features. `GNNBackboneEquivariant` defaults to `init_eps = 1e-2`. Any sensitivity or
invariance test on an EGNN must run at trained-scale weights, or an untrained model reports
invariance it does not have.

### aggregation-and-scale

Nothing the policy sees may scale with `n`.

1. **Mean aggregation.** With dense all-pairs passing a node receives `n − 1` messages, so a sum
   grows with `n`. GINE uses `aggr="mean"` and EGNN `m_pool_method="mean"`.
2. **No EGNN coordinate update.** The coordinate update is a hardcoded sum over `j` that re-enters
   later layers through `rel_dist` and compounds. `update_coors=False` is also semantically right,
   since `coors` are ground-truth poses and the backbone discards the returned `coors`.
3. **Normalized counts.** Degree and common neighbors are divided by `m_req/n`, the mean degree of a
   minimally rigid graph. Flex magnitude is normalized by its own total. Legacy presets keep
   `normalize_counts=False`.
4. **Initial-graph spread** proportional to `m_req` ([initial-edge-count](#initial-edge-count)).

`tests/test_scale_invariance.py` checks that the fixed configuration is flat in `n` and that the
broken ones still drift, so the guard cannot become vacuous.

### backbone-num-layers

The layer count is a constructor argument, so checkpoints at other depths load
(`agent_loader.rebuild_backbone`). Submodules are named `conv1..convN`.

---

## Experiments and reporting

### experiment-scripts

`experiments/` holds one runnable script per experiment. Each parses its own arguments, loads its
own environment and models, and writes its own run directory under `runs_outputs/`. What they share
is the methods (`baselines.py`), figures and tables (`report.py`), the harnesses (`repair.py`,
`ablation.py`, `convergence.py`) and the setup (`experiments/common.py`), so a method or a palette
has one implementation. Each script replaces `sys.path[0]` with the repository root, which is what
lets `experiments/repair.py` import the root's `repair.py`. `generalisation.py` and `scale.py`
aggregate earlier run directories and record each row's source directory and commit.

### observation-compatibility

A checkpoint loads only against an observation of the same width and meaning.
`observation_mismatch` compares the config keys that decide the observation (`OBS_KEYS`) and names
the differing key before anything loads. The environment name is only a note, since configs get
regenerated under the same name.

### method-identity

`report.configure_methods(models)` rebuilds the method tables once per run. With one policy every
method keeps a categorical hue (`SINGLE_MODEL_STYLE`). With several, the hues go to the models and
the baselines move to ink tones and dashes. `method_style(name)` is the only lookup and never hands
two methods the same style. One model keeps the label `learned`. Several are labeled by the name
tokens that differ, and a collision is an error.

### palette

Colors come from the data-viz reference palette in its documented order, validated for adjacent
pairs under color-vision deficiency. `initial` and `optimal` are reference points and take neutral
inks with dashed strokes. Every series carries a direct label, so identity never rests on color
alone. Validate a palette change with the ported checks rather than by eye.

### plain-figures

Every figure is written twice, the second as `<name>-plain`. `report.plain()` sets a module flag that
`_figure`, `_finish`, `_save` and `plot_table` read. The plain variant drops the header and notes
card and keeps titles, axis labels, units, reference lines, legends and series labels. Callers draw
the whole set, then draw it again inside `plain()`.

### run-info

`run_info` is one figure recording the environment, instances, methods, and per model the
algorithm, backbone, widths, training length, learning rate, seed, commit, and the objective it was
trained on beside the one it is scored by.

### at-bound

`m_req` is stamped per row in the episode loop and reaches `results.csv`. `at_bound` (rigid on
exactly `m_req` edges) is exact, and `is_MBR` is the heuristic, so both columns are reported.

### paired-comparison

Every method runs on the same instances, so `report.paired_stats` differences each method against a
reference per network and bootstraps over the differences. `shape_err` is differenced in `log10`.
The output is an interval with wins, losses and ties, and no p-value.

### decision-analysis

`edit_landscape` scores every legal single-edge change at each step by both `Δφ` and `Δ shape_err`,
and `decision_record` ranks the policy's choice among them by midrank. The policy is trained on `φ`,
so the two rankings separate what the objective rewards from what it leaves behind. The error
ranking is absent while the graph is flexible. Gated by `MAX_DECISION_ANALYSIS_N`.

### noise-sweep

`--noise-sweep` perturbs every bearing of each method's reported graph, recovers the formation and
compares measured against predicted error. Flexible graphs get no sweep, since their error is
infinite.

### formation-figures

`uncertainty`, `softest_mode` and `sensitivity` draw the network in real 3-D axes, with marker shape
for the domain. Ellipsoids share one exaggeration factor set from the median panel, with the true
size printed, while softest-mode arrows are scaled per panel. Flexible networks get no panel.
`FORMATION_PRIORITY` decides which methods survive the panel cap, separately from `METHOD_ORDER`.
These figures place their own panels instead of using `tight_layout`.

### ablation-protocol

`ablation.py` destroys one observation channel at a time. Sensitivity measures how often the argmax
changes along a reference trajectory, and outcome re-runs the episode with the channel perturbed.

- Every variant is scored on the same instance, restored by deep copy between rollouts.
- The reference stops at its first repeated state, and every variant is capped at that budget, or
  ablating a channel buys extra exploration.
- `φ` must be a function of the state, which is why per-episode references are reseeded.
- Shuffling a channel that is constant across nodes is a no-op and is reported as not ablated.
  `adj` feeds the action mask, so its row is marked.
- `--live-env` uses the working-tree environment instead of the archived one.
- `coord_features` is a null control for GINE, which never reads coordinates.

### control-harness

`convergence.py` is the harness and `experiments/control.py` the driver. Each case is one goal shape
and one start displaced within each agent's admissible DOFs (`convergence.admissible`), shared by
every method, so the edge set is the only difference. One edge set serves both the goal bearings and
the controller's `B`, and it is chosen on the goal poses. Position error is measured modulo the
gauge, bearing error per edge. A run that never reaches tolerance reports `None` rather than the
step budget. The 3-D figure normalizes each formation (center, unit radius) before drawing. A global
rotation is not removed, so a converged `SE(3)` formation can appear rotated.

---

## repair.py

### repair-harness

`repair.py` damages a converged network and scores how each method restores it.

- **The headline is a percentile** among all repairs of the same size, by shape error, 100 = best.
  The population is complete: exhaustive below a cap, rejection-sampled above it, decided before
  enumerating. Only a method that spent exactly the minimum is placed in it.
- **The converged graph favors no method.** It is a greedy construction pruned to minimality in
  random order by "stays rigid".
- **Damage is keyed on the instance**, `event_rng(seed, index, event, k)`, so adding a method
  cannot change what another faces.
- **Three stopping rules from one rollout** (`first_rigid`, `best_phi`, `final`). `headline_rule`
  picks `first_rigid` where rigidity broke and `best_phi` where it survived, and exactly one row per
  instance, event and method is reported.
- **Node events resync the observation space** (`sync_observation_space`) whenever `n` changes, or a
  model returns a wrongly sized output without raising.
- **`bearing_corrupt` is not implemented.** `B` is built from true poses, so corrupting a measured
  bearing changes nothing a method can observe. Doing it properly needs per-edge weights in `BᵀWB`
  and a reliability channel.

---

## scope

### distributed-feasibility

The long-term motivation is a distributed protocol for maintaining rigid formations, and the
centralized formulation is a first step that should not foreclose it. Rigidity and minimality are
global, and the current policy is centralized (dense all-pairs backbones, a global action index,
global features). A distributed version is still tractable with centralized training and
decentralized execution: an actor restricted to `K` rounds of message passing over the communication
graph is a `K`-hop local rule, and per-node action heads replace the global index. In homogeneous
`R^d`, Henneberg-style vertex addition reaches minimal rigidity without any agent computing a global
rank. A distributed policy can at best certify rigidity with local minimality.

#### Feature availability tiers

- **Tier 1, locally available.** Own pose and domain, own degree, bearings currently measured,
  whatever one-hop neighbors communicate.
- **Tier 2, available under a sensing-radius assumption.** Bearings to agents not currently
  measured (candidate edges). Admissible if detection is cheap and only maintaining a link costs,
  not if every measurement costs what an edge costs. `include_candidate_bearings` switches it.
- **Tier 3, not available.** `rank(B)`, rank deficit, `c_k`, `is_IBR`, the null-space and spectral
  channels, the graph centralities.

GNN depth stays a constructor argument, per-node and per-edge heads stay first-class, and
candidate-bearing inclusion stays switchable, to keep this direction open.
