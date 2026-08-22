# Design notes

Rationale that would otherwise sit in the source as long comment blocks. The code carries a short
comment and nothing more; search here for the symbol or flag you are looking at.

`CLAUDE.md` describes what the code is, `THEORY.md` the mathematics. This file is the "why is it
written this way" layer.

---

## environment.py

### state-score

`compute_state_score()` lives outside `step()` because the reward uses the *improvement* of the
score between steps, so `reset()` needs the initial graph's score as a baseline
(`begin_episode()`).

### weighted-normalized

`Weighted` (`20*rank - 10*m`) measures rank and edge count in raw units, which do not mean the same
thing across domains. A rigidity-matrix edge block has rank **2 in R^3** and **1 in R^2**, so a
rank-adding edge is worth +30 in R^3 but only +10 in R^2, against +10 for pruning a redundant edge
in both. R^3 is therefore three times more eager to add than to prune while R^2 is neutral. The
optimum also moves with the configuration - 50 at n=4/R^2, 270 at n=8/R^3, 590 at n=16/R^3 - which
shifts the critic's target range whenever `n` or the domain changes. One policy cannot span both.

`WeightedNormalized` puts both terms in units of rank and divides by `rank_K`:

```
phi = w_rank * rank/rank_K  -  w_edge * (m * c_max)/rank_K
```

`m * c_max` is "the rank this many edges could have carried at best", so the edge term is the
fraction of the required rank the agent has spent edges on. Both numerators are ranks and both
denominators are `rank_K`, so the score is dimensionless and its scale no longer moves with `n` or
the domain.

**Deliberately not normalized by `m_req`.** `m_req` is a lower bound from greedy block-rank
accumulation, not a ground truth (see [required-edge-count](#required-edge-count)); it can only
ever *understate* the true minimum, and an understated denominator over-penalizes edges. `rank_K`
and `c_max` are both plain rank computations - exact, and asserting nothing about achievability.

The payoff is that the central guarantee becomes structural rather than empirical. A maximally
informative edge gains `w_rank * c_max/rank_K` and costs `w_edge * c_max/rank_K`, so it is worth
adding **iff `w_rank > w_edge`** - for any geometry, domain mix or `n`, because the same
`c_max/rank_K` factor appears on both sides. Under `m/m_req` the two factors were `c_max/rank_K`
and `1/m_req`, which coincide only when `m_req` happens to equal `rank_K/c_max`.

`w_rank/w_edge = 4` reproduces R^3's existing 3:1 preference for adding rank over pruning a
redundant edge, now identically in every domain. That ratio is the meaningful knob; the overall
scale only sets the reward magnitude. phi's ceiling is `w_rank - w_edge = 75` when the poses admit
a perfectly packed rigid graph and slightly below otherwise - a fact about the geometry, not a
tuning issue.

Measured (greedy baseline, same 100/25 weights):

| config | rank_K | c_max | m_req | greedy phi | old `Weighted` phi |
|---|---|---|---|---|---|
| n=4 / R^2 | 5 | 1 | 5 | 75.00 (= brute-force optimal) | 50 |
| n=8 / R^2 | 13 | 1 | 13 | - | 130 |
| n=8 / R^3 | 20 | 2 | 10 | 73.00 | 270 |
| n=16 / R^3 | 44 | 2 | 22 | - | 590 |

### margin-in-phi

`margin_kappa` (env config, default `0.0` = off) adds the rigidity margin to
`WeightedNormalized` as `kappa * one_edge * q(lam)`, with `q` a sigmoid of `log10(lam/lam_ref)`.
Derivation, the two obstacles to using `lam` raw, and every measurement are in `THEORY.md` §15.
What matters here is the plumbing:

- **`lam` costs nothing.** `step()` already gets it from the single `rigidity_decomposition` it
  performs for the rank, and `begin_episode()` from its own. `compute_state_score` takes it as an
  optional `lam=None`, so the older call signature still works and `kappa = 0` is byte-identical.
- **`lam_ref` is an episode constant**, built in `compute_episode_constants` from
  `rigidity.reference_margin`. All of the cost is here: reset goes 2.7 -> 46.8 ms at n=8/`R^3`
  (`margin_ref_samples=3`), while **per-step cost is unchanged**. Over a 50-step episode that is
  +55% wall clock at n=8/`R^3`, +27% on `mixed`, and it is not currently a blocker anywhere. If it
  becomes one, the addition oracle (`candidate_gain`) can pick rank-raising edges from one
  nullspace instead of `O(n^2)` rank computations per round - deliberately not done, because it
  would make the reference construction differ from the `constructive` baseline's.
- **`self.margin_rng` is private and must stay private.** `lam_ref`'s construction order draws from
  it, never from `np.random` - that is the stream instances are drawn from, and using it would move
  the networks every method is scored on. This is the exact regression recorded for `constructive`
  once, so `test_enabling_the_margin_does_not_move_the_instance_stream` pins it.
- **One construction, shared.** `rigidity.greedy_rigid_construction` is the loop;
  `baselines._construct_once` is now a thin wrapper on it. A reference construction that drifted
  from the baseline would silently change what `lam_ref` means. Verified byte-identical to the
  previous inline loop over 4 seeds x 3 configurations, and `bench_n8_R3` reproduces its
  `initial`/`greedy`/`constructive` rows exactly.
- **`baselines.score_network` now takes rank *and* `lam` from one `rigidity_decomposition`** instead
  of `matrix_rank` via `is_IBR_explicit`. Roughly cost-neutral, since `matrix_rank` already performs
  an SVD, and necessary: without it every `greedy` candidate would be scored with the margin term at
  zero, which at `kappa > 0` is not the configured phi. The side effect is that `greedy` becomes
  margin-aware for free.

### episode-constants

`compute_episode_constants()` depends on the poses but not the edge set, so it runs once per
episode. `B_K` is built once and shared because it is the expensive part.

- `rank_K` - rank of the fully-connected graph's rigidity matrix; the rank a rigid graph must
  reach (`3n-4` in R^3, `2n-3` in R^2). **Exact.**
- `c_max` - the most rank one edge could contribute at these poses. **Exact.**
- `m_req` - fewest edges that could possibly make these poses rigid. **A lower bound.** Reported,
  and used for the MBR metric; never in the reward.

### initial-edge-count

Uniformly random edge counts are almost always far above what rigidity needs - the requirement
grows ~linearly in `n` while `n^2-n` grows quadratically - so the agent would only ever see graphs
that need edges removed. `sample_initial_edge_count()` samples around the minimum requirement
instead. The mean is only exact for homogeneous R^d networks; for other domains it is below the
true requirement, which is acceptable.

### training-metrics

`Best *` and `Final *` cannot tell "found a good graph" from "searched until it stumbled on one".
Two runs died in ways those metrics never showed: a policy that scored 100% rigid / 85% minimal on
best-state-visited but was no better than random under argmax, and two runs that reached 99.3%
minimal and then spent 240k steps emitting invalid no-ops. These metrics are chosen to be blind to
best-state-visited, so a searcher cannot score well on them.

#### The one chart

`set_writer()` registers a `Decision/quality` multiline layout over four per-episode scalars:

| tag | definition | reads as |
|---|---|---|
| `Decision/ useful` | steps where phi strictly increased / steps | knows *which* edit helps |
| `Decision/ wasted` | (noop + skip) / steps | is not stalling |
| `Decision/ overshoot` | `max(0, m_final - m_req)/m_req` | is not padding edges |
| `Decision/ converge` | `best_step / steps` | decides fast vs searches |

`overshoot` is unbounded above while the others are in [0,1]; in a healthy run it sits near 0, so
sharing the axis is fine.

#### Action kinds

Everything above needs to know what a step *did*. Derived centrally in `step()` rather than in each
of the ten `action_*` functions, extending the convention `acc["skips"]` already used:

```
m increased -> add     m decreased -> remove     "skip" in info -> skip
"select" in info -> select   (the pointer's first pick: protocol, not waste)
otherwise -> noop            (add-existing / remove-absent)
```

The strings come from the same file, so this stays a local convention. Verified by construction on
scripted episodes for both `SelectNodesSequentially` and `AddRemoveEdgeDiscreteNoSelfLoops`, whose
no-op semantics differ.

Logged as `Actions/ {add,remove,noop,skip,select} fraction`, plus `Actions/ index` as a real
histogram - a collapsed policy puts all its mass on one index.

#### Trajectory shape

`Steps to first rigid`, `Steps to first minimal` and their difference, `Steps rigid to minimal`.
The last isolates the n=16 failure: reaching rigidity is fine, *pruning* is where it stalls.

`Edit efficiency` = `|m_final - m_initial| / edits`: 1 = every edit moved the count the same way,
0 = pure oscillation. Measured 1.00 on a monotone deletion sequence and 0.33 on an add/remove cycle.

#### The probe (`probe.py`)

Every `PROBE_INTERVAL` timesteps, `PROBE_EPISODES` **fixed seeded** instances are rolled out in
argmax, sample and uniform-random modes. Fixed instances mean the curve tracks the policy, not
instance noise. Hooked into the `post_interaction` wrapper both training scripts already had.

`Probe/ argmax-sample gap` is the headline: ~0 means a genuine decision rule, strongly negative
means a sampling search. `Probe/ useful (argmax)` against `Probe/ useful (random)` answers "better
than chance?" directly. `Probe/ max abs logit` catches logit drift *before* it crosses the mask.

Calibrated against three known policies:

| policy | argmax phi | gap | useful (argmax) | useful (random) | max abs logit |
|---|---|---|---|---|---|
| good checkpoint (phase4 @300k) | **75.00** (optimum) | **0.00** | **0.725** | 0.237 | 1e9 |
| known sampler (phase3 AllBearings) | 58.50 | **-16.00** | 0.354 | 0.166 | - |
| collapsed checkpoint (phase4 @600k) | 55.83 | 0.00 | **0.080** | 0.237 | **2.5e23** |

The collapsed policy is *below* the random floor on useful-action rate, and its logits are fourteen
orders of magnitude out. Both would have been obvious in real time.

DQN has no sampling distribution, so its argmax and sample coincide; the gap is logged as 0 rather
than faking a second mode.

Cost: ~8% of training throughput at a 1.5k interval, so ~0.5% at the 25k default.

### horizon

`MAX_STEPS = 4*m_req + 10`, not the old `4*n*(n-1)`.

The old budget was 20-30x the measured `best@` (6.6-12.8 steps across every run), and
`Edit efficiency` ended training at 0.018 ~ 1/56: the policy reached its answer around step 7 and
then ran a two-cycle for the remaining ~217 steps because `skip_enabled: false` forced an edit
every step.

The argument for cutting it is about **data**, not wall clock. A run sees
`total_timesteps / max_steps` distinct instances, and its replay buffer holds
`memory_size * num_envs / max_steps` of them (skrl buffers are `(memory_size, num_envs, ...)`, so
10000 x 4):

| config | old `max_steps` | instances seen | in buffer | new | instances seen | in buffer |
|---|---|---|---|---|---|---|
| n=8 R^3 | 224 | 1785 | 178 | 50 | 8000 | 800 |
| `mixed` n=10 | 360 | 1111 | 111 | 78 | 5128 | 512 |
| n=16 R^3 | 960 | 416 | 41 | 98 | 4081 | 408 |

Raising `memory_size` is not the alternative: each transition stores a 1050-float observation
twice, so the buffer already costs ~336 MB at n=10.

**Verified free.** `generaldqngine` re-evaluated at the 50-step horizon reproduces its 224-step
result exactly -- 10.05 edges, 100% rigid, 95% minimal, best@ 8.0. The `random` row does get worse
(85% -> 60% rigid), which is correct: it was benefiting from a longer search budget, so the shorter
horizon makes it a fairer floor.

Shorter episodes mean far more resets, and `reset()` rebuilds `B_K` and `m_req`. Measured, that
amortises to 0.06-0.23 ms/step against a 1.2-8.5 ms step -- 2-3%, not a concern.

`m_req` depends only on `(n, domain mix)`, not the poses, so the config generator settles it from a
single draw.

**skip is an arm, off by default.** Generated configs keep `skip_enabled: false`; the stop arm is
`skip_enabled: true` + `skip_is_stop: true` + a small `time_penalty_value`. The reason skip was
masked out is that as a *free no-op* it is an absorbing zero-reward cycle that on-policy methods
collapse onto; as a terminating action it is not, and it makes the final state a meaningful headline
metric instead of best-state-visited. The penalty has to stay well under one edge's worth of phi
(`w_edge*c_max/rank_K`, 1.1-2.5 across the configurations in use) or the agent stops before it is
finished.

#### Is the stop action worth having? (measured, unresolved)

`skip_enabled` / `skip_is_stop` / `time_penalty_value` are env keys, so both terminations are
trainable arms. Four 150k-step DQN runs at n=8/R^3, identical apart from those keys, evaluated by
argmax on the frozen `bench_n8_R3`:

| arm | time penalty | stops? | steps | **final** min% | best-visited min% |
|---|---|---|---|---|---|
| A  no stop | -- | no | 50 | 55% | **95%** |
| B  stop | 0.05 | yes | 7.7 | **85%** | 85% |
| C  stop | 0.20 | yes | 7.5 | 70% | 75% |
| D  stop | 0.01 | yes | 7.0 | 50% | 50% |
| greedy (reference) | -- | -- | 6.2 | 50% | 50% |

**It does not collapse**, which was the thing worth ruling out. `Q(s, stop) = -c` exactly -- the
graph does not change, so the shaping term is zero and the episode ends -- making it a constant and
trivially learnable, and the danger is that a *guaranteed* value beats a badly estimated one early
on. It does not happen: initial graphs are far from optimal, so improving actions have clearly
positive `d phi` from the start. Episode length falls 30 -> ~7 over training and settles;
`Episode/ Terminated` reaches 1.00; the policy stops *on* its best graph (`Best-final score gap`
1.79 -> 0.04).

**The two columns say opposite things.** As a deployed policy (final state) the stop arm is the best
thing measured: 85% against 55% for no-stop and 50% for greedy, at 6.5x fewer edits. As a search
(best of everything visited) no-stop wins 95% to 85% -- but that 95% costs 50 edits and takes the
max over the trajectory; arm A reaches its best at step 6.8 and then wanders for 43 more because it
is forced to act.

**Not resolved, and it would take seeds to resolve.** tp = 0.01 -> 50%, 0.05 -> 85%, 0.20 -> 70% is
non-monotone, and D stops at the same ~7 steps as B. A 35-point swing between penalty values that
produce identical behaviour is more likely seed noise than sensitivity, and one seed per arm cannot
separate them. Treat both terminations as arms; do not quote either number as settled.

That variance is itself worth recording: **at n=8/R^3 a single seed spans at least 35 points of
minimality**, so any single-seed headline is exactly that, and a
three-seed protocol is not optional.

**A trap in reading the training curves.** The TensorBoard averages make the stop arms look far
worse than the argmax evaluation does (`Best is min rigid` 0.97 vs 0.57) because training episodes
still carry epsilon = 0.05. Over 50 steps that is 2.5 random edits, which arm A absorbs; over ~7
steps it is 0.35, and one of them can be *stop*, ending the episode early. Judge terminations on an
argmax evaluation, never on the curves.

### rotation-augmentation

`rotation_augmentation` (env config, **default `False` everywhere** -- generator, `initialize()` and
`load()` -- so it is an arm and archived runs replay unchanged). Applies a random global rotation in
`reset()`.

The task is invariant to a global rotation. The observation is not, in `R^2`/`R^3`: `get_bearing`
returns a global-frame vector when the agent has no frame of its own, so rotating the network moves
the policy's output. Audited end to end -- `R^d` logits move, `R^2xS^1`/`R^3xS^1`/`SE(3)` are
invariant to 6e-08. This is free data augmentation for the frameless half and a no-op for the rest,
and it matters on `mixed`, where four of ten agents are frameless.

**Measured effect is small.** Same 20 instances with and without a global rotation
(`bench_n8_R3` vs `bench_n8_R3_rot`): `generaldqngine` scores 95% minimal / 10.05 edges unrotated,
90% / 10.10 rotated. One instance in twenty, within noise at that sample size. The augmentation is
free and principled, not a fix for a large defect -- do not oversell it.

**But the effect is model-dependent, and larger than this row suggests.** The same paired benchmark
run against `letsgo_dqn_gine` (trained on `mixed` **with** `rotation_augmentation` on, evaluated
out of distribution at n=8/`R^3`) flips **8 of 20** instances: 10 minimal unrotated against 14
rotated. Every classical method is byte-identical across the pair, so the churn is entirely the
policy. Read the flip count, not the net, and do not conclude the augmentation fixed anything --
this policy had it enabled during training and still moves.

**Only the z axis is admissible when any planar agent is present** -- an arbitrary axis would lift
it out of its plane. `rotate_network` rotates about the centroid and leaves the z component
untouched under a z rotation, so planar agents stay at z=0 exactly (asserted).

While adding this: `random_scenario` now carries `rotation_axes` the way it already carried
`domains`. `set_domain` resets the axis to `e3`, so an `R^dxS^1` agent with a scenario-specified
axis silently lost it on every reset. Nothing measured depended on it (`e3` is the only axis in
use), but the maths is correct for arbitrary axes and the environment could not produce one.

### benchmarks

`benchmark.py` freezes evaluation instances (poses, orientations, domains, rotation axes, initial
edges) into `benchmarks/<name>.npz`; `baselines.py --benchmark <name>` evaluates on them instead of
sampling, and records the name and a content digest in `meta.json`.

This exists because regenerating an env config silently resamples the instance distribution, and
that has already invalidated one comparison: the two n=16 evaluations ran against initial graphs of
52.25 +- 46.53 and 23.70 +- 10.64 edges, so reading "31.60 -> 23.85 edges" as progress conflated a
better policy with an easier instance set. A change to `m_req` or to the horizon moves
`max_steps`, so every config is being regenerated at once.

Verified faithful: `--benchmark bench_n8_R3` reproduces the sampled seed-0 run exactly
(initial 11.20 +- 5.33, greedy 10.50 / 100% / 50%).

**Tracked, not gitignored** -- unlike everything under `runs/`, `train/` and `environments/`, a
benchmark is a fixture rather than an output, and an untracked one defeats the purpose. Three
20-instance sets cost 32 KB in total.

### episode-logging

Environment metrics are written once per episode, not once per step. A step-resolution scalar costs
a TensorBoard event per step and is then downsampled and averaged for display anyway, so the
resolution was paid for and never seen; the summary is both cheaper and closer to what the plots
were already showing.

`step()` folds each step into `episode_accum` (`new_episode_accum()`: sums and counts only, so an
episode costs the same whether it is 100 or 2000 steps long). `episode_summary()` then emits the
whole episode as one flat, float-valued record - where it ended up (`Final ...`), the best graph it
visited (`Best ...`), and what it looked like throughout (`Mean ...`) - so `write_episode()` can
dump it without knowing what any of it means.

Scalars are written against `writer_counter` (the global env step) rather than the episode index,
so the curves share an x-axis with skrl's loss/reward plots.

### dict-observation

There used to be six `Dict*` observation types differing only in which keys they populated -
`DictNodeFeaturesAndAdj`, `...AndSelection`, `...AndEdgeProposal`, `DictEquivariant...`,
`DictBearing...`, `DictNodeFeaturesAndEdgeFeatures...`. Every model already selects what it needs
by string key and ignores the rest, so the split bought nothing and coupled two unrelated
decisions: *what the environment exposes* and *which network consumes it*.

They are now one type, `"Dict"`, always emitting `node_features`, `coord_features`,
`edge_features`, `adj`, `selection` (plus `proposed_edge` when the action space is `DecideOnEdge`).
Choosing EGNN vs GINE is now a `BACKBONE` constant in the training script, which is where a model
hyperparameter belongs. Contents:

| key | width | contents |
|---|---|---|
| `node_features` | 10 | domain one-hot (5), in/out degree (2), closeness, eigenvector centrality, node betweenness |
| `coord_features` | 3 | pose-normalized positions, see [pose-normalization](#pose-normalization) |
| `edge_features` | 7 | bearing (3), `edge_exists` (1), edge betweenness, reciprocity, common neighbours |
| `adj` | n×n | adjacency |
| `selection` | n | current pointer state |

**The pre-merge names still work.** Each maps to a preset of the builder's flags that reproduces
its old layout exactly - verified element-wise against the pre-merge code for the EGNN variant:
`node_features` (10), raw un-normalized `coord_features`, and 6-channel `edge_features` with
bearings on existing edges only. Reproducing *raw* coordinates matters as much as the shapes: a
checkpoint trained before pose normalization would otherwise be fed differently-scaled inputs and
quietly mis-evaluated.

| legacy name | node set | coords | edge ch. | selection |
|---|---|---|---|---|
| `DictEquivariantNodeFeaturesAndAdjAndSelection` | graph (10) | raw | 6 | yes |
| `DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection` | graph (10) | - | 6 | yes |
| `DictNodeFeaturesAndAdj` | domain+sign-bearing (5+3n) | - | - | no |
| `DictNodeFeaturesAndAdjAndSelection` | domain+sign-bearing (5+3n) | - | - | yes |
| `DictNodeFeaturesAndAdjAndEdgeProposal` | domain+bearing (5+3n) | - | - | no (+`proposed_edge`) |
| `DictBearingNodeFeaturesAndAdjAndSelection` | bearing (3n) | - | - | yes |

`OBS_BACKBONE` records which GNN each legacy name implied, and the training scripts prefer it over
their `BACKBONE` constant - so an old GINE config trains a GINE model even when the constant says
`Equivariant`. An unknown `obs_type` raises listing the known ones.

### graph-features

`graph_features` (env config, default `True`) toggles closeness centrality, eigenvector centrality
and node/edge betweenness. Domain one-hot and in/out degree are always present.

Measured, and the case for turning them off is strong:

| feature | corr. with flex magnitude | share of feature-build cost (n=16) |
|---|---|---|
| **out-degree** (free) | **-0.401** | 0.2% |
| closeness | -0.395 | 40.0% |
| node betweenness | -0.328 | 10.7% |
| eigenvector | -0.309 | 5.4% |

All three expensive centralities carry *less* rigidity-relevant signal than out-degree, which costs
nothing; closeness is also 0.933-correlated with out-degree, so it is close to a rescaling of it.
Edge betweenness predicts "removing this edge drops rank" at r = +0.146 (means 6.07 critical vs 4.59
redundant, heavily overlapping) - near-useless for the pruning decision the policy has to make.

Turning them off takes an n=16 step from **43.4 ms to 9.2 ms (4.7x)**, because closeness and
Brandes betweenness are O(n^3) pure Python and dominate everything else in the environment.

Left as a flag rather than deleted so the removal is an ablation arm (`_lean` configs) rather than
an assumption. Correlational evidence is not proof that a GNN cannot use them nonlinearly.

### rigidity-features

Tier-3 information ([distributed-feasibility](#distributed-feasibility)): quantities derived from
the rigidity matrix, which no local
decision maker could compute. Off by default - this is an **ablation arm**, and the gap between arms
is the result. Three graded flags, so several information levels can be compared:

| flag | node channels | edge channels |
|---|---|---|
| `rigidity_global` | `(rank_K-rank)/rank_K`, `m/m_req`, `is_IBR` | - |
| `rigidity_flex` | `flex_mag` | `add_gain` |
| `rigidity_edge` | - | `c_k / c_max`, `add_rank` |

**`c_k` is nearly useless on its own, and that is why the flags are graded.** Per-edge block rank is
*constant* in every homogeneous configuration - measured 2 for every edge in R^3 and 1 in R^2, at
n=4/8/16 - so it is a dead channel in all three configurations currently trained and evaluated. It
varies only on heterogeneous networks. It is kept as its own flag rather than bundled, so it never
silently pads the feature vector in the runs where it means nothing.

### null-space-features

`rigidity_decomposition(B, rank_K)` returns `(rank, singular values, lam)` from **one** thin SVD.
`step()` used to do three decompositions of the same matrix: `matrix_rank(B)`, one rank per edge
inside `is_MBR`, and an `eigvalsh(B^T B)` for the rigidity eigenvalue. The rank now flows into
`is_MBR` as `rank_brm`, and the margin is `s[rank_K - 1]**2` off the same singular values, which is
the rigidity eigenvalue by definition (`THEORY.md` §4). This is what makes the margin term affordable
rather than a second full decomposition per step.

`lam` is 0 unless the framework is rigid, deliberately: below `rank_K` the `rank_K`-th singular
value is a numerical zero and reporting it as a margin would be meaningless.

The remaining `rigidity_eigenvalue()` calls in `compute_state_score` are the score types that
actually read it. `WeightedNormalized` has `w_eig = 0` and does not reach them, so the shared value
covers every path currently trained.

#### The flex features

Both channels come from `ker(B)` of the **whole** matrix, positions and attitudes together.

- `add_gain[i,j] = ||b_ij Z||_F / ||b_ij||_F` - the fraction of edge `i->j`'s row block that lies
  outside the current row space. It is zero exactly on the pairs that would add no rank, which is
  not an approximation: rank gain **is** `rank(b_ij Z)` (`THEORY.md` §13.1).
- `add_rank[i,j] = rank(b_ij Z) / c_max` - the same thing as an integer. It rides on `rigidity_edge`
  with `c_k`, since both are per-edge rank quantities.
- `flex_mag[i]` - how free node `i` is, from `flex_space(Z, Z_K)`, the non-trivial part of the null
  space. `ker(B_K)` *is* the trivial variation set (Michieletto Theorem 1), so nothing has to be
  enumerated by hand, in any domain or mix.

**This replaced a position-only construction, and the replacement is not cosmetic.** The previous
`flex_align` used a projector built from `B_p = B[:, :3n]` alone and measured *destroyed flex*
rather than *added rank*. Blind to the attitude columns, it was at chance in the oriented domains:
AUC 0.634 in `SE(3)`, 0.678 in `R^2xS^1`, against 1.000 with a clean split for `add_gain` in all
five domains and three mixes. `flex_tensor` / `flex_constraint_power` are kept and tested as the
reference for `THEORY.md` §10, but the environment no longer calls them.

`candidate_gain_reference` is the readable statement of this: it loops over pairs, builds `b_ij` by
calling `extended_bearing_rigidity_matrix` on a network carrying only that edge, and takes
`||b_ij Z||` and `rank(b_ij Z)` directly. `candidate_gain` is the same thing with the three nonzero
blocks expanded by hand into batched products, ~3x faster, and the tests hold it to the reference in
every domain. That pairing is the point: flipping the attitude sign in the fast version makes
`test_fast_candidate_gain_matches_the_readable_one` fail in `R^2xS^1`, `R^3xS^1` and `SE(3)` and
**pass** in `R^2`/`R^3`, since `P_i = 0` there and the term is invisible. That is exactly why the
bug survived the first round of checking.

Three implementation details that each cost a debugging cycle:

1. **`Ē_o` contributes `-y_i`**, so the attitude term enters `b_ij Z` with a minus. With a plus the
   AUC was 0.906-0.947 and looked plausible - good enough to ship, wrong enough to matter.
2. **`ker(B)` is not scale-invariant.** `B`'s position columns carry `1/length` and its attitude
   columns are dimensionless, so a uniform scaling of the formation moves the null space. The length
   unit is fixed to the formation's own RMS radius (`characteristic_length` /
   `nullspace_in_scaled_units`), the same normalisation `coord_features` uses. Related and separate:
   `P_i = v_i v_i^T` is in world coordinates, so `rotate_network` has to rotate
   `agent.rotation_axis` too - it did not, which broke `R^3xS^1` rotation invariance.
3. **The rank threshold has to be measured.** `add_rank` cuts at `add_gain > 1e-6`, which sits in
   the middle of an eight-order-of-magnitude gap (`THEORY.md` §13.3). The first cut was at `1e-18`
   relative, below the noise floor of a Gram matrix in double precision, so the channel flipped by a
   whole rank unit when the geometry was translated or scaled.

Normalisation is **per pair**, by that pair's own `||b_ij||`, not against the spread over pairs. On
a rigid framework every raw gain is at machine zero, and dividing those by their own RMS turns
rounding noise into an O(1) feature.

#### Cost

The per-component breakdown is `THEORY.md` §13.6. Two choices earned most of it: `nullspace` uses
`eigh(B^T B)` rather than an SVD of `B` whose left factor is (3m, 3m) and never used (13.15 -> 2.50
ms at n=16), and `candidate_gain` reads norm and rank off the 3x3 Gram matrix rather than a batched
SVD (1.77 -> 0.59 ms). The rank still comes from the thin SVD, not from `eigh`: squaring halves the
precision of the eigenvalues, and thresholding them disagreed with `matrix_rank` on 840 of 840
cases. Only the eigen*vectors* come from `eigh`.

Profile pinned to one BLAS thread. Unpinned, the same 144x144 `eigh` was timed at anywhere from 0.26
to 16 ms on the same input.

#### Ordering

`step()` used to build the observation *before* the rigidity matrix, so rigidity channels would have
described the previous step's graph. `_get_obs()` now runs after the state score, and
`compute_rigidity_features()` caches everything the observation needs from the `brm` already built.
`initialize()` also has to populate the cache before declaring the observation space, or the space
is missing channels that `reset()` then produces (`ValueError: Output array is the wrong shape` from
the vector-env concatenate).

#### Cost of the arm as a whole

Same fixed graph, single-threaded, ms/step. n=8: `{}` 2.13, `{global}` 2.59, `{global,flex}` 2.66,
`{global,flex,edge}` 2.67 (+25% for the widest arm). n=16: 7.85 -> 9.16 -> 9.58 -> 9.52 (+21%).

Most of the cost is `{global}`, which is `is_IBR` and `m/m_req`; the null-space channels on top of
it are nearly free, because `rigidity_decomposition` and `nullspace` run for the state score
anyway. `rigidity_edge` measures at or below `rigidity_flex` here - `add_rank` comes out of the same
Gram matrix `add_gain` already formed, so the marginal cost is within noise.

### all-pairs-bearings

`get_bearings_explicit()` zeroes `b[i,j]` unless the edge exists, so for every edge the agent might
*add* - the decision it is actually making - the bearing that determines whether that edge adds
rank was invisible. All that reached the policy about a candidate pair was EGNN's internal
`rel_dist` and `common_neighbors`. Bearing rigidity is invariant to uniform scaling and depends on
**directions**, so distance is close to the wrong invariant. This was the first-order cause of the
generalization failure.

The `Dict` observation now carries `get_all_pairs_bearings()` - every ordered pair, edge or not -
plus an explicit binary `edge_exists` channel, so adjacency is stated rather than implied by a
zeroed bearing.

**`include_candidate_bearings` (env config, default `True`)** reverts to bearings on existing edges
only, keeping the observation shape identical. This is not a tuning knob but a modelling one:
candidate-edge bearings are tier-2 information ([distributed-feasibility](#distributed-feasibility))
- an agent does not know its
bearing to a node it has not measured. Whether a distributed version may use them depends on
whether detection is cheaper than maintaining a link, which is an open question. The flag exists so
that a later tier-1-only variant is a config change, not a rewrite.

### pose-normalization

`coord_features` are centred on the centroid and scaled to unit RMS radius. Bearings are already
unit vectors and so scale-invariant, but EGNN's internal `rel_dist = ||x_i - x_j||^2` is not - which
is the only reason changing `random_scenario`'s `pos_limits` from ±100 to ±1 ever mattered. It
should not have. Normalizing per instance also makes n=8 and n=16 comparable when both are drawn
from the same box but at different densities.

Normalization is applied to the *observation* only. The rigidity maths keeps the true poses; rank
is scale-invariant anyway, but the rigidity eigenvalue is not.

### min-eig-caching

When tracking is on, the rigidity eigenvalue is needed for logging anyway, so `step()` computes it
once and hands it to the best-state tracker rather than letting that recompute it. `trace_min_eig`
asks for the same value without a writer attached - `baselines.py` records the rigidity eigenvalue
over time.

---

## rigidity.py

### per-node-dof

`extended_bearing_rigidity_matrix` builds `B = [Dp Ēᵀ S̄ | Da Ē_oᵀ P̄]`, applying each agent's own
DOF projector on the **column** side, rather than the previous `[Dp U Ēᵀ | Da V Ē_oᵀ]` with a
per-**edge** `U_ij`, `V_ij` from `bearing_DOFs`.

Why the difference matters, in one line: `U_ij` multiplies the relative displacement
`(p_j − p_i)`, so it applies the *same* restriction to both endpoints. That is correct exactly when
`S_i = S_j`, i.e. in a homogeneous network, and wrong in every mixed one - Michieletto's own Table
III sets `U = I₃` for a planar agent measuring a spatial one, which re-enables the *planar* agent's
z DOF. Measured consequence: `rank_K = 36` against `Σ dim D_i = 36` on the `mixed` scenario (zero
trivial motions, impossible), `rank_K = 14 > 13 = Σ dim D_i` on 5×R²+1×R³, and IBR verdicts
differing from the corrected matrix on 2-40% of random graphs depending on the mix. Full derivation
in `THEORY.md` §12.

Three things worth knowing about the implementation:

- **Homogeneous output is bit-identical** (max abs difference 0.0 over 60 graphs in each of the five
  domains), so no existing homogeneous result moved. `bearing_DOFs` is kept unused precisely so
  `test_matches_michieletto_table_I_on_homogeneous_networks` can assert that. It reproduces Table I
  only at `v = e₃` though: it stores the `R^3xS^1` rotational entry as `e₃vᵀ` where the paper has
  `[0_{3x2} v]`. Every scenario uses `e₃`, so nothing measured depends on it, but a Table I
  comparison off the default axis has to build the entry itself, as
  `docs/verify_dof_restriction.py` does.
- **It is also faster**, because the two `(3m, 3m)` dense `U`/`V` allocations are gone: 1.3× at
  n=8, 2.2× at n=16, **6.1× at n=32** on the complete graph. That is a real contribution to the
  large-`n` scaling study, which was blocked on step cost.
- **`P_i` is a projector `v vᵀ`, not a row placement.** The old `V_ij = [0; 0; rax]` (as rows)
  agrees with Michieletto's `[0_{3x2} v]` (a column) only at `v = e₃`, which is the only axis ever
  used - so nothing measured depended on it, but the parameter is exposed and the row form is wrong
  for any other axis.

The check is not a regression comparison but the definition itself:
`test_matrix_is_the_numerical_jacobian_of_the_bearings` central-differences the bearing function and
asserts `B δ` matches to 1e-6 relative, over all five domains and eight heterogeneous mixes, with a
non-default rotation axis. Removing the DOF restriction fails 35 tests.

### max-edge-rank

`max_edge_rank()` returns `max_k rank(B_K[3k:3k+3, :])` over the fully-connected graph: the most
rank a single edge could possibly contribute at these poses.

It is **exact** - a max over plain rank computations, making no claim about what is jointly
achievable. That is why the state score normalizes with this rather than with an edge count: it
turns "one edge" into a comparable unit across domains (`d-1` in homogeneous R^d, so 2 in R^3 and 1
in R^2 - exactly the factor that made un-normalized `Weighted` non-transferable) without asserting
a minimum.

### required-edge-count

`required_edge_count()` is the fewest edges that could possibly make *these poses* rigid - an
episode constant, unlike `is_MBR`'s `m_req`, which is derived from whatever edges the graph
currently has.

Homogeneous R^d uses the closed form. Otherwise: every edge contributes at most
`rank(B[3k:3k+3, :])` to `rank(B)`, so taking the highest-rank blocks of the fully-connected graph
first and accumulating until `rank_K` is reached gives a bound by rank subadditivity. The two agree
exactly on homogeneous R^2 and R^3, where every block has rank `d-1` and the bound reduces to
`ceil(rank_K / (d-1))`.

**This is a lower bound, not a ground truth.** Rank subadditivity says no edge set smaller than
this can be rigid; it does not say one of this size exists. Heterogeneous networks are exactly
where the greedy sum over the highest-rank blocks may not be jointly realizable.

Brute force finds it tight on every case small enough to check exhaustively:

| n | mixes tested | tight |
|---|---|---|
| 4 | 8 domain mixes × 3 seeds | 24/24 |
| 5 | 6 domain mixes | 6/6 |

covering all five domains and both homogeneous and mixed networks. That is evidence, not proof.

Use it for reporting and for the MBR metric. **Do not put it in the reward** - see
[weighted-normalized](#weighted-normalized).

Cost is `n(n-1)` small rank computations, so call it once per episode and cache it;
`Environment.compute_episode_constants()` does.

---

## train_ppo.py

### constructive-baseline

`baselines.py --methods constructive`. From the empty graph, keep any edge that raises `rank(B)`,
stop at `rank_K`; best of `--restarts` independent random orders. This is the classical algorithm
for the problem and the one the learned policy has to beat. `tools/constructive_greedy.py` is the
standalone version, used for difficulty sweeps that do not need an env config.

**It is the only method that does not start from the initial graph.** It throws those edges away and
builds from nothing, because it is a construction and not an edit. The report says so, in the
`summary.txt` legend and on the figure card, since "all methods see the same networks" is otherwise
the reader's default assumption and it would be wrong here.

**Why it is the right opponent, and why `greedy` is not enough on its own.** `greedy` hill-climbs phi
from the initial random graph, so it mostly prunes an over-dense graph downward. `constructive`
never sees the initial edges and has to find an independent set from scratch, which is the harder
and more standard framing. Reporting only the first invites "why not compare against the obvious
constructive algorithm".

Measured at n=8/`R^3` (`m_req` = 10), 4 instances: 11.50 edges at 1 restart, 11.00 at 5, 10.75 at 20.
Every instance is order-sensitive, never a matroid, as `c_max = 2` predicts. In the `c_max = 1`
domains (`R^2`, `R^2xS^1`) the independent sets *are* a matroid and greedy is optimal by
construction, so a "beats greedy" claim is only meaningful in the spatial domains.

**It gets its own RNG.** `np.random` is the stream `reset()` draws instances from, so shuffling the
candidate order there changes which networks *every other method* is scored on - enabling the method
silently moved the `initial` row from 15.33 to 13.00 edges. `run_constructive` takes an
`np.random.default_rng(seed)` of its own. `greedy` uses no randomness and `random` uses the action
space's own seeded RNG, so the instance sequence stays independent of `--methods`, which is what
makes two runs comparable.

Selection among restarts is by phi rather than by edge count. Among rigid graphs the two agree,
since `WeightedNormalized` is monotone decreasing in `m` at fixed rank, and phi keeps the choice
consistent with the column the table reports. Per-step statistics are computed only for the winning
restart, by replaying its additions from empty.

### ppo-rollout-size

One constant feeds both the memory size and `cfg.rollouts`, and they must stay equal. skrl's
`PPO.update()` runs `compute_gae()` over the **whole** memory ring and then samples
`batch_size=len(memory)`, so a memory larger than one rollout trains on stale off-policy data -
7/8 of it at `memory_size=8192, rollouts=1024` - with `last_values` bootstrapped at the ring's wrap
point instead of the trajectory end. The stale samples fall outside the ratio clip band and
contribute no gradient. This is what broke
`bigPPOSelectEquivariant3e-4lrNormalizedPositions`.

### ppo-discount-factor

`discount_factor` must stay `< 1`. The environment's reward is potential-based (`phi(s') - phi(s)`),
so at γ=1 the return telescopes to `phi(s_T) - phi(s_0)` and the advantage becomes
`E[phi(s_T)|s'] - E[phi(s_T)|s]`, which is ≈0 under a near-uniform policy because the walk over
edge sets mixes and forgets `s`. There is then no gradient to bootstrap from - that is what froze
the earlier run's entropy at ~1.9 nats of a ~2.0 ceiling.

At γ<1, Abel summation turns the same reward into

```
-phi(s_0) + (1 - gamma) * sum_t gamma^(t-1) phi(s_t)
```

i.e. maximize the discounted average of phi along the trajectory: converge fast and stay converged.
DQN uses 0.99 and solves n=8/R^3; PPO now matches it.

γ=1 used to be set so the logged return matched the optimized objective. Read `Episode/ Return` for
that instead - it is undiscounted by construction.

---

## policy/

### model-registry

`policy/registry.py` maps `(role, backbone, action_type)` to a model class, replacing the if/elif
chains that used to select models in both training scripts (they lost ~180 and ~110 lines). Roles
are skrl's own model-dict keys - `policy`, `value`, `q_network` - so `build_models()` output goes
straight to the agent.

Constructors differ: every model takes `n`, `node_feat_dim`, `gnn_hidden_dim`, `head_hidden_dim`,
`observation_space`, `action_space`, `device`; the EGNN and GINE ones also take `edge_feat_dim`;
the `SelectNodesSequentially` ones also take `allow_skip`. `instantiate()` filters a superset of
kwargs against `inspect.signature(cls.__init__)`, so callers pass everything and each class picks
up what it declares. `agent_loader` uses the same function, so there is one implementation of that
rule.

A `(role, backbone, None)` entry is the fallback for a role and backbone, which is how the critics
cover every action space that has no selection stage without enumerating them.

## policy/gnn_backbone.py

### action-masking

Invalid actions are masked in the model, not the environment, by writing `MASK_VALUE` into their
logits / Q-values. `MASK_VALUE` is `-inf`, and it **must stay scale-free**.

It used to be `-1e9`, which is a sentinel only while every real logit stays above it. In a collapsed
run the logits reached `-1e23`, at which point `-1e9` became the *largest* value in the row and
argmax started deliberately selecting masked actions - the policy locked onto invalid no-ops and the
symptom looked like an exploration failure rather than a masking bug. `-inf` cannot invert, and it
makes `softmax` give the masked action exactly zero probability rather than merely a small one.

The cost is that `softmax` of an all-`-inf` row is NaN, which is reachable for an add-only action
space once the graph is complete. `unmask_if_all_masked` catches that row and falls back to a
uniform distribution over everything.

DQN Q-networks mask in `random_act` as well, or epsilon-greedy exploration would propose exactly the
actions the greedy path forbids.

Regression coverage: `tests/test_masking_and_skip.py` drives real logits to `-1e12` and asserts a
masked action still cannot win an argmax.

### egnn-dense-all-pairs

`GNNBackboneEquivariant.forward` accepts `adj_mat` but does not forward it to `EGNN`. In
`egnn_pytorch`, `adj_mat` is read *only* inside `if use_nearest:`, which needs
`num_nearest_neighbors > 0` or `only_sparse_neighbors=True`; the backbone constructs
`EGNN(dim, m_dim, edge_dim)` with both at their defaults. Passing it was therefore a silent no-op -
verified, `max abs diff 0.0` between an all-zeros and an all-ones adjacency.

Dense all-pairs message passing is the right choice here (the whole task is reasoning about edges
you do not have), so the fix is to make it deliberate rather than to sparsify: the graph reaches
the model through `edge_features`, where `edge_exists` now states adjacency explicitly. The
argument stays in the signature so archived model sources that pass it keep working.

`EGNN` also accepts a `mask` argument the backbone never passes, which is what variable-`n`
batching will need.

### gine-dense-all-pairs

GINE used to build `edge_index` from `adj.nonzero()` and gather `edge_features[i][src, dst]`, i.e.
message passing over **existing edges only** - with a comment reading "we get all possible edges'
features from the observation but we only need existing edges'". Once the observation carried
all-pairs bearings that became exactly backwards: the candidate-edge geometry was computed and then
discarded, so [all-pairs-bearings](#all-pairs-bearings) reached the EGNN arm and not the GINE one.

`GNNBackboneGINE.forward(nodes, edges)` now takes the dense `(B, N, N, E)` edge tensor and builds a
complete digraph internally (no self loops, row-major so it lines up with the dense tensor
flattened the same way, cached per `(batch_size, n, device)`). Both backbones now do dense
all-pairs message passing and differ only in *how* they mix, which is what makes a backbone
comparison meaningful. `edge_exists` is what distinguishes a real edge from a candidate, in both.

This also removed the per-sample Python loop that had been duplicated across all seven GINE models.

Verified: perturbing a **non-edge**'s features changes the GINE output (0.86 on a random init), and
with a single layer `d(h_0)/d(edge_attr[0,1])` is nonzero while `d(h_0)/d(edge_attr[1,0])` is
exactly zero - so the outgoing-edge direction below is preserved.

### egnn-input-embedder

`EGNN` preserves the feature dimension: `dim` in equals `dim` out. `GNNBackboneEquivariant` was
constructed with `dim=node_feat_dim`, so the node representation it handed the action head was as
wide as the raw observation - **11 on `mixed`** (5 domain + 2 degree + 3 rigidity_global + 1
flex_mag) - while `GNNBackboneGINE` output `gnn_hidden_dim = 128`. Confirmed in a checkpoint:
`gnn.conv1.edge_mlp.0.weight` had shape `(62, 31) = (2*m_dim, 2*11+1+8)`. Every EGNN-vs-GINE
comparison run before the embedder was added was an 11-dimensional model against a 128-dimensional
one, not a comparison of message-passing schemes.

`self.embed` is `Linear(node_feat_dim, hidden) -> LeakyReLU -> Linear(hidden, hidden)` applied
before the stack, and the stack now runs at `dim=hidden_dim`. Two properties it must not break, both
measured:

- **Equivariance survives.** The EGNN's equivariance is with respect to `coors`; `feats` are
  invariant scalars throughout, so embedding them is free. Embedding `coors` would break it.
  Rotating `coors` alone moves the output by 3.0e-8 at `init_eps=1e-2` and 9.5e-6 at trained-scale
  `1e-1` - float32 accumulation through three 128-wide layers, ~3e-7 relative to `mean|h| = 7.8`.
- **No `n` dependence.** The embedder is applied per node, so it cannot introduce the scaling that
  [aggregation-and-scale](#aggregation-and-scale) is about. `mean|h|` is 7.80 / 4.48 / 5.46 at
  n = 8 / 16 / 32, non-monotone, i.e. sampling noise rather than drift.

**Equal width is not equal capacity, and the roadmap's "~18k parameters" was wrong.** It counted the
embedder only. Raising `dim` from 11 to 128 also grows the EGNN stack itself, because every layer's
`edge_mlp` and `node_mlp` widen with `dim`:

| | width | params | x GINE |
|---|---|---|---|
| EGNN before (`dim=11`, `m_dim=128`) | 11 | 40,407 | 0.5x |
| **EGNN after (`dim=m_dim=128`)** | **128** | **940,956** | **10.9x** |
| GINE (`hidden=128`) | 128 | 86,499 | 1.0x |
| embedder alone | - | 18,048 | - |

So the two controls cannot both be satisfied: matched width puts the EGNN at 10.9x the parameters,
and matched parameters (~86k) puts it at `dim ~= 32`, a quarter of GINE's width. **Width is the one
implemented**, because width was the diagnosed defect. A matched-parameter arm would need `m_dim` separated
from `hidden_dim` in the constructor; it is not currently a knob. Whichever is reported, say which
control it is - an EGNN that wins at 10.9x the parameters has not beaten GINE at message passing.

### egnn-init-eps

`egnn_pytorch` applies `nn.init.normal_(weight, std=init_eps)` to *every* Linear in an `EGNN` layer,
with `init_eps=1e-3` by default - a guard against deep stacks going NaN. Stacked three deep and set
against the node residual (`node_out = node_mlp(...) + feats`), the edge-feature path starts at
about **1e-10** of the output. The dependence is structural, not absent - all gradient entries are
nonzero - but the model begins effectively blind to every edge feature, bearings included, and has
to grow those weights before geometry can matter at all.

It does escape: the trained `bigDQN8SelectEquivariant3e-4lrNormalizedPositions` checkpoint has
`edge_mlp` and `node_mlp` weight std at 1.2e-1 … 4.3e-1, up from 1e-3, and reached 98.2% minimally
rigid. So this is a slow start, not a ceiling.

It is worth knowing for two reasons. It is a plausible contributor to the policy latching onto
node-level statistics (degree, centralities, which arrive via `feats` and the residual) instead of
geometry. And it is an asymmetry against GINE, whose Linears use the PyTorch default, ~5e-2 … 2e-1
- roughly where the EGNN *finishes*. So the two backbones do not start on equal footing.

`GNNBackboneEquivariant` exposes `init_eps`, and **the default is now 1e-2**, raised from
`egnn_pytorch`'s 1e-3. 1e-3 is what the one working run used, so this does invalidate strict
comparison against it - but that run is a single-configuration result that does not generalize, and
a start where geometry is 1e-10 of the output is a direct contributor to the shortcut learning
documented in [aggregation-and-scale](#aggregation-and-scale). Checkpoints are unaffected: this is a
constructor default, not a shape, and manifest-bearing runs replay their archived backbone source.

The same blindness is a **measurement trap**. At 1e-3 an untrained EGNN reports invariance it does
not have, and reports sum- and mean-pooling as identical to three decimals. Any test of what an EGNN
is sensitive to must run at trained-scale weights (`std ~= 0.15`), which is what
`tests/test_scale_invariance.py::_trained_scale` and the invariance tests do.

### aggregation-and-scale

**Nothing the policy sees may scale with `n`.** A policy trained at `n=8` and evaluated at `n=16`
was no better than random, and the first-order reason was not the task - it was that the inputs and
the internal activations were both quantitatively different at the two sizes, so the trained
network was being evaluated far outside the range it ever saw. Four separate places did this.

**1. Message aggregation.** Both backbones do dense all-pairs message passing
([egnn-dense-all-pairs](#egnn-dense-all-pairs), [gine-dense-all-pairs](#gine-dense-all-pairs)), so
every node aggregates `n-1` messages. With a sum/add aggregator the pooled message is `O(n)` by
construction. GINE now uses `aggr="mean"` and `EGNN` `m_pool_method="mean"`. Measured at
trained-scale weights, activations relative to `n=8`:

| | n=8 | n=16 | n=32 | n=64 |
|---|---|---|---|---|
| GINE `add` | 1.00x | 8.27x | 71.99x | 580.11x |
| GINE `mean` | 1.00x | 0.98x | 0.99x | **1.00x** |

**2. The EGNN coordinate update, which `m_pool_method` does not cover.** `m_pool_method` governs
only the feature message `m_i`. The coordinate update is a separate and *hardcoded* sum over `j`:

```python
coors_out = einsum('b i j, b i j c -> b i c', coor_weights, rel_coors) + coors
```

That result re-enters the next layer through `rel_dist = ||x_i - x_j||^2`, which is part of
`edge_input` - so the growth compounds across the three layers and squares each time. Mean pooling
alone therefore fixes almost nothing on the EGNN arm:

| EGNN config | n=8 | n=16 | n=32 | n=64 |
|---|---|---|---|---|
| `sum`, `update_coors=True` | 1.00x | 1.99e3 | 2.43e5 | 4.23e8 |
| `mean`, `update_coors=True` | 1.00x | 1.11e3 | 6.89e4 | 6.09e7 |
| `mean`, `coor_weights_clamp_value=1.0` | 1.00x | 20.12x | 295.70x | 6292.88x |
| `mean`, `norm_coors=True` | 1.00x | 1.19x | 1.10x | 1.14x |
| `sum`, `update_coors=False` | 1.00x | 5.42x | 30.85x | 307.22x |
| **`mean`, `update_coors=False`** | 1.00x | 0.92x | 0.76x | **0.88x** |

Both switches are load-bearing - the `sum, update_coors=False` row is the control. `update_coors`
is off by default now, over the `norm_coors` alternative, because it is also the semantically right
choice here: `coors` are pose-normalized ground-truth positions, EGNN reads them only through
`rel_dist`, and `GNNBackboneEquivariant.forward` **discards the returned `coors`**. A layer that
moves them just hands later layers inter-agent distances the network does not have. It is cheaper
too - `update_coors=False` drops `coors_mlp` entirely (4 parameter tensors).

This also answers the standing `TODO: should we recalculate bearings (edges) from c_out?` - no. The
geometry is fixed within an episode; only the edge set moves.

**3. Unnormalized node and edge channels.** `degree` and `common_nbrs` were raw counts, and the flex
channels carried a `sqrt(n)` that assumed a fixed flex dimension. Degree is now divided by `m_req/n`
(the mean degree a *minimally rigid* graph would have) rather than by `n-1`, which over-corrected -
`n-1` is the mean degree of the complete graph, and rigid graphs are sparse, so dividing by it drove
the channel to zero as `n` grew (0.307 -> 0.170). `common_nbrs` likewise. Flex is normalized by its
own total power, so it is comparable across domains with different `rank_K` and different deficits.

Legacy observation presets pin `normalize_counts=False`, so pre-merge checkpoints still see the
scales they were trained on ([dict-observation](#dict-observation)).

**4. The initial-graph sampler.** `sample_initial_edge_count` drew with a spread that grew like
`n^4`, so at `n=16` the sd was 72.7 against a mean of 22 and instances actually started at ~41.6
edges - a systematically harder and differently-distributed problem than at `n=8`. It is now
`sd = max(0.5 * m_req, 1.0)`, making `m0/m_req` centred on 1 at every size: measured 1.04 / 1.00 /
1.00 at n=8/16/32 in R^3 and 0.97 at n=8 in SE(3).

**What this does not fix.** Scale invariance is necessary, not sufficient. The ablation that
motivated this work - perturbing one channel at a time and reading the change in phi - showed
degree at **+21.00** against bearings **+0.25**, `flex_mag` **-0.25** and `flex_align` **+0.00**.
The policy was making its decisions almost entirely from a node-degree statistic and ignoring both
the geometry and the rigidity features. Fixing the scaling removes the excuse for that shortcut; it
does not by itself force the model to use the geometry. That is what the retraining and the mixed
`n`/domain runs are meant to establish.

Regression coverage: `tests/test_scale_invariance.py`.

### backbone-num-layers

The layer count used to be hardcoded at 3 and was 2 in older runs. It is a constructor argument now
so a checkpoint trained at a different depth can still be loaded (see
`agent_loader.rebuild_backbone`). Submodules keep the names `conv1..convN`, so state dicts of
3-layer models are unaffected.

### gine-edge-direction

GIN(E) message passing adds the *inward* edge features to the neighbour's features. Here it is the
outward edge that carries the meaning - "I measure this bearing to that node" - so `edge_index` is
flipped before message passing.

---

## report.py

### palette

Colours come from the data-viz reference palette, used unchanged and in its documented order.
`greedy` / `learned` / `random` take categorical slots 1-3, which are certified for the all-pairs
case (overlapping lines) in both modes. `initial` and `optimal` are *reference points* rather than
methods under comparison, so they take neutral inks and dashed strokes instead of a categorical
hue - which also keeps the categorical count at 3.

### panel-titles

Static output for a thesis, so light mode only and no hover layer. Identity is never colour alone:
every series carries a direct label at its line end (or a value label on its bar), and every figure
ships the notes card that says what it is showing.

Each panel is titled with *what the quantity is*, with the reading direction on a second line -
"edges used" told a reader nothing about why they should care.

## scope

### distributed-feasibility

The long-term motivation is a *distributed* protocol for maintaining rigid formations in swarms.
The centralized formulation here is a deliberate first step, and it should not foreclose that.
Assessment: feasible, with one genuine limitation.

**What blocks a naive distributed version.** Rigidity is a global algebraic property - `rank(B)`
cannot be computed or certified locally, so no local rule can *verify* rigidity, and minimality is
global by definition. The current policy is also centralized in all three ways that matter: both
backbones are dense all-pairs, the action is a global index over all node pairs, and the centrality
features are global.

**What makes it tractable anyway.** Centralized training with decentralized execution fits almost
exactly: train the critic on the full graph, restrict the *actor* to `K` rounds of message passing
over a communication graph with only locally computable features, and the actor is literally a
`K`-hop local rule. Factored per-node actions replace the global index: each agent emits a
distribution over its own candidate out-neighbours, parameter-shared across nodes, which is the same
property "one policy, any n" already requires. And in homogeneous `R^d`, minimally bearing-rigid
graphs admit a Henneberg-style vertex-addition construction attaching each new agent with `d`
edges - exactly what `MBR_required_Rd` counts - so a distributed *constructive* protocol reaches
minimality by construction without any agent computing a global rank.

**The limitation:** a distributed policy achieves at best rigidity plus *local* minimality.
Certified global minimality requires global information. That is a result to state, not a failure.

#### Feature availability tiers

Which "global" features are global in which way decides what a distributed version could reuse.

- **Tier 1, locally available unconditionally.** Own pose and domain, own in/out degree, the
  bearings the agent is currently measuring, whatever one-hop neighbours communicate.
- **Tier 2, available only under a sensing-radius assumption.** Bearings to agents the agent is
  *not* currently measuring - the geometry of candidate edges, which
  [all-pairs-bearings](#all-pairs-bearings) supplies. **This is not free information**: an agent
  does not know `p_hat_ij` before measuring it, so no purely local decision maker can evaluate
  "would adding `i -> j` help?" without first obtaining it. Whether it is admissible depends on an
  unmade modelling choice. If detection is cheap and only *maintenance* is expensive - omnidirectional
  vision within radius `R`, against an edge as a persistent tracked link - then all-pairs bearings
  within `R` are a legitimate local observation and the current observation carries over unchanged.
  If every measurement costs what an edge costs, they are not, and a distributed protocol needs an
  explicit exploration phase or a policy reasoning from communicated *positions*. This is the single
  most important open modelling question for the distributed direction, and the centralized work
  does not depend on resolving it.
- **Tier 3, not available at all.** `rank(B)`, rank deficit, per-edge block rank `c_k`, `is_IBR`,
  the null-space features and the graph centralities.

The tier-2 / tier-3 split is what the observation arms price, in two separable steps: *informed minus
geometry-only* is the cost of losing tier 3, and *geometry-only minus a tier-1-only variant* is the
cost of losing tier 2 under the pessimistic measurement model. The second arm does not exist yet;
`include_candidate_bearings` already makes it a config flag rather than a rewrite.

To keep this open: GNN depth stays an explicit constructor argument (`num_layers`), per-node and
per-edge action heads stay first-class rather than flattened to a global index, and candidate-bearing
inclusion stays switchable.
