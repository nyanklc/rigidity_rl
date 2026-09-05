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

### softest-mode-features

`rigidity_stiffness` (env config, default `False`) adds two channels: `add_stiffness[i,j] = ||b_ij v||`
over all ordered pairs and `node_slack[i]` = the position and attitude norms of `v_i`, where `v` is
the mode at the rigidity eigenvalue. Derivation and every measurement are in `THEORY.md` §16. The
implementation facts:

- **No new algebra.** `candidate_gain(network, Z, L)` already returns `||b_ij Z||` for an arbitrary
  `(6n, k)` matrix, so `add_stiffness` is that call with `k = 1`. `nullspace_and_softest` returns the
  kernel and `v` from **one** `eigh(B^T B)`: `v` is the column immediately after the kernel, since
  the eigenvalues come back ascending. Cost at n=10 is 9.45 -> 9.90 ms per step.
- **Both are gated on `is_IBR` and written as zeros otherwise.** `v` is meaningless on a flexible
  framework, and zero is the honest encoding: stiffness does not exist there. This makes the two
  feature sets exactly complementary, which is the point -- `add_independence`/`add_rank`/`node_freedom` carry
  information only while flexible, `add_stiffness`/`node_slack` only once rigid.
- **Normalised per channel by its own mean**, like `node_freedom`: which pair is soft, not how soft in
  absolute terms, which is what transfers across n and domain.

### removal-channels

`rigidity_removal` (env config, default `False`) adds `remove_rank` and `remove_stiffness`: what the
rank and the stiffness would lose if an existing edge were deleted. Both exact, derivation in
`THEORY.md` §17. Implementation facts:

- **No rebuilding.** `extended_bearing_rigidity_matrix` writes one 3-row block per directed edge in
  `np.nonzero(edges)` order, so an existing edge's block is the slice `brmat[3k:3k+3]`. A test pins
  that layout, since both channels rest on it and nothing else did.
- **No extra decomposition.** `nullspace_and_softest` now returns `(w, V)` alongside the kernel and
  the softest mode, so `(B^T B)^+` for the leverage comes from the eigh already performed.
- **Two skips carry the cost.** The `eigvalsh` of the downdate is not run when the rank drops (the
  answer is 1 by definition) nor when the framework is flexible (there is no stiffness to lose).
  Pinned to one BLAS thread, 3.46 -> 5.76 ms per step at n=10 with ~35 edges, so about +66%, growing
  as `m * (6n)^3`. Unpinned these numbers are BLAS contention rather than the algorithm, the same
  trap [null-space-features](#null-space-features) records.
- **Own flag rather than folded into `rigidity_edge`.** They are exact one-step oracles like
  `add_rank`, but keeping them separable is what lets the ablation price add-side against
  remove-side information, and it keeps every existing config byte-identical.

### stiffness-in-phi

`stiffness_kappa` (env config, default `0.0` = off) adds the stiffness to
`WeightedNormalized` as `kappa * one_edge * q(lam)`, with `q` a sigmoid of `log10(lam/stiffness_ref)`.
Derivation, the two obstacles to using `lam` raw, and every measurement are in `THEORY.md` §15.
What matters here is the plumbing:

- **`lam` costs nothing.** `step()` already gets it from the single `rigidity_decomposition` it
  performs for the rank, and `begin_episode()` from its own. `compute_state_score` takes it as an
  optional `lam=None`, so the older call signature still works and `kappa = 0` is byte-identical.
- **`stiffness_ref` is an episode constant**, built in `compute_episode_constants` from
  `rigidity.reference_stiffness`. All of the cost is here: reset goes 2.7 -> 46.8 ms at n=8/`R^3`
  (`stiffness_ref_samples=3`), while **per-step cost is unchanged**. Over a 50-step episode that is
  +55% wall clock at n=8/`R^3`, +27% on `mixed`, and it is not currently a blocker anywhere. If it
  becomes one, the addition oracle (`candidate_gain`) can pick rank-raising edges from one
  nullspace instead of `O(n^2)` rank computations per round - deliberately not done, because it
  would make the reference construction differ from the `constructive` baseline's.
- **`self.stiffness_rng` is private and must stay private.** `stiffness_ref`'s construction order draws from
  it, never from `np.random` - that is the stream instances are drawn from, and using it would move
  the networks every method is scored on. This is the exact regression recorded for `constructive`
  once, so `test_enabling_stiffness_does_not_move_the_instance_stream` pins it.
- **One construction, shared.** `rigidity.greedy_rigid_construction` is the loop;
  `baselines._construct_once` is now a thin wrapper on it. A reference construction that drifted
  from the baseline would silently change what `stiffness_ref` means. Verified byte-identical to the
  previous inline loop over 4 seeds x 3 configurations, and `bench_n8_R3` reproduces its
  `initial`/`greedy`/`constructive` rows exactly.
- **`baselines.score_network` now takes rank *and* `lam` from one `rigidity_decomposition`** instead
  of `matrix_rank` via `is_IBR_explicit`. Roughly cost-neutral, since `matrix_rank` already performs
  an SVD, and necessary: without it every `greedy` candidate would be scored with the stiffness term at
  zero, which at `kappa > 0` is not the configured phi. The side effect is that `greedy` becomes
  stiffness-aware for free.

### spectral-functional

`WeightedNormalizedSpectral` is `WeightedNormalized` with the spectral bonus read off
whichever functional `spectral_functional` names -- `eigenvalue`, `trace` or `logdet`.

It exists for a negative result, not for an arm that is expected to win.
`THEORY.md` §18.4-18.5 measures that the trace is a monotone restatement of λ
(rank correlation 0.99) and that log-det, though decorrelated from λ, predicts the
*measured* error worse than either. A config key makes that reproducible without
a training run; a second `state_score_type` would have implied there was something
to choose between.

Three things it has to get right:

- **`eigenvalue` must be bit-identical to `WeightedNormalized`**, or the comparison
  is against a moving baseline. `reference_spectral(functional="eigenvalue")` returns
  `median(log10 lam)` over the same greedy constructions with the same RNG that
  `reference_stiffness` uses, so `g - g_ref` reproduces `log10(lam/stiffness_ref)`
  exactly. A test asserts equality, not closeness.
- **The sigmoid width is per functional.** §15.2's 0.75 decades came from λ's own
  p10-p90 spread; `logdet` is measured in nats and spans 17-25 of them, so copying
  0.75 would put the whole achievable band inside a hair of the sigmoid.
  `SPECTRAL_SIGMOID_WIDTH` scales each width by the functional's measured spread
  (`tools/spectral_criteria.py` prints them).
- **`trace` and `logdet` cost an extra SVD per step**, since they need
  `estimation_error_of` on the length-normalised `B` where λ comes free from the
  decomposition `step()` already performs. `eigenvalue` pays nothing.

### state-quality

`rigidity_quality` adds one tiled node channel: how good the current graph is on the axis
nothing else covers. `rigidity_global` gives rank deficit, `m/m_req` and `is_IBR`, and every
pair channel answers a *difference* question -- what adding or removing an edge would do.
Once the graph is rigid, none of them says whether its conditioning is good, so the policy
cannot distinguish "already good, stop" from "bad, keep going".

`Environment.state_quality()` reuses the state score's own sigmoid: 0 while flexible, ~0.5
for a graph as good as a typical greedy construction on the same poses, rising towards 1.

Two choices worth stating:

- **Not raw lambda.** It has no absolute scale and tracks the pose range, so a policy would
  learn a threshold that means nothing at another `n` or domain. The sigmoid against a
  per-episode reference is dimensionless and bounded, which is what makes the channel
  transferable at all.
- **Its own flag, not folded into `rigidity_global`.** The ablation has to be able to price
  it separately, and turning it on at the same time as a reward change would make that
  ablation unreadable for the reason THEORY.md 16 gives.

The reference is an episode constant, so the channel costs nothing per step beyond the
decomposition `step()` already performs -- except under `spectral_functional != eigenvalue`,
which needs one more SVD.

### formation-figures

`uncertainty`, `softest_mode` and `sensitivity` draw the network itself rather than a
statistic of it. Four decisions, each of which was wrong first:

- **Real 3-D axes.** A bearing formation is a spatial object, and a projection onto its
  two widest directions hides the axis it is worst determined along -- which for these
  formations is often exactly the interesting one.
- **Marker shape is the domain.** On a five-domain mixture a plain scatter throws away
  the thing that makes the formation heterogeneous. `DOMAIN_MARKER` fixes the shape per
  domain and the card carries the key, so identity is never colour-alone.
- **One exaggeration factor for the ellipsoids, set from the MEDIAN panel.** A true-scale
  1-sigma shell is a few percent of the formation and invisible. Setting the factor from
  the worst panel made every other panel invisible instead; the factor is shared so the
  panels stay comparable, and the true percentage is printed above each one. The softest
  mode is the opposite case -- the eigenvector is normalised, so lengths do not compare
  between panels anyway and each is scaled to its own largest arrow.
- **The panels place themselves.** `tight_layout` does not know about the title band a
  3-D panel needs above it and silently undoes an explicit `set_position`, so these
  figures compute their own cell rects from the header/card band and pass `tight=False`.

Panel subtitles say `<m> edges, final` or `<m> edges, best visited`, because a rollout
row reports the best state it saw and an edit-space row reports where it stopped. Which
one is being drawn is not guessable from the picture.

### measurement-sensitivity

Perturbing one bearing at a time, and perturbing one agent's bearings at a time, are
not two Monte-Carlo sweeps. Noise on measurement `k` reaches the estimate through
`B^+`, so its contribution to the squared shape error is the squared norm of the
matching columns of `B^+`, and those contributions sum to `tr((B^T B)^+)` exactly.
One pseudo-inverse gives every share, and the shares read as percentages of the
total because they are.

`measurement_sensitivity` returns both groupings: per edge, and per agent over the
edges it measures. Validated against actually perturbing a single bearing.

The per-agent version is the *input* side and is not the uncertainty ellipse, which
is the output side -- one says whose sensing the error comes from, the other says
whose position is badly determined. Drawn together they are the two halves of the
same question.

### decision-analysis

At each step `edit_landscape` scores every legal single-edge change, and
`decision_record` ranks the one the policy took among them.

**It is ranked twice, and the second ranking is the point.** The policy is trained on
phi, which rewards rank and charges for edges; it is never asked about shape error. A
high phi percentile beside a chance-level error percentile therefore says the
*objective* is what leaves error behind, not the policy -- which is a different
conclusion from "the policy chooses badly", and the two are indistinguishable without
both rankings.

Two details that would otherwise mislead:

- **Percentiles are midrank.** Several edits are often exactly as good, especially on
  a minimal graph where every redundant removal scores the same. Counting only
  strictly-worse alternatives scored a genuinely optimal choice at 67%. `phi_best` is
  the separate flag for "was it actually one of the best".
- **The error ranking is absent while the graph is flexible**, where every edit leaves
  the error infinite. Those steps carry a phi rank and no error rank, rather than a
  zero.

Gated by `MAX_DECISION_ANALYSIS_N`: it costs `n(n-1)` score evaluations per step.

### noise-sweep

`outputs.py --noise-sweep` measures what each method's graph actually costs: perturb
every bearing by that many degrees, recover the formation, compare against the truth
(`estimation.py`). The table prints measured against predicted, and `plots/*/noise.*`
draws both.

It runs on **the graph each row reports**, which for a rollout is the best state
visited rather than whatever the episode stopped on -- so `result()` carries the edge
set it is about. Rows that came out flexible carry no sweep at all: their error is
infinite, not large, and a number there would be read as a measurement.

The point of showing prediction beside measurement is that the gap is itself the
result. They track while the error is small; measurement falling below prediction is
the signature of leaving the regime where the analytic metric means anything
(`THEORY.md` §18.3).

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
edges) into `benchmarks/<name>.npz`; `outputs.py --benchmark <name>` evaluates on them instead of
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
is the result. Six graded flags, so several information levels can be compared:

| flag | node channels | edge channels |
|---|---|---|
| `rigidity_global` | `(rank_K-rank)/rank_K`, `m/m_req`, `is_IBR` | - |
| `rigidity_quality` | `quality` | - |
| `rigidity_flex` | `node_freedom` | `add_independence` |
| `rigidity_edge` | - | `c_k / c_max` (`pair_max_rank`), `add_rank` |
| `rigidity_stiffness` | `node_slack` (position, attitude) | `add_stiffness` |
| `rigidity_removal` | - | `remove_rank`, `remove_stiffness` |

The two global channels (`rigidity_global`, `rigidity_quality`) are tiled identically across nodes,
which is why `ablation.py` cannot shuffle them and needs `--mode noise` there. What each flag costs
is [observation-cost](#observation-cost).

**`c_k` is nearly useless on its own, and that is why the flags are graded.** Per-edge block rank is
*constant* in every homogeneous configuration - measured 2 for every edge in R^3 and 1 in R^2, at
n=4/8/16 - so it is a dead channel in all three configurations currently trained and evaluated. It
varies only on heterogeneous networks. It is kept as its own flag rather than bundled, so it never
silently pads the feature vector in the runs where it means nothing.

### null-space-features

`rigidity_decomposition(B, rank_K)` returns `(rank, singular values, lam)` from **one** thin SVD.
`step()` used to do three decompositions of the same matrix: `matrix_rank(B)`, one rank per edge
inside `is_MBR`, and an `eigvalsh(B^T B)` for the rigidity eigenvalue. The rank now flows into
`is_MBR` as `rank_brm`, and stiffness is `s[rank_K - 1]**2` off the same singular values, which is
the rigidity eigenvalue by definition (`THEORY.md` §4). This is what makes the stiffness term affordable
rather than a second full decomposition per step.

`lam` is 0 unless the framework is rigid, deliberately: below `rank_K` the `rank_K`-th singular
value is a numerical zero and reporting it as a stiffness would be meaningless.

The remaining `rigidity_eigenvalue()` calls in `compute_state_score` are the score types that
actually read it. `WeightedNormalized` has `w_eig = 0` and does not reach them, so the shared value
covers every path currently trained.

#### The flex features

Both channels come from `ker(B)` of the **whole** matrix, positions and attitudes together.

- `add_independence[i,j] = ||b_ij Z||_F / ||b_ij||_F` - the fraction of edge `i->j`'s row block that lies
  outside the current row space. It is zero exactly on the pairs that would add no rank, which is
  not an approximation: rank gain **is** `rank(b_ij Z)` (`THEORY.md` §13.1).
- `add_rank[i,j] = rank(b_ij Z) / c_max` - the same thing as an integer. It rides on `rigidity_edge`
  with `c_k`, since both are per-edge rank quantities.
- `node_freedom[i]` - how free node `i` is, from `flex_space(Z, Z_K)`, the non-trivial part of the null
  space. `ker(B_K)` *is* the trivial variation set (Michieletto Theorem 1), so nothing has to be
  enumerated by hand, in any domain or mix.

**This replaced a position-only construction, and the replacement is not cosmetic.** The previous
`flex_align` used a projector built from `B_p = B[:, :3n]` alone and measured *destroyed flex*
rather than *added rank*. Blind to the attitude columns, it was at chance in the oriented domains:
AUC 0.634 in `SE(3)`, 0.678 in `R^2xS^1`, against 1.000 with a clean split for `add_independence` in all
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
3. **The rank threshold has to be measured.** `add_rank` cuts at `add_independence > 1e-6`, which sits in
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
Gram matrix `add_independence` already formed, so the marginal cost is within noise.

### observation-cost

What every observation flag costs, where the cost is paid, and which parts of the environment set
the ceiling on `n`. Reproduced by `tools/flag_cost.py`, `tools/rigidity_cost.py` and
`tools/policy_cost.py`. All of it single-threaded BLAS on `R^3`; absolute milliseconds are machine
and load dependent, the ratios are not.

#### The six flags are not independent arms

`compute_rigidity_features()` computes the null space and `candidate_gain` **before** it branches on
which flag is set, so the first flag turned on pays for both and the rest are comparatively cheap.
`rigidity_global` is three tiled scalars that `step()` already has in hand, and it still costs
1.5x a step, because it drags an `eigh(6n)` and a full all-pairs candidate scan behind it.

The consequence for an ablation is that a one-flag-at-a-time sweep pays the shared cost once per
arm and gives six baselines that are each ~1.5x the true zero-information floor. Leave-one-out from
the full arm is the cheaper design and the fairer comparison, since every arm then differs from the
reference only by its own marginal work.

ms per `env.step()`, and the same as a multiple of the all-off baseline:

| flags | n=8 | n=16 | x base n=8 | x base n=16 |
|---|---|---|---|---|
| baseline (all off) | 1.80 | 4.56 | 1.00 | 1.00 |
| `graph_features` | 2.64 | 8.85 | 1.46 | 1.94 |
| `rigidity_global` | 2.67 | 7.04 | 1.48 | 1.54 |
| `rigidity_quality` | 2.67 | 7.22 | 1.48 | 1.58 |
| `rigidity_flex` | 2.88 | 9.73 | 1.59 | 2.13 |
| `rigidity_edge` | 2.04 | 8.50 | 1.13 | 1.86 |
| `rigidity_stiffness` | 2.63 | 6.59 | 1.46 | 1.45 |
| `rigidity_removal` | 3.81 | 14.22 | 2.11 | 3.12 |
| all six | 5.01 | 23.83 | 2.78 | 5.23 |
| all six + `graph_features` | 4.73 | 26.09 | 2.62 | 5.72 |

#### Two flags are paid somewhere other than the step

ms per `env.reset()`. Every flag not listed sits between 4.8 and 6.9 at n=8 and 17 and 47 at n=16:

| flags | n=8 | n=16 |
|---|---|---|
| baseline (all off) | 4.24 | 21.15 |
| `rigidity_quality` | 61.53 | **771.81** |
| all six | 90.65 | 1076.63 |

`rigidity_quality` makes `compute_episode_constants` call `reference_spectral`, which runs
`stiffness_ref_samples` (default 3) full greedy rigid constructions, each `O(n^2)` rank evaluations
of a growing `B`. At n=16 that is 0.77 s per episode, which over a `4*m_req + 10` horizon amortizes
to roughly 8 ms per step and more than doubles the step budget on its own. It is the same machinery
`stiffness_kappa > 0` already pays for, so with the stiffness reward on the channel is free and
standalone it is the most expensive flag by a wide margin.

`rigidity_edge` is the other one: `pair_max_rank` is an episode constant, but computing it forces
`edge_block_ranks(brmat_K)`, `n(n-1)` rank computations on `(3, 6n)` blocks in a Python loop (2.0 /
2.8 / 14.4 ms at n = 8 / 16 / 32). Per step it is the cheapest informative pair channel, since
`add_rank` is the second return value of the `cand` already computed.

#### `candidate_gain` does not do an SVD per candidate edge

Worth stating because the name of the reference implementation suggests otherwise.
`candidate_gain` forms `b_ij Z` for all pairs at once with einsums over `(n, n, 3, k)` and reads
both norm and rank off the **3x3 Gram matrix** per pair, one batched `eigvalsh`, no Python loop and
no `b_ij` ever built. Cost is `O(n^2 k)` with `k = 6n - rank`, so `O(n^3)`.
`candidate_gain_reference` is the per-pair-SVD form and exists only as the test oracle.

#### Per-primitive, on a near-minimal graph

ms, from `tools/rigidity_cost.py`:

| primitive | n=8 (m=12) | n=16 (m=26) | n=32 (m=52) |
|---|---|---|---|
| build `B` `(3m, 6n)` | 0.73 | 1.31 | 2.13 |
| `rigidity_decomposition` `svd(B)` | 0.10 | 0.32 | 1.60 |
| `is_MBR`, `m` x `rank(3, 6n)` | 0.44 | 0.91 | 1.23 |
| `nullspace` `eigh(6n)` | 0.20 | 0.76 | 2.45 |
| `candidate_gain` | 0.54 | 1.43 | 4.50 |
| `flex_space` + magnitude | 0.16 | 0.62 | 1.11 |
| `removal_costs` | 0.58 | 3.83 | 22.15 |
| `[reset]` build `B_K` | 2.78 | 7.94 | 131.48 |
| `[reset]` `edge_block_ranks(B_K)` | 1.98 | 2.81 | 14.39 |
| `closeness` (`graph_features`) | 0.30 | 0.96 | 15.19 |
| `_brandes_betweenness` (`graph_features`) | 0.11 | 0.29 | 1.33 |
| `get_all_pairs_bearings` | 0.62 | 0.74 | 5.91 |

The first four rows and `get_all_pairs_bearings` are paid on every step with every flag off. Two
incidental things the table exposes. `get_node_betweenness_features` and
`get_edge_betweenness_features` each call `_brandes_betweenness()` separately, so it runs twice per
step under `graph_features`. And `get_all_pairs_bearings` is a Python double loop costing more at
n=32 than the whole of `candidate_gain`, while `get_all_pairs_bearings_world` directly above it is
already vectorized.

#### `removal_costs` scales with density, not with `n`

`removal_costs` does one `eigvalsh(G - b^T b)`, a full `6n x 6n` eigendecomposition, per **redundant**
edge, so its cost is `O(#redundant * n^3)` and a near-minimal test graph hides it. Measured against
`m / m_req`:

| n | `m/m_req` | m | redundant | ms |
|---|---|---|---|---|
| 8 | 1.0 | 11 | 0 | 0.12 |
| 8 | 2.0 | 22 | 22 | 1.18 |
| 16 | 1.0 | 24 | 2 | 1.09 |
| 16 | 2.0 | 48 | 48 | 12.55 |
| 24 | 2.0 | 80 | 78 | 91.33 |
| 32 | 1.0 | 51 | 29 | 28.51 |
| 32 | 2.0 | 102 | 100 | 137.49 |

This is the worst scaling in the repository, and it peaks exactly mid-episode, when the policy has
surplus edges to prune and the channel is at its most useful. A rank-3 downdate does not need a full
eigendecomposition, since only the eigenvalue at index `6n - rank_K` moves, so it is fixable, but
that is an algorithmic change rather than a tweak.

#### The ceiling on `n` is not a flag

`extended_bearing_rigidity_matrix` allocates `Dp` and `Da` as dense `(3m, 3m)`. On the **complete**
graph `m = n^2`, so `reset()` makes a `Theta(n^4)` allocation with every flag off:

| n | `m = n^2` | `Dp` shape | `Dp + Da` MB | ms |
|---|---|---|---|---|
| 8 | 56 | (168, 168) | 0 | 1.6 |
| 16 | 240 | (720, 720) | 8 | 21.6 |
| 32 | 992 | (2976, 2976) | 142 | 128.1 |
| 64 | 4032 | (12096, 12096) | **2341** | **3665.5** |

2.3 GB and 3.7 s per episode at n=64. `Dp` is block diagonal, one 3x3 block per edge, so applying it
blockwise instead of materializing it is worth more than every flag decision combined if n=64 is
wanted. This is the allocation behind `CLAUDE.md`'s "0.1-6 s at n=64".

#### Policy side

The flags widen the per-pair edge tensor, 6 to 12 channels for all six, and that tensor is both what
the backbones spend their time on and what the replay buffer stores. Observation width at n=8:
baseline 7 node / 6 edge / 536 floats, all six 14 / 12 / 976. At `MEM_SIZE = 10000` with `obs` and
`next_obs` in float32, that is 43 MB against 78 MB at n=8, ~290 MB at n=16 and ~1 GB at n=32.

Both backbones do dense all-pairs message passing, so a forward is `O(B n^2 (H^2 + E H))` per layer.
ms at `hidden=128`, 3 layers, one thread:

| model | batch | n=8 | n=16 | n=32 |
|---|---|---|---|---|
| GINE | 1 | 1.0 | 1.1 | 2.1 |
| GINE | 256 | 37.6 | 102.4 | 540.5 |
| EGNN | 1 | 2.0 | 5.8 | 31.1 |
| EGNN | 256 | 531.8 | 1897.5 | 8240.2 |

EGNN is 14-15x GINE at every batch and `n`. That is a constant factor, not a worse exponent: at
batch 256 both are cleanly quadratic in `n`, and the batch-1 rows only look like a scaling
difference because GINE is overhead-dominated there. At n=32 and the training batch of 256 a single
EGNN forward is 8.2 s single-threaded, an order of magnitude above the environment step.

Doubling the edge width costs 1.8% of GINE's parameters (85,155 to 86,721) and 2.6% of EGNN's
(932,304 to 956,100), so the flags change what the network reads rather than how much capacity it
has, and an arm comparison is not confounded by capacity.

#### What scales with what

| quantity | scaling | paid | flag |
|---|---|---|---|
| build `B_K` | `Theta(n^4)` memory | reset | none, always |
| `reference_spectral` / `reference_stiffness` | `O(samples * n^2)` rank evaluations | reset | `rigidity_quality`, `stiffness_kappa > 0` |
| `edge_block_ranks(B_K)` | `n^2` small SVDs, Python loop | reset | `rigidity_edge`, or any mixed-domain scenario |
| `removal_costs` | `O(#redundant * n^3)` | step | `rigidity_removal` |
| `nullspace`, `candidate_gain` | `O(n^3)` | step | any rigidity flag |
| `flex_space` | `O(n^3)` | step | `rigidity_flex` |
| build `B`, `svd(B)`, `is_MBR` | `O(n^3)` at `m ~ m_req` | step | none, always |
| `closeness`, `_brandes_betweenness` | `O(n^3)` / `O(nm)` pure Python | step | `graph_features` |
| `get_all_pairs_bearings` | `O(n^2)` pure Python | step | none, always |
| backbone forward | `O(B n^2 H^2)` | step and update | edge width doubles with flags |
| replay buffer | `O(MEM_SIZE * n^2 * E)` | memory | edge width doubles with flags |

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
asks for the same value without a writer attached - `outputs.py` records the rigidity eigenvalue
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

### repair-edge-count

`repair_edge_count()` is the same question asked of a graph that is already there:
**how few edges could make THIS graph rigid again**, after an agent left or a link
failed. `required_edge_count` cannot answer it -- it starts from the empty graph and
so ignores everything the survivors still have.

The argument is the same subadditivity, moved: the deficit is `rank_K - rank(B)`, and
no `k` edges can close more of it than the `k` largest marginals available now,
because rank is monotone submodular ([THEORY.md §14.2](THEORY.md)) and adding edges
can only shrink a marginal.

Two choices worth stating:

- **The marginals are the exact per-pair gains `rank(b_ij Z)` (§13), not the complete
  graph's block ranks.** An edge whose own block has rank 2 may contribute only 1 to
  this particular graph, and using the block rank would understate the count -- which
  would make the bound unsound, not merely loose.
- **Only absent pairs are counted.** An existing edge already lies in the row space,
  so its marginal is 0 and it would be filtered out anyway; excluding them explicitly
  is what makes the function say "how many *more*".

It returns 0 on a rigid graph, and from the empty graph it reproduces
`required_edge_count` exactly in every domain and on the `mixed` mixture -- the two
are the same construction seen from different starting points, so that agreement is a
real check rather than a coincidence.

**Prior work, and what is new.** Karimian and Tron (CDC 2017) solve the homogeneous
2-D case exactly: decompose into rigid components and count
`2n - 3 - sum_X (2|X| - 3)`, attained by a greedy algorithm they prove optimal. In
`R^2`, `c_max = 1`, every useful edge closes exactly one unit of deficit, and this
bound collapses to `rank_K - rank(B)` -- their formula, since the component ranks add.
Their stated open problems are the 3-D extension and a criterion for *which* edges to
add; this is the former, and phi with the shape-error metric is aimed at the latter.

Same status as `required_edge_count`: **a lower bound, not a ground truth**, and it
stays out of the reward for the same reason.

---

## estimation.py

### estimation-monte-carlo

`rigidity.estimation_error` predicts how far a recovered formation lands from the true
one. `estimation.py` measures it, so the prediction stops being an assertion. It runs
in evaluation only -- never in `step()`.

Four decisions, each of which was arrived at by getting it wrong first:

- **The noise is full 2-DOF tangent in every domain.** A bearing is a unit vector, so
  noise lives in its tangent plane, and `sigma` is then an angle in radians. It is
  tempting to restrict a planar agent's bearing noise to its plane; that is wrong. The
  DOF restriction is on the agent's *motion*, not on its camera, and the component the
  restriction makes unobservable is exactly the one `B` already carries as a zero row.
- **The solver's Jacobian must be `B`, and `B` must differentiate THIS module's bearing
  map.** `true_bearings` therefore builds `R_i^T p_hat_ij` the way
  `extended_bearing_rigidity_matrix` does, rather than calling `Agent.get_bearing`,
  which returns the *world* vector for `R^2`/`R^3` and agrees only because those agents
  happen to keep an identity orientation. A test checks it against central differences
  in all five domains.
- **Steps are restricted through `node_dof_projectors`.** `lstsq`'s minimum-norm
  solution already lands there, since `B` zeroes those columns, but doing it explicitly
  makes "an agent never leaves its domain" a property of the code rather than of the
  solver's tolerance.
- **The gauge quotient is linear, and that makes it a small-error metric.** The
  unobservable directions are `ker(B_K)`, and projecting the error off them is exact
  only to first order -- which is the same linearisation the Cramer-Rao prediction
  makes, so the two are comparable by construction rather than by luck.

**Do not replace the linear quotient with an exact one.** Centring and rescaling both
formations looks safer and is not: 3-D centring shifts `z` for a planar agent, whose
`z` is not a free coordinate, so it deletes real error. Measured on a five-domain mix,
that alone moves agreement with the bound from 0.93 to 0.82. `ker(B_K)` is the only
description of the gauge that is right in every domain.

**One thing the linear quotient genuinely cannot do** is score a gross failure. A
scaling by zero is a gauge direction like any other, so a formation collapsed to a
single point projected to *no error at all* -- found by implementing the anchored
linear solve below and watching it collapse. `max_scale_ratio` catches it and returns
`inf`. It cannot fire in the Monte-Carlo experiment, where the solver starts from the
truth under small noise; it is there for callers handing in an estimate from elsewhere,
which the robustness harness will.

**Why Gauss-Newton rather than the anchored linear solve.** In `R^d` each bearing says
`(p_j - p_i)` is parallel to `z_ij`, i.e. `(I - z z^T)(p_j - p_i) = 0`, which is linear
in the positions -- fix a couple of agents and solve. Measured at n=8, `sigma = 1e-3`,
it works: 1.86e-03 against Gauss-Newton's 2.03e-03, slightly better because fixing two
agents pins 6 numbers where only 4 are free and the extra 2 are true values handed to
the solver. It does **not** carry over to the oriented domains: there `z_ij = R_i^T
p_hat_ij` with `R_i` unknown, so the equation is bilinear in `(R, p)`, and running it
anyway gives 3.3e-01 against 1.7e-03 in `SE(3)`. The number of coordinates that may be
anchored is also the gauge dimension, which is domain dependent (4 in `R^3`, 7 in
`SE(3)`) -- anchor too few and the formation collapses, too many and the estimate is
being fed the answer. Gauss-Newton needs none of that bookkeeping, covers all five
domains in one path, and is the maximum-likelihood estimator the Cramer-Rao bound
describes, which is why the agreement lands at 1.00 rather than merely near it.

---

## train_ppo.py

### greedy-vs-policy

Whether the learned policy is worth having rests on what greedy is actually doing, and greedy turns
out to be computing something the observation already holds. Reproduced by
`tools/greedy_landscape.py`.

#### What greedy costs

`run_greedy` scores all `n(n-1)` single-edge toggles per improvement step, and each score is a
`score_network` call that **rebuilds `B` and takes a fresh SVD**. One improvement step is therefore
`O(n^2)` evaluations of an `O(n^3)` quantity, and a run takes `O(n)` improvement steps:

| n | phi evals / step | ms / step | improvement steps / episode | phi evals / episode |
|---|---|---|---|---|
| 6 | 30 | 6.0 | 3.6 +- 2.1 | 108 |
| 8 | 56 | 14.4 | 4.8 +- 1.2 | 269 |
| 12 | 132 | 53.5 | 7.6 +- 2.7 | 1,003 |
| 16 | 240 | 152.2 | 11.4 +- 3.3 | 2,736 |

`O(n^5)` per improvement step, `O(n^6)` per episode. Measured growth is nearer `n^3.5`, because at
these sizes the fixed per-call overhead across `n^2` calls still outweighs the matrix work; it
steepens as `n` grows.

#### The kappa = 0 case, which is not the thesis case

With `stiffness_kappa = 0`, phi is affine in rank, so a toggle moves it by exactly the rank it adds
or costs - which is what `add_rank` and `remove_rank` already are:

```
dphi(add i->j)    = ( 100*rk_ij - 25*c_max) / rank_K
dphi(remove i->j) = (-100*rl_ij + 25*c_max) / rank_K
```

Verified against greedy's own landscape to machine precision (max `4.9e-15` over 24 states per
configuration, `R^3` / `SE(3)` / `R^3xS^1` at n=6 and 8), with the top move agreeing 24/24
everywhere. So at kappa = 0 the observation contains greedy's decision outright, a policy reading
it can at best learn the argmax, and a channel-based greedy would be `O(n^3)` per improvement step
rather than `O(n^5)`, measured 9x to 38x faster over n=6 to 16 and widening. Recorded because it
explains the ablation result that `add_rank` dominates every other channel, not because kappa = 0
is a configuration the thesis argues about.

Two artifacts of that regime worth knowing if a kappa = 0 run is ever read: any rank-adding edge
improves phi, since `100*rk > 25*c_max` for `rk >= 1` in every domain, so greedy is barely
selective about *which* rank it adds; and in `R^3` adding a rank-1 edge and removing a redundant
edge are an exact tie at `50/rank_K`, broken by row-major enumeration order, so an implementation
detail decides between growing and pruning.

#### What changes at kappa > 0

The stiffness term needs `lambda` of the **updated** matrix, and no channel holds that for an
addition. `remove_stiffness` is exact, so deletions stay covered; additions have only
`add_stiffness`, a ranking prior. The rank-only closed form stops being greedy's landscape, and how
far it stops depends on the domain:

| domain | n | kappa | rank-only picks greedy's move | mean phi lost | worst | one edge |
|---|---|---|---|---|---|---|
| `R^3` | 6 | 0.9 | 14/24 | 0.091 | 0.298 | 3.57 |
| `R^3` | 6 | 2.0 | 14/24 | 0.203 | 0.662 | 3.57 |
| `R^3` | 8 | 0.9 | 9/24 | 0.158 | 0.971 | 2.50 |
| `R^3` | 8 | 2.0 | 9/24 | 0.352 | 2.157 | 2.50 |
| `SE(3)` | 6 | 0.9 | 0/24 | 0.465 | 1.329 | 1.72 |
| `SE(3)` | 6 | 2.0 | 0/24 | 1.033 | 2.954 | 1.72 |
| `SE(3)` | 8 | 0.9 | 0/24 | 0.177 | 0.738 | 1.22 |
| `SE(3)` | 8 | 2.0 | 0/24 | 0.394 | 1.640 | 1.22 |
| `R^3xS^1` | 6 | 0.9 | 4/24 | 0.371 | 1.237 | 2.63 |
| `R^3xS^1` | 8 | 0.9 | 2/24 | 0.083 | 0.607 | 1.85 |

`one edge` is `25*c_max/rank_K`, phi's own price of an edge, so the losses read as fractions of one.
Following the rank channels costs between 3% and 60% of an edge on average and up to 1.7 edges at
worst. **The oriented domains are where it breaks hardest**: `SE(3)` never agrees, at either kappa.

Two things fall out of the table. Agreement is identical at kappa 0.9 and 2.0 in every row while
the phi lost scales by about 2.2, which is `2.0/0.9`: kappa multiplies the whole stiffness bonus, so
within a state it rescales rather than reorders, and greedy's top move was the same at both in all
144 states here. And a hand-scaled combination of `add_stiffness` and `remove_stiffness` recovers
greedy's move 15-24 of 24 at kappa = 0.9 but 0-2 of 24 at kappa = 2.0. **That collapse is not
evidence the information is missing**, it is evidence the hand scaling is wrong at the larger kappa
- `add_stiffness` is normalised per channel and carries no phi units, so something has to supply
them. Fitting that scaling from data is what a policy is for.

#### So is the policy worth it

This analysis does not answer that, and it is worth being exact about which half it does settle.

Settled: **at kappa > 0 the policy is not a trivially-learnable oracle reader.** At kappa = 0 it is
one, and the honest reading there is that a correctly implemented greedy is both as good and
cheaper, since it stops after ~`m_req` edits while the policy runs a `4*m_req + 10` horizon. That
argument does not carry over, because the closed form it rests on does not.

Also settled, the cost side, per episode:

| | per improvement step or action | per episode |
|---|---|---|
| greedy, as implemented | `O(n^5)` | `O(n^6)` |
| greedy, from the rank channels (kappa = 0 only) | `O(n^3)` | `O(n^4)` |
| policy + observation, rigidity flags on | `O(n^3)` | `O(n^4)` |
| policy + observation, rigidity flags off | `O(n^2)` | `O(n^3)` |

The policy is an order cheaper than greedy only in the arm where it has no rigidity channels, which
is also the arm where `ablation.py` says the geometric channels currently cost it nothing. With the
channels on it is the same order as a well-implemented greedy, so cost is not the argument.

Not settled, and this is the measurement that decides it: whether greedy's local optima at kappa > 0
are worse than what the policy reaches. The structural reason to expect a gap is already recorded -
the stiffness is **not** submodular ([THEORY.md](THEORY.md) 14, 59% of tested triples violating
diminishing returns), so greedy carries no guarantee there, unlike the rank objective where it sits
0-5% above `m_req`. Expecting a gap is not measuring one. `outputs.py` at `stiffness_kappa > 0`
with `greedy` and `learned` on the same instances is the experiment, read on `shape err` and
stiffness rather than on edge count, and no such comparison is recorded anywhere in this repository
yet.

One thing that makes it a fair fight and is easy to miss: `score_network` takes `rank` and `lam`
from the same decomposition, so `greedy` is stiffness-aware at kappa > 0 for free. It is hill
climbing the same phi the policy is trained on, not a rank-only opponent handicapped by its
objective.

### constructive-baseline

`outputs.py --methods constructive`. From the empty graph, keep any edge that raises `rank(B)`,
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

### spectral-baseline

`outputs.py --methods spectral`. Greedy rescores all `n(n-1)` toggles per improvement step. This
computes the same landscape in closed form from the rigidity algebra, ranks the toggles by it, and
rescores only the `--spectral-shortlist` best (default 5) before applying one.

**Why the closed form is exact at `stiffness_kappa = 0`.** phi is affine in rank there, so a toggle
moves it by exactly the rank it adds or costs, and `candidate_gain` / `removal_costs` return that
for every pair at once. `greedy-vs-policy` above already measured the two landscapes agreeing to
`4.9e-15`; this turns that measurement into a method. Measured, `spectral` reproduces `greedy` edge
for edge on 60/60 instances across all five domains at n = 6 and 8, at **3.7x to 6.9x** fewer
rigidity computations. Over n = 6, 8, 12 the counts grow as `n^0.81` against greedy's `n^2.95`, so
the gap widens. Both tables come from

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. uv run tools/cost_scaling.py --episodes 6
```

**Why it verifies rather than trusting the ranking.** At `stiffness_kappa > 0` the removal side is
still exact (`remove_stiffness` is a rank-3 downdate) but the addition side is not: an addition's
true lambda is not available from the current matrix, so the term is `add_stiffness`, a ranking
prior normalised by its own maximum. Normalised that way it always shows *some* candidate as
positive, so a hill climb that trusted it never reached a local optimum and ran to its step cap -
23.5 edges against greedy's 14.3 on the first version of this. Rescoring the shortlist with the
real phi fixes that at 5 phi evaluations per edit against greedy's `n(n-1)`, and costs nothing at
`kappa = 0` where the ranking is already exact.

**Two things that are load bearing.** The tie break must be stable in row-major order, because
greedy's `delta > best_delta` keeps the *first* of a tie and in the `c_max = 2` domains adding a
rank-1 edge and dropping a redundant one are an exact tie - `np.argsort(...)[::-1]` reverses that
order and cost 1 instance in 6 in `R^3` and `R^3xS^1`. And `phi_landscape` **raises** on a state
score outside `WeightedNormalized` / `WeightedNormalizedSpectral`: it hardcodes their `w_rank = 100`
and `w_edge = 25`, and returning those deltas for `Weighted` would be silently wrong rather than
loudly wrong.

`removal_costs` grew a `need_stiffness` flag for this. Its rank half is `m` 3x3 `eigvalsh`; its
stiffness half is one `eigvalsh(6n)` **per redundant edge**, the worst scaling in the repository
(`#observation-cost`). At `kappa = 0` nothing reads it, and computing it anyway would throw away
the cost advantage the method exists to demonstrate.

`tools/greedy_landscape.py` imports `phi_landscape` rather than keeping its own copy. That script
exists to check the closed form against brute force, so a second implementation of the formula
there would make the check meaningless.

### anneal-baseline

`outputs.py --methods anneal`. Simulated annealing over single-edge toggles on the configured
phi: accept an improvement, accept a worsening with probability `exp(dphi/T)`, cool geometrically.
Scored on best-state-visited, since the last state of an annealer is not its best.

**Why it is here.** Greedy's guarantee comes from submodularity, and the stiffness is not
submodular - 59% of tested triples violate diminishing returns ([THEORY.md](THEORY.md) 14.4). Above
`stiffness_kappa = 0` greedy is hill climbing an objective it has no claim on, and a method that
assumes nothing is the right thing to hold it to.

**What it has actually shown so far is nothing conclusive, in either direction.** Two six-instance
runs disagree: at `kappa = 2` on `random8` it beat greedy (14.00 edges and 100% minimal against
14.33 and 67%), at `kappa = 10` on `mixed` it lost (17.50 and 67% against 17.17 and 100%). Six
instances and one seed cannot separate those, and the controlled experiment - both arms on a frozen
50-instance benchmark, three seeds, read on `shape err` rather than on edges - has not been run.
Treat the rows as a working implementation rather than a result.

**Temperature is denominated in phi's own units.** `one_edge = w_edge*c_max/rank_K` is what phi
charges for an edge, so `T0 = one_edge` accepts a one-edge worsening at `e^-1` and `T1 = T0/100`
essentially refuses it. Stated that way the schedule transfers across `n` and domain the way phi
does; stated in absolute phi it would not.

**The budget is greedy's, measured rather than guessed.** `--anneal-budget` counts phi evaluations,
and the default is exactly what `greedy` spent on *that instance*, read off the `cost.py` counters
in the same episode loop. Falls back to `4n(n-1)` when greedy is not in `--methods`. "Budget
matched" is a claim that has to be shown, so the realised count is in the cost block. Own RNG, for
the reason in `#constructive-baseline`.

### degree-baseline

`outputs.py --methods degree`. Add the absent pair minimising `outdeg(i) + indeg(j)` until the
network is rigid, then repeatedly drop the highest-degree edge whose removal keeps it rigid.

It exists to price the tier-1 row of `#distributed-feasibility`: everything it reads is locally
available except the rigidity test itself, which is global and unavoidable. No marginal ranks, no
spectrum, no phi. `tests/test_outputs_reference.py` pins that by asserting it never calls
`candidate_gain`, `removal_costs`, `nullspace_and_softest` or `score_network` - a test that fails
the moment someone "improves" it with information a distributed agent could not have.

Measured on `mixed` (n=10, 6 instances, `stiffness_kappa = 10`): 17.67 edges and 50% minimal at
**79** rigidity computations, against greedy's 17.17 / 100% at **2252**. Within half an edge for 3%
of the compute. Two caveats before that means anything: six instances is far too few, and its
*conditioning* is bad - shape error `2.4e+02` against greedy's `1.2e+01`, which is what an algorithm
that never looks at geometry should be expected to produce. Reproduce with the command under
`#cost-counters`.

### cost-counters

`cost.py`. The compute comparison had to be measured rather than argued, and the counting had to
be non-invasive enough that `rigidity.py` still reads as the mathematics it implements.

**Decorators, not wrapped call sites.** `@counted` on ~18 primitives, one line each, and no call
site anywhere changes. The alternative considered and rejected was monkey-patching `np.linalg`
inside the meter: it needs no source change at all, but it is action at a distance in the one
module where a reader most needs to see what is happening, and it would catch every unrelated
`np.linalg` call in the process. `functools.wraps` is not optional - `manifest.py` archives sources
and `agent_loader.build_class_from_source` replays them by name.

**Counting calls to named primitives, not decompositions.** A raw decomposition count would weigh
`nullspace` (one `eigh` of `6n x 6n`) the same as `edge_block_ranks` (`m` ranks of a `3 x 6n`
slice), and `candidate_gain` does `n(n-1)` 3x3 eigendecompositions in *one* batched call. Naming
the primitive and publishing `cost.OPERATION` beside the number says what one call is; the wall
time is what weighs them. This is also the shape `#observation-cost`'s tables are already in, so
the two read together.

**`LEAVES` is checked, not asserted.** The headline total sums only primitives that call no other
counted one, and membership is not obvious from a function's name. `Network.eigenvalues` builds `B`
itself, and `estimation.solve_shape` builds one per Gauss-Newton iteration, so counting either as a
leaf would count those builds twice. `tests/test_cost.py` calls every leaf and asserts it tallies
only itself, which is the only version of this list that survives someone adding a primitive.

**Measurement work is not the method's cost.** Per-step tracing costs an eigendecomposition per
step and `stats_now` adds a `rigidity_eigenvalue` and a `shape_error_now` to every recorded point.
Without a split, the same method measured with `--no-plots` and with plots differs by more than the
methods differ from each other. `@cost.measurement` puts `stats_now`, `edit_landscape`,
`measure_noise` and `repair_spread` in a separate bucket, so a cost number means the same thing in
both modes and the instrumentation's own cost is visible as its own row.

**The meter wraps the method, never the restore.** `restore()` deep-copies the instance and calls
`env.reset()`, which recomputes `rank_K`, `c_max`, `m_req` and the spectral references. That is
shared setup every method gets for free, so it sits outside the `Meter`.

**What it says, and it is not flattering to the policy.** greedy 2252 rigidity computations,
`learned` **1424**. Same order, not a different one - the policy's observation does the algebra its
reward never asks it to use. `#greedy-vs-policy` predicted exactly this ("cost is not the argument")
from the scaling; this is the measurement. `spectral` is 190. From the cost block of

```bash
uv run outputs.py <mixed-config> --episodes 6 --methods all \
    --model estimation_k10_dqn_gine --no-plots
```

Six instances and one seed, so the ordering is the finding and the digits are not.

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

### pair-head

The `AddRemoveEdgeDiscreteNoSelfLoops` heads score a pair from
`[h_i, h_j, e_ij]`, not `[h_i, h_j, adj_ij]`. Pair scalars would otherwise reach the head only
through three rounds of mean aggregation over `n-1` pairs, while being ~6.6x concentrated on a few
pairs. The measured precedent: a held-out linear probe for "does adding i->j raise the rank" scores
1.000 from `e_ij` alone and 0.955 from `[h_i, h_j]`, so the backbone does carry most of it but
degrades it, and a continuous channel like `add_stiffness` has more to lose than a near-binary one.

`edge_exists` is inside `e_ij`, so the separate `adj_ij` scalar was redundant. The `adj` tensor
itself stays, because what it is really for is the action mask. Head width follows the
`edge_feat_dim` constructor argument, so widening the observation widens the head.

`SelectNodesSequentially` is deliberately unchanged: its pair is (selected, candidate) rather than
(i, j), and it is not the action space in use.

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
node_freedom) - while `GNNBackboneGINE` output `gnn_hidden_dim = 128`. Confirmed in a checkpoint:
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
degree at **+21.00** against bearings **+0.25**, `node_freedom` **-0.25** and `flex_align` **+0.00**.
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

Three baselines took the count to seven, which the all-pairs certification does not cover. Ordered
as `METHOD_ORDER` draws them - `random, degree, greedy, spectral, anneal, constructive, learned` -
the set clears the **adjacent**-pair gates, which is the right pairlist for lines, bars and boxes:
worst CVD Delta E 9.1, worst normal-vision Delta E 22.9, both against the light surface. That 9.1
is the reference palette's own worst adjacent pair, so it is a documented-acceptable level rather
than a new concession. Aqua, yellow and magenta sit below 3:1 contrast, so the relief rule applies,
which the figures already satisfy: every series carries a direct label at its line end or its tick,
and the table view always ships.

The one real trade was `spectral`. Giving it magenta maximises adjacent CVD separation at 15.2 but
puts the headline new baseline at 2.11 contrast; green drops separation to 9.1 and buys 4.82.
Separation was already clear of its gate, so green won. Red was available and deliberately not
used: a red series reads as a verdict, and a figure is not where the conclusion goes.

**Validate, do not eyeball.** The skill ships `scripts/validate_palette.js` and this machine has no
node, so the six checks were ported to Python and checked against the reference palette's published
numbers (9.1 / 19.6) before being trusted. Reproduce that before changing a hue rather than
reasoning about Delta E.

### sections

`outputs.py` produces every figure and table the thesis reports, and `--sections` picks which.
Default `baselines`, and `all` for everything.

**Why one script rather than four.** The figures share a palette, a method identity, a plain-variant
mechanism and a run directory. Split across scripts those drift, which is exactly what happened to
the presentation figures in `dummy/rl_course`: they redefine `GRID` one shade lighter than
`report.GRID`, alias `C_DQN = METHOD_STYLE["greedy"]["color"]` so blue means DQN there and greedy
here, and reimplement the plain variant as a module global mutated across modules.

**Ablation is why sections are opt-in.** It re-runs every episode once per channel per mode, so at
`19 channels x 3 modes x N models` it dwarfs the comparison it sits beside. `ablation.py` keeps its
own CLI; the section calls `ablation.measure(...)`, lifted out of its `main()` verbatim, against
the agent and the live environment the baselines section already built. Two sections in one
directory reporting numbers from two different environments is the failure that would be hardest
to notice.

**`generalisation` is the one section that mixes vintages.** It aggregates earlier run directories,
which were produced at different times against possibly different code. `generalisation.csv`
carries each row's `source_dir` and `git_commit` and the figure card says so. It also writes an
empty cell rather than a zero where a prior run predates `m_req` in `results.csv`, since a zero
bar there would read as "never reached the bound" instead of "not recorded".

**`training` reuses `tools/compare_runs.py:load`**, which groups event files by PID. The naive
version in `dummy/rl_course` globs the directory and merges, so training twice under one name
silently splices two runs into one curve.

**`generalisation` draws only what it can compare.** A method present on one instance set says
nothing about generalising and puts a "not run" marker on every other column, so those series are
listed on the card instead of drawn. Its width is capped rather than scaled per set: at a few
inches each, a dozen sets produce a figure no display can show.

### formation-panels

The three formation figures (`uncertainty`, `softest_mode`, `sensitivity`) draw one 3-D panel per
method, and two rules decide which methods get one.

**A flexible network has no panel at all.** There is no error ellipsoid and no softest mode without
rigidity, so `_panel_rows` filters on `is_IBR`. This is why `initial` is usually absent: on `mixed`
it is rigid on 15% of instances.

**What survives the panel cap is chosen by `FORMATION_PRIORITY`, not by `METHOD_ORDER`.** Those are
different questions. The table wants the classical methods first and the policy last, which is a
reading order; the figure wants whatever makes the comparison legible, which puts `learned` first
because the figure exists to show it. Using display order as figure priority silently dropped the
policy from all three figures the moment the method count passed the cap, since `learned` sits
second to last in `METHOD_ORDER`. Selection is by priority and *drawing* is still by `METHOD_ORDER`,
so the panels read in the same order as the table's rows.
`tests/test_outputs_reference.py` asserts `learned` survives a cap tight enough to bite.

**The cap bounds height, not width.** Every panel is a fixed 6.8 inches, so `_grid_for` choosing
more columns widens the figure rather than shrinking the panels. It picks the column count in
{3, 4} leaving the fewest empty cells, ties going to the narrower grid, which puts seven panels at
4x2 with one hole instead of 3x3 with two - a third of the figure left blank. At nine the cap and
the grid coincide.

### method-identity

`report.configure_methods(models)` rebuilds `METHOD_ORDER`, `METHOD_STYLE`, `METHOD_BLURB` and
`FORMATION_PRIORITY` in place, once per run, from the models the run was given.

**Why in place rather than a registry threaded through every plot function.** The seven
`METHOD_ORDER` read sites and the `_rank_in` sort keys then do not change at all, against roughly
forty signature and call-site edits for the alternative. The hidden-global objection is real in
general and weak here: `report.py` is single-threaded, is drawn once per process, and already
carries `_PLAIN` for exactly this reason. What the alternative would have bought is not safety but
plumbing.

**The fallback was the actual bug.** Every lookup used to be
`METHOD_STYLE.get(name, {"color": INK_2})`, which hands *every* unregistered method the same ink.
With one policy that never fired; with three it would have drawn them as one line. `method_style`
takes the next unused categorical hue instead and remembers it.

**Colour belongs to the models, once there is more than one.** A run with one policy has a hue to
spare for every method and keeps the palette the figures had before, `SINGLE_MODEL_STYLE`. Past
one, the hue has to answer "which model" and the baselines carry none, which is also the only
thing that scales: the validated palette has eight categorical slots and there are already eight
baselines. `SURFACE` supports three usable line tones (`INK` ~19:1, `INK_2` ~6.5:1, `MUTED` ~2.9:1;
`AXIS` and `GRID` cannot carry a line), so the dash does the rest, and the honest limit is that
`--methods all` with three models puts six grey lines in one panel, separated by dash and by the
direct end label rather than by tone. The default `--methods` is five and a thesis figure should
subset.

**One model keeps the label `learned`.** Every existing `results.csv`, `tools/verify_results.py`
and the tests key on that literal, and `--methods learned` still means "every model" whatever they
are labelled, so the default `--methods` string means what it always meant.

### observation-compatibility

A checkpoint only loads against an environment whose observation has the same width and meaning.
Three of the five models that looked like obvious comparison partners predate the
`rigidity_quality` channel, so they cannot be scored on a current config at all.

The environment *name* does not predict this: those three name the same environment as the models
that do fit, because the config was regenerated in place. `observation_mismatch` compares
`OBS_KEYS` instead and names the differing key before anything loads, which turns
`ValueError: parameter shapes do not match the checkpoint`, raised from inside
`agent_loader.resolve_model`, into `rigidity_quality was None, this environment has True`. The
name check survives as a note, since the same environment does also get regenerated under
different names.

### run-info

The title block used to repeat the environment, model, network and instance set on every figure.
Removing it gives each figure back about four lines of height, and `run_info` carries that
information once.

It also carries what the table cannot: per model the algorithm, backbone, widths, training length,
learning rate, seed and git commit, and **the objective the model was trained on beside the one it
is scored by**. Every row in a run is scored by the current environment's phi, which is what makes
the table comparable, but a model trained at `stiffness_kappa = 10` sitting in a table scored at
`2` is a fact that changes how its row reads and it was previously written down nowhere.

### at-bound

`m_req` is pose-dependent, so it belongs to the row rather than to the run. It is stamped in the
episode loop next to `episode`, where no `run_*` can forget it, and reaches `results.csv`.

That is what makes `at_bound` possible without hardcoding a benchmark constant. The presentation
scripts in `dummy/rl_course` set `M_REQ = 17` and threshold four figures on it, which is correct
for exactly one benchmark and silently wrong for every other. It is also what lets the
`generalisation` section put instance sets with different bounds on one axis, as edges over the
bound.

`is_MBR` is kept beside it rather than replaced. The two disagree on most methods on `bench_mixed`
(`greedy` 65% at bound against 80% minimal, `degree` 30% against 45%), and since the bound is
proven and the heuristic is not, seeing the gap is worth a column.

### plain-figures

Every figure is written twice, the second under `<name>-plain`: one carries the title block and the
notes card, one is the panels alone, for a document that supplies its own caption.

**Plain does not mean raw.** Panel titles, axis labels, units, reference lines, legends and the
direct labels that identify a series stay in both, because without them the plot cannot be read.
What comes off is the header block (which repeats the environment and model names a caption would
carry) and the notes card. Nothing inside an axes changes, so the two variants cannot disagree.

**One flag, consulted in four places.** `report.plain()` is a context manager over a module-level
`_PLAIN`; `_figure` zeroes the header and card bands and sets `top = 1.0`, `_finish` skips
`_draw_card`, `_save` appends the suffix, and `plot_table` does the same three because it builds
its own figure rather than going through `_figure`. A plot function needs no argument and no
knowledge of the mechanism, which is why this reaches all twelve figures rather than the one it
was first hand-rolled for. Callers draw their whole set, then draw it again inside `plain()`.

The 3-D formation figures fall out for free: they place panels inside the band `(top, bottom)` left
by header and card, and with both zero the panels fill the figure.

### panel-titles

Static output for a thesis, so light mode only and no hover layer. Identity is never colour alone:
every series carries a direct label at its line end (or a value label on its bar), and every figure
ships the notes card that says what it is showing.

Each panel is titled with *what the quantity is*, with the reading direction on a second line -
"edges used" told a reader nothing about why they should care.

### ablation-protocol

Three things the outcome columns depend on, all of which were silently wrong until 2026-08-23 and
all of which move the numbers more than most channels do.

**Stop at the reference's convergence, and give every variant that budget.** With `skip_enabled:
False` the policy must act every step, so an unperturbed argmax policy reaches its answer and then
runs a cycle: measured 20/20 episodes enter a repeated state, median step 14 of 78. Since the
unperturbed policy is a function of the edge set, a repeated state *is* an infinite cycle, so
stopping there costs the reference nothing. What matters is that the perturbed rollouts are then
capped at the same number of steps. Without the cap, ablating a channel buys extra exploration:
before the fix every live channel showed a *negative* cost, because ~3 of 20 references were stuck
short of minimal and any perturbation rescued them.

**phi must be a function of the state.** `stiffness_ref` is built from a seeded construction, but its
rng used to advance on every `reset()`, so repeated restores of one instance drew references two
decades apart and every variant was scored under a different phi. `compute_episode_constants`
reseeds per episode now, so the same poses always give the same reference.

**`--live-env` exists because the archive is right for the policy and wrong for the measurement.**
`load_run` replays the environment a checkpoint was trained against, which is what keeps an old
checkpoint runnable. It also keeps environment-side *measurement* fixes out: the reseed above did
not reach the ablation until the run was repeated with `--live-env`. The header states which was
used.

`coord_features` is the built-in null control: GINE never receives coordinates, so its cost must be
exactly 0.00 in every mode. It reads 0.00 now and read +0.29 before these fixes, which is how the
residual bias was found.

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
