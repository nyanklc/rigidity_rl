# Roadmap — margin and heterogeneity

**Branch:** `margin-and-heterogeneity` (from `formulation-overhaul` @ `cb074cd`)
**Started:** 2026-08-12

This file is the live plan *and* the work log. It is written so that a session with **no
conversation history** can read it and continue. If you are picking this up cold, read
§0 first.

> **Never commit.** See `CLAUDE.md`. Work stays in the tree; the user reviews and commits.

---

## Status board

| WP | Title | Tier | State |
|---|---|---|---|
| WP1 | Per-node DOF restriction in the rigidity matrix | 0 | **done** (2026-08-12) |
| WP6 | Cut the horizon; stop action as a trainable arm | 0 | **done** (2026-08-12) |
| WP12 | Freeze benchmark instances to disk | 0 | **done** (2026-08-12) |
| WP11 | Random global rotation at reset | 0 | **done** (2026-08-12) |
| — | `mixed5` scenario as the ground-truth debug case | 0 | **done** (user-provided) |
| WP2 | Null-space features: fix flex, add the exact addition oracle | 1 | **done** (2026-08-14) |
| WP5 | Pairwise action head (level 2 default, level 3 arm) | 1 | **next** |
| WP10 | Input embedder for the EGNN | 1 | **done (2026-08-21)** |
| WP13 | DQN hygiene: target time constant, DDQN, seeding | 1 | **code done (2026-08-21), control run pending** |
| WP7 | Heterogeneous training (phase A / phase B) | 2 | **phase A run, partial** (2026-08-15) |
| WP3 | Rigidity margin in the reward (κ = 0.9) | 2 | not started |
| WP4 | Margin-aware observation (softest mode) | 2 | not started |
| WP8 | Baselines: constructive restart greedy | 3 | **partly done** (2026-08-15) |
| WP9 | Multi-n training | parked | — |
| — | UCT / model-based planning | future work | deferred by decision |
| — | Sensing range, degree budgets, geometric limits | future work | deferred by decision |

---

## §0 Picking this up cold

**What the project is.** See `CLAUDE.md`. Master's thesis: choose the edge set of a directed
multi-agent network so it is bearing rigid with as few edges as possible, via deep RL + GNNs, for
any `n` and any mix of agent domains.

**Where it stood before this branch.** The formulation worked at n=8/R^3 (95% minimally rigid,
against 50% for phi-greedy) and failed everywhere else. §1.5 below has the cross-domain numbers.

**The two findings that motivated this branch:**

1. **`rank(B)` is generically a function of the graph alone, so the current reward contains no
   geometry** — and therefore neither the bearings nor the equivariant backbone have anything to
   do. See §1.1. The fix is to put the rigidity margin into the reward (WP3), which is also what
   the thesis wants scientifically.
2. **The heterogeneous rigidity matrix is wrong.** The DOF restriction is attached to the edge
   rather than the node, so a planar agent gets three position DOFs whenever it is adjacent to a
   spatial one. See §1.2. Everything heterogeneous rests on this, including the new `mixed` and
   `mixed5` scenarios. This is WP1 and it blocks the rest.

**Evidence.** Every number quoted in §1 was measured on this machine on 2026-08-11/12. The
one-off scripts live in the session scratchpad and are *not* committed; §1 records the numbers and
the method so they can be reproduced. Where a claim is inference rather than measurement it says so.

**Reading order for a fresh session:** this file §0–§2 → `THEORY.md` §3 and §12 → `CLAUDE.md`
"Architecture" → `DESIGN_NOTES.md` for whichever component you are touching.

---

## §1 Corrected diagnosis

### 1.0 Measured results, 2026-08-15

`letsgo_dqn_gine`: DQN + GINE, trained 400k steps on the `mixed` scenario (n=10, two agents each of
the five domains), informed arm (`rigidity_global`/`flex`/`edge` all on), single seed. All rows are
20 frozen benchmark instances, `--policy-mode sample`, 100-step budget.

**Reproducing these numbers:** `PYTHONPATH=. uv run tools/verify_results.py` re-derives every
digest, `rank_K`/`c_max`/`m_req`, and both classical baselines from the tracked `benchmarks/`, with
the environment built programmatically so a fresh clone suffices. The `learned` rows need the
checkpoint, which `models/`/`train/` do not track. Instance sets, by digest:
`bench_mixed` 83a53b8677d9, `bench_mixed5` ee7ce6e6da7d, `bench_n8_R3` a678a0266a20,
`bench_n8_R3_rot` 803938347f8e, `bench_n8_SE3` 94c9396becab, `bench_n8_R3xS1` 72b1d517025f,
`bench_n8_R2xS1` 7805a3bd2f6f, `bench_n16_R3` 333864562507.

**In distribution** (`bench_mixed`, `m_req` = 17, phi ceiling 74.24):

| method | edges | phi | rigid | minimal | best at |
|---|---|---|---|---|---|
| initial | 18.15 | 48.26 | 15% | 0% | — |
| random | 22.10 | 63.64 | 40% | 10% | 19.1 |
| greedy | 17.40 | 73.64 | 100% | 80% | 9.6 |
| constructive (20 restarts) | 17.70 | 73.18 | 100% | 50% | 17.7 |
| **learned** | **17.05** | **74.17** | **100%** | **95%** | 10.3 |

Reaching `m_req` with a rigid graph *proves* per-instance optimality, since `m_req` is a sound lower
bound (rank subadditivity over edge blocks). This does not rely on the `is_MBR` heuristic.

Under argmax rather than sampling (the ablation reference row): 17.20 edges, 100% rigid, 90%
minimal.

**Transfer, no retraining:**

| benchmark | learned | best classical | verdict |
|---|---|---|---|
| `bench_mixed5` (n=5) | 8.15, 85% | 8.00, 100% (= brute force) | task saturated, policy just behind |
| `bench_n8_R3` | 10.75, 50% | 10.45, 55% | tie |
| `bench_n16_R3` | 23.20, 0%, 95% rigid | **greedy 22.65, 45%**; constructive 23.20, 0% | **loses to greedy** |
| `bench_n8_SE3` (`m_req` 21) | **26.10, 25%** | **21.00, 100%** (both) | **fails** |

**Correction, from `tools/verify_results.py`:** the n=16 row above originally read "tie", because
`greedy` was absent from that run and only `constructive` was there to compare against. Measured
since on the same frozen instances, greedy reaches 22.65 edges and 45% minimal, against 23.20 and 0%
for both constructive and the policy. At n=16 the policy therefore **loses to greedy**, and ties
only the weaker baseline. The one configuration where it beats both classical methods remains the
heterogeneous mixture it trained on.

**Transfer degrades monotonically with agent DOF.** Homogeneous n=8, all five run against the same
policy, 20 frozen instances each:

| domain | DOF/agent | `c_max` | `m_req` | learned | greedy | constructive | learned edges, step 0 -> 24 |
|---|---|---|---|---|---|---|---|
| `R^3` | 3 | 2 | 10 | 10.75, **50%** | 10.50, 50% | 10.45, 55% | 11.2 -> 10.6, prunes |
| `R^2xS^1` | 3 | 1 | 20 | 20.00, **100%** | 20.00, 100% | 20.00, 100% | 21.4 -> 19.4, prunes |
| `R^3xS^1` | 4 | 2 | 14 | 15.40, **10%** (85% rigid) | 14.15, 85% | 14.00, 100% | 15.2 -> 16.7, accumulates |
| `SE(3)` | 6 | 2 | 21 | 26.10, **25%** | 21.00, 100% | 21.00, 100% | 23.7 -> 31.9, accumulates |

The split is at 3 versus 4 DOF per agent, and the accumulation magnitude is monotone in DOF: +0 at
3, +1.5 at 4, +8.2 at 6. Wherever the policy works it prunes toward `m_req`; where it fails it never
enters a pruning phase and reaches rigidity by accumulation instead. On `SE(3)` it scores phi 68.17
against 70.37 for a *uniform random policy*, while remaining rigid on 100% of instances, so the
failure is purely over-density.

Two qualifications on the two successes. `R^2xS^1` has `c_max = 1`, so its independent sets form a
matroid, greedy is optimal by construction and even a random policy reaches 45% minimal; that row
does not discriminate. `R^3` does discriminate (random 0% minimal, learned 50% = both baselines) and
is a genuine transfer success.

**Coverage is the leading explanation, and is not yet established.** The training mixture is 2 agents
each of the five domains, so only 20% of nodes are 6-DOF and 20% are 4-DOF, and the policy never saw
a network where high-DOF agents dominate. In favour: it handles `SE(3)` and `R^3xS^1` agents *well*
when they are a minority, reaching 95% minimal in distribution, and fails only when they are the
majority, which is a composition-shift signature rather than an inability to represent those domains.
Not ruled out: a capacity or architecture limit that only bites when the constraint density is high.
The two are separated by one experiment, training on a high-DOF-weighted mixture or on homogeneous
`SE(3)` and re-evaluating. Cross-domain generalization (issue 4) is **not** resolved by these
results.

**Cost.** Per instance, measured by counting rigidity-matrix builds: constructive greedy 1864,
learned 7 — **266x**. Wall clock is only 2.6x on CPU, where the GNN forward dominates.

**Rotation dependence, quantified.** `bench_n8_R3_rot` is `bench_n8_R3` under an exact global
rotation (residual 4e-16, det R = +1, identical edges and pairwise distances). Every classical
method is byte-identical across the two, as invariance requires. Only the policy moves. **8 of the
20 instances change their minimality verdict** (10 minimal unrotated, 14 rotated, but 8 individual
flips), from a transformation that changes nothing about the problem. Read the churn, not the net:
40% of instances changing outcome establishes the dependence, while the net direction (+4) is noise
at this sample size. This is §2.3 / known issue 7 measured on a deployed policy rather than argued.
Note the policy was trained *with* `rotation_augmentation` on, so the augmentation did not remove
the dependence here.

**Ablation** (`dummy/abl_letsgo_dqn_gine*`, three modes). The large `degree` / `rigidity_glob` /
`add_rank` costs seen under `--mode zero` are out-of-distribution artifacts: zeroing normalized
degree asserts every node has degree 0. Under `shuffle`, which preserves the marginal, everything
collapses to noise except `degree` (35% of minimality) and the `adj` mask control. What holds in
**all three modes** is the finding that matters: `bearings`, `coord_features`, `add_gain` and
`flex_mag` cost nothing. The policy solves the task from graph structure and reads no geometry.

**Margin collapse over training:** mean min-eig 0.058 -> 0.003 (19x), best 0.009 -> 0.001. The
policy trades robustness for sparsity because phi has `w_eig = 0`. In the baselines table the
*random* policy holds a better margin than any method that solves the problem.

Together these are one finding, and the case for WP3: the objective is combinatorial, so the policy
solved it combinatorially, discarded the geometry, and produced fragile topologies. WP3's acceptance
test is this same ablation — the geometric channels must start costing something.

*Caveats:* single seed; 20 instances, so the 95/80/50 gaps are 3, 4 and 10 instances; the informed
arm carries an exact rank oracle (`add_rank`), so this is closer to constructive-greedy-with-learned-
ordering than to learning rigidity from scratch. The uninformed arm is untrained and is the
comparison that would settle it.

### 1.1 The objective is combinatorial; the geometry is not in it

`B` is the Jacobian of the bearing function and does depend on the configuration. **`rank(B)` does
not**, generically: every entry of `B(χ)` is rational in χ, so `rank(B(χ))` is lower
semi-continuous and drops only on the zero set of a minor determinant — a proper algebraic subset,
measure zero. For χ from any continuous distribution, `rank(B(χ))` equals its generic value almost
surely, and that generic value is a function of the graph and the domain assignment alone.

Michieletto's noncollinearity assumption (Definition 2, and §VII, which treats collinear
formations separately) is exactly the assumption that puts us in the generic set.

*Measured:* 30 graphs × 5 domains × 100 pose resamples each — rank never moved once. Repeated for
heterogeneous mixes with the same result.

**Consequence.** Both state scores are pure functions of the edge set: `Weighted` is
`20·rank − 10·m` (its `w_ibr` and `w_eig` terms are hardcoded to 0) and `WeightedNormalized` is
`(100·rank − 25·m·c_max)/rank_K`. Neither reads anything but rank and edge count. This is why the
ablation shows every geometric channel costing exactly 0.00 phi: **the policy is not failing to use
the geometry, the reward is not asking it to.**

**What does *not* follow.** Geometry is not useless as an *input*:

- It is the computational route to the combinatorial answer. The exact test for "does adding
  `i -> j` raise the rank" is `‖b_ij · Z‖ > 0` with `Z` spanning `ker(B)` — a geometric computation
  returning a combinatorial fact. Measured AUC 1.000 with a clean separation in all five domains.
- The **margin** (the rigidity eigenvalue) is not combinatorial at all. It spans 10^5 across
  equally-optimal minimal graphs on the *same* poses.

**The reframing that drives this branch:** the problem splits into a combinatorial half (which
graph is rigid — what the GNN and the action space solve) and a geometric half (which of the rigid
graphs is *good* — what bearings and equivariance are for). The reward currently contains only the
first. WP3 adds the second.

### 1.2 The heterogeneous rigidity matrix

Michieletto's framework requires an infeasible variation to appear as a **null column** of `B⁺` —
that is what makes `rk(B⁺) = 6n − q_v − q_i` and Theorem 2 work. Table I delivers that for
homogeneous manifolds. Table III (the heterogeneous case study) sets `U_(1,4) = I₃` for a planar
robot measuring the aerial platform, which reactivates the planar agent's z-column; the paper
handles it by observing those columns come out linearly dependent *in their particular 4-agent
example*. That is configuration-specific. `rigidity.bearing_DOFs` implements Table III faithfully
and inherits the gap.

*Measured:*

| mix | Σ DOF | `rank_K` as built | corrected | IBR verdict disagreement |
|---|---|---|---|---|
| `mixed` (2 of each of the five) | 36 | **36** | 33 | 2.0% of 300 random graphs |
| 3×R^2 + 1×SE(3) | 15 | 11 | 9 | 13.0% |
| 4×R^2 + 2×R^3 | 14 | 14 | 11 | 14.0% |
| 5×R^2 + 1×R^3 | 13 | **14** | 10 | **40.0%** |
| 3×R^3 + 3×SE(3) | 27 | 23 | 23 | 0.0% (no planar agents) |

`rank_K = Σ DOF` on `mixed` means **zero trivial motions are being counted**, which is impossible
for any framework. Direct check on 4×R^2 + 4×R^3: a pure +z motion of a planar agent gives
`‖B v‖ = 2.1` — the model resists a motion the agent cannot make, and spends rank doing it.

Homogeneous networks are unaffected: the corrected construction reproduces the current rank exactly
on 20 random graphs in each of the five domains.

### 1.3 The margin: two facts that constrain the reward

**λ is monotone increasing in edges** (Weyl: adding rows cannot decrease any eigenvalue). Measured
at n=8/R^3 on pose-normalized coordinates, from a minimal graph outward:

```
m  = 12    16     20     24     28     32     36
λ  = .001  .005   .218   .259   .365   .620   .701       ~700x over the range
```

So "maximize λ" alone has the complete graph as its optimum and is not a sparsity objective at all.
It must be bounded against the edge term — which is what κ = 0.9 does (WP3).

**λ decays with n and no fixed normalizer fixes it.** On greedy-minimal graphs, pose-normalized,
median of 10 instances:

| domain | n=4 | n=6 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|
| R^3 | 1.5e-01 | 2.4e-02 | 1.2e-02 | 3.9e-03 | 1.3e-03 |
| R^2 | 1.0e-01 | 2.6e-02 | 2.8e-03 | 9.5e-04 | 4.0e-05 |
| SE(3) | 3.6e-02 | 1.7e-02 | 3.9e-03 | 1.7e-04 | 7.1e-04 |

Dividing by λ of the complete graph removes the length-scale dependence but not the n-dependence
(0.32 at n=4 → 1.0e-3 at n=12 in R^3). **But the spread is n-stable** — the p10–p90 band of
log10 λ among minimal graphs is 1.1–1.9 decades at every n measured. Since the reward is
potential-based, the n-dependent *level* cancels and only the spread enters the return. That is
what makes a log-λ term viable, and it dictates the `λ_ref` normalizer in WP3.

### 1.4 Where the combinatorial difficulty actually is

Randomized constructive greedy (start empty, add any rank-raising edge, stop at `rank_K`),
25 runs per configuration:

| domain | n | `c_max` | `m_req` | min | max | mean | reading |
|---|---|---|---|---|---|---|---|
| R^2 | 8 | 1 | 13 | 13 | 13 | 13.00 | matroid |
| R^2xS^1 | 8 | 1 | 20 | 20 | 20 | 20.00 | matroid |
| R^3 | 8 | 2 | 10 | 10 | 13 | 11.72 | order matters |
| R^3xS^1 | 8 | 2 | 14 | 14 | 17 | 15.28 | order matters |
| SE(3) | 8 | 2 | 21 | 21 | 23 | 21.96 | order matters |

In the `c_max = 1` domains every run terminates at exactly `m_req`: the independent sets behave as
a matroid and the problem is trivially solved by any greedy. **R^2 is therefore a control and a
debugging case with ground truth, never a headline result.** The `c_max = 2` domains are not a
matroid, and that gap is the substance of the task.

Darvariu et al. (2024) make the same point structurally: their survey's inclusion criterion is that
no satisfactory exact or approximate algorithm exists, and only the `c_max = 2` and heterogeneous
regimes qualify.

**The matroid statement generalizes, and it cuts both ways** (`THEORY.md` §14). `S -> rank(B_S)` is
monotone **submodular**: an edge contributes a subspace, rank is the dimension of their sum, and the
part not already spanned can only shrink as the graph grows. Proved in §14.2, and measured at 0
violations in 1920 triples across all five domains and a mix (`tools/submodularity.py`). So minimum-edge rigidity is **minimum
submodular cover**, and by Wolsey (1982) the constructive baseline is an `H(c_max)` approximation:
exact at `c_max = 1` (the matroid rows above), 1.5 at `c_max = 2`.

That makes `constructive` the principled opponent rather than an ad-hoc one. It also caps the prize:
measured against `m_req`, greedy is only 0–5% above the bound, so **no method can gain much on edge
count**, and §1.0 shows the policy already closing ~88% of that gap in distribution. Darvariu §6.2
predicts exactly this — RL gains little where shallow horizons suffice.

**The margin escapes it.** `lambda_r(S)` is monotone (Weyl) but **not** submodular: 887 of 1493
triples violate diminishing returns, worst gap -4.16e-01. Two edges can be worth more together than
apart, which is invisible to any one-edge-at-a-time method. Greedy therefore has **no guarantee** on
the margin, and that is the structural case for WP3 — the geometric objective is a different problem
class, not just a different reward.

### 1.5 The baseline that has to be beaten

Constructive restart greedy — ten lines, no learning, no geometry — against the best checkpoint
(`generaldqngine`, trained at n=8/R^3), 20 instances, % solved at exactly `m_req`:

| config | `m_req` | k=1 | k=5 | k=20 | mean m (k=20) | learned |
|---|---|---|---|---|---|---|
| n=8 R^3 | 10 | 5% | 25% | 65% | 10.35 | **10.05 · 95% min** |
| n=12 R^3 | 16 | 0% | 0% | 20% | 16.80 | — |
| n=16 R^3 | 22 | — | — | — | **23.15** | 23.85 · 10% min |
| n=8 R^3xS^1 | 14 | 15% | 75% | **100%** | 14.00 | 13.95 · **25% rigid** |
| n=8 SE(3) | 21 | 40% | 85% | **100%** | 21.00 | 17.30 · **5% rigid** |
| n=8 R^2 | 13 | **100%** | 100% | 100% | 13.00 | — |

The learned policy wins in exactly one configuration: the one it was trained on. Cross-domain it is
below the *untouched initial graph* (5% vs 35% rigid on SE(3)). Mechanism, measured in the weights:
training in R^3 leaves four of the five domain one-hot columns identically zero, so their
first-layer weights never receive a gradient and stay at initialization —
`generaldqnequi`'s `gnn.conv1.edge_mlp.0.weight` has mean |w| 0.007–0.009 on the four unseen domain
columns against 0.074 on the trained one. Evaluating in SE(3) flips the one-hot onto an untrained
column: the domain identity cannot inform the policy, only inject noise. **No observation or
architecture change fixes this; only training across domains does** (WP7).

**Update, 2026-08-15: WP7 phase A has now run and the prediction is half right.** Training on
`mixed` (2 agents of each domain) removed the catastrophic failure — `letsgo_dqn_gine` is rigid on
100% of `SE(3)` instances where `generaldqngine` managed 5%, i.e. the one-hot column is no longer
untrained. But it did **not** deliver minimality: 25% against 100% for both classical baselines, and
transfer now degrades with agent DOF rather than collapsing outright. Full table in §1.0. Domain
coverage was necessary and is not sufficient; the open question is whether *composition* coverage
(high-DOF agents in the majority) closes the rest.

### 1.6 Checked and found fine — do not spend time here

- **Observation channel scales are all O(1)** on `mixed` (domain 0.20±0.40, degree 1.18±0.81,
  rig_glob 0.52±0.61, flex_mag 0.79±0.61, bearings 0.01±0.55, flex_align 0.70±0.79, max |·| 3.47).
  `observation_preprocessor = None` is defensible; no `RunningStandardScaler` needed.
- **Near-coincident agents do not confound λ.** Correlation between log min-pairwise-distance and
  log λ is +0.23 at n=8/R^3 and +0.09 on `mixed`, against a λ spread of 3–5 decades. No
  minimum-separation sampler needed.
- **The action index ↔ (i,j) mapping is consistent** between
  `action_AddRemoveEdgeDiscreteNoSelfLoops` and the models' boolean masks (both row-major over
  off-diagonal entries).
- **PPO's config is genuinely fixed**: one `ROLLOUT_SIZE` for memory and rollouts, γ = 0.99,
  `time_limit_bootstrap = True`.
- **`is_MBR`'s false-positive path does not fire on `mixed`** — 0 in 122 rigid graphs. The mechanism
  is real (it derives `m_req` from the current graph's blocks rather than the complete graph's) but
  you cannot build a rigid graph out of rank-1 edges on that mix. Fixed as hygiene in WP7.

---

## §2 Work packages

### Tier 0 — before anything else

#### WP1 — per-node DOF restriction in the rigidity matrix

*Why:* §1.2. Blocks WP2, WP3, WP7 and every heterogeneous result.

*What:* build the position block with the true measurement Jacobian and restrict per endpoint,
rather than applying a per-edge `U_ij` to the relative displacement:

```
B[3k:3k+3, 3j:3j+3] += Dp_k @ S_j        # S_i = diag(1,1,0) planar, I_3 spatial
B[3k:3k+3, 3i:3i+3] -= Dp_k @ S_i        # per endpoint, NOT per edge
B[3k:3k+3, 3n+3i:3n+3i+3] += Da_k @ P_i  # P_i = I_3 SE(3), v vᵀ for R^dxS^1, 0 for R^d
```

Also repairs `V_ij` for `R^3xS^1`, currently `[0; 0; rax]` as *rows* where Michieletto's
`[0_{3x2} v]` is a *column*; they coincide only at `rax = e₃`, the only value ever used. The
projector form `v vᵀ` is right for any axis.

*Acceptance:* (i) rank identical to today on all five homogeneous domains — already verified on 20
random graphs each; (ii) `rank_K ≤ Σ_i DOF_i − |trivial|` asserted for every mix; (iii) IBR verdict
unchanged by appending an all-zero column; (iv) a planar agent's z-column is exactly zero.

*Not in this WP:* `trivial_modes` is also wrong for mixes (it hardcodes three translations plus
scaling; a mixed planar/spatial framework admits only two plus scaling). It is fixed in WP2,
because the correct replacement — an orthonormal basis of `ker(B_K)` — changes the shape from
`R^{3n}` to `R^{6n}` and only makes sense together with the full-null-space flex rework.

#### WP6 — cut the horizon, restore the stop action

*Why:* the horizon governs **instance diversity**, not just wall clock. Instances seen in 400k
steps is `total_timesteps / max_steps`; instances held in the replay buffer is
`memory_size × num_envs / max_steps` (skrl buffers are `(memory_size, num_envs, …)`, so 10000 × 4).

| config | `max_steps` | instances seen in 400k | instances in the buffer |
|---|---|---|---|
| n=8 R^3 | 224 | 1785 | 178 |
| `mixed` n=10 | 360 | 1111 | **111** |
| n=16 R^3 | 960 | 416 | 41 |
| `mixed` at `4·m_req+10` = 78 | 78 | **5128** | **513** |

Raising `memory_size` is not the alternative: each transition stores a 1050-float observation
twice, so the buffer already costs ~336 MB at n=10.

The horizon is oversized 20–30×: measured `best@` across every run is 6.6–12.8 steps, and
`Edit efficiency` ends training at 0.018 ≈ 1/56 — the policy reaches its answer around step 7 and
runs a two-cycle for the remaining ~217 because `skip_enabled: false` forces an edit every step.

*What:* `max_steps = 4·m_req + 10`; re-enable the stop action with a small time penalty (the reason
it was disabled — `select→skip` as an absorbing zero-reward 2-cycle — was a γ=1 problem, and γ is
0.99 now); keep the constructive (empty-start) formulation as an arm, not a replacement.

*Acceptance:* no quality change at n=8/R^3 at 1/4 the horizon. If quality drops, that tells you the
reported numbers were relying on best-of-N search rather than on the policy.

#### WP12 — freeze benchmark instances to disk

*Why:* this already cost one comparison. The two n=16 evaluations ran against initial graphs of
52.25 ± 46.53 and 23.70 ± 10.64 edges because the config was regenerated with the fixed sampler in
between, so "31.60 → 23.85 edges" conflates a better policy with an easier instance distribution.
WP1 changes `m_req` and WP6 changes the horizon, so every config is about to be regenerated.

*What:* `benchmarks/<name>.npz` holding N instances (positions, orientations, domains, initial
edges). `baselines.py --benchmark <name>` loads them instead of sampling. Record the benchmark name
and a hash in `meta.json`; refuse to plot two runs with different hashes on one axis.

#### WP11 — random global rotation at reset

*Why:* bearings for `R^2`/`R^3` agents are returned in the global frame, so the policy's output
changes under a global rotation although the task does not. Free data augmentation; on `mixed`,
four of ten agents are frameless.

*What:* one call to `Network.rotate_network` in `reset()`. It already rotates orientations for
oriented agents, so it is correct as-is. **Constraint: when any planar agent is present only
rotations about z are admissible** — an arbitrary axis would lift them out of the plane.

### Tier 1 — observation and architecture

#### WP2 — null-space features  **[done 2026-08-14; see the work log entry]**

*Why:* `flex_tensor` takes `Bp = brmat[:, :3n]`, the position block only, so the flex space ignores
that agents can rotate. In `R^d` that is the whole space; in the oriented domains it is the wrong
subspace, and `flex_align` — the feature designed to answer "would this edge help" — degrades
exactly where the project fails:

| domain | `flex_align` (current) | `‖b_ij Z‖`, Z = ker(B) of the full matrix |
|---|---|---|
| R^2 | 1.000 | **1.000** clean split |
| R^3 | 1.000 | **1.000** clean split |
| R^2xS^1 | 0.748 | **1.000** clean split |
| R^3xS^1 | 0.847 | **1.000** clean split |
| SE(3) | **0.632** | **1.000** clean split |

(AUC for predicting "adding i→j raises rank(B)", ~250 candidate pairs per domain.)

*What:*
1. **One SVD replaces three decompositions.** `step()` currently does `matrix_rank(B)`, `is_MBR`
   (one rank per edge) and `eigvalsh(BᵀB)`. A single `svd(B)` gives the rank, an orthonormal `Z`
   spanning `ker(B)` from the trailing right singular vectors, *and* the rigidity eigenvalue as
   `σ_r²`. Prerequisite for WP3/WP4 being affordable.
2. `add_gain[i,j] = ‖b_ij Z‖` and `add_rank[i,j] = rank(b_ij Z)/c_max`, **all ordered pairs**.
3. Fix `flex_mag` to use the full null space, projected against `ker(B_K)` (which also replaces
   `trivial_modes`).
4. `block_rank` from the **complete graph, all pairs**. Today `compute_rigidity_features`
   (`environment.py:897`) fills it only for existing edges, so every candidate reads 0 —
   indistinguishable from "contributes nothing". On `mixed` the complete graph's block ranks are
   `{1: 12, 2: 78}`, so the channel is finally informative and currently blind on exactly the pairs
   the agent chooses between. The values are already computed inside `required_edge_count`.

*Honest framing:* `add_gain` makes the informed arm approximately
constructive-greedy-with-learned-ordering. Say so explicitly. It is still the right arm: a
20-restart constructive greedy reaches the optimum only 65% of the time at n=8/R^3 and 20% at n=12,
so a perfect one-step oracle is demonstrably **not** sufficient, and the arm isolates "can RL learn
the non-myopic part". The uninformed arm stays the headline.

#### WP5 — pairwise action head

*Why:* held-out linear probes on the trained GINE backbone, predicting "does adding i→j raise the
rank":

| head input | R^3 | R^2xS^1 | SE(3) |
|---|---|---|---|
| `[h_i, h_j]` — what the head sees today | 0.955 | 0.956 | 0.972 |
| `[h_i, h_j, e_ij]` | 0.998 | 0.948 | 0.977 |
| `[e_ij]` alone | **1.000** | 0.892 | 0.759 |

The backbone carries most of it — an earlier claim that the head was structurally blind was too
strong. The real loss is narrower and still matters: a perfect pairwise signal exists in `e_ij` and
reaches the head degraded to 0.955, and that gap is where the hard end-game decisions live. In
SE(3) the pair feature was the *worse* source only because `flex_align` was broken there; WP2 fixed
that, so `e_ij` should now be re-probed before WP5 fixes a head width to it.
**WP5 without WP2 is half-pointless; WP2 without WP5 is half-used.** (Linear probes
lower-bound what an MLP head could extract.)

*What — level 2 is the default, decided:*
- **Level 1:** head input `[h_i, h_j, h_i ⊙ h_j, e_ij, adj_ij]`.
- **Level 2 (default):** additionally route the action-specific rigidity scalars around the GNN as
  a skip connection — `[add_gain_ij, add_rank_ij, block_rank_K_ij, soft_align_ij]`. These have a
  near-affine relationship to the action's value, and three layers of mean aggregation over `n−1`
  pairs is the wrong thing to put between them and the Q-value.
- **Level 3 (arm):** explicit one-step decomposition `Q(s,a) = Δφ̂(s,a) + f_θ(h_i, h_j, e_ij)`,
  where the rank part of `Δφ̂` is exactly computable —
  `Δφ_rank = (100·add_rank_ij·c_max − 25·c_max)/rank_K`. Needs `rank_K` and `c_max` exposed as
  observation scalars, which they currently are not.

`SelectNodesSequentially` gets the same plus the old phase-5 items: separate first-pick and
second-pick heads, explicit phase flag instead of the zero-vector sentinel. Note this action space
has a structural disadvantage — the first pick earns zero reward so its value is pure bootstrap, and
its Q must implicitly encode `max_j Q(second pick)`, which interacts badly with WP13's frozen
target. Run `AddRemoveEdge` as the main line; keep `SelectNodes` for when n grows.

#### WP10 — input embedder for the EGNN

*Why:* `EGNN` preserves the feature dimension and `GNNBackboneEquivariant` is constructed with
`dim = node_feat_dim`. On `mixed`, `node_feat_dim = 11` (5 domain + 2 degree + 3 rigidity_global +
1 flex_mag). So:

```
                        node representation width    action head input width
GNNBackboneGINE                   128                      2*128 + 1 = 257
GNNBackboneEquivariant             11                      2* 11 + 1 =  23
```

confirmed in the checkpoint: `gnn.conv1.edge_mlp.0.weight` has shape `(62, 31) = (2*m_dim, 2*11+1+8)`.
**The EGNN-vs-GINE comparison as run is an 11-dimensional model against a 128-dimensional one**,
not a comparison of message-passing schemes; that claim has to come down until this is fixed,
independently of the p ≈ 0.34 power problem.

*What:* `nn.Sequential(Linear(node_feat_dim, 128), LeakyReLU(), Linear(128, 128))` before the EGNN
stack, then `EGNN(dim=128, …)`. **This does not break equivariance** — the EGNN's equivariance is
with respect to `coors`, and `feats` are invariant scalars throughout. (Embedding `coors` would
break it.) ~~~18k parameters.~~

*Done 2026-08-21.* `GNNBackboneEquivariant.embed`, stack at `dim=hidden_dim`; the seven
`Equivariant_*` models now size their heads on `gnn_hidden_dim` instead of `node_feat_dim`, so
**both backbones output the same width** and the head arithmetic is the same either way.
`test_both_backbones_output_the_same_width` pins it. Equivariance and n-invariance both re-measured
and intact (numbers in `DESIGN_NOTES.md#egnn-input-embedder`). Four archived checkpoints
(`generaldqnequi`, `phase4_dqn_equi_n8_SE3`, `heynewppo`, `letsgo_dqn_gine`) re-fingerprint
bit-identically through `tools/checkpoint_fingerprint.py`.

**The "~18k parameters" above was wrong, and the error matters.** It counted the embedder only.
Raising `dim` from 11 to 128 widens every layer's `edge_mlp` and `node_mlp` too, so the stack goes
40,407 -> 940,956 parameters — **10.9x GINE's 86,499**. Matched width and matched parameters cannot
both hold: parameter parity would need `dim ~= 32`, a quarter of GINE's width. Width is what is
implemented, because width was the diagnosed defect and stage 1's criterion is stated as equal
width — but the comparison must now be reported as *equal width, unequal capacity*. An EGNN that
wins at 10.9x the parameters has not beaten GINE at message passing. A matched-parameter arm needs
`m_dim` separated from `hidden_dim` in the constructor and is deliberately not a knob yet; `tools/backbone_capacity.py` prints what it would buy at any given `node_feat_dim`/`hidden_dim`.

*The harder question this exposes.* EGNN consumes coordinates only through `‖x_i − x_j‖²`. Bearing
rigidity is scale-invariant and depends on *directions*, so distance is close to the wrong
invariant, and the directional information arrives as invariant scalar edge features that GINE
receives identically. The ablation on the EGNN checkpoint agrees: `coord_features` flips 22.5% of
decisions but costs **−0.50 phi** (nothing, within noise at 10 episodes) and `bearings` flips 8.3%
for **−0.25 phi**; only `adj`, `rigidity_glob` and `degree` have a positive cost, the same picture
as GINE. So fix the width, re-run at 3 seeds, and be prepared for the honest answer to be
*"equivariance is not exercised by this objective"* — a legitimate negative result, and one WP3 may
overturn, since the margin *is* geometric. A genuinely directional architecture (GVP, vector
neurons, e3nn) is thesis-scale and should not start before WP3 shows the geometry matters.

#### WP13 — DQN hygiene

*Why:* skrl applies `polyak = 0.005` every `target_update_interval = 200` **updates**, and updates
happen every `update_interval = 4` timesteps — one 0.5% step every 800 timesteps, giving a target
time constant of **160,000 timesteps**. Across a 400k-step run that is 2.5 time constants, i.e.
two or three Bellman backups of value propagation for the whole of training. Almost certainly an
unintended hybrid of the two standard conventions.

*What:* pick one convention — hard update (`polyak = 1`) every 200 updates, or soft update
(`polyak ≈ 0.005`) every update. Either gives a time constant near 800 timesteps. Also: **Double
DQN** (skrl supports it as a drop-in; with 181 actions on `mixed` the max-operator overestimation
bias is not negligible).

*Done 2026-08-21 (code only; the control run is still owed).*

- **Soft convention chosen**, by decision: `cfg.polyak = 0.005`, `cfg.target_update_interval = 1`,
  so the constant is `update_interval / polyak` = 800 timesteps against the previous ~160k.
  `polyak` is now set explicitly rather than inherited from skrl's default, so it reaches the
  manifest. Hard updates remain a one-line alternative if soft disappoints.
- **DDQN is an `ALGORITHM` env-var arm, default `DQN`.** skrl's `DDQN` shares `DQN`'s config
  fields, models dict and argmax rollout, so it is a genuine drop-in. `train_dqn.py` writes the
  manifest and the checkpoint under the selected algorithm, and `agent_loader.build_agent` rebuilds
  the class the run actually used instead of always DQN.
- **Seeding needed no change — the roadmap's claim above was wrong.** Measured directly, in
  `train_dqn.py`'s own construction order (`SyncVectorEnv` built, *then* seeded): two runs at
  `SEED=0` produce a byte-identical digest over the first reset and 20 vector steps, and `SEED=1`
  differs. `initialize()` does draw a network from the unseeded global stream at construction, but
  the first `reset()` discards it, and sharing one global stream across sub-envs is deterministic
  under `SyncVectorEnv`'s fixed stepping order. *Scope of the check:* CPU environment sampling only.
  Bitwise reproducibility of a CUDA training run is a separate question and was not tested.

*Acceptance, still open:* one control run before and after, at n=8/`R^3` or on `mixed`. This
changes learning dynamics, and it lands together with the head fix below, so the two are
confounded unless run separately — see the work log.

### Tier 2 — the science

#### WP7 — heterogeneous training

*Phase A — the `mixed` scenario, no code required.* Two agents in each of the five domains means
all 25 ordered domain pairs occur every episode, so every `(U_ij, V_ij)` combination is exercised
on every reset. The observation and action shapes do not depend on the mix. Blocked only on WP1.

*Phase B — resample the composition.* New env key
`domain_sampler: "fixed" | "uniform" | {weights}`, drawn in `reset()` before `random_scenario`.
Two ordering details: compute the mix first, then `m_req`, then the edge count (today
`sample_initial_edge_count` reuses the previous episode's `m_req` as its mean, which is stale the
moment the mix moves); and pass `env.m_req` into `is_MBR` rather than letting it recompute from the
current graph's blocks.

*Prediction, stated in advance so it can be wrong:* a mix-trained policy at n=10 should reach ≥90%
rigid on **every** homogeneous corner at n=8 (against today's 5% on SE(3), 25% on R^3xS^1), and
should not lose more than ~10 points of minimality at n=8/R^3 relative to the specialist. If it
does not, the diagnosis is wrong and the problem is capacity or credit assignment, not domain
coverage — and the next move is the constructive formulation, not more features.

*Outcome of phase A, scored against that prediction (2026-08-15).* **Clause 1 nearly holds.** Rigid
at n=8: `SE(3)` 100%, `R^3` 100%, `R^2xS^1` 100%, `R^3xS^1` **85%** — three corners clear the 90%
bar and one just misses, against 5% and 25% before. The untrained-one-hot failure of §1.5 is gone.
**Clause 2 fails badly.** At n=8/`R^3` the mix-trained policy is 50% minimal against the specialist
`generaldqngine`'s 95%, a 45-point loss where the allowance was ~10. It now merely ties greedy (50%)
and constructive (55%) where the specialist beat them outright.

So phase A bought reliability and cost specialism, and did not deliver minimality on the high-DOF
corners (§1.0). Taken literally the pre-registered rule points at capacity or credit assignment
rather than coverage. Two reasons not to conclude that yet: phase A used a **fixed** 2-of-each
composition, so high-DOF agents are never in the majority and homogeneous corners are never drawn —
that is precisely what phase B changes; and the specialist comparison is across two different
training budgets and one seed each.

*Pre-register phase B before running it,* or the coverage hypothesis becomes unfalsifiable and the
answer to every failure is more coverage. Proposed criterion: with `domain_sampler: "uniform"` over
compositions (so homogeneous corners are drawn during training), the policy must reach **≥80%
minimal on homogeneous `SE(3)` and `R^3xS^1` at n=8**, where both classical baselines reach 100%.
Below that, coverage is not the explanation and the next move is the constructive formulation or a
capacity change, as the original rule says.

#### WP3 — the rigidity margin in the reward

```
phi = (w_rank * rank  -  w_edge * m * c_max) / rank_K   +   w_eig * 1[IBR] * q(lam)

  lam      = sigma_r(B_hat)^2  on the POSE-NORMALIZED network (centred, unit RMS radius)
  lam_ref  = the same quantity for ONE greedy-built minimal graph on the same poses,
             computed once per episode in compute_episode_constants
  q(lam)   = sigmoid( log10(lam / lam_ref) / s ),   s = 0.75 decades     -> q in (0,1)
  w_eig    = kappa * w_edge * c_max / rank_K,       kappa = 0.9   (DECIDED)
```

- **κ = 0.9 makes the margin a strict tie-breaker.** Since `q ≤ 1`, the whole margin range is worth
  less than one edge, so the sparsity ordering can never be overturned and `is_MBR` stays the
  primary metric. One edge costs `w_edge·c_max/rank_K` = 2.50 at n=8/R^3, 1.52 on `mixed` after
  WP1, 1.14 at n=16/R^3 — expressing κ as a multiple of one edge makes it scale automatically.
  κ = 0 reproduces today byte-for-byte. This is the bound that makes *"most rigid configuration
  with the minimal number of edges"* well-posed rather than self-contradictory (§1.3). Weighting
  rigidity above sparsity later is exactly κ > 1; the natural figure is a Pareto front over
  κ ∈ {0, 0.9, 2, 4}.
- **`λ_ref` per episode, and it is cheap.** Measured: 0.09 s at n=8, 0.12 s on `mixed`, 0.26 s at
  n=16 for one greedy construction — 133 s over a whole 400k-step run on `mixed`. It is a
  per-episode constant, so potential-based shaping is preserved.
- **Why not `λ_K`.** `λ/λ_K` still decays two decades from n=4 to n=12, so `q` would saturate at
  large n. The greedy reference is by construction a typical solution at *this* n, these poses and
  this mix, so `q ≈ 0.5` for a typical answer everywhere. *Fallback if the oracle becomes a
  problem:* estimate the offset `log10(λ_min/λ_K)` once at env init from ~8 pose draws — measured
  stable to 0.15 decades at n=8/R^3 and 0.38 on `mixed`, against a within-configuration signal of
  1.1–1.9 decades. Usable but noisier.
- **s = 0.75 decades** because the p10–p90 spread among minimal graphs is 1.1–1.9 decades, so the
  logistic spends its range on the achievable band instead of saturating.
- **`q ≥ 0` and gated on IBR** so becoming rigid is never punished; a raw `log λ` term would make
  the transition to rigidity a large negative jump.

*To state in the thesis:* λ mixes translational stiffness (columns scaling as `1/‖p_ij‖`) with
rotational stiffness (dimensionless) through an implicit length unit. Pose-normalizing to unit RMS
radius pins that unit to the formation's own size. Defensible and similarity-invariant, but it is a
modelling choice and must be named as one — it matters most in `SE(3)` and in any mix.

*Acceptance:* edge count and minimality unchanged within noise at κ = 0.9; geometric-mean margin up
5–10× at n=8/R^3. And the decisive one: **the ablation should finally show a nonzero cost for
destroying the geometric channels.** That single figure is the acceptance test for the entire
geometric half of the thesis. Run it in at least two ablation modes — `zero` alone produced three
false positives in the 2026-08-15 ablation (`DESIGN_NOTES.md#rigidity-features`).

**Two questions, and only one of them needs a weight.** They are worth separating in the writing
because they have different scientific status:

1. *Among equally sparse graphs, pick the best-conditioned one.* At `m = m_req` the margin spans
   ~10^5 on the same poses (§1.3), so this is a **tie-break, not a trade-off**, and κ < 1 is the
   principled choice rather than an arbitrary one: it is exactly the condition that the entire
   margin range is worth less than one edge, so sparsity can never be overturned. There is no free
   parameter to justify here, and the available headroom is enormous. This is the cleaner result
   and should carry the geometric half of the thesis.
2. *Buy margin with extra edges.* This is the genuine trade-off, it is κ > 1, and there is no
   optimal κ because the answer is application-specific. The deliverable is a **Pareto front**, not
   a number.

**Covering the κ range without one training run per κ.** Condition the policy on κ: sample it per
episode, feed it as a global channel tiled across nodes like the other globals, and train once. The
front is then traced at inference by sweeping κ on a single checkpoint. Potential-based shaping
survives because κ is fixed within an episode, so `phi_κ` is still a per-episode potential. This is
standard preference-conditioned multi-objective RL and it is the difference between a workstation
and a cluster: one run instead of |κ| × seeds. **Validate it** against two specialist runs at the
endpoints (κ = 0, κ = κ_max); a conditioned policy can underfit relative to specialists, and if it
does the whole front is biased inward and the trade-off curve is not trustworthy.

#### WP4 — margin-aware observation (softest mode)

With `v` the singular vector at the rigidity eigenvalue, `Δλ ≈ ‖b_ij v‖²` for adding `(i,j)` — and
the same number is what removing an existing edge would cost. Measured on minimal graphs at n=7:

| domain | log-log correlation | predictor's top pick in the true top-3 |
|---|---|---|
| R^2 | 0.962 | 50% |
| R^3 | 0.925 | 40% |
| SE(3) | 0.874 | 30% |

A strong ranking signal, **not** an oracle — first-order theory breaks down because adding an edge
is not an infinitesimal perturbation and eigenvalues cross. That is the ideal position, and it
pairs an approximate channel with `add_gain`'s exact one.

*What:* `soft_node[i] = ‖v_i‖` (position and orientation parts separately) and
`soft_align[i,j] = ‖b_ij v‖²` over all pairs, each normalized by its own mean. Gate behind a
`rigidity_margin` flag, default off, so existing configs stay byte-identical.

### Tier 3

#### WP8 — baselines and evaluation protocol

phi-greedy is the wrong opponent: it hill-climbs from a random start into exactly the trap phi's
landscape sets — from (rank 19, m 10) in R^3, adding a rank-1 edge scores +2.5 and lands on
(rank 20, m 11), from which no single edit improves anything. Good illustration, bad reference
point. Add `constructive_greedy` with the restart count as an explicit compute knob and report
phi-evaluations used per method; add a **margin-aware greedy** once WP3 lands (ties broken by
`‖b_ij v‖²`) or greedy will look bad on the new objective for the wrong reason; report **final
state** as the headline with best-visited as a second column; **three seeds minimum** and paired
statistics; extend brute force at n ≤ 5 to return the **margin-optimal** minimal graph, which is
what makes `mixed5` worth having.

#### WP9 — multi-n training (parked)

Two decisions to fix now so WP5 and WP6 do not need redoing: pad every sub-env to `n_max` and
thread a boolean node mask through both backbones (`EGNN` already takes a `mask` that
`GNNBackboneEquivariant` never passes; GINE's complete-digraph builder needs to respect it), and
make the action masks mask padded nodes. WP6's constructive arm is what makes a curriculum over
n = 4…32 affordable.

---

## §3 Sequencing

| stage | work | question it answers | go / no-go |
|---|---|---|---|
| 0 | WP1 · WP6 · WP12 · WP11 | is the physics right, are runs cheap, are numbers comparable? | homogeneous rank identical · `rank_K ≤ ΣDOF−3` on every mix · quality unchanged at 1/4 horizon |
| 1 | WP2 · WP5 · WP10 · WP13 | can the policy see and use "which edge helps", in every domain, at a fair width? | `e_ij` probe AUC ≈ 1.0 in SE(3) · no regression at n=8/R^3 · EGNN ≈ GINE at equal width |
| 2a | WP7 phase A on `mixed` | does domain coverage fix cross-domain transfer? | ≥90% rigid on all five homogeneous corners |
| 2b | WP3 · WP4 on n=8/R^3 and `mixed5` | does it find *good* rigid graphs, not just rigid ones? | margin up 5–10× at unchanged minimality · ablation charges for the geometric channels |
| 3 | WP7 phase B, then WP3 × WP7 | one policy across compositions and objectives? | held-out compositions within 10 points of specialists |
| 4 | WP8 | does any of it beat a ten-line algorithm? | beats restart-greedy at matched phi-evaluations for n ≥ 12 |

**2a and 2b are deliberately separate branches of work.** Darvariu et al. (2024) report the same
method generalizing under one objective and collapsing under a closely related one, so changing the
domain distribution and the reward in one run would make a bad result uninterpretable.

---

## §4 Decisions log

| date | decision |
|---|---|
| 2026-08-12 | κ = 0.9 — sparsity first, margin as tie-breaker. Pareto front over κ deferred. |
| 2026-08-12 | The greedy `λ_ref` oracle is in. Fallback (once-per-config offset) documented in WP3. |
| 2026-08-12 | R^2 is a control and a ground-truth debugging case, never a headline result. |
| 2026-08-12 | Constructive (empty-start) stays an arm; the action space and empty-graph start already exist. |
| 2026-08-12 | WP5 defaults to level 2; level 3 (`Q = Δφ̂ + residual`) is an arm. |
| 2026-08-12 | UCT / model-based planning deferred to future work; possibly a baseline after everything else. |
| 2026-08-12 | Sensing range, degree budgets and other geometric limits are out of scope for now. |
| 2026-08-12 | `mixed5` (one agent of each of the five domains, n=5) is the ground-truth verification case. |
| 2026-08-12 | Nothing is committed to a thesis chapter, so WP1 costs only re-runs. |

---

## §5 Work log

Newest first. One entry per work package or per material finding.

### 2026-08-21 — WP10: the EGNN input embedder

The EGNN-vs-GINE comparison was 11 dimensions against 128; it is now 128 against 128. Details and
the measured invariance/scale checks in WP10 above and `DESIGN_NOTES.md#egnn-input-embedder`.

**The finding worth carrying forward is the one that contradicts the WP10 spec.** Equalizing width
costs parameter parity, by 10.9x, because `dim` widens the EGNN's own MLPs and not just the input.
The spec's "~18k parameters" counted the embedder in isolation. There is no setting where both
controls hold, so the backbone comparison has to name which control it ran. This does not change the
decision — width was the defect, and stage 1 says equal width — but it changes what a win would
mean, and it should be stated in the thesis rather than discovered by a reader.

WP10's own text already anticipated the honest outcome being *"equivariance is not exercised by this
objective"*. That is now more likely rather than less: a 10.9x-larger EGNN that still ties GINE
would be strong evidence for it. As recorded there, the real test comes after WP3 puts geometry in
the reward — running the backbone comparison now, on a combinatorial objective, measures the wrong
thing whichever way it lands.

Checkpoint replay verified rather than assumed, on three Equivariant runs plus the GINE headline.
`tools/checkpoint_fingerprint.py` reports `(archived source)` for all four and identical digests, so
a backbone change is exactly the case the archive was built for.

Suite: **578 passed**, 25 skipped, 9 xfailed.

### 2026-08-21 — WP13 code, and the affine q-head

**The GINE q-networks had no nonlinearity in their pair head.** `nn.Sequential(Linear, Linear)`
composes to one affine map, so `DQN_QNetwork_GINE_AddRemoveEdgeDiscreteNoSelfLoops` scored every
candidate pair *linearly* in `[h_i, h_j, adj_ij]`, however wide the head looked. Its sibling
`GINE_AddEdgeDiscreteNoSkipNoSelfLoops` had the same gap; every `Equivariant` model and every
`*_SelectNodesSequentially` model carried the `LeakyReLU`, and so did this file's own `skip_head`,
so it was an outlier rather than a convention. The obsolete `Default` backbone is affected
throughout (13 heads) and is deliberately left alone.

`letsgo_dqn_gine` — every number in §1.0 — was trained with the affine head, confirmed from the
manifest's archived `q_network_architecture`. **This does not invalidate those numbers**; it says
what architecture produced them. It does sharpen WP5: the `[h_i, h_j]` linear probe at 0.955 AUC was
measuring the deployed head's actual ceiling, not a lower bound on what an MLP head could extract,
so the gap WP5 is meant to close was partly this.

`tests/test_models_registry.py::test_live_models_have_no_linear_stacked_on_a_linear` walks every
registered `Equivariant`/`GINE` model and names the offending `Sequential`. Mutation-checked: reverting
the fix fails it with the offending layer pair in the message.

**Reproducibility held, and was verified rather than assumed.** `tools/checkpoint_fingerprint.py`
loaded `letsgo_dqn_gine` and digested its q-values on a fixed observation before and after the edit: the
loader reports `DQN_QNetwork_GINE_AddRemoveEdgeDiscreteNoSelfLoops changed since this run; using the
archived version`, and the q-sum, min/max and argmax are bit-identical (1.995017700195e+02,
argmax 150). The `backbone_source` + `q_network_architecture` archive does exactly what it was built
for.

**WP13 as recorded in its own section:** soft updates at a 800-timestep constant, DDQN as an
`ALGORITHM` arm, and the seeding item withdrawn after measurement.

**What is confounded, and what to do about it.** The head fix and the target time constant both land
in the next run and both change learning dynamics. A single new run against `letsgo_dqn_gine` cannot
attribute a difference to either. Cheapest honest split, in order of value: (1) soft-update alone,
head reverted, is *not* worth a run — the head fix is a defect repair, not an arm; (2) run the fixed
head + soft updates as the new baseline and compare to `letsgo_dqn_gine` as a package, stating it as
a package; (3) only if that package regresses, bisect. DDQN stays off until (2) has a number.

Suite: 564 -> **576 passed**, 24 skipped, 9 xfailed. One new slow test runs the DDQN arm end to end
and asserts the manifest's algorithm, the `models/complete/DDQN/` path and the target time constant.

### 2026-08-15 — first full evaluation of a heterogeneously-trained policy

`letsgo_dqn_gine` (DQN + GINE, 400k steps on `mixed`, informed arm, single seed) evaluated on frozen
benchmarks in distribution and on four homogeneous corners. Numbers in §1.0; what they mean:

**It beats both classical baselines in distribution** — 95% minimal against 80% (greedy) and 50%
(constructive) at n=10 on the five-domain mixture, at the proven lower bound of 17 edges and 266x
fewer rigidity-matrix builds. This is the first configuration where the learned policy beats the
classical algorithm rather than matching it, and it is the heterogeneous case, which is the one WP1
made physically correct.

**Transfer degrades monotonically with agent DOF**, and the mechanism is a clean behavioural switch:
where the policy works it prunes toward `m_req`, where it fails it never enters a pruning phase and
accumulates instead (+0 edges at 3 DOF, +1.5 at 4, +8.2 at 6). On homogeneous `SE(3)` it scores
below a *uniform random policy* on phi while staying rigid everywhere, so the failure is purely
over-density. WP7 phase A's pre-registered prediction is scored in its own section: clause 1 nearly
holds, clause 2 fails by 45 points.

**The ablation's negative result is the robust one.** Destroying any geometric channel costs the
policy nothing in all three modes. The large `degree` / `rigidity_glob` / `add_rank` costs seen
under `--mode zero` do not survive `shuffle` and are out-of-distribution artifacts — zeroing a
normalized degree asserts every node has degree 0. This is now recorded as a methodological rule in
`CLAUDE.md`: never believe a positive from one ablation mode. The negative survives because `zero`
is the aggressive ablation.

**Two of my own earlier claims were corrected in this pass.** The rotation finding was first written
as "a 20-point swing"; the honest statement is that 8 of 20 instances change verdict while the net
direction is noise at that sample size, and the policy had `rotation_augmentation` enabled during
training and still moves. And the README briefly claimed transfer to unseen homogeneous compositions
in general, which the `SE(3)` and `R^3xS^1` runs falsified within the hour; it is now scoped to
comparable agent complexity.

**Tooling.** `constructive` wired into `baselines.py` with a private RNG (drawing from the global
stream changed the instances every *other* method was scored on). `report.py`'s env-name shortener
was returning the literal string `env` for every current-format config, so every recent run
directory and figure title was unlabelled. README figures regenerated from the `mixed` benchmark.

### 2026-08-15 — constructive greedy wired into baselines; the problem is harder than assumed

`baselines.py --methods constructive` (`--restarts K`, default 20). Implementation notes in
`DESIGN_NOTES.md#constructive-baseline`. WP8's *measurement* protocol is still open; this is the
baseline itself.

**The premise this was built on turned out to be wrong, in a useful direction.** The working
assumption was that finding a minimally rigid graph is easy for classical methods, so RL would have
to move to the margin objective (WP3) to have a defensible claim. Measured on `R^3` against the
closed-form optimum:

| | n=8 (`m_req` 10) | n=12 (`m_req` 16) |
|---|---|---|
| 1 restart | 11.25 | 18.67 |
| 5 restarts | 10.75 | 17.33 |
| 20 restarts | 10.50 | 16.33 |
| hit the optimum at 20 restarts | 2 of 4 | 2 of 3 |
| cost per instance | 0.7 s | 4.5 s |

Every instance reported `order matters`, never `matroid`, which is what `c_max = 2` predicts. The
gap to optimum *widens* with n (12.5% at n=8, 16.7% at n=12 for a single restart). So the
combinatorial problem is not solved by the classical algorithm at these sizes, and WP3 is no longer
a prerequisite for having a result — it remains the more distinctive contribution.

Sample sizes are 4 and 3 instances. This is a signal to run the experiment properly, not a result.

**Measured since (2026-08-15):** the paired comparison ran on frozen instances and the learned
policy wins in distribution, 95% minimal against 80% for greedy and 50% for constructive. See §1.0.

**A caveat on the existing resume claim.** In a 3-instance smoke run, `greedy` (phi hill-climbing
from the initial graph) reached 10.00 edges and 100% minimal — against the 50% minimal recorded for
it in the current claim. Three instances is far too small to conclude anything, but it is a reason
to re-measure `greedy` on the frozen benchmark before quoting the 50% figure anywhere.

**One regression caught during verification, worth remembering.** The first version shuffled the
candidate order with `np.random`, which is the stream `reset()` draws instances from. Enabling the
method therefore changed the networks *every other method* was scored on — the `initial` row moved
from 15.33 to 13.00 edges. `greedy` uses no RNG and `random` uses the action space's own seeded
stream, so the instance sequence had been independent of `--methods`; the fix gives `constructive` a
private `default_rng(seed)` and restores that. Verified byte-identical `initial`/`random` rows with
and without the method selected.

### 2026-08-14 — WP2 done

`flex_align` is gone, replaced by two channels derived from `ker(B)` of the **whole** matrix.

**The criterion is exact, not a heuristic.** With `Z` an orthonormal basis of `ker(B)`, adding edge
`i -> j` raises the rank by exactly `rank(b_ij Z)`, because row space and null space are orthogonal
complements (`THEORY.md` §13.1). So `add_gain = ||b_ij Z||/||b_ij||` is zero precisely on the pairs
that contribute nothing, and `add_rank = rank(b_ij Z)/c_max` is the gain itself. Measured against
ground truth (rebuild `B` with the edge added, recompute the rank): **AUC 1.000 with a clean split
in all five domains and three heterogeneous mixes, exact rank on 1,501 pairs**, against
`flex_align`'s 0.634 in `SE(3)` and 0.567 on the `R^2xS^1`+`R^3xS^1` mix. `flex_mag` now comes from
`flex_space(Z, Z_K)` and needs no hand-built trivial modes, since `ker(B_K)` *is* the trivial
variation set by Michieletto Theorem 1 — that closes the `trivial_modes()` item flagged under WP1.
`block_rank` is filled from the complete graph for all pairs, so candidates no longer read 0.

Done as specified except that item 1's "one SVD" is one SVD *plus* one `eigh`, for a measured
reason below.

**Four things went wrong, all worth recording.**

1. **Sign.** `Ē_o` contributes `-y_i`, so the attitude term enters `b_ij Z` with a minus. With a
   plus the AUC was 0.906-0.947 — plausible enough to have shipped, wrong enough to matter. Only
   the exact-rank check caught it, not the AUC.
2. **`ker(B)` is not scale-invariant.** `B`'s position columns carry `1/length` and its attitude
   columns are dimensionless (§12.4 / `THEORY.md` §13.4), so a uniform scaling genuinely moves the
   null space. Fixed by pinning the length unit to the formation's own RMS radius, the same
   normalisation `coord_features` uses. Separately, `rotate_network` did not rotate
   `agent.rotation_axis`, and `P_i = v v^T` is in world coordinates — that broke `R^3xS^1` rotation
   invariance for reasons unrelated to these features.
3. **The rank threshold was below the noise floor.** The first cut was `1e-18` relative, which for
   a Gram matrix in double precision is *inside* the rounding error, so `add_rank` flipped by a
   whole rank unit whenever the geometry was translated or scaled. Measured separation: pairs that
   add nothing reach `add_gain` at most `1.59e-10`, pairs that add rank at least `1.43e-02`. The cut
   is now `1e-6`, in the middle of eight empty orders of magnitude.
4. **Normalising against the spread amplified noise.** On a rigid framework every raw gain is at
   machine zero; dividing by their RMS turned rounding error into an O(1) feature. Normalisation is
   per pair now, by that pair's own `||b_ij||`, which also bounds the channel to `[0, 1]`.

**Cost.** The whole rigidity block is 0.63 ms at n=8/`R^3` and 2.50 ms at n=16/`SE(3)`,
single-threaded. As a fraction of the step, the widest arm is +25% at n=8 and +21% at n=16 over no
rigidity features at all — and nearly all of that is `{global}`, since the null-space channels reuse
a decomposition the state score already needs.

Two decisions bought this. `nullspace` takes `eigh(B^T B)` rather than an SVD of `B`, whose left
factor is (3m, 3m) and never used: 13.15 -> 2.50 ms at n=16. The rank is **not** taken from `eigh`,
though — squaring halves the precision of the eigenvalues and thresholding them disagreed with
`matrix_rank` on 840 of 840 cases, so the rank still comes off the thin SVD and only the
eigenvectors come from `eigh`. And `candidate_gain` reads norm and rank off the 3x3 Gram matrix
`b_ij Z (b_ij Z)^T` rather than a batched SVD of the (3, k) blocks: 1.77 -> 0.59 ms.

Profile this pinned to one BLAS thread. Unpinned, the same 144x144 `eigh` timed anywhere from 0.26
to 16 ms on identical input, which is contention, not the algorithm — worth stating because it is
the kind of number that silently justifies a wrong optimisation.

**Verification.** Invariance holds to 1e-13 under translation, scaling and rotation in every domain
and on the mix, with the single expected exception of the `R^d` global-frame bearing artefact
(§2.3), which is not these channels. Full suite 527 passed. Six new tests in `tests/test_flex.py`
pin the addition criterion against ground truth per domain, `nullspace` against an SVD basis,
`flex_space`'s dimension, `rigidity_decomposition` against `matrix_rank` and `rigidity_eigenvalue`,
and scale invariance.

**Honest framing, restated.** `add_gain` makes the informed arm approximately
constructive-greedy-with-learned-ordering, and now it does so *exactly* rather than approximately.
That raises the bar on the framing rather than lowering it: the arm is only interesting because a
20-restart constructive greedy reaches the optimum 65% of the time at n=8/`R^3` and 20% at n=12, so
a perfect one-step oracle is demonstrably not sufficient. The uninformed arm stays the headline, and
WP8 has to actually measure that greedy baseline before the comparison can be made.

**Readable form kept as the oracle** (added 2026-08-15, after the `CLAUDE.md` instruction on
readability). `candidate_gain_reference` loops over pairs and builds `b_ij` by calling
`extended_bearing_rigidity_matrix` on a one-edge network, so it states (13.1) as written and cannot
drift from the construction it checks. `candidate_gain` stays as the hand-expanded version, ~3x
faster, and a test pins it to the reference in all five domains and the mix. The 1.2 ms/step this
buys is not a bottleneck by the standard now recorded in `CLAUDE.md`; it is kept because the pairing
costs nothing and turns the derivation into something executable.

Worth noting what the pairing catches: flipping the attitude sign in the fast version fails the
comparison in `R^2xS^1`, `R^3xS^1` and `SE(3)` and **passes** in `R^2`/`R^3`, because `P_i = 0` there
and the term does not exist. Every configuration trained to date is homogeneous `R^d`, so this class
of bug is structurally invisible to the experiments actually being run.

**Not carried over.** `flex_tensor` / `flex_constraint_power` stay in `rigidity.py`, tested, as the
reference implementation `THEORY.md` §10's ground-truth check runs against. The environment no
longer calls them. `THEORY.md` §9 is marked superseded rather than deleted, since the derivation
explains why a position-only construction fails in the oriented domains.

### 2026-08-14 — tools/ started

`tools/` now collects scripts worth re-running, per the new standing instruction in `CLAUDE.md`
(tests go to `tests/`, document-specific scripts stay in `docs/`). Seeded with three reconstructed
from this branch's scratch work:

- `constructive_greedy.py` — the restart greedy baseline WP8 needs. Reproduces the matroid split:
  R^2 and R^2xS^1 terminate at `m_req` on every restart, the `c_max = 2` domains do not.
- `env_report.py` — switches, observation layout with per-channel statistics, episode constants and
  cost for any config. Warns when the layout has drifted from `build_dict_obs`.
- `compare_runs.py` — tail averages and per-fifth trajectories across runs, reading `runs/` directly.

### 2026-08-14 — rigidity matrix re-audited; note and verification finished

Full independent audit of `extended_bearing_rigidity_matrix` across all five domains, homogeneous
and heterogeneous, no code changes. It is correct. What the audit produced:

| check | result |
|---|---|
| `B δ` equals the central-difference Jacobian | 432 frameworks, worst rel err **1.1e-9** |
| `B(I − AAᵀ) = 0` (vanishes on the virtual subspace) | **2.2e-16** |
| homogeneous `rank_K` vs closed forms | exact, all 5 domains, n = 3…16 |
| heterogeneous `rank_K ≤ Σ dim D_i − trivial` | holds, tight in 7 of 8 mixes |
| trivial motions | exactly the predicted set per domain, nothing extra in the kernel |
| similarity invariance of rank | holds |
| block structure | position block touches only `{i,j}`, attitude block only `i` |
| Table I equivalence, homogeneous | bitwise, 25 graphs per domain |

Two findings, neither a defect in the matrix:

- **`bearing_DOFs` is only a faithful Table I reference at `v = e₃`.** It uses the row form `e₃vᵀ`
  for `R^3xS^1`; Table I has the column form `[0_{3x2} v]`; the projector is `v vᵀ`. All three
  coincide at `e₃`, the only axis in use, so nothing measured is affected. Caught because building
  the edge-indexed form from it gave a relative error of **1.8** against the Jacobian. Recorded in
  `THEORY.md` §12.4 and `CLAUDE.md`.
- **"Null columns" is the wrong way to count `q_v` in general.** The invariant statement is
  `B(I − AAᵀ) = 0`; the column count is equivalent only when the admissible subspace is coordinate
  aligned, which fails for `R^3xS^1` with a generic axis under the projector parametrisation. The
  note now states (R2) only as the identity.

A knock-on: the two forms use *different coordinates* for the `R^3xS^1` rotational freedom (Table I
stores θ̇ in slot 3; the projector keeps ω in `span{v}`), so comparing them against one Jacobian
needs matched bases.

**Verification consolidated.** Seven scratch scripts replaced by `docs/verify_dof_restriction.py`,
sectioned to match the note's environments, 10/10 checks, ~11 s (`--quick` 0.6 s). It was not
reproducible at first: `random_scenario` draws poses from the global numpy stream, which was never
seeded, so figures drifted between runs. Both streams are seeded now and three consecutive runs are
identical.

`docs/verify_dof_restriction_2.py` (independent, shares no code, re-derives both forms from the
paper) agrees on every qualitative claim and contributed a sharpening now in the note: when
`rank(S_i − S_j) = 2`, which happens if agents are confined to *different* planes, the obstruction
cannot be met at any configuration rather than merely generically. For the five manifolds of Table I
the rank is at most 1, so there the failure stays generic.

Numbers in the note updated to what the script prints (worst rel err 1.1e-9; identity violated
517/1200; rank condition wrong 81/1200 = 6.8%). The earlier 7.7% predated the Table I fix.

### 2026-08-13 — WP1 claim audited and written up as a proof

Re-read Michieletto et al. in full and re-derived the WP1 claim from scratch, because "a published
IEEE paper has an error" needs to be right. It holds, and it is sharper than first recorded.

**What is actually wrong.** Not Table III's entries: the *form* of eq. (10). In `B_p = D_p U Ēᵀ`
the same `U_ij` multiplies both endpoints' column blocks. Faithfulness plus annihilation forces
`D_p,k (S_i − S_j) = 0`, i.e. `p̂_ij = ±e₃` for a planar/spatial pair. Generically impossible, so
*no* per-edge factor can encode two different endpoint restrictions.

**Why it matters.** Definition 13 only pins the matrix down on the admissible subspace (δ⁺ is
zero-padded, so inadmissible columns are multiplied by zero). Both constructions therefore satisfy
Definition 13 and both reproduce the true Jacobian on that subspace to 1.2e-10. They differ off it,
and Theorem 2's proof reads the columns off it — it needs `q_v` to cancel between `G` and `K`.
Under the per-edge form `q_v` moves with the graph, so it does not cancel.

**Measured.** Rank test disagrees with true IBR on **81/1200 (6.8%)** of random heterogeneous
frameworks; the rank identity `rk = 6n − q_v − q_i` is violated on 517/1200. Under the per-node form:
0/1200 and 0/1200. An independent reimplementation sharing no code
(`docs/verify_dof_restriction_2.py`) gets 116/1197 and 492/1197 on its own sample, and 0/1197 for
the per-node form. Explicit 4-agent counterexample (3× R^2×S^1 + 1× SE(3), 6 edges): per-edge says
rk 11 vs 13, "not IBR"; ground truth says IBR.

**Two corrections to our own earlier account:**
- `q_v` must be defined **structurally** as `6n − Σ dim D_i`, not as "the number of null columns"
  (the paper's phrasing). An isolated agent contributes null columns for its *admissible*
  coordinates too, and those belong in `q_i`. With the structural definition the identity holds
  identically for the per-node form.
- The `R^3xS^1` rotational entry `[0_{3×2} v]` is **not** an error: it is a different, valid
  coordinatisation of the same 1-D freedom (θ̇ stored in the third slot), with the same rank and the
  same `q_v`. Only the *translational* block carries the defect. Our repo's earlier row-form
  `e₃vᵀ` was a genuine third variant and was wrong; that part of the WP1 note stands.

Also caught: reproducing the paper's Section VI-B gives rk = 13 with 6 null columns, matching the
values printed there, which confirms our implementation of Table III is faithful and the
disagreement is in the construction rather than in our code.

Written up with full proofs in `docs/dof_restriction_note.tex`;
`docs/verify_dof_restriction.py` reproduces every number.

### 2026-08-13 — README rewritten for a research audience; docs de-duplicated

`README.md` rewritten as a research description (451 → 206 lines): problem, research questions,
approach, current state, evaluation figures. Citations with DOIs at the top, since
`resources/papers/` is no longer tracked. Implementation detail removed (it was duplicated in
`CLAUDE.md` and `THEORY.md` anyway).

**Documentation policy, now recorded in `CLAUDE.md`:** measured results live in exactly one place,
this file §1. Nothing else carries policy numbers; superseded numbers get deleted rather than
archived, because git history is the archive and a stale number in a file that loads every session
costs more than it is worth. Applied by cutting `CLAUDE.md`'s 70-line historical results section and
the checkpoint numbers that had leaked into its other sections.

Swept all five docs for dead file references and fixed: `gpu_environment.py` / `gpu_network.py` /
`gpu_rigidity.py` (deleted from the repo, still referenced in three places), `dummy/test_mbr.py`,
and a wrong path for `tests/test_environment_api.py`. Every `DOC.md#anchor` cross-reference now
resolves.

**README figures regenerated.** The four in `resources/` were from a PPO run on the legacy
`Weighted` objective, dated 2026-08-07, and contradicted the README text. Regenerated from
`generaldqngine` on `bench_n8_R3` so figures and prose agree, and converted SVG → PNG.

One bug found doing it: `report.py`'s figure header hardcoded "N random networks · seed S" even
under `--benchmark`, so a figure that outlives its run would misstate its own provenance. It now
names the benchmark.

### 2026-08-12 — arms turned off by default, config ownership, a flawed test fixed

The generator now emits the **validated baseline**: `4*m_req + 10` horizon, stop action off
(`skip_enabled`/`skip_is_stop` false, `time_penalty_value` 0), `rotation_augmentation` false. Both
new behaviours stay switchable env keys, so either can be trained as an arm without a source edit.
`environments/` and `scenarios/` regenerated once to match, and are **the user's to manage from here
— do not regenerate or hand-edit them; make a new config under a new name if an experiment needs
different settings** (recorded in `CLAUDE.md`).

`benchmarks/` refreshed against the restored configs. The WP11 pair re-measures identically
(95% unrotated / 90% rotated), so that number does not depend on the arms being on.

**A pre-existing test was wrong and is fixed.**
`test_greedy_is_at_least_as_good_as_random_and_far_cheaper` ran `run_random` on the graph
`run_greedy` had just optimised, with no restore between them — so random could only improve on
greedy's answer, and the assertion held only while random failed to find the two-edit swap greedy
cannot. It now deep-copies the instance between methods, which is what `baselines.py` does and what
`CLAUDE.md` already required. Unrelated to WP1/6/11/12; it surfaced because the comparison finally
got exercised from a different starting graph.

### 2026-08-12 — stop-action experiment, cleanup

Four A/B/C/D runs and their artefacts (`train/run_stop*`, `models/complete/DQN/run_stop*`,
`runs/run_stop*`, `environments/env_stop*`, `runs_baselines/*stopeval*`) removed after the
measurement below; the finding lives in `DESIGN_NOTES.md#horizon`. Every `environments/*rigGFE*`
config regenerated so none is stale (all now carry the `4*m_req+10` horizon and the
`rotation_augmentation` key) — that includes the `SelectNodesSequentially` and non-`_lean` variants.

Two fixes that came out of it:

- **`benchmark.digest()` hashed the file, not the instances.** npz is a zip and stores timestamps,
  so an identical benchmark rewritten got a different digest — exactly the false mismatch the digest
  exists to detect. It now hashes the arrays; verified stable across a rewrite.
- **`benchmark.py rotate <source> <name>`** added, so the rotated pair behind the WP11 number is
  reproducible from the CLI instead of a scratch script.

### 2026-08-12 — WP6, WP11, WP12 done (tier 0 complete)

**WP6 — horizon.** `MAX_STEPS = 4*m_req + 10` in the config generator (n=8/R^3 → 50, `mixed5` → 42,
`mixed` → 78, n=16/R^3 → 98, n=8/SE(3) → 94). The stop action (`skip_enabled` + `skip_is_stop` +
`time_penalty_value`) is a trainable **arm, off in generated configs**. Also fixed: the
scenario branch of the generator took `config["domains"][0]`, writing a homogeneous label into every
mixed config; it now carries the full per-agent list, which `MAX_STEPS` needs anyway.

| | instances seen in 400k | in the replay buffer |
|---|---|---|
| n=8 R^3 | 1785 → **8000** | 178 → **800** |
| `mixed` | 1111 → **5128** | 111 → **512** |
| n=16 R^3 | 416 → **4081** | 41 → **408** |

**Acceptance met, measured:** `generaldqngine` re-evaluated at 50 steps reproduces its 224-step
result exactly — 10.05 edges, 100% rigid, 95% minimal, best@ 8.0. (`random` drops 85% → 60% rigid,
correctly: it was living off the longer search budget.) Reset cost amortises to 0.06–0.23 ms/step
against a 1.2–8.5 ms step, so the extra resets are 2–3%.

**The stop action: measured, and kept as an arm.** Four 150k-step runs at n=8/R^3 differing only in
`skip_enabled` / `skip_is_stop` / `time_penalty_value`, argmax on `bench_n8_R3`:

| arm | tp | stops? | steps | **final** min% | best-visited min% |
|---|---|---|---|---|---|
| A no stop | -- | no | 50 | 55% | **95%** |
| B stop | 0.05 | yes | 7.7 | **85%** | 85% |
| C stop | 0.20 | yes | 7.5 | 70% | 75% |
| D stop | 0.01 | yes | 7.0 | 50% | 50% |
| greedy | -- | -- | 6.2 | 50% | 50% |

- **It does not collapse.** `Q(s, stop) = -c` exactly, so it is a constant and trivially learnable —
  the risk was that a guaranteed value beats a badly estimated one early on. It does not, because
  initial graphs are far from optimal and improving actions have clearly positive `d phi` from the
  start. Episode length settles at ~7, `Episode/ Terminated` reaches 1.00, and the policy stops *on*
  its best graph (`Best-final score gap` 1.79 → 0.04).
- **The two columns disagree.** As a deployed policy (final state) the stop arm is the best measured
  — 85% vs 55% no-stop and 50% greedy, at 6.5x fewer edits. As a search (best-visited) no-stop wins
  95% vs 85%, but that costs 50 edits and takes the max over the trajectory.
- **Not resolved.** tp 0.01 → 50%, 0.05 → 85%, 0.20 → 70% is non-monotone while D stops at the same
  ~7 steps as B, so this is most likely seed noise. One seed per arm cannot separate them.

**Decision (2026-08-12): keep both terminations as trainable arms**, selected by the env keys, with
the generator emitting the no-stop baseline. No further training spent on resolving it now; revisit
with seeds when WP8's protocol lands. To reproduce the arms, set `SKIP_ENABLED`/`SKIP_IS_STOP` true
and `TIME_PENALTY_VALUE` in `environment.py`'s `__main__` and generate a config under a new name —
the `env_stop*` configs used for the measurement were deleted with the runs.

**Side finding, and it matters for every later number:** a single seed at n=8/R^3 spans **at least
35 points of minimality**. The historical "95% minimal" headline is a single-seed number too.
WP8's three-seed protocol is not optional.

**A trap:** the TensorBoard averages make the stop arms look far worse than argmax evaluation does
(`Best is min rigid` 0.97 vs 0.57), because training still carries epsilon = 0.05 — 2.5 random edits
over 50 steps (absorbed) versus 0.35 over ~7 steps, one of which can be *stop*. Judge terminations
on an argmax evaluation, never on the curves.

**WP11 — rotation augmentation.** *(Effect size measured 2026-08-12, and it is small — see below.)*
`rotation_augmentation` env key, **default `False` everywhere** — an arm, like the stop action, so
archived runs replay unchanged. `Environment.randomly_rotate()`
in `reset()`, z-axis only when a planar agent is present. 10 new tests in `tests/test_environment_api.py`
(rank/`m_req`/edges invariant, planar agents stay at z=0 exactly, the network actually moves).

**How much it is worth, measured.** `bench_n8_R3` and `bench_n8_R3_rot` are the same 20 instances,
one set given a random global rotation (rank and edge sets verified identical, 0/20 changed).
Through `baselines.py`, `generaldqngine` scores **95% minimal / 10.05 edges** unrotated and
**90% / 10.10** rotated — one instance out of twenty, i.e. **below the resolution of a 20-instance
evaluation**. `initial` and `greedy` are bit-identical across the pair, confirming the pairing.

So WP11 is justified on principle (the observation genuinely is not invariant while the task is,
and the augmentation is free) but it is **not** a large effect at n=8/R^3. Do not claim otherwise.
A scratch rollout script initially suggested a 30-point drop; it disagreed with `baselines.py` on
the same instances and was wrong. Lesson recorded: **measure through `baselines.py`**, which is the
path every reported number uses, rather than through an ad-hoc rollout.

While there: `random_scenario` now carries `rotation_axes` the way it already carried `domains`, and
`Environment` captures `self.rotation_axes` at init. `set_domain` resets the axis to `e3`, so an
`R^dxS^1` agent with a scenario-specified axis silently lost it on every reset — harmless while `e3`
is the only axis in use, but WP1 made the maths correct for arbitrary axes and the environment could
not produce one. Found by a test, not by reading.

**WP12 — frozen benchmarks.** New `benchmark.py`: `save`/`load`/`digest`/`available`, CLI
`uv run benchmark.py <env> <name> [--instances N] [--seed S]` and `list`. `baselines.py --benchmark
<name>` evaluates on the stored set and records the name + digest in `meta.json`. Verified faithful:
`--benchmark bench_n8_R3` reproduces the sampled seed-0 run exactly (initial 11.20±5.33, greedy
10.50 / 100% / 50%). Created `bench_n8_R3`, `bench_mixed5`, `bench_mixed` (20 instances each, 32 KB
total). **`benchmarks/` is tracked on purpose** — a fixture, not an output.

Suite: 508 → **527 passed**, 28 skipped, 9 xfailed.

Docs: `DESIGN_NOTES.md#horizon`, `#rotation-augmentation`, `#benchmarks`; `CLAUDE.md` commands,
config keys, `skip_enabled`, gitignore section.

### 2026-08-12 — WP1 done

`rigidity.py`: new `node_dof_projectors(agent)` → `(S_i, P_i)`;
`extended_bearing_rigidity_matrix` now builds `[Dp Ēᵀ S̄ | Da Ē_oᵀ P̄]`. `bearing_DOFs` retained,
unused by the matrix, as the Table I reference the equivalence test compares against.

Verified:

| check | result |
|---|---|
| homogeneous output identical to the old construction | max abs diff **0.0**, 60 graphs × 5 domains |
| `B δ` equals the central-difference Jacobian | rel err ≤ 1.1e-9 over 432 frameworks, 5 domains + 16 mixes, random rotation axes |
| inadmissible variations annihilated | `max ‖B @ inadmissible‖ = 0.0`; admissible dim = Σ dim D_i |
| `rank_K ≤ Σ dim D_i − trivial` | holds for all 8 mixes (was violated: 36 vs 33, 14 vs 13) |
| existing suite | 472 passed, unchanged |
| mutation check (remove the restriction) | **23 tests fail** — the new tests discriminate |

Effect on the configurations in use:

| config | `rank_K` | `c_max` | `m_req` | phi\* |
|---|---|---|---|---|
| `mixed5` | 15 | 2 | 8 | 73.33 |
| `mixed` | 33 (was 36) | 2 | 17 (was 18) | 74.24 |
| n=8 R^3, n=16 R^3, n=8 SE(3), n=4 R^2 | unchanged | | | 75.00 / 75.00 / 74.39 / 75.00 |

Side benefit: dropping the two `(3m, 3m)` dense `U`/`V` allocations makes `B` **1.3× faster at
n=8, 2.2× at n=16, 6.1× at n=32** on the complete graph.

New tests in `tests/test_rigidity_matrix.py`:
`test_matrix_is_the_numerical_jacobian_of_the_bearings`, `test_rank_K_respects_the_dof_budget`,
`test_infeasible_coordinates_are_zero_columns`,
`test_matches_michieletto_table_I_on_homogeneous_networks`,
`test_rotation_axis_is_a_projector_not_a_row`. `tests/conftest.py` gains `DOF_PER_AGENT`, `MIXES`
and `max_rank_K()`. The stale `R^3xS^1` xfail in
`test_homogeneous_string_domain_accepts_every_domain` is removed — the branch it claimed was
commented out is present, and `pytest.xfail()` short-circuits so it could never have reported the
fix.

Docs updated in the same change: `THEORY.md` §2 (assembly), §4 (`c_k` for mixes) and new §12;
`DESIGN_NOTES.md#per-node-dof`; `CLAUDE.md` rigidity-core bullets and known issues 1 and 4.

**Still open, deliberately deferred to WP2** (*closed there on 2026-08-14*): `trivial_modes()`
hardcodes three translations plus scaling and is wrong for mixes. The correct replacement is an
orthonormal basis of `ker(B_K)`, which changes its shape from `R^{3n}` to `R^{6n}` and only makes
sense with the full-null-space flex rework. WP2's `flex_space(Z, Z_K)` does exactly this;
`trivial_modes()` survives only for the `THEORY.md` §10 reference check.

### 2026-08-12 — branch created

- Branch `margin-and-heterogeneity` off `formulation-overhaul` @ `cb074cd`.
- `ROADMAP.md` replaced with this plan; the phase 1–6 record is condensed into appendix B.
- `CLAUDE.md`: added the never-commit and keep-docs-current standing instructions.

---

## Scenarios in use

| name | n | composition | purpose |
|---|---|---|---|
| `mixed5` | 5 | one of each of the five domains | ground-truth debug case; brute-forceable (`m_req` 8, 125,970 subsets ≈ 2 min) and inside `MAX_BRUTE_FORCE_N = 5` |
| `mixed` | 10 | two of each of the five domains | the heterogeneous experiment; all 25 ordered domain pairs every episode; 5.88 ms/step, 0.65 h per 400k steps |

Both use `only_randomize_edges = False`, so poses and edges are redrawn each episode and only the
domain mix carries over — which is what makes them generalization experiments rather than case
studies. Two caveats recorded so they are not rediscovered:

- The **saved** poses put planar agents off-plane (`set_domain` ran after the positions were
  drawn). Harmless while `only_randomize_edges = False`, because `reset()` redraws poses and
  `Agent.randomize_position` respects the domain — but it silently becomes a wrong geometry if that
  flag is ever flipped for a case-study figure.
- `"domains"` in the env config is vestigial when a scenario is set (the scenario wins in
  `initialize()`); `mixed`'s config says `"R^2"`, which is misleading but has no effect.

---

## Appendix A — feasibility of a distributed formulation

Context: the long-term motivation is a *distributed* method for maintaining rigid formations in
swarms. The current centralized formulation is a deliberate first step. Not being pursued now, but
the design should not foreclose it. Assessment:

**Feasible, with one genuine limitation.**

What blocks a naive distributed version:
1. Rigidity is a global algebraic property. `rank(B)` cannot be computed or certified locally, so
   no local decision rule can *verify* rigidity. Minimality is worse — it is global by definition.
2. The current policy is centralized in all three ways that matter: the EGNN is dense all-pairs
   (every agent would need every other agent's pose), the action is a global pointer over all
   nodes, and the centrality features are global.

What makes it tractable anyway:
1. **CTDE (centralized training, decentralized execution)** fits this problem almost exactly.
   Train the critic on the full graph; restrict the *actor* to `K` rounds of message passing over
   a communication graph and give it only locally computable features. The actor is then literally
   a `K`-hop local rule, executable by each agent from information within `K` hops.
2. **Factored per-node actions.** Replace the global pointer with a per-node head: each agent
   emits a distribution over its own candidate out-neighbours (or a keep/drop bit per candidate
   out-edge), all executed simultaneously. Parameter-shared across nodes, so it is one network
   regardless of `n` — which is the same property the current GNN already has, and the same
   property the "one policy, any n" target requires.
3. **Constructive / Henneberg-style protocols.** In homogeneous `R^d`, minimally bearing-rigid
   graphs admit a vertex-addition construction where each new agent attaches with `d` edges —
   which is exactly what `MBR_required_Rd(n, d) = 1 + k*d + r + sgn` counts. A distributed
   *constructive* protocol built on that reaches minimality **by construction**, without any agent
   ever computing a global rank. This is the same thing as open question 3 (constructive vs
   editing) and is the most promising bridge.

The limitation: a distributed policy will at best achieve **rigidity plus local minimality**.
Certified global minimality requires global information. That is a real result to state, not a
failure.

### A.1 Feature availability tiers

Not every "global" feature is global in the same way, and the distinction decides what a
distributed version could reuse. Three tiers:

**Tier 1 — locally available unconditionally.** Own pose and domain; own in/out degree; the
bearings the agent is *currently measuring*; whatever one-hop neighbours communicate.

**Tier 2 — available only under a sensing-radius assumption.** *Bearings to agents the agent is
not currently measuring* — i.e. the geometry of **candidate** edges. This is what the all-pairs
bearing observation adds, and it is worth being explicit that **it is not free information**: an
agent does not know its bearing to another node before measuring it. There is no way for a purely
local decision maker to evaluate "would adding `i -> j` help?" without first obtaining `p_hat_ij`.

Whether it is admissible in a distributed setting therefore depends on a modelling choice that has
not been made yet, and should be made deliberately:

- If **detection is cheap and maintenance is expensive** — e.g. omnidirectional vision gives
  bearings to every agent within radius `R` essentially for free, while an *edge* denotes a
  persistent, tracked, communicated measurement link with a real cost — then all-pairs bearings
  within `R` are a legitimate local observation, and the current observation carries over unchanged
  (with the all-pairs set truncated to the sensing radius).
- If **any measurement costs what an edge costs**, candidate bearings are genuinely unavailable
  and a distributed protocol needs either an explicit exploration phase, a construction phase in
  which agents acquire bearings as they join, or a policy that reasons from communicated
  *positions* rather than measured bearings.

This is the single most important open modelling question for the distributed direction, and the
current centralized work does not depend on resolving it.

**Tier 3 — not available at all.** `rank(B)`, rank deficit, per-edge block rank `c_k`, `is_IBR`,
the null-space features of WP2/WP4, and the graph centralities. Global by construction, too
expensive to compute and to communicate.

### A.2 What this buys the current ablation

The tier-2 / tier-3 split is what the observation arms measure, so the ablation prices
decentralization in two separable steps before committing to any of it:

- **informed minus geometry-only** = the cost of losing tier 3 (rigidity algebra).
- **geometry-only minus a tier-1-only variant** = the cost of losing tier 2 (candidate-edge
  geometry), under the pessimistic measurement model.

The second comparison is worth adding as a third arm later if the distributed direction is
revived; it is not needed now, but `include_candidate_bearings` already makes it a config flag
rather than a rewrite.

Concrete things to preserve so this stays open:
- keep GNN depth an explicit constructor argument (already true, `num_layers`);
- keep per-node / per-edge action heads as a first-class option rather than flattening to a global
  index;
- keep the observation builders tagged by tier, and keep candidate-bearing inclusion switchable.

---

## Appendix B — the phase 1–6 overhaul, completed on `formulation-overhaul`

Condensed record of the previous roadmap. Full text in git history at `cb074cd:ROADMAP.md`.

| phase | what | outcome |
|---|---|---|
| 1 | PPO discount factor and memory sizing | **done, confirmed.** `memory_size == cfg.rollouts`; γ = 1.0 → 0.99. With purely potential-based shaping and γ=1 the advantage collapses to ≈0 under a near-uniform policy, which is what froze PPO's entropy at ~1.9 of ~2.0 nats. |
| 2 | dimension-normalized state score | **done.** `WeightedNormalized` = `(100·rank − 25·m·c_max)/rank_K`; optimum ≈ 75 at any n and domain. The old `Weighted` made R^3 three times more eager to add than to prune while R^2 was neutral. |
| 3 | all-pairs bearing observation | **done.** Bearings for every ordered pair plus an explicit `edge_exists` channel. Large transfer gain at the time (n=16 edge count 50.80 → 32.95, rigid 80% → 90%). |
| 4 | rigidity-feature observation arms | **done.** `rigidity_global` / `rigidity_flex` / `rigidity_edge`, default off. Note `c_k` is constant in every homogeneous domain, which is why it has its own flag — and why `mixed` is the first configuration where it carries information. |
| 5 | pairwise pointer head | **not done** — superseded by and folded into WP5. |
| 6 | correctness and cost cleanups | **partly done.** `fully_connected()` diagonal fixed. Still open and now folded into WP2/WP6/WP13: gating `is_MBR`/`rigidity_eigenvalue` behind the shared SVD, the shorter horizon, the dead code in `action_SelectNodesSequentially`, the broken `is_MBR_Rd` branch, and the seeding order. |

**What phases 1–4 did not fix, and this branch exists to address:** cross-domain transfer stayed
below random (5% rigid on n=8/SE(3) against 35% for the untouched initial graph), because the
reward contains no geometry (§1.1), the heterogeneous physics is wrong (§1.2), and a
single-domain-trained policy cannot use the domain channel at all (§1.5).
