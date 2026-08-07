# Roadmap — formulation overhaul

Branch: `formulation-overhaul`. Started 2026-08-07.

This document is the working plan and the record of *why* each change is being made. It is
tracked in git so it can be reviewed at any point. `CLAUDE.md` describes what the code **is**;
this file describes what is **wrong with it and what is being done about it**.

---

## 1. Where the project actually stands

### 1.1 The formulation works at n=8 / R^3

`bigDQN8SelectEquivariant3e-4lrNormalizedPositions` (DQN, `SelectNodesSequentially`, EGNN,
`Weighted` score, 600k trainer steps x 4 envs) converges to essentially the optimum. Training
curves in eighths:

| metric | start | end | optimum |
|---|---|---|---|
| Best state score | 257.0 | **299.8** | 300 |
| Best nr edges | 13.75 | **10.02** | 10 |
| Best is rigid | 0.795 | **1.000** | 1 |
| Best is min rigid | 0.013 | **0.982** | 1 |
| Final nr edges | 23.45 | 10.94 | 10 |

`Final` tracking `Best` matters: the policy converges *and* roughly holds, despite having no skip
action. In `baselines.py` it reaches its best graph at step ~15.5, i.e. about 8 edge toggles —
against `greedy`'s 11 hill-climbing steps, each of which costs `n(n-1)` full phi evaluations.

**This is the result the thesis currently rests on, and it is a good one.**

### 1.2 PPO is broken, for two identifiable reasons

`bigPPOSelectEquivariant3e-4lrNormalizedPositions` (same environment) learns nothing in 127k
steps: Best score 251 -> 248, Best is rigid 0.64 -> 0.72, Best is min rigid ~1%. Entropy loss sits
at -0.017..-0.019, i.e. entropy ~1.7-1.9 nats against a ceiling of ~2.0 (8 nodes on the first
pick, 7 on the second, skip masked). The policy never leaves uniform. Policy loss ~= -0.001.

**(a) `memory_size != cfg.rollouts`.** At commit `809f13a`:

```python
MEM_SIZE = 2048 * 4          # 8192
cfg.rollouts = 1024
memory = RandomMemory(memory_size=MEM_SIZE, num_envs=env.num_envs, device=device)
```

skrl's `PPO.update()` runs `compute_gae` over the whole `memory.get_tensor_by_name("rewards")`
ring and then `memory.sample(batch_size=len(self.memory))`. With `rollouts=1024` every update
trains on 8192 steps of which **7/8 are up to eight rollouts stale**, and `last_values` bootstraps
at the ring's wrap point rather than at the true trajectory end. Stale samples land outside the
`[0.8, 1.2]` ratio band, get clipped, and contribute no gradient. Every commit before `809f13a`
had `cfg.rollouts = MEM_SIZE`; the mismatch was introduced in that one commit, which is exactly
the commit this run was trained at.

**(b) `discount_factor = 1.0` makes the MDP degenerate.** With gamma=1, `time_penalty=0`, no stop
action and a purely potential-based reward, the episode return is

    sum_t ( phi(s_{t+1}) - phi(s_t) )  =  phi(s_200) - phi(s_0)

so the advantage is

    A^pi(s,a) = E[ phi(s_200) | s' ] - E[ phi(s_200) | s ]

Under a near-uniform policy the 200-step random walk over edge sets mixes and forgets `s`, so
**A ~= 0 everywhere at initialisation**. There is no gradient to bootstrap from. This is not a
tuning problem — it is the reason PPO cannot start.

DQN escapes it because it uses **gamma = 0.99**, and that is not incidental. With
`r_t = phi_{t+1} - phi_t` and gamma < 1, Abel summation gives

    sum_t gamma^t r_t  =  -phi_0 + (1 - gamma) * sum_{t>=1} gamma^{t-1} phi_t

so **gamma < 1 silently converts potential-based shaping into "maximise the discounted average of
phi along the trajectory"** — get good fast and stay good. That is precisely why DQN's final edge
count tracks its best edge count while PPO's does not (PPO's `Best-final score gap` ~= 100, about
ten spurious edges).

### 1.3 Generalization is where it actually fails

Both trained policies collapse off-distribution. `baselines.py`, 20 instances, seed 0:

| model | env | edges | rigid | minimal | required |
|---|---|---|---|---|---|
| DQN(n8/R^3) | n8/R^3 | 12.95 | 100% | 75% | 10 |
| DQN(n8/R^3) | **n4/R^2** | 4.20 | **45%** | 40% | 5 |
| DQN(n8/R^3) | **n16/R^3** | 53.9 | **65%** | **0%** | 22 |
| PPO(old, n8/R^3) | n8/R^3 | 10.60 | 100% | 70% | 10 |
| PPO(old, n8/R^3) | **n4/R^2** | 4.00 | **35%** | 35% | 5 |
| *random* | n4/R^2 | 5.50 | *80%* | 25% | |
| *random* | n16/R^3 | 54.3 | *60%* | 0% | |

At n=4/R^2 the learned policy is **worse than uniform random** (45% vs 80% rigid): it deletes down
to ~4 edges when 5 are required. At n=16 it is indistinguishable from random. The policy learned
*an edge-count prior for n=8 / R^3*, not a rigidity criterion — which is exactly what you would
expect given sections 2.2 and 2.4 below.

**Target claim (decided): one policy, any n, any domain mix.** That makes this the blocking
problem, not a nice-to-have.

---

## 2. Conceptual errors, ranked

### 2.1 `adj_mat` is a no-op in the EGNN backbone — verified

```
adj all-zeros vs all-ones identical: True     max abs diff: 0.0
```

In `egnn_pytorch`, `adj_mat` is read **only inside `if use_nearest:`**, which requires
`num_nearest_neighbors > 0` or `only_sparse_neighbors=True`. `GNNBackboneEquivariant` constructs
`EGNN(dim, m_dim, edge_dim)` with both at their defaults, so the adjacency passed at
`policy/gnn_backbone.py` is silently discarded. The EGNN runs **dense all-pairs** message passing;
topology reaches the network only through the edge-feature channel, where `||bearing|| in {0,1}`
acts as a de-facto adjacency bit.

For this task dense all-pairs is arguably the *right* choice — you want to reason about edges you
do not have — but it must be deliberate, and the adjacency must then be an explicit feature rather
than an accident of zero-padding.

### 2.2 The policy cannot see the geometry of candidate edges

`Network.get_bearings_explicit()` writes `b[i,j] = 0` unless `edges[i,j]`. So for every edge the
agent might *add*, the bearing — the quantity that decides whether that edge adds rank — is zeroed
out. What remains is EGNN's internal `rel_dist = ||x_i - x_j||^2` (all-pairs) and
`common_neighbors = A@A` (topological). But bearing rigidity is **invariant to uniform scaling**
and depends only on **directions**; pairwise distance is close to the wrong invariant.

The GINE path is worse: `GINE_SelectNodesSequentially` builds `edge_index` from `adj.nonzero()`,
so it message-passes only over existing edges, and `DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection`
carries **no geometric node feature at all** (domain one-hot, in/out degree, three centralities).
A GINE policy is structurally blind to the geometry of any edge it does not already have.

**This is the first-order cause of 1.3.**

### 2.3 The EGNN's invariance is broken by its own inputs

Raw bearing vectors are fed as `edges`, i.e. as **invariant scalar edge features**, when they are
rotation-*equivariant* quantities.

The original rationale was sound and is worth keeping in the record: a bearing is measured in the
**measuring agent's local frame**, so for a node that *has* a local frame (`R^2xS^1`, `R^3xS^1`,
`SE(3)`) the measurement is genuinely invariant to a global rotation of the network — rotating the
world rotates the agent's frame with it, and `R_i^T p_hat_ij` is unchanged. That reasoning does
**not** hold for `R^2` / `R^3` agents, where `Agent.get_bearing` returns the global-frame vector
(`network.py`, the `if self.domain not in ["R^3", "R^2"]` branch). Since every current experiment
is homogeneous `R^d`, in practice the features are global-frame vectors consumed as invariants:
rotate the whole network and the policy output changes.

Scale has the same problem from the other direction. Rigidity rank is invariant to uniform
scaling, `rel_dist` is not — which is the only reason changing `pos_limits` from +-100 to +-1
mattered at all. It should not have.

### 2.4 phi is built from `rank(B)`, which the network has no way to compute

`phi = 20*rank(B) - 10*m`. Rank of a `3m x 6n` matrix is a global algebraic property; a 3-layer
MPNN over 8 nodes will not recover it from positions and centralities. **The observation contains
no rigidity information whatsoever.** Meanwhile the environment already computes, every step:
`rank_brm`, `rank_K`, `is_IBR`, and — inside `is_MBR` — the per-edge block ranks `c_k`.

**Decision: run this as an ablation, not as a default.** Two observation variants that differ
*only* in whether rigidity-derived features are present, so the gap between them is itself the
result: *how much rigidity structure a GNN recovers from geometry alone.* Given the n=4/R^2 number
(45% rigid, below random's 80%) the gap is expected to be large, which makes it a better figure
than either arm on its own. See appendix A for why this also matters for the long-term
distributed formulation — these features are tier 3, unavailable to any local decision maker.

### 2.5 phi's weights are dimension-dependent, so nothing transfers

Verified: an edge's rigidity-matrix block has rank **2 in R^3** and **1 in R^2** (the orthogonal
projection `P` has rank 2 in 3D, and rank 1 once `U_ij` restricts it to the plane). With
`w_rank=20, w_edge=10`:

| | rank-adding edge | pruning a redundant edge |
|---|---|---|
| R^3 | **+30** | +10 |
| R^2 | **+10** | +10 |

So in R^3 the score is three times more eager to add than to prune, and in R^2 the two are
symmetric. phi's optimum also moves with the configuration — 50 at n=4/R^2, 300 at n=8/R^3 — so
the critic's target range shifts whenever n or the domain changes. A single policy cannot span
both under this phi.

### 2.6 The pointer head has no pairwise term

```python
add_remove_logits = self.head(torch.cat([h, selected_repeated], -1))
```

The logit for target `j` given selected `i` is `MLP([h_j, h_i])` — no `e_ij`, no `adj_ij`, no
`p_hat_ij`, and no flag distinguishing the first pick from the second. "Nothing selected" is
encoded as `h_sel = 0`, which is ambiguous with a genuinely-zero embedding, and the *same* MLP
scores both picks from two quite different input distributions.

### 2.7 The forced-action / no-stop design

With `skip_enabled: false`, `skip_is_stop: false`, `time_penalty: 0`, `max_steps: 200`, the agent
**must** toggle an edge every second step for 200 steps. It can only hold a good graph by learning
a 4-step add/remove oscillation. DQN essentially managed it (final 10.94 vs best 10.02); PPO did
not. 200 steps is also ~10x more budget than needed — DQN's `best@` is 15.5. Once gamma < 1 the
"hold" problem largely dissolves (1.2), but the horizon should still come down.

### 2.8 Unnormalized, largely redundant features

`degree` (0-7), `betweenness` (0-~30), `common_neighbors = A@A` (0-8) all scale with density and
n, and `observation_preprocessor` is `null`. The three centralities are also things a GNN can
compute for itself, and they are pure-Python Floyd-Warshall plus Brandes on every step.

### 2.9 Smaller correctness items

- `environment.py`, `action_SelectNodesSequentially`: `didnt_exist` / `existed` are computed
  *after* the branch condition, so both are always `False`. Dead code in a live path.
- `network.py`, `fully_connected()`: sets `edges = np.ones((n, n))` **including the diagonal**.
  `rank_K` survives only because `extended_bearing_rigidity_matrix` has an `if i == j: continue`
  and because `E[i,k]` gets `-1` then `+1`. Fragile.
- `rigidity.py`, `is_MBR_Rd`: `if brmat: isIBR, rank = is_IBR_explicit()` calls with no arguments,
  and `if brmat` on an ndarray raises. Unused path, but broken.
- **Per-step cost**: `step()` calls `is_MBR` unconditionally (a full-matrix rank *plus* one rank
  per edge — ~25 SVDs at n=8) and, when tracking, `rigidity_eigenvalue` (which rebuilds `B` and
  does a 48x48 `eigvalsh`). `Weighted` needs **neither**. Env stepping is 8.7 ms against 2.6 ms of
  inference: three times the network's cost spent on metrics that do not enter the reward.
- Seeding: `np.random.seed(SEED)` runs *after* the sub-envs are constructed, and the envs use
  global `np.random` rather than `self.np_random`, so all four sub-envs share one stream.
- `CLAUDE.md`'s "Known issues" 4, 5 and 6 were already fixed in code and have been corrected.

---

## 3. Phased plan

Each phase is independently testable and leaves the repository in a working state. Old
`environment.py` dispatch names are kept intact throughout, so existing checkpoints keep replaying
through the manifest system.

### Phase 1 — PPO discount factor and memory sizing
*Fixes 1.2. Two lines, largest expected effect per unit of work.*

- `cfg.discount_factor = 1.0` -> `0.99`.
- `memory_size == cfg.rollouts` (already in the working tree as `ROLLOUT_SIZE`).
- Record in the source comment *why* gamma=1 is degenerate here, since the existing comment argues
  for it.

**Acceptance:** a PPO run on `...termMaxSteps_n8_R3` shows entropy falling below ~1.4 nats and
`Episode/ Best is min rigid` rising above 0.5 within 300k trainer steps. Target is parity with
DQN's 0.98.

### Phase 2 — dimension-normalized state score
*Fixes 2.5. Prerequisite for anything multi-n or multi-domain.*

- New `state_score_type = "WeightedNormalized"`:
  `phi = w_r * rank/rank_K - w_e * m/m_req`, both terms O(1), optimum ~ `w_r - w_e` regardless of
  n and domain.
- `rank_K` is already cached per episode. `m_req` comes from `MBR_required_Rd` for homogeneous
  `R^d` and from the `is_MBR` greedy lower bound otherwise (cache it per episode alongside
  `rank_K`).
- `Weighted` stays exactly as it is.

**Acceptance:** phi's optimum is within a few percent of the same value for n=4/R^2, n=8/R^3 and
n=16/R^3; `baselines.py` `optimal` and `greedy` rows confirm it.

### Phase 3 — all-pairs bearing observation (geometry-only ablation arm)
*Fixes 2.2 and 2.1.*

- `Network.get_all_pairs_bearing_features()` — bearings for every ordered pair `i != j`,
  regardless of whether the edge exists.
- Explicit binary `edge_exists` channel in the edge features.
- Resolve the `adj_mat` no-op: either enable `only_sparse_neighbors` or drop the argument and
  document that message passing is dense all-pairs by design.
- Per-instance pose normalization (centre, scale by RMS radius) so the policy stops depending on
  `pos_limits`.
- Keep candidate-bearing inclusion behind a flag rather than hardcoding it: candidate bearings are
  **tier 2** information (appendix A.1) — an agent does not know its bearing to a node it has not
  measured — so a later distributed variant needs to be able to switch them off without a rewrite.

**Acceptance:** a policy trained at n=8/R^3 evaluated zero-shot at n=4/R^2 is no longer *below*
the random baseline on `rigid %`.

### Phase 4 — rigidity-feature observation (informed ablation arm)
*Addresses 2.4 as a measured comparison.*

- Same as Phase 3 plus: rank deficit `rank_K - rank(B)`, `m/m_req`, `is_IBR` as graph-level
  features; per-edge block rank `c_k` as an edge channel.
- `c_k` is already computed inside `is_MBR`; expose it rather than recomputing.

**Acceptance:** both arms train to completion at n=8/R^3 and the gap between them is measured on
the n=4/R^2 and n=16/R^3 transfer tables. The gap is the deliverable, whichever way it points.

### Phase 5 — pairwise pointer head
*Fixes 2.6.*

- Separate first-pick and second-pick heads; explicit phase flag instead of the zero-vector
  sentinel.
- Second-pick logit from `[h_i, h_j, h_i*h_j, e_ij, adj_ij, p_hat_ij]`.
- Applies to actor, critic and q_func variants of `SelectNodesSequentially`.

**Acceptance:** no regression at n=8/R^3; improvement expected on transfer.

### Phase 6 — correctness and cost cleanups
*Fixes 2.7 (partly), 2.8, 2.9.*

- `fully_connected()` diagonal; dead code in `action_SelectNodesSequentially`; the broken
  `is_MBR_Rd` branch.
- Gate `is_MBR` and `rigidity_eigenvalue` behind an explicit flag / episode-end computation, so a
  `Weighted`-scored run stops paying for them every step.
- `RunningStandardScaler` as the observation preprocessor in both training scripts.
- Shorter default horizon (~50 steps at n=8) once gamma < 1 is in.

**Acceptance:** `Stats / Env stepping time (ms)` drops materially; no change in learning curves.

### Later (not in this branch)
- Multi-n training: curriculum over n within one run, or padded sub-envs with different n. The
  latter needs a node mask threaded through the backbone — note `EGNN` already accepts a `mask`
  argument that `GNNBackboneEquivariant` never passes.
- Heterogeneous domains at scale, where the `is_MBR` greedy bound can produce false negatives
  (known issue 1).
- The constructive (add-only, start-from-empty) formulation, which is also the natural bridge to
  appendix A.

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
not currently measuring* — i.e. the geometry of **candidate** edges. This is exactly what Phase 3
adds, and it is worth being explicit that **it is not free information**: an agent does not know
its bearing to another node before measuring it. There is no way for a purely local decision maker
to evaluate "would adding `i -> j` help?" without first obtaining `p_hat_ij`.

Whether Phase 3 is admissible in a distributed setting therefore depends on a modelling choice
that has not been made yet, and should be made deliberately:

- If **detection is cheap and maintenance is expensive** — e.g. omnidirectional vision gives
  bearings to every agent within radius `R` essentially for free, while an *edge* denotes a
  persistent, tracked, communicated measurement link with a real cost — then all-pairs bearings
  within `R` are a legitimate local observation, and Phase 3 carries over unchanged (with the
  all-pairs set truncated to the sensing radius).
- If **any measurement costs what an edge costs**, candidate bearings are genuinely unavailable
  and a distributed protocol needs either an explicit exploration phase, a construction phase in
  which agents acquire bearings as they join, or a policy that reasons from communicated
  *positions* rather than measured bearings.

This is the single most important open modelling question for the distributed direction, and the
current centralized work does not depend on resolving it.

**Tier 3 — not available at all.** `rank(B)`, rank deficit, per-edge block rank `c_k`, `is_IBR`,
and the graph centralities (`betweenness`, `closeness`, `eigenvector`). Global by construction,
too expensive to compute and to communicate.

### A.2 What this buys the current ablation

The Phase 3 / Phase 4 split maps onto tiers 2 and 3, so it measures *the price of
decentralization* in two separable steps, before committing to any of it:

- **Phase 4 minus Phase 3** = the cost of losing tier 3 (rigidity algebra).
- **Phase 3 minus a tier-1-only variant** = the cost of losing tier 2 (candidate-edge geometry),
  under the pessimistic measurement model.

The second comparison is worth adding as a third arm later if the distributed direction is
revived; it is not needed now, but the observation code should be structured so that dropping
candidate bearings is a config flag rather than a rewrite.

Concrete things to preserve in the current code so this stays open:
- keep GNN depth an explicit constructor argument (already true, `num_layers`);
- keep per-node / per-edge action heads as a first-class option rather than flattening to a global
  index;
- keep the observation builders tagged by tier, and keep candidate-bearing inclusion switchable
  rather than baked in.

---

## Appendix B — what the split actually is

**Config-level (fixable now, mostly already fixed):**
gamma=1.0 in PPO; `memory_size != rollouts`; a 200-step horizon; DQN's `polyak=0.005` combined
with `target_update_interval=200`, which gives a target time constant of ~160k steps (it worked,
but it is not what those numbers suggest was intended).

**Structural (no amount of tuning fixes these):**
candidate-edge geometry invisible (2.2); no rigidity information in the observation (2.4);
phi's weights dimension-dependent (2.5); EGNN invariance broken by its inputs (2.3) and `adj_mat`
ignored (2.1).

Fixing the config items should get PPO to roughly where DQN already is at n=8/R^3. Only the
structural items move the transfer numbers in 1.3.
