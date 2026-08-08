# Design notes

Rationale that would otherwise sit in the source as long comment blocks. The code carries a short
comment and a pointer here (`see DESIGN_NOTES.md#anchor`).

`CLAUDE.md` describes what the code is. `ROADMAP.md` is the live plan and diagnosis. This file is
the "why is it written this way" layer.

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
optimum also moves with the configuration — 50 at n=4/R^2, 270 at n=8/R^3, 590 at n=16/R^3 — which
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
and `c_max` are both plain rank computations — exact, and asserting nothing about achievability.

The payoff is that the central guarantee becomes structural rather than empirical. A maximally
informative edge gains `w_rank * c_max/rank_K` and costs `w_edge * c_max/rank_K`, so it is worth
adding **iff `w_rank > w_edge`** — for any geometry, domain mix or `n`, because the same
`c_max/rank_K` factor appears on both sides. Under `m/m_req` the two factors were `c_max/rank_K`
and `1/m_req`, which coincide only when `m_req` happens to equal `rank_K/c_max`.

`w_rank/w_edge = 4` reproduces R^3's existing 3:1 preference for adding rank over pruning a
redundant edge, now identically in every domain. That ratio is the meaningful knob; the overall
scale only sets the reward magnitude. phi's ceiling is `w_rank - w_edge = 75` when the poses admit
a perfectly packed rigid graph and slightly below otherwise — a fact about the geometry, not a
tuning issue.

Measured (greedy baseline, same 100/25 weights):

| config | rank_K | c_max | m_req | greedy phi | old `Weighted` phi |
|---|---|---|---|---|---|
| n=4 / R^2 | 5 | 1 | 5 | 75.00 (= brute-force optimal) | 50 |
| n=8 / R^2 | 13 | 1 | 13 | — | 130 |
| n=8 / R^3 | 20 | 2 | 10 | 73.00 | 270 |
| n=16 / R^3 | 44 | 2 | 22 | — | 590 |

### episode-constants

`compute_episode_constants()` depends on the poses but not the edge set, so it runs once per
episode. `B_K` is built once and shared because it is the expensive part.

- `rank_K` — rank of the fully-connected graph's rigidity matrix; the rank a rigid graph must
  reach (`3n-4` in R^3, `2n-3` in R^2). **Exact.**
- `c_max` — the most rank one edge could contribute at these poses. **Exact.**
- `m_req` — fewest edges that could possibly make these poses rigid. **A lower bound.** Reported,
  and used for the MBR metric; never in the reward.

### initial-edge-count

Uniformly random edge counts are almost always far above what rigidity needs — the requirement
grows ~linearly in `n` while `n^2-n` grows quadratically — so the agent would only ever see graphs
that need edges removed. `sample_initial_edge_count()` samples around the minimum requirement
instead. The mean is only exact for homogeneous R^d networks; for other domains it is below the
true requirement, which is acceptable.

### episode-logging

Environment metrics are written once per episode, not once per step. A step-resolution scalar costs
a TensorBoard event per step and is then downsampled and averaged for display anyway, so the
resolution was paid for and never seen; the summary is both cheaper and closer to what the plots
were already showing.

`step()` folds each step into `episode_accum` (`new_episode_accum()`: sums and counts only, so an
episode costs the same whether it is 100 or 2000 steps long). `episode_summary()` then emits the
whole episode as one flat, float-valued record — where it ended up (`Final ...`), the best graph it
visited (`Best ...`), and what it looked like throughout (`Mean ...`) — so `write_episode()` can
dump it without knowing what any of it means.

Scalars are written against `writer_counter` (the global env step) rather than the episode index,
so the curves share an x-axis with skrl's loss/reward plots.

### dict-observation

There used to be six `Dict*` observation types differing only in which keys they populated —
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
its old layout exactly — verified element-wise against the pre-merge code for the EGNN variant:
`node_features` (10), raw un-normalized `coord_features`, and 6-channel `edge_features` with
bearings on existing edges only. Reproducing *raw* coordinates matters as much as the shapes: a
checkpoint trained before pose normalization would otherwise be fed differently-scaled inputs and
quietly mis-evaluated.

| legacy name | node set | coords | edge ch. | selection |
|---|---|---|---|---|
| `DictEquivariantNodeFeaturesAndAdjAndSelection` | graph (10) | raw | 6 | yes |
| `DictNodeFeaturesAndEdgeFeaturesAndAdjAndSelection` | graph (10) | — | 6 | yes |
| `DictNodeFeaturesAndAdj` | domain+sign-bearing (5+3n) | — | — | no |
| `DictNodeFeaturesAndAdjAndSelection` | domain+sign-bearing (5+3n) | — | — | yes |
| `DictNodeFeaturesAndAdjAndEdgeProposal` | domain+bearing (5+3n) | — | — | no (+`proposed_edge`) |
| `DictBearingNodeFeaturesAndAdjAndSelection` | bearing (3n) | — | — | yes |

`OBS_BACKBONE` records which GNN each legacy name implied, and the training scripts prefer it over
their `BACKBONE` constant — so an old GINE config trains a GINE model even when the constant says
`Equivariant`. An unknown `obs_type` raises listing the known ones.

### all-pairs-bearings

`get_bearings_explicit()` zeroes `b[i,j]` unless the edge exists, so for every edge the agent might
*add* — the decision it is actually making — the bearing that determines whether that edge adds
rank was invisible. All that reached the policy about a candidate pair was EGNN's internal
`rel_dist` and `common_neighbors`. Bearing rigidity is invariant to uniform scaling and depends on
**directions**, so distance is close to the wrong invariant. This was the first-order cause of the
generalization failure (`ROADMAP.md` §2.2).

The `Dict` observation now carries `get_all_pairs_bearings()` — every ordered pair, edge or not —
plus an explicit binary `edge_exists` channel, so adjacency is stated rather than implied by a
zeroed bearing.

**`include_candidate_bearings` (env config, default `True`)** reverts to bearings on existing edges
only, keeping the observation shape identical. This is not a tuning knob but a modelling one:
candidate-edge bearings are tier-2 information (`ROADMAP.md` A.1) — an agent does not know its
bearing to a node it has not measured. Whether a distributed version may use them depends on
whether detection is cheaper than maintaining a link, which is an open question. The flag exists so
that a later tier-1-only variant is a config change, not a rewrite.

### pose-normalization

`coord_features` are centred on the centroid and scaled to unit RMS radius. Bearings are already
unit vectors and so scale-invariant, but EGNN's internal `rel_dist = ||x_i - x_j||^2` is not — which
is the only reason changing `random_scenario`'s `pos_limits` from ±100 to ±1 ever mattered. It
should not have. Normalizing per instance also makes n=8 and n=16 comparable when both are drawn
from the same box but at different densities.

Normalization is applied to the *observation* only. The rigidity maths keeps the true poses; rank
is scale-invariant anyway, but the rigidity eigenvalue is not.

### min-eig-caching

When tracking is on, the rigidity eigenvalue is needed for logging anyway, so `step()` computes it
once and hands it to the best-state tracker rather than letting that recompute it. `trace_min_eig`
asks for the same value without a writer attached — `baselines.py` records the rigidity eigenvalue
over time.

---

## rigidity.py

### max-edge-rank

`max_edge_rank()` returns `max_k rank(B_K[3k:3k+3, :])` over the fully-connected graph: the most
rank a single edge could possibly contribute at these poses.

It is **exact** — a max over plain rank computations, making no claim about what is jointly
achievable. That is why the state score normalizes with this rather than with an edge count: it
turns "one edge" into a comparable unit across domains (`d-1` in homogeneous R^d, so 2 in R^3 and 1
in R^2 — exactly the factor that made un-normalized `Weighted` non-transferable) without asserting
a minimum.

### required-edge-count

`required_edge_count()` is the fewest edges that could possibly make *these poses* rigid — an
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

Use it for reporting and for the MBR metric. **Do not put it in the reward** — see
[weighted-normalized](#weighted-normalized).

Cost is `n(n-1)` small rank computations, so call it once per episode and cache it;
`Environment.compute_episode_constants()` does.

---

## train_ppo.py

### ppo-rollout-size

One constant feeds both the memory size and `cfg.rollouts`, and they must stay equal. skrl's
`PPO.update()` runs `compute_gae()` over the **whole** memory ring and then samples
`batch_size=len(memory)`, so a memory larger than one rollout trains on stale off-policy data —
7/8 of it at `memory_size=8192, rollouts=1024` — with `last_values` bootstrapped at the ring's wrap
point instead of the trajectory end. The stale samples fall outside the ratio clip band and
contribute no gradient. This is what broke
`bigPPOSelectEquivariant3e-4lrNormalizedPositions`.

### ppo-discount-factor

`discount_factor` must stay `< 1`. The environment's reward is potential-based (`phi(s') - phi(s)`),
so at γ=1 the return telescopes to `phi(s_T) - phi(s_0)` and the advantage becomes
`E[phi(s_T)|s'] - E[phi(s_T)|s]`, which is ≈0 under a near-uniform policy because the walk over
edge sets mixes and forgets `s`. There is then no gradient to bootstrap from — that is what froze
the earlier run's entropy at ~1.9 nats of a ~2.0 ceiling.

At γ<1, Abel summation turns the same reward into

```
-phi(s_0) + (1 - gamma) * sum_t gamma^(t-1) phi(s_t)
```

i.e. maximize the discounted average of phi along the trajectory: converge fast and stay converged.
DQN uses 0.99 and solves n=8/R^3; PPO now matches it.

γ=1 used to be set so the logged return matched the optimized objective. Read `Episode/ Return` for
that instead — it is undiscounted by construction.

---

## policy/

### model-registry

`policy/registry.py` maps `(role, backbone, action_type)` to a model class, replacing the if/elif
chains that used to select models in both training scripts (they lost ~180 and ~110 lines). Roles
are skrl's own model-dict keys — `policy`, `value`, `q_network` — so `build_models()` output goes
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

### egnn-dense-all-pairs

`GNNBackboneEquivariant.forward` accepts `adj_mat` but does not forward it to `EGNN`. In
`egnn_pytorch`, `adj_mat` is read *only* inside `if use_nearest:`, which needs
`num_nearest_neighbors > 0` or `only_sparse_neighbors=True`; the backbone constructs
`EGNN(dim, m_dim, edge_dim)` with both at their defaults. Passing it was therefore a silent no-op —
verified, `max abs diff 0.0` between an all-zeros and an all-ones adjacency.

Dense all-pairs message passing is the right choice here (the whole task is reasoning about edges
you do not have), so the fix is to make it deliberate rather than to sparsify: the graph reaches
the model through `edge_features`, where `edge_exists` now states adjacency explicitly. The
argument stays in the signature so archived model sources that pass it keep working.

`EGNN` also accepts a `mask` argument the backbone never passes, which is what variable-`n`
batching will need.

### gine-dense-all-pairs

GINE used to build `edge_index` from `adj.nonzero()` and gather `edge_features[i][src, dst]`, i.e.
message passing over **existing edges only** — with a comment reading "we get all possible edges'
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
exactly zero — so the outgoing-edge direction below is preserved.

### egnn-init-eps

`egnn_pytorch` applies `nn.init.normal_(weight, std=init_eps)` to *every* Linear in an `EGNN` layer,
with `init_eps=1e-3` by default — a guard against deep stacks going NaN. Stacked three deep and set
against the node residual (`node_out = node_mlp(...) + feats`), the edge-feature path starts at
about **1e-10** of the output. The dependence is structural, not absent — all gradient entries are
nonzero — but the model begins effectively blind to every edge feature, bearings included, and has
to grow those weights before geometry can matter at all.

It does escape: the trained `bigDQN8SelectEquivariant3e-4lrNormalizedPositions` checkpoint has
`edge_mlp` and `node_mlp` weight std at 1.2e-1 … 4.3e-1, up from 1e-3, and reached 98.2% minimally
rigid. So this is a slow start, not a ceiling.

It is worth knowing for two reasons. It is a plausible contributor to the policy latching onto
node-level statistics (degree, centralities, which arrive via `feats` and the residual) instead of
geometry. And it is an asymmetry against GINE, whose Linears use the PyTorch default, ~5e-2 … 2e-1
— roughly where the EGNN *finishes*. So the two backbones do not start on equal footing.

`GNNBackboneEquivariant` now exposes `init_eps`. The default is unchanged, because 1e-3 is what the
working run used and changing it silently would invalidate the one result the project rests on.
Raising it to ~1e-1 for a depth-3 stack is a cheap experiment, not a settled fix.

### backbone-num-layers

The layer count used to be hardcoded at 3 and was 2 in older runs. It is a constructor argument now
so a checkpoint trained at a different depth can still be loaded (see
`agent_loader.rebuild_backbone`). Submodules keep the names `conv1..convN`, so state dicts of
3-layer models are unaffected.

### gine-edge-direction

GIN(E) message passing adds the *inward* edge features to the neighbour's features. Here it is the
outward edge that carries the meaning — "I measure this bearing to that node" — so `edge_index` is
flipped before message passing.

---

## report.py

### palette

Colours come from the data-viz reference palette, used unchanged and in its documented order.
`greedy` / `learned` / `random` take categorical slots 1–3, which are certified for the all-pairs
case (overlapping lines) in both modes. `initial` and `optimal` are *reference points* rather than
methods under comparison, so they take neutral inks and dashed strokes instead of a categorical
hue — which also keeps the categorical count at 3.

### panel-titles

Static output for a thesis, so light mode only and no hover layer. Identity is never colour alone:
every series carries a direct label at its line end (or a value label on its bar), and every figure
ships the notes card that says what it is showing.

Each panel is titled with *what the quantity is*, with the reading direction on a second line —
"edges used" told a reader nothing about why they should care.
