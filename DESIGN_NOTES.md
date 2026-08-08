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

## policy/gnn_backbone.py

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
