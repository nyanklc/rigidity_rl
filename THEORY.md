# Theory

The algebra behind the environment: what the rigidity matrix is, what its rank and null space mean,
and how every derived quantity in the code (`rank_K`, `c_max`, `m_req`, the state score, the flex
features) follows from it.

`CLAUDE.md` is what the code is, `DESIGN_NOTES.md` is why it is written that way, `ROADMAP.md` is
the plan. This file is the maths.

---

## 1. Notation

`n` agents. Agent `i` has position `p_i ∈ R³` and orientation `R_i ∈ SO(3)`; a *domain* fixes which
of those degrees of freedom actually exist (`R^2`, `R^3`, `R^2xS^1`, `R^3xS^1`, `SE(3)`).

A directed edge `i -> j` means **`i` measures the bearing to `j`**:

```
p_ij  = p_j - p_i          p̂_ij = p_ij / ||p_ij||        b_ij = R_iᵀ p̂_ij
```

`b_ij` is the measurement in `i`'s own frame. For `R^2` / `R^3` there is no frame and `R_i = I`, so
`b_ij = p̂_ij` is a **global-frame** vector — the fact that makes the observation's use of raw
bearings rotation-dependent (`ROADMAP.md` §2.3).

Two standard objects:

```
P(x) = I - x xᵀ          orthogonal projector onto x⊥, for unit x
[a]×                     skew-symmetric matrix with [a]× b = a × b
```

`P(x)` is symmetric, idempotent, `P(x) x = 0`, and `rank P(x) = 2` for `x ∈ R³`.

---

## 2. The bearing derivative

Differentiating `p̂_ij` along a motion `ṗ`:

```
d/dt p̂_ij = (1/||p_ij||) · P(p̂_ij) · (ṗ_j - ṗ_i)                                   (2.1)
```

The projector appears because `p̂` is unit-norm: only the component of relative velocity
**perpendicular** to `p̂` can rotate it. The component *along* `p̂` changes `||p_ij||` but not the
bearing — this is the scale blindness of bearing measurements, and it is the single most important
fact in this document. It reappears in §5, §7 and §9.

With orientation, `b_ij = R_iᵀ p̂_ij` and (world-frame angular velocity `ω_i`, `Ṙ_i = [ω_i]× R_i`):

```
ḃ_ij = Ṙ_iᵀ p̂_ij + R_iᵀ (d/dt p̂_ij)
     = -R_iᵀ [ω_i]× p̂_ij + (1/||p_ij||) R_iᵀ P(p̂_ij) (ṗ_j - ṗ_i)
     = +R_iᵀ [p̂_ij]× ω_i + (1/||p_ij||) R_iᵀ P(p̂_ij) (ṗ_j - ṗ_i)                    (2.2)
```

using `[ω]× p̂ = -[p̂]× ω`. Rotating the observer's own frame moves the measurement even when nothing
translates.

### Matching the code

`rigidity.extended_bearing_rigidity_matrix` builds, per edge `k = (i,j)`:

```
s      = 1/||p_ij||
Dp_k   = s · R_iᵀ P(p̂_ij)
Da_k   = -R_iᵀ [p̂_ij]×
```

and assembles

```
B = [ Dp Ēᵀ S̄ | Da Ē_oᵀ P̄ ]
```

with incidence `E[i,k] = -1, E[j,k] = +1` and `E_o[i,k] = -1`, and with
`S̄ = blkdiag(S_1 … S_n)`, `P̄ = blkdiag(P_1 … P_n)` the **per-node** DOF projectors of §12.

The signs work out: the position half contracts `E` to give `(ṗ_j - ṗ_i)`, matching (2.1); the
orientation half picks up `E_o`'s `-1`, so `Da_k · (-1) = +R_iᵀ [p̂]×`, matching (2.2). The apparent
minus in `Da_k` is cancelled by the incidence matrix, not an error. **Only rank and null space are
ever used, so an overall sign would be immaterial anyway.**

`B` is `(3m × 6n)`: **one 3-row block per directed edge**, in `np.nonzero(edges)` order. Everything
below exploits that block structure.

The construction is checked against its own definition by central differences —
`tests/test_rigidity_matrix.py::test_matrix_is_the_numerical_jacobian_of_the_bearings` asserts
`B δ = d/dt bearings` to 1e-6 relative for random admissible `δ`, in all five domains and eight
heterogeneous mixes. That validates `Dp`, `Da`, both incidence signs and both projectors at once.

---

## 3. Trivial motions and `rank_K`

A motion is *trivial* when it changes no bearing. From (2.1)–(2.2):

- **Translation** `ṗ_i = v` for all `i`. Then `ṗ_j - ṗ_i = 0`, so every block vanishes. `d` of these.
- **Uniform scaling** `ṗ_i = c(p_i - p̄)`. Then `ṗ_j - ṗ_i = c·p_ij = c||p_ij||·p̂_ij`, and
  `P(p̂_ij) p̂_ij = 0`. **One** of these. This is (2.1)'s scale blindness again.
- **Global rotation** is trivial only when every agent carries a frame that rotates with it: then
  `R_i` and `p̂_ij` rotate together and `b_ij = R_iᵀ p̂_ij` is unchanged. In `R^d` there is no frame,
  the bearings are global, and rotation is **not** trivial.

The pattern generalises: each domain's trivial space is `d` translations, `1` scaling, plus however
much of the rotation group the agents' own frames absorb. Verified at n=8/16/32/64:

| domain | DOF/agent | trivial dim | `rank_K` | `c_max` |
|---|---|---|---|---|
| `R^2` | 2 | 3 | `2n - 3` | 1 |
| `R^3` | 3 | 4 | `3n - 4` | 2 |
| `R^2xS^1` | 3 | 4 (+1 rotation) | `3n - 4` | 1 |
| `R^3xS^1` | 4 | 5 (+1 rotation) | `4n - 5` | 2 |
| `SE(3)` | 6 | 7 (+3 rotations) | `6n - 7` | 2 |

`c_max` is 1 in the planar domains and 2 in the spatial ones for the reason in section 4: a bearing
is one angle in the plane and two in 3-space.

So for homogeneous `R^d` the trivial space is `d + 1`-dimensional, giving

```
rank_K = d·n - d - 1          3n - 4 in R³ ,  2n - 3 in R²                          (3.1)
```

for the fully connected graph. Measured: `rank_K = 20` at n=8/R³, `44` at n=16/R³, `5` at n=4/R²,
`13` at n=8/R². All match (3.1).

**A framework is Infinitesimally Bearing Rigid (IBR) iff `rank(B) = rank_K`** — it admits no motions
beyond the trivial ones. The code computes `rank_K` numerically from the complete graph rather than
from (3.1), so heterogeneous networks are handled without a closed form.

`rank_K_pos = rank(B_K[:, :3n])` is the same quantity restricted to the position columns. It equals
`rank_K` whenever no domain contributes orientation (all current experiments), and is what §9 needs.

---

## 4. Per-edge block rank `c_k`, and why it is constant

Edge `k`'s 3-row block touches only its two endpoints' columns. `R_iᵀ` is orthogonal, so

```
c_k = rank [ P(p̂_ij) S_i | P(p̂_ij) S_j | [p̂_ij]× P_i ]
```

In a homogeneous domain `S_i = S_j = S` and the first two terms have the same range, which recovers
the old `c_k = rank(P(p̂_ij) U_ij)` with `U = S`:

- **R³**: `S = I`, `P_i = 0`, so `c_k = rank P(p̂) = 2` for every edge.
- **R²**: all positions are coplanar, so `p̂_ij` lies in the plane. `S = diag(1,1,0)` restricts to
  that 2-D plane; `P(p̂)` kills `p̂`, which is *in* the plane. A 2-D space minus one direction leaves
  **1**. So `c_k = 1` for every edge.

Generally `c_k = d - 1` in homogeneous `R^d`, for **every** edge, independent of geometry. In a mix
the three terms differ and `c_k` genuinely varies — on the `mixed` scenario the complete graph's
block ranks are `{1: 12, 2: 78}`, the 12 being the ordered pairs whose measurer *and* target are
both planar.

This is why `rigidity_edge` is nearly useless on its own: as an observation channel `c_k` is a
constant in every homogeneous configuration. It varies only in heterogeneous networks, where the
endpoints' domains differ (measured: a mix of 1s and 2s). Verified empirically at n=4/8/16 in both
R² and R³ — all edges identical.

`c_max = max_k c_k` over the *complete* graph is exact and cheap, and is what the state score
normalizes with (§7).

---

## 5. The minimum edge count `m_req`

Rank is subadditive over the block rows:

```
rank(B) ≤ Σ_{e ∈ E} c_e ≤ m · c_max
```

A rigid graph needs `rank(B) = rank_K`, hence

```
m ≥ rank_K / c_max        i.e.   m_req = ⌈ rank_K / c_max ⌉                          (5.1)
```

This is a **lower bound, not a ground truth**: subadditivity says no smaller edge set can be rigid,
not that one of this size exists. `rigidity.required_edge_count` implements the greedy version of
the same argument (accumulate the highest block ranks of the complete graph until `rank_K` is
reached), which reduces to (5.1) when all blocks are equal.

For homogeneous `R^d`, (5.1) gives `⌈(dn - d - 1)/(d - 1)⌉`, which coincides exactly with the closed
form of Trinh et al. (`MBR_required_Rd`). Checked at n=4…16 in R² and R³ — all agree.

| config | rank_K | c_max | m_req |
|---|---|---|---|
| n=4 / R² | 5 | 1 | 5 |
| n=8 / R² | 13 | 1 | 13 |
| n=8 / R³ | 20 | 2 | 10 |
| n=16 / R³ | 44 | 2 | 22 |

Brute force finds the bound tight on every instance small enough to check exhaustively (n=4 across
8 domain mixes × 3 seeds, n=5 across 6 mixes, all five domains) — evidence, not proof. **It is kept
out of the reward for exactly this reason** (§7).

---

## 6. `is_MBR` — the minimality heuristic

Sort the current graph's `{c_e}` descending, accumulate until the sum reaches `rank_K`, call the
count `m_req'`; declare minimal iff IBR and `m = m_req'`. Sound as a lower bound by the same
subadditivity, and exact for homogeneous `R^d`. It can produce **false negatives** on heterogeneous
networks, where the highest-rank blocks may not be jointly realizable.

Note `m_req'` here is derived from the *current* edge set, so it is not an episode constant — unlike
`required_edge_count`, which uses the complete graph and therefore depends only on the poses.

---

## 7. The state score

### Why `Weighted` does not transfer

`φ = w_r·rank(B) - w_e·m` with `(20, 10)`. Per §4 a rank-adding edge contributes `c_max`:

| | gain from a rank-adding edge | gain from pruning a redundant one |
|---|---|---|
| R³ (`c_max = 2`) | `2·20 - 10 = +30` | `+10` |
| R² (`c_max = 1`) | `1·20 - 10 = +10` | `+10` |

The rank/edge exchange rate is an accident of the dimension: R³ prefers adding 3:1, R² is neutral.
The optimum also moves with the configuration (50 at n=4/R², 270 at n=8/R³, 590 at n=16/R³), so the
critic's target range shifts whenever `n` or the domain changes.

### `WeightedNormalized`

Put both terms in units of rank and divide by `rank_K`:

```
φ = ( w_r · rank(B)  -  w_e · m · c_max ) / rank_K                                   (7.1)
```

`m·c_max` is "the rank this many edges could have carried at best", so the second term is the
fraction of the required rank spent on edges. Both numerators are ranks, both denominators `rank_K`
— dimensionless.

**The central guarantee.** Adding an edge contributing `Δr` rank:

```
Δφ = ( w_r·Δr - w_e·c_max ) / rank_K
```

At the best case `Δr = c_max` this is `(w_r - w_e)·c_max/rank_K`, which is **positive iff
`w_r > w_e`** — for any geometry, any domain mix, any `n`, because the same `c_max/rank_K` factor
appears on both sides. Under the earlier `m/m_req` normalization the two factors were `c_max/rank_K`
and `1/m_req`, which coincide only when `m_req` happens to equal `rank_K/c_max`; the guarantee was
then contingent on a heuristic being tight. This is why `m_req` was removed from the reward.

Removing a redundant edge (`Δr = 0`) gains `w_e·c_max/rank_K`. So

```
(rank-adding edge) / (pruning a redundant edge)  =  (w_r - w_e)/w_e  =  3   at (100, 25)
```

reproducing R³'s original 3:1 preference, now identically in every domain.

**Optimum.** At `rank = rank_K` and `m = m_min`:

```
φ* = w_r - w_e · m_min · c_max / rank_K   ≤   w_r - w_e = 75
```

with equality iff the poses admit a perfectly packed rigid graph (`m_min = rank_K/c_max`). Check
against measurement: greedy at n=8/R³ reaches `m = 10.80`, giving
`100 - 25·10.80·2/20 = 100 - 27.0 = 73.0` — exactly the 73.00 reported. At n=4/R², `m = 5` and
`c_max = 1` give `100 - 25·5·1/5 = 75.00`, also exact.

### Potential-based shaping

The step reward is `r_t = φ(s_{t+1}) - φ(s_t)`, so the reward is the *improvement*, not the level.

---

## 8. Why the discount factor is not free

With `r_t = φ_{t+1} - φ_t`, Abel summation gives

```
Σ_{t=0}^{T-1} γᵗ (φ_{t+1} - φ_t)
  = γ^{T-1} φ_T  -  φ_0  +  (1-γ) Σ_{t=1}^{T-1} γ^{t-1} φ_t                          (8.1)
```

**At `γ = 1`** everything but the ends cancels: `G = φ_T - φ_0`. Only the final graph matters, and
nothing rewards reaching it quickly or holding it.

**At `γ < 1`** the middle term survives: the objective becomes `-φ_0` plus a *discounted average of
`φ` along the trajectory* — get good fast and stay good. That is the intended behaviour, and it
arrives purely from the discount factor, not from any reward change.

### Advantage collapse at `γ = 1`

`V^π(s) = E[φ_T | s] - φ(s)`, so

```
A^π(s,a) = Q^π - V^π
         = E[ r + V^π(s') ] - V^π(s)
         = E[ φ(s') - φ(s) + E[φ_T|s'] - φ(s') ] - ( E[φ_T|s] - φ(s) )
         = E[φ_T | s'] - E[φ_T | s]                                                  (8.2)
```

Under a near-uniform policy the walk over edge sets mixes long before `T`, so the law of `s_T`
becomes independent of the current state: both terms tend to the same constant and **`A → 0`
everywhere**. There is no gradient to bootstrap from. This is the analytical explanation of the
observed PPO failure (entropy frozen at ~1.9 nats of a ~2.0 ceiling, policy loss ≈ 0). DQN was
unaffected because it used `γ = 0.99`.

---

## 9. Flexes — the null space features

### The flex space

Let `B_p = B[:, :3n]` be the position block and `N = ker(B_p) ⊆ R^{3n}`. `N` contains the trivial
motions `T` of §3 (translations and uniform scaling; `dim T = 3n - rank_K_pos`). Define

```
F = N ∩ T⊥                                                                           (9.1)
```

the **infinitesimal flexes**: motions that change no bearing and are not trivial. Immediately

```
dim F = dim N - dim T = (3n - rank B_p) - (3n - rank_K_pos) = rank_K_pos - rank(B_p)
      = the rank deficit                                                             (9.2)
```

So the flex space is empty exactly when the framework is rigid, and its dimension *is* the rank
deficit. Verified numerically: on a graph with deficit 1, `Σ_i tr(G_i) = 1.0000` (below).

### Two traps

**(a) `eigh` gives an arbitrary basis of the whole null space.** The trivial modes are *not* the
first few eigenvectors — any orthonormal basis of `N` mixes them arbitrarily. Skipping columns by
index therefore does not remove them. `trivial_modes()` builds the 3 translations and the uniform
scaling analytically, orthonormalizes them, and they are projected out of `N` explicitly. Before
this fix the feature failed to localise on an obviously under-constrained node and was not
rotation-invariant.

**(b) A single eigenvector is a basis artefact.** `dim F > 1` is normal, and any individual vector
inside a degenerate eigenspace is arbitrary — not even reproducible between calls. The
basis-independent object is the **projector** onto `F`:

```
Π = Σ_c v_c v_cᵀ        for any orthonormal basis {v_c} of F                         (9.3)
```

`flex_tensor` returns `Π` as `(n, n, 3, 3)` blocks, `Π[i,j] = Σ_c v_{c,i} v_{c,j}ᵀ`.

When `F` is empty (rigid), the smallest **non-zero** mode is used instead — the direction the
framework resists least, i.e. the rigidity eigenvalue's eigenvector. The feature degrades gracefully
from "where it is free" to "where it is nearly free".

### Feature 1 — how free a node is

```
flex_mag_i = sqrt( tr Π[i,i] ) · sqrt(n)                                             (9.4)
```

`tr Π[i,i] = Σ_c ||v_{c,i}||²` is node `i`'s share of the flex space. Summing over nodes gives
`tr Π = dim F`, the rank deficit — so per-node magnitudes scale as `sqrt(deficit/n)` and the
`sqrt(n)` keeps the feature `O(1)` as `n` grows. That matters directly for a policy meant to span
several `n`.

### Feature 2 — would this edge help

This is the one that has to be derived rather than guessed. Adding edge `i -> j` imposes

```
P(p̂_ij) (v_j - v_i) = 0
```

by (2.1). A flex `v` **survives** the new edge iff `v_j - v_i` is *parallel* to `p̂_ij` — the
perpendicular part is what the bearing constrains, the parallel part is the scale freedom it cannot
see. So the amount of flex the edge destroys is

```
A[i,j]² = Σ_c || P(p̂_ij) (v_{c,j} - v_{c,i}) ||²
        = Σ_c ||D_c||²  -  Σ_c (p̂_ij · D_c)² ,        D_c = v_{c,j} - v_{c,i}        (9.5)
```

Both terms come straight out of `Π`:

```
Σ_c ||D_c||²      = tr Π[i,i] + tr Π[j,j] - 2 tr Π[i,j]
Σ_c (p̂ · D_c)²   = p̂ᵀ ( Π[i,i] + Π[j,j] - Π[i,j] - Π[j,i] ) p̂
```

`flex_constraint_power` computes exactly this, for **all ordered pairs**, from the projector.

> **An earlier version of this feature was wrong.** It used `sqrt(p̂ᵀ Π[i,i] p̂)` — the projection of
> node `i`'s flex *onto* the bearing. That is the component the edge does **not** constrain, and it
> ignores node `j` entirely. It was rotation-invariant and basis-independent, so the invariance
> tests passed; only the ground-truth test in §10 exposed it.

### Invariance

Under a global rotation `R`, positions rotate and the null vectors rotate blockwise, `v_i → R v_i`,
so `Π[i,j] → R Π[i,j] Rᵀ`. Then

- `tr Π[i,i]` is invariant (trace is similarity-invariant),
- `p̂ᵀ Π p̂ → (Rp̂)ᵀ R Π Rᵀ (Rp̂) = p̂ᵀ Π p̂` is invariant,

so both features are rotation-invariant scalars. This is deliberate: feeding a flex **vector** as a
node feature would repeat the mistake `ROADMAP.md` §2.3 records for bearings — rotation-equivariant
data consumed as invariant scalars, making the policy rotation-dependent.

---

## 10. Ground-truth validation of (9.5)

Take nodes 0–6 fully connected (rigid) and node 7 held by a single bearing `7 -> 0`, in R³. Then
`rank_K_pos - rank(B_p) = 1`, so `dim F = 1`.

| check | result |
|---|---|
| `Σ_i tr Π[i,i]` vs `dim F` from (9.2) | `1.0000` vs `1` |
| `flex_mag` argmax | node 7 (2.501, others 0.27–0.72) |
| mean `A[i,j]` over candidates that **do** raise rank | **2.166** |
| mean `A[i,j]` over candidates that **do not** | **0.0000** |
| rotation invariance, `flex_mag` / `A` | 1.1e-15 / 4.2e-08 |

The separation is exact: every candidate edge that raises the rank has `A > 0`, and the one that
does not has `A = 0` identically. That is (9.5) behaving as derived — a flex is destroyed iff the
relative motion has a component perpendicular to the new bearing.

---

## 11. Known caveats

1. **`m_req` is a bound, not a truth** (§5). Kept out of the reward; used for reporting and the MBR
   metric only.
2. **`is_MBR` can false-negative on heterogeneous networks** (§6), so a genuinely minimal graph may
   never be recognised and a `MinimallyRigid` episode may never terminate.
3. **`flex_mag`'s scale changes meaning at the rigid boundary.** For `dim F ≥ 1`,
   `Σ_i tr Π[i,i] = dim F`; in the rigid fallback the single unit eigenvector gives `Σ = 1`. The
   feature is therefore comparable *within* a regime but jumps by a factor of `dim F` when the
   framework becomes rigid. Normalizing `Π` by `dim F` would fix it; not done, so that the existing
   verification stands.
4. **Degenerate eigenvalues** inside `F` are handled (the projector is basis-independent), but a
   near-degeneracy *between* the last trivial mode and the first flex could mix them numerically.
   The tolerance is relative (`w ≤ max(w)·1e-9`).
5. **Bearings for `R^d` are global-frame vectors** (§1), so feeding them as invariant edge features
   makes the policy rotation-dependent. Audited end to end: `R^2`/`R^3` policy logits move under a
   global rotation, while `R^2xS^1`/`R^3xS^1`/`SE(3)` are invariant to 6e-08. Translation and
   uniform scaling are invariant in every domain to 1e-14. **The defect is confined to the `R^d`
   testbed and vanishes in the target domains.**

   A trap when testing this: at `egnn_pytorch`'s default `init_eps = 1e-3` an untrained EGNN is
   numerically blind to edge features and reports *zero* rotation sensitivity even in `R^3`. The
   dependence only appears at trained-scale weights (`init_eps ~ 1e-1`: Δlogit 4.8e-03).

6. **The frame mismatch that this audit caught.** `flex_constraint_power` must contract the flex
   tensor with **world-frame** bearings. `Network.get_all_pairs_bearings()` returns the body-frame
   *measurement* `R_iᵀ p̂_ij`, which coincides with the world frame only when `R_i = I`, i.e. only in
   `R^d`. Using it made `flex_align` rotation-dependent in `R^3xS^1`, `SE(3)` and any heterogeneous
   mix — invisible in every `R^d` test. `get_all_pairs_bearings_world()` exists for this.

---

## 12. The DOF restriction is a property of the node, not of the edge

This section records why `extended_bearing_rigidity_matrix` looks the way it does, and what was
wrong with the previous version. It only matters for **heterogeneous** networks; every homogeneous
result is bit-identical under both constructions.

### 12.1 The requirement

Michieletto embeds every domain `D_i ⊆ SE(3)` into `SE(3)^n` and pads the variation vector with
zeros for the coordinates an agent cannot vary (Definition 13, eq. (8)). The rank identity that
Theorem 2 rests on,

```
rk(B⁺_G) = 6n - q_v - q_i ,
```

counts `q_v`, the *virtual* variations, as the **number of null columns** of `B⁺`. So the framework
requires: **a coordinate an agent cannot vary must correspond to an identically-zero column.**

### 12.2 What Table III does, and why it is not enough

Table I gives `U_ij`, `V_ij` per manifold for *homogeneous* formations, and there it satisfies the
requirement: for `R^2`, `U_ij = [e₁ e₂ 0]` for every edge, so the z columns are zero.

Table III is their heterogeneous case study (three `R^2xS^1` terrestrial robots, one `R^3xS^2`
aerial platform) and sets

```
U_(1,4) = U_(2,4) = U_(3,4) = I₃          # planar robot measuring the aerial platform
```

`U_ij` multiplies the whole relative displacement `(p_j - p_i)`, so this reactivates the *planar*
agent's z column as well as the aerial one's. The paper's own accounting notices the consequence —
it reports `q_v = 6` (only the terrestrial robots' unfeasible x/y rotations) and then has to
classify the planar agents' z columns as *linearly dependent on the rest* rather than as null. That
happens to hold for their particular four-agent configuration. It is not a general fact.

### 12.3 What goes wrong

Measured on random configurations (`ROADMAP.md` §1.2):

| mix | Σ dim D_i | `rank_K` under Table III | corrected | IBR verdicts differing |
|---|---|---|---|---|
| 2 of each of the five domains | 36 | **36** | 33 | 2.0% of 300 graphs |
| 5×`R^2` + 1×`R^3` | 13 | **14** | 10 | **40%** |
| 3×`R^3` + 3×`SE(3)` | 27 | 23 | 23 | 0% (no planar agent) |

`rank_K = Σ dim D_i` means *zero* trivial motions, and `rank_K > Σ dim D_i` is outright impossible:
the matrix cannot have more independent columns than the system has coordinates. Directly: for
4×`R^2` + 4×`R^3`, a pure `+z` motion of a planar agent gives `‖B v‖ = 2.1` — the framework resists
a motion the agent cannot make, and spends rank doing it.

The cause is structural rather than a typo. The true derivative is

```
ḃ_ij = Dp_k (ṗ_j - ṗ_i)  with  ṗ_i ∈ range(S_i) , ṗ_j ∈ range(S_j)
```

and a single `U_ij` applied to the difference cannot express two different restrictions. It
coincides with the truth exactly when `S_i = S_j`, i.e. in the homogeneous case.

### 12.4 The construction in use

`rigidity.node_dof_projectors(agent)` returns, per node:

| domain | `S_i` (translational) | `P_i` (rotational) |
|---|---|---|
| `R^2` | `diag(1,1,0)` | `0` |
| `R^3` | `I₃` | `0` |
| `R^2xS^1` | `diag(1,1,0)` | `e₃e₃ᵀ` |
| `R^3xS^1` | `I₃` | `v vᵀ` |
| `SE(3)` | `I₃` | `I₃` |

and the matrix applies them on the column side, `B = [Dp Ēᵀ S̄ | Da Ē_oᵀ P̄]`. Every infeasible
coordinate is then an exactly-zero column, which is what §12.1 asks for.

Two notes.

- **`P_i` is a projector, not a placement.** For `R^3xS^1` the previous code used
  `V_ij = [0; 0; rax]` read as *rows*, where Table I's `[0_{3x2} v]` is a *column*. The two agree
  only at `v = e₃` — the only axis ever used, so nothing measured depended on it — but `v vᵀ` is
  right for any axis, and it is the form that survives the numerical-Jacobian test with
  `v = (1, 2, -0.5)/‖·‖`.
- **`bearing_DOFs` is retained**, unused by the matrix, as the reference implementation of Table I.
  `test_matches_michieletto_table_I_on_homogeneous_networks` asserts the two constructions produce
  the *same matrix* (max abs difference 0.0, 60 graphs per domain), which is what guarantees no
  homogeneous result moved.

### 12.5 Consequences for the trivial space

§3's table stays correct for homogeneous domains. For a mix the trivial dimension is no longer read
off a table: the z-translation is trivial only when no agent is planar, and a coordinated rotation
only when *every* agent carries a frame (an `R^d` agent measures in the global frame, so rotating
the world changes what it sees). The robust statement is

```
rank_K  ≤  Σ_i dim D_i  −  (3 if any agent is planar else 4)
```

which is what `tests/conftest.py::max_rank_K` asserts, and the exact trivial space is just an
orthonormal basis of `ker(B_K)` — by Theorem 1 that *is* the trivial variation set. `trivial_modes`
still hardcodes three translations plus scaling and is therefore wrong for mixes; it is replaced by
the `ker(B_K)` basis in WP2, together with the rest of the flex rework.
