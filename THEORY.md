# Theory

The algebra behind the environment: what the rigidity matrix is, what its rank and null space mean,
and how every derived quantity in the code (`rank_K`, `c_max`, `m_req`, the state score, the flex
features) follows from it.

`CLAUDE.md` is what the code is, `DESIGN_NOTES.md` is why it is written that way. This file is the
maths.

---

## 1. Notation

`n` agents. Agent `i` has position `p_i ∈ R³` and orientation `R_i ∈ SO(3)`; a *domain* fixes which
of those degrees of freedom actually exist (`R^2`, `R^3`, `R^2xS^1`, `R^3xS^1`, `SE(3)`).

A directed edge `i -> j` means **`i` measures the bearing to `j`**:

```
p_ij  = p_j - p_i          p̂_ij = p_ij / ||p_ij||        b_ij = R_iᵀ p̂_ij
```

`b_ij` is the measurement in `i`'s own frame. For `R^2` / `R^3` there is no frame and `R_i = I`, so
`b_ij = p̂_ij` is a **global-frame** vector - the fact that makes the observation's use of raw
bearings rotation-dependent (§11).

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
bearing - this is the scale blindness of bearing measurements, and it is the single most important
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

The construction is checked against its own definition by central differences -
`tests/test_rigidity_matrix.py::test_matrix_is_the_numerical_jacobian_of_the_bearings` asserts
`B δ = d/dt bearings` to 1e-6 relative for random admissible `δ`, in all five domains and eight
heterogeneous mixes. That validates `Dp`, `Da`, both incidence signs and both projectors at once.

---

## 3. Trivial motions and `rank_K`

A motion is *trivial* when it changes no bearing. From (2.1)-(2.2):

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

**A framework is Infinitesimally Bearing Rigid (IBR) iff `rank(B) = rank_K`** - it admits no motions
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
the three terms differ and `c_k` genuinely varies - on the `mixed` scenario the complete graph's
block ranks are `{1: 12, 2: 78}`, the 12 being the ordered pairs whose measurer *and* target are
both planar.

This is why `rigidity_edge` is nearly useless on its own: as an observation channel `c_k` is a
constant in every homogeneous configuration. It varies only in heterogeneous networks, where the
endpoints' domains differ (measured: a mix of 1s and 2s). Verified empirically at n=4/8/16 in both
R² and R³ - all edges identical.

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
form of Trinh et al. (`MBR_required_Rd`). Checked at n=4…16 in R² and R³ - all agree.

| config | rank_K | c_max | m_req |
|---|---|---|---|
| n=4 / R² | 5 | 1 | 5 |
| n=8 / R² | 13 | 1 | 13 |
| n=8 / R³ | 20 | 2 | 10 |
| n=16 / R³ | 44 | 2 | 22 |

Brute force finds the bound tight on every instance small enough to check exhaustively (n=4 across
8 domain mixes × 3 seeds, n=5 across 6 mixes, all five domains) - evidence, not proof. **It is kept
out of the reward for exactly this reason** (§7).

---

## 6. `is_MBR` - the minimality heuristic

Sort the current graph's `{c_e}` descending, accumulate until the sum reaches `rank_K`, call the
count `m_req'`; declare minimal iff IBR and `m = m_req'`. Sound as a lower bound by the same
subadditivity, and exact for homogeneous `R^d`. It can produce **false negatives** on heterogeneous
networks, where the highest-rank blocks may not be jointly realizable.

Note `m_req'` here is derived from the *current* edge set, so it is not an episode constant - unlike
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
- dimensionless.

**The central guarantee.** Adding an edge contributing `Δr` rank:

```
Δφ = ( w_r·Δr - w_e·c_max ) / rank_K
```

At the best case `Δr = c_max` this is `(w_r - w_e)·c_max/rank_K`, which is **positive iff
`w_r > w_e`** - for any geometry, any domain mix, any `n`, because the same `c_max/rank_K` factor
appears on both sides. Under an `m/m_req` normalization the two factors would be `c_max/rank_K`
and `1/m_req`, which coincide only when `m_req` happens to equal `rank_K/c_max`; the guarantee was
then contingent on a heuristic being tight. This is why `m_req` was removed from the reward.

Removing a redundant edge (`Δr = 0`) gains `w_e·c_max/rank_K`. So

```
(rank-adding edge) / (pruning a redundant edge)  =  (w_r - w_e)/w_e  =  3   at (100, 25)
```

reproducing R³'s 3:1 preference, identically in every domain.

**Optimum.** At `rank = rank_K` and `m = m_min`:

```
φ* = w_r - w_e · m_min · c_max / rank_K   ≤   w_r - w_e = 75
```

with equality iff the poses admit a perfectly packed rigid graph (`m_min = rank_K/c_max`). Check
against measurement: greedy at n=8/R³ reaches `m = 10.80`, giving
`100 - 25·10.80·2/20 = 100 - 27.0 = 73.0` - exactly the 73.00 reported. At n=4/R², `m = 5` and
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
`φ` along the trajectory* - get good fast and stay good. That is the intended behaviour, and it
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

## 9. Flexes - the null space features

> **Superseded by §13 for the observation.** Everything below works on the *position block*
> `B_p = B[:, :3n]` alone, so it is blind to the attitude columns and measured AUC 0.634 at
> predicting rank gain in `SE(3)`. §13 derives the replacement from the full `ker(B)`.
> `flex_tensor` / `flex_constraint_power` are retained and tested as the reference the §10
> ground-truth check runs against; the environment no longer calls them.

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
first few eigenvectors - any orthonormal basis of `N` mixes them arbitrarily. Skipping columns by
index therefore does not remove them. `trivial_modes()` builds the 3 translations and the uniform
scaling analytically, orthonormalizes them, and they are projected out of `N` explicitly. Before
this fix the feature failed to localise on an obviously under-constrained node and was not
rotation-invariant.

**(b) A single eigenvector is a basis artefact.** `dim F > 1` is normal, and any individual vector
inside a degenerate eigenspace is arbitrary - not even reproducible between calls. The
basis-independent object is the **projector** onto `F`:

```
Π = Σ_c v_c v_cᵀ        for any orthonormal basis {v_c} of F                         (9.3)
```

`flex_tensor` returns `Π` as `(n, n, 3, 3)` blocks, `Π[i,j] = Σ_c v_{c,i} v_{c,j}ᵀ`.

When `F` is empty (rigid), the smallest **non-zero** mode is used instead - the direction the
framework resists least, i.e. the rigidity eigenvalue's eigenvector. The feature degrades gracefully
from "where it is free" to "where it is nearly free".

### Feature 1 - how free a node is

```
node_freedom_i = sqrt( tr Π[i,i] ) · sqrt(n)                                             (9.4)
```

`tr Π[i,i] = Σ_c ||v_{c,i}||²` is node `i`'s share of the flex space. Summing over nodes gives
`tr Π = dim F`, the rank deficit - so per-node magnitudes scale as `sqrt(deficit/n)` and the
`sqrt(n)` keeps the feature `O(1)` as `n` grows. That matters directly for a policy meant to span
several `n`.

### Feature 2 - would this edge help

This is the one that has to be derived rather than guessed. Adding edge `i -> j` imposes

```
P(p̂_ij) (v_j - v_i) = 0
```

by (2.1). A flex `v` **survives** the new edge iff `v_j - v_i` is *parallel* to `p̂_ij` - the
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

### Invariance

Under a global rotation `R`, positions rotate and the null vectors rotate blockwise, `v_i → R v_i`,
so `Π[i,j] → R Π[i,j] Rᵀ`. Then

- `tr Π[i,i]` is invariant (trace is similarity-invariant),
- `p̂ᵀ Π p̂ → (Rp̂)ᵀ R Π Rᵀ (Rp̂) = p̂ᵀ Π p̂` is invariant,

so both features are rotation-invariant scalars. This is deliberate: feeding a flex **vector** as a
node feature would repeat the mistake §11 records for bearings - rotation-equivariant
data consumed as invariant scalars, making the policy rotation-dependent.

---

## 10. Ground-truth validation of (9.5)

Take nodes 0-6 fully connected (rigid) and node 7 held by a single bearing `7 -> 0`, in R³. Then
`rank_K_pos - rank(B_p) = 1`, so `dim F = 1`.

| check | result |
|---|---|
| `Σ_i tr Π[i,i]` vs `dim F` from (9.2) | `1.0000` vs `1` |
| `node_freedom` argmax | node 7 (2.501, others 0.27-0.72) |
| mean `A[i,j]` over candidates that **do** raise rank | **2.166** |
| mean `A[i,j]` over candidates that **do not** | **0.0000** |
| rotation invariance, `node_freedom` / `A` | 1.1e-15 / 4.2e-08 |

The separation is exact: every candidate edge that raises the rank has `A > 0`, and the one that
does not has `A = 0` identically. That is (9.5) behaving as derived - a flex is destroyed iff the
relative motion has a component perpendicular to the new bearing.

---

## 11. Known caveats

1. **`m_req` is a bound, not a truth** (§5). Kept out of the reward; used for reporting and the MBR
   metric only.
2. **`is_MBR` can false-negative on heterogeneous networks** (§6), so a genuinely minimal graph may
   never be recognised and a `MinimallyRigid` episode may never terminate.
3. **`node_freedom`'s scale changes meaning at the rigid boundary.** For `dim F ≥ 1`,
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
   mix - invisible in every `R^d` test. `get_all_pairs_bearings_world()` exists for this.

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

> A self-contained write-up with full proofs, the counterexample and the numerical
> protocol is in `docs/dof_restriction_note.tex`; `docs/verify_dof_restriction.py`
> reproduces every number in it.

### 12.2 What Table III does, and why it is not enough

Table I gives `U_ij`, `V_ij` per manifold for *homogeneous* formations, and there it satisfies the
requirement: for `R^2`, `U_ij = [e₁ e₂ 0]` for every edge, so the z columns are zero.

Table III is their heterogeneous case study (three `R^2xS^1` terrestrial robots, one `R^3xS^2`
aerial platform) and sets

```
U_(1,4) = U_(2,4) = U_(3,4) = I₃          # planar robot measuring the aerial platform
```

`U_ij` multiplies the whole relative displacement `(p_j - p_i)`, so this reactivates the *planar*
agent's z column as well as the aerial one's. The paper's own accounting notices the consequence -
it reports `q_v = 6` (only the terrestrial robots' unfeasible x/y rotations) and then has to
classify the planar agents' z columns as *linearly dependent on the rest* rather than as null. That
happens to hold for their particular four-agent configuration. It is not a general fact.

### 12.3 What goes wrong

Measured on random configurations:

| mix | Σ dim D_i | `rank_K` under Table III | corrected | IBR verdicts differing |
|---|---|---|---|---|
| 2 of each of the five domains | 36 | **36** | 33 | 2.0% of 300 graphs |
| 5×`R^2` + 1×`R^3` | 13 | **14** | 10 | **40%** |
| 3×`R^3` + 3×`SE(3)` | 27 | 23 | 23 | 0% (no planar agent) |

`rank_K = Σ dim D_i` means *zero* trivial motions, and `rank_K > Σ dim D_i` is outright impossible:
the matrix cannot have more independent columns than the system has coordinates. Directly: for
4×`R^2` + 4×`R^3`, a pure `+z` motion of a planar agent gives `‖B v‖ = 2.1` - the framework resists
a motion the agent cannot make, and spends rank doing it.

**The obstruction is structural, not a wrong table entry.** In `B_p = D_p U Ēᵀ` the same `U_ij`
multiplies the column blocks of *both* endpoints of an edge. Requiring it to be faithful on
admissible variations *and* to annihilate inadmissible ones forces `D_p,k (S_i − S_j) = 0`, i.e.
`range(S_i − S_j) ⊆ span{p̂_ij}`. For a planar `i` and a spatial `j` that says `p̂_ij = ±e₃`: the two
agents must be vertically stacked. So no choice of `U_ij` works off a measure-zero set. Proof in
`docs/dof_restriction_note.tex`, Theorem 1.

**Consequence.** Whether a planar agent's z-column vanishes becomes a property of the *graph*
(it vanishes iff that agent has no spatial neighbour), so `q_v` no longer cancels between `G` and
`K` - which is exactly the step the paper's Theorem 2 proof relies on. Measured: the resulting rank
test returns the wrong IBR verdict on 6.8% of 1200 random heterogeneous frameworks. An explicit
4-agent counterexample is in the note.

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
  only at `v = e₃` - the only axis ever used, so nothing measured depended on it - but `v vᵀ` is
  right for any axis, and it is the form that survives the numerical-Jacobian test with
  `v = (1, 2, -0.5)/‖·‖`.
- **`bearing_DOFs` is retained**, unused by the matrix, as the reference implementation of Table I.
  `test_matches_michieletto_table_I_on_homogeneous_networks` asserts the two constructions produce
  the *same matrix* (max abs difference 0.0, 60 graphs per domain), which is what guarantees no
  homogeneous result moved. **It is a faithful Table I reference only at `v = e₃`**: it stores the
  `R^3xS^1` rotational entry as `e₃vᵀ` (rows) where Table I has `[0_{3x2} v]` (columns), and the
  projector is `v vᵀ`. All three coincide at `e₃`, the only axis any scenario uses, so nothing
  measured depends on it; but a Table I comparison off `e₃` must not use it.
  `docs/verify_dof_restriction.py` implements Table I directly for that reason.

### 12.5 Consequences for the trivial space

§3's table stays correct for homogeneous domains. For a mix the trivial dimension is no longer read
off a table: the z-translation is trivial only when no agent is planar, and a coordinated rotation
only when *every* agent carries a frame (an `R^d` agent measures in the global frame, so rotating
the world changes what it sees). The robust statement is

```
rank_K  ≤  Σ_i dim D_i  −  (3 if any agent is planar else 4)
```

which is what `tests/conftest.py::max_rank_K` asserts, and the exact trivial space is just an
orthonormal basis of `ker(B_K)` - by Theorem 1 that *is* the trivial variation set. `trivial_modes`
still hardcodes three translations plus scaling and is therefore wrong for mixes; it is replaced by
the `ker(B_K)` basis of §13.

---

## 13. The exact addition criterion

§9 asked "would this edge help?" and answered it with a projector built from the position block.
That is a heuristic in two ways: it discards the attitude columns, and it measures destroyed flex
rather than added rank. Both are avoidable - the exact criterion is one line of linear algebra.

### 13.1 The criterion

Adding edge `i -> j` appends its 3-row block `b_ij` to `B`. Let `Z` be an orthonormal basis of
`ker(B)`. Then

```
rank([B; b_ij]) - rank(B) = rank(b_ij Z)                                            (13.1)
```

*Proof.* Row space and null space are orthogonal complements, so `b_ij` raises the rank by exactly
the dimension of its component outside `rowspace(B) = ker(B)^⊥`. Projecting onto `ker(B)` is
`b_ij Z Zᵀ`, and `rank(b_ij Z Zᵀ) = rank(b_ij Z)` since `Z` has orthonormal columns. ∎

In particular `rank(B)` is unchanged iff `b_ij Z = 0`. This is exact, needs no threshold on a
difference of two ranks, and holds in every domain and every heterogeneous mix, because it says
nothing about what `B` is - only that it is a matrix.

### 13.2 The two features

```
add_independence[i,j] = ||b_ij Z||_F / ||b_ij||_F  ∈ [0, 1]
add_rank[i,j] = rank(b_ij Z) / c_max       ∈ [0, 1]
```

over **all ordered pairs**, not just the absent ones. `add_rank` is the answer to the question;
`add_independence` is its continuous relaxation, and carries strictly more information - it distinguishes
an edge that barely escapes the row space from one that is fully outside it, which is what a value
function needs in order to prefer one of two rank-1 edges.

The normalisation is per pair, by that pair's own `||b_ij||`. Normalising against the spread over
pairs (as the `node_freedom` channel does) is wrong here: on a rigid framework `ker(B)` is exactly the
trivial space, every raw gain is at machine zero, and dividing those by their own RMS turns
rounding noise into an O(1) feature.

`candidate_gain_reference` states (13.1) directly: one pair at a time, with `b_ij` built by calling
the matrix routine on a network carrying only that edge. `candidate_gain` expands the three nonzero
blocks by hand,

```
b_ij Z = D_p (S_j Z_j − S_i Z_i) − D_a P_i Z_i
```

into batched products over all pairs, and is checked against the reference in every domain. Note the
minus: `Ē_o` places `−1` at the measuring node only. It is invisible in `R^2`/`R^3`, where `P_i = 0`.

`b_ij` has 3 rows, so `G = (b_ij Z)(b_ij Z)ᵀ` is 3×3. `tr G` gives the norm and `eigvalsh(G)` the
rank, at a cost that does not grow with `dim ker(B)` - a batched SVD of the (3, k) blocks would.

### 13.3 The rank threshold

`add_rank` counts eigenvalues of `G` above `1e-12 · ||b_ij||²`, i.e. `add_independence > 1e-6`. That cut is
measured, not guessed. Over 1,501 candidate pairs across all five domains and three mixes:

| | `add_independence` |
|---|---|
| pairs that add no rank | max `1.59e-10` |
| pairs that add rank | min `1.43e-02` |

Eight orders of magnitude of empty space, and `1e-6` sits in the middle of it. The original cut was
at `1e-18` relative, which is *below* the noise floor of a Gram matrix in double precision, so
borderline eigenvalues flipped whenever the geometry was translated or scaled and the channel
drifted by a full rank unit. See §13.5.

### 13.4 Scale invariance and the length unit

`B` is dimensionally inhomogeneous: the position columns carry `1/||p_ij||` and the attitude columns
are dimensionless (§12.4). Scaling the whole formation by `α` therefore does **not** scale `B`
uniformly, and `ker(B)` genuinely moves. The features would then depend on the units the poses
happen to be in, which is unacceptable for a policy meant to transfer.

The fix is to fix the length unit to something the formation carries itself. `characteristic_length`
returns the RMS radius about the centroid, `nullspace_in_scaled_units` divides the position rows of
`Z` by it and re-orthonormalises, and `candidate_gain` multiplies `Dp` by the same factor. This is
the same normalisation `coord_features` already applies to positions, so the observation is
consistent about what "unit length" means. The rigidity maths itself still runs on the true poses.

A second, easily missed prerequisite: `P_i = v_i v_iᵀ` for `R^dxS^1` is expressed in **world**
coordinates, so a global rotation has to rotate `agent.rotation_axis` too. `rotate_network` did not,
which broke `R^3xS^1` rotation invariance for reasons that had nothing to do with these features.

### 13.5 Validation

`add_independence` against ground truth (rebuild `B` with the edge added, recompute the rank), 6-node
frameworks at 40% density:

| domain | pairs | AUC `add_independence` | AUC `flex_align` (§9) | clean split | exact rank |
|---|---|---|---|---|---|
| `R^2` | 111 | **1.000** | 1.000 | yes | 111/111 |
| `R^3` | 155 | **1.000** | 1.000 | yes | 155/155 |
| `R^2xS^1` | 264 | **1.000** | 0.678 | yes | 264/264 |
| `R^3xS^1` | 250 | **1.000** | 0.807 | yes | 250/250 |
| `SE(3)` | 268 | **1.000** | 0.634 | yes | 268/268 |
| all five mixed | 160 | **1.000** | 0.767 | yes | 160/160 |
| `R^2` + `SE(3)` | 118 | **1.000** | 0.648 | yes | 118/118 |
| `R^2xS^1` + `R^3xS^1` | 115 | **1.000** | 0.567 | yes | 115/115 |

"Clean split" means every rank-adding pair scores above every non-adding pair, with no overlap.
`flex_align` is at chance in the oriented domains, which is the failure §9's position-only
derivation predicts: it cannot see a bearing that pins down an attitude.

`dim flex_space` equals `rank_K - rank(B)` exactly in every case, which is (9.2) generalised - and
note it needs no hand-built trivial modes, because `ker(B_K)` **is** the trivial variation set by
Michieletto Theorem 1, in every domain and mix.

Invariance, worst channel over 6 instances each (`tests/test_invariance.py`):

| domain | translate | scale | rotate |
|---|---|---|---|
| `R^2` | 1.7e-12 | 4.5e-15 | 8.7e-01 (bearings, §11.3) |
| `R^3` | 2.6e-14 | 5.0e-15 | 8.3e-01 (bearings, §11.3) |
| `R^2xS^1` | 1.8e-13 | 4.4e-12 | 3.6e-14 |
| `R^3xS^1` | 1.8e-11 | 3.9e-13 | 1.2e-13 |
| `SE(3)` | 7.4e-14 | 3.2e-13 | 2.1e-13 |
| mixed | 2.0e-13 | 3.6e-13 | 8.7e-01 (bearings, §11.3) |

The rotation column is the known `R^d` global-frame bearing artefact of §11.3 and is not these
features; `add_independence` and `add_rank` are themselves invariant to 1e-13 under all three transforms in
every domain.

### 13.6 Cost

Single-threaded, at 60% of `m_req` edges, milliseconds:

| n | domain | build `B` | rank (SVD) | `ker` (eigh) | `candidate_gain` | flex | total |
|---|---|---|---|---|---|---|---|
| 8 | `R^3` | 0.21 | 0.04 | 0.07 | 0.23 | 0.08 | **0.63** |
| 8 | `SE(3)` | 0.38 | 0.10 | 0.14 | 0.22 | 0.02 | **0.85** |
| 16 | `R^3` | 0.44 | 0.19 | 0.28 | 0.69 | 0.29 | **1.90** |
| 16 | `SE(3)` | 0.89 | 0.48 | 0.52 | 0.59 | 0.03 | **2.50** |
| 24 | `R^3` | 0.77 | 0.52 | 0.70 | 1.52 | 0.65 | **4.16** |
| 24 | `SE(3)` | 1.52 | 1.28 | 1.10 | 1.21 | 0.07 | **5.17** |

Two things paid for this. `nullspace` takes `eigh(BᵀB)` rather than an SVD of `B`, whose left factor
is (3m, 3m) and never used - 13.15 ms to 2.50 ms at n=16. Squaring costs precision in the
eigen*values*, which is why the rank is still read off the thin SVD in `rigidity_decomposition` and
only the eigen*vectors* come from `eigh` (taking the rank from `eigh` disagreed with `matrix_rank`
on 840 of 840 cases). And `candidate_gain` uses the 3×3 Gram matrix instead of a batched SVD, 1.77
ms to 0.59 ms at n=16.

Measure this pinned to one BLAS thread. Unpinned, `eigh` on a 144×144 matrix was reported at
anywhere from 0.26 to 16 ms on the same input, which is thread contention rather than the algorithm.

---

## 14. Submodularity: why greedy is strong on the edge count and weak on the stiffness

This section explains what kind of optimization problem each objective is. It is the reason the
constructive baseline is hard to beat, and the reason stiffness is worth switching to.

### 14.0 `rank(B)` is generically a function of the graph alone

`B(χ)` is the Jacobian of the bearing map and does depend on the configuration χ. **Its rank does
not**, generically. Every entry of `B(χ)` is rational in χ, so `rank(B(χ))` is lower semi-continuous
and drops only on the zero set of a minor determinant - a proper algebraic subset, of measure zero.
For χ drawn from any continuous distribution, `rank(B(χ))` equals its generic value almost surely,
and that generic value is fixed by the graph and the domain assignment. Michieletto's
noncollinearity assumption is exactly the assumption that puts us in the generic set.

*Measured:* 30 graphs x 5 domains x 100 pose resamples each - the rank never moved once. Repeated
for heterogeneous mixes with the same result.

**So a rank-based state score contains no geometry.** `WeightedNormalized` at `stiffness_kappa = 0` is
a function of the edge set alone, which is why a policy trained on it reads none of the geometric
observation channels: the reward never asks it to. What does *not* follow is that geometry is
useless as an input - it is the computational route to the combinatorial answer (§13 turns a
geometric computation into an exact rank prediction), and the **stiffness** of §15 is not combinatorial
at all, spanning ~10^5 across equally-minimal graphs on the same poses.

### 14.1 What submodular means

A **set function** `f(S)` assigns a number to every subset `S` of the candidate edges. It is
**monotone** if adding an edge never decreases it, and **submodular** if it has *diminishing
returns*: an edge is worth less once you already have more edges. Formally, for `S ⊆ T` and
`e ∉ T`,

```
f(S ∪ {e}) − f(S)   ≥   f(T ∪ {e}) − f(T)                                            (14.1)
```

The left side is the edge's marginal value in the small graph, the right side in the big one.
Submodular means the small graph gets at least as much out of it.

### 14.2 `rank(B_S)` is monotone submodular

Each edge `e` contributes a 3-row block to `B`, so it contributes a small subspace `V_e` (that
block's row space). The rank of the assembled matrix is the dimension of the sum of those
subspaces:

```
f(S) = dim( Σ_{e ∈ S} V_e )
```

Monotone is immediate. For submodularity, the marginal value of `e` is the part of `V_e` not
already spanned:

```
f(S ∪ {e}) − f(S) = dim(V_e) − dim(V_e ∩ V_S)                                        (14.2)
```

`S ⊆ T` gives `V_S ⊆ V_T`, so `dim(V_e ∩ V_S) ≤ dim(V_e ∩ V_T)` and the marginal can only shrink.
That is (14.1). ∎

*Measured* (`tools/submodularity.py`): 1920 (S, T, e) triples across all five domains and a
heterogeneous mix, **0 violations**, worst marginal gap exactly 0.

### 14.3 What that buys, and what it costs us

"Fewest edges making the framework rigid" is therefore **minimum submodular cover**, a named
problem. Wolsey (1982) proved that greedy - repeatedly take the edge with the largest marginal
gain - is an `H(d)` approximation for integer-valued monotone submodular cover, where
`d = max_e f({e})` and `H(k) = 1 + 1/2 + … + 1/k`. Here `d = c_max`:

| domain class | `c_max` | greedy guarantee |
|---|---|---|
| `R^2`, `R^2xS^1` | 1 | `H(1) = 1`, i.e. **exact** |
| `R^3`, `R^3xS^1`, `SE(3)` | 2 | `H(2) = 1.5` |

The `c_max = 1` row is the matroid statement of §1.4 recovered from a different direction.

Two consequences, and the second is uncomfortable:

- **The constructive baseline is not ad hoc.** It is the standard algorithm for this problem class,
  with a proof behind it. That makes it the right opponent, and makes beating it meaningful.
- **The headroom above greedy is small.** Measured against the proven lower bound `m_req`, greedy
  lands 0-5% above it (0% in `SE(3)` and `R^2xS^1` at n=8, +2.4% on `mixed`, +5.0% at n=8/`R^3`,
  +3.0% at n=16/`R^3`) - far better than its 50% worst case. So no method, learned or otherwise,
  can gain much on edge count. A trained policy closes about 88% of that gap on the training
  mixture, which is close to all there was.

This is the structural reason a rank-based objective cannot carry the thesis on its own, and it
agrees with Darvariu et al. (2024) §6.2: RL is not expected to gain much where shallow decision
horizons already suffice.

### 14.4 Stiffness is **not** submodular

The rigidity eigenvalue `λ_r(S)` is still **monotone** - adding an edge adds a PSD term to `BᵀB`,
and by Weyl's inequality every eigenvalue can only move up. But it is not submodular.

*Measured* (`tools/submodularity.py`): 1493 triples with both `S` and `T` rigid, across all five
domains and a mix: **887 violations of (14.1), 59.4%**, worst marginal gap −4.16e-01.
Non-submodularity needs only one valid counterexample; there are 887, far above numerical noise.

The intuition is that eigenvalues are global and coupled in a way ranks are not. Two edges can be
worth more together than the sum of their separate contributions - a complementary pair bracing a
direction that neither braces alone. That is *increasing* returns, the exact opposite of (14.1), and
it is invisible to a method that only ever evaluates one edge at a time.

**So greedy carries no approximation guarantee on the stiffness.** That is the principled reason to
expect a sequential, long-horizon method to have room there when it has almost none on edge count,
and it is the argument the stiffness objective (§15) rests on. Stated as a prediction rather than a
result: a stiffness-aware
policy should beat greedy on stiffness by a wider relative margin than the ~2% it wins on edge count.
A spectral first-order heuristic is the honest opponent to hold it to, since
greedy-on-stiffness is a weak one.

*Caveat.* §14.2 is proved and then confirmed numerically; §14.4 is numerical only, but for a
*negative* result that is the stronger position - one counterexample refutes submodularity, whereas
no number of confirmations would prove it.

## 15. The stiffness in the state score

§14 says the edge-count problem is nearly solved by greedy and the stiffness problem is not. This
section is the objective that follows from that.

### 15.0 What λ means, and why it is called stiffness

**Rigidity is binary and generic** (§14.0). At a non-degenerate configuration a framework either
attains `rank_K` or it does not, and if it does, the shape is recoverable from the bearings. λ adds
nothing to *whether*, and calling a large-λ framework "more rigid" is wrong.

What λ is: with `δχ` a variation orthogonal to the trivial motions (`ker B`),

```
‖B δχ‖  ≥  √λ · ‖δχ‖                                                                 (15.0)
```

so λ lower-bounds **how much the bearings move when the shape is deformed**. Inverting it gives the
statement that matters:

```
shape error  ≤  (1/√λ) · bearing error
```

λ is therefore the **conditioning of the bearing → shape inverse problem**, not a degree of rigidity.
`BᵀB` is the Hessian of the bearing error and, for isotropic bearing noise, proportional to the
Fisher information, so λ is also the worst-direction observability of the shape and `1/√λ` the
Cramér-Rao error amplification. **This document calls λ the stiffness of the weakest non-trivial
mode**, because `ẋ = -BᵀB x` makes that literal and because "margin" names only the Eckart-Young
reading in point 3 below, which is a proxy rather than the meaning.
Exact bearings recover the shape at any `λ > 0`; noisy bearings recover it to within a factor
`1/√λ`. *Measured*, three rigid graphs on one set of 8 poses in `R^3`, unit-norm bearing
perturbations pushed through `B⁺`:

| λ | `1/√λ` | measured worst amplification |
|---|---|---|
| 1.4e-05 | 264 | 132 |
| 5.7e-03 | 13.2 | 6.9 |
| 9.8e-02 | 3.2 | 1.8 |

(Random draws rarely hit the worst-case direction, so the measured column sits below the bound; the
*scaling* is what matters, and it drops two decades alongside `1/√λ`.)

Three consequences, worth keeping separate because they are usually conflated:

1. **Estimation.** The above: λ is the amplification from measurement noise into shape error. This
   is the one that justifies the objective.
2. **Control.** For a bearing-based formation controller the linearised error dynamics are
   `ẋ = -BᵀB x`, so the slowest non-trivial mode decays at rate λ. This is where the name comes from
   in the formation-control literature. Nothing here runs a controller, so it is motivation rather
   than a measured claim.
3. **Distance to degeneracy.** By Eckart-Young `√λ` is exactly the spectral-norm distance from `B`
   to the nearest rank-deficient matrix, so a low-λ graph sits close to losing rigidity outright.
   The caveat: that nearest matrix need not be the rigidity matrix of any realisable configuration,
   so this bounds distance-to-singularity in *matrix* space, not in configuration space. A good
   proxy, not a theorem about poses.

The collinear case is the limit of all three: as a configuration approaches degeneracy λ → 0
continuously, and rigidity fails at exactly the point λ reaches zero. λ is the continuous quantity
whose vanishing *is* the binary failure, which is what makes it the thing to maximise away from.

### 15.1 Why λ cannot enter φ raw

Two obstacles stop λ going straight into φ.

**λ is monotone in edges.** Adding an edge appends rows to `B`, hence a PSD term to `BᵀB`, so by
Weyl's inequality every eigenvalue can only rise. Measured at n=8/`R^3`:

```
m  = 12    16     20     24     28     32     36
λ  = .001  .005   .218   .259   .365   .620   .701      ~700x over the range
```

So `max λ` alone has the **complete graph** as its optimum: added with any positive weight it pays
the agent to add edges.

**λ has no fixed scale.** It decays with `n` (1.5e-01 at n=4 to 1.3e-03 at n=16 in `R^3`) and scales
with the size of the formation, because `B`'s position entries carry `1/‖p_ij‖`. A weight tuned at
one `n` and one pose range means nothing at another, destroying the property §7 exists for.

λ therefore has to be squashed into a bounded, dimensionless number, and a squashing function needs
a **centre**. That is what `λ_ref` is.

### 15.2 The formula

```
phi = (w_rank*rank - w_edge*m*c_max)/rank_K  +  w_eig * 1[is_IBR] * q(lam)        (15.1)

  q(lam) = sigmoid( log10(lam / stiffness_ref) / s ),      s = 0.75 decades      -> q in (0,1)
  w_eig  = kappa * w_edge * c_max / rank_K
```

- **`λ_ref`** is the median λ of `stiffness_ref_samples` graphs built by the constructive greedy **on
  this episode's own poses** (`rigidity.reference_stiffness`), so `log10(λ/λ_ref)` reads "how much
  stiffer than a typical decent graph on these exact poses" - dimensionless, comparable at any `n`,
  domain and formation size, and near 0 for a typical answer. λ of the *complete* graph would
  saturate instead: `λ/λ_K` still decays two decades from n=4 to n=12. The **median**, not the best:
  a yardstick has to be a typical answer for `q ≈ 0.5` to mean "typical".
- **`s = 0.75` decades**, because the p10-p90 spread of `log10 λ` among minimal graphs is 1.1-1.9
  decades, so the logistic spends its range on the achievable band.
- **`w_eig` is denominated in edges.** `w_edge·c_max/rank_K` is what one edge costs, so the whole
  stiffness term is worth `κ` edges and rescales with `n` and domain by itself. `κ = 0` reproduces the
  rank-only score exactly, and the term is bounded in `[0, κ·one_edge)`, so the agent can profit by
  at most about `κ` extra edges. `κ < 1` is a tie-break sparsity always wins; `κ > 1` is a real
  trade-off with no principled value, answered by a front over κ rather than a number.
- **`q ≥ 0`, gated on IBR**, so becoming rigid is never punished; a raw `log λ` term would make the
  transition to rigidity a large negative jump.

`κ`, `λ_ref`, `rank_K` and `c_max` are constant within an episode, so (15.1) is still a **potential**
and the shaping stays potential-based (§7).

### 15.3 `λ_ref` is noisy, and that is why it is a median

A single greedy construction is a poor centre. Measured across construction orders on *fixed* poses:

| | p10-p90 of `log10 λ_ref` | sd | cost/episode |
|---|---|---|---|
| n=8/`R^3`, k=1 | 2.26 decades | 0.99 | 15.6 ms |
| n=8/`R^3`, k=3 | **0.48** | **0.22** | 46.4 ms |
| `mixed`, k=1 | 2.31 decades | 1.07 | 43.3 ms |
| `mixed`, k=3 | **1.31** | **0.65** | 123.3 ms |

At k=1 the centre wobbles by more than the 1.1-1.9 decade signal it is meant to centre, and further
than the sigmoid is wide, so in many episodes `q` would sit saturated and contribute no gradient.
The **log-median** of k=3 narrows it 4.5x at n=8/`R^3` - better than `1/sqrt(k)`, because the median
is robust to the occasional bad construction that dominated the spread. `stiffness_ref_samples` is a
config key; raise it if the residual variance matters. The cost is entirely in `reset()`
(2.7 -> 46.8 ms at n=8/`R^3`); **per-step cost is unchanged**, since λ comes from the SVD `step()`
already performs.

### 15.4 Invariance, and the one place it is only approximate

λ and λ_ref are computed on the same poses, so a similarity transform applies to both and cancels.
That is why (15.1) needs no pose normalization, and so avoids a second SVD per step. Measured,
complete graph against its own greedy reference:

| domain | translate | rotate | scale x1.5 | x2.7 | x10 |
|---|---|---|---|---|---|
| `R^2`, `R^3` | exact | exact | **0.0000** | **0.0000** | **0.0000** |
| `R^2xS^1`, `R^3xS^1`, `SE(3)` | exact | exact | 0.077 | 0.13-0.16 | 0.16-0.21 |

(decades of drift in `log10(λ/λ_ref)`.)

**Translation and rotation are exact in every domain, and so is scaling in `R^d`** - there every
column of `B` carries `1/length`, so a rescale is a scalar factor that cancels in the ratio. In the
oriented domains it does not: the position columns carry `1/length` while the attitude columns are
dimensionless (§13.4), so a rescale genuinely reweights them against each other and moves the
spectrum. At worst 0.21 decades, i.e. **~7% of one edge**.

That is acceptable **only because every instance shares a pose scale** - `random_scenario` draws
from a fixed `pos_limits`, and neither `rotation_augmentation` nor the benchmark sets change it. A
standing condition, not a proof: if instances ever carry genuinely different physical scales, λ must
be pose-normalized (scale `B`'s first `3n` columns by the formation's RMS radius before the
decomposition). `tests/test_state_score.py` asserts the three regimes separately.

### 15.5 What it buys, measured

`greedy` hill-climbs on whatever φ is configured, so it is stiffness-aware at `κ > 0` and gives a
reading without any training. 12 instances, n=8/`R^3`, identical poses across arms
(`tools/kappa_sweep.py`):

| κ | edges | stiffness (gmean) | vs κ=0 | rigid |
|---|---|---|---|---|
| 0 | 10.42 | 1.01e-03 | 1.00x | 100% |
| 0.9 | 10.42 | 2.01e-03 | **2.0x** | 100% |
| 2.0 | 10.50 | 1.25e-02 | **12.4x** | 100% |
| 4.0 | 10.42 | 2.08e-02 | **20.6x** | 100% |

Edge count is flat to within 0.08 of an edge across the whole range while stiffness moves 20x, and
rigidity never drops. Two caveats: this is greedy, not a learned policy, and 12 instances is a smoke
test rather than a result.

`constructive` does **not** adapt - it is the rank-based classical algorithm scored on the stiffness
objective, which is the comparison §14.4 predicts it loses.

## 16. The softest mode as an observation

§15 puts stiffness in the objective. This section is the feature that lets a policy act on it.

### 16.1 Why the existing rigidity channels cannot serve

`add_independence`, `add_rank` and `node_freedom` all derive from `ker(B)` (§13). On a **rigid** framework the
kernel is exactly the trivial variation set (Michieletto Thm 1), and a trivial variation changes no
bearing, so `b_ij Z = 0` for every ordered pair. Measured on one `mixed` instance, the same graph
before and after being made rigid:

| | add_independence | add_rank | node_freedom |
|---|---|---|---|
| flexible, rank 32/33 | 9/100 nonzero | 9/100 | 1/10 |
| rigid, rank 33/33 | **0/100** | **0/100** | **0/10** |

Stiffness is only defined once rigid. So in the exact regime where it is the only thing
still varying, every rigidity channel is identically zero. A policy trained on (15.1) has no
spectral information at the moment it needs it.

### 16.2 The feature

Let `v` be the eigenvector of `BᵀB` at the rigidity eigenvalue `λ`, i.e. the softest non-trivial
mode. First-order perturbation of a simple eigenvalue gives, for adding edge `i -> j`,

```
dlambda  ~=  ||b_ij v||^2                                                            (16.1)
```

and the same quantity is what removing an existing edge would cost. Two channels follow:

```
add_stiffness[i, j] = ||b_ij v||        over all ordered pairs        (pair channel)
node_slack[i]     = (||v_i^pos||, ||v_i^att||)                      (node channel)
```

Both reuse §13's machinery unchanged: `candidate_gain(network, v, L)` is already
`||b_ij Z||` for an arbitrary `(6n, k)` matrix, so passing the single column `v` in place of the
kernel basis gives `add_stiffness` with no new algebra, and `v` itself is one column past the kernel in
the `eigh(BᵀB)` that `nullspace` already performs (`nullspace_and_softest`). Measured cost at n=10:
9.45 -> 9.90 ms per environment step.

### 16.3 A ranking prior, not an oracle

Measured on `mixed`, against rebuilt-matrix ground truth:

| | |
|---|---|
| predicts the true `dlambda` of **adding** (log-log correlation) | **+0.93** |
| predicts the true cost of **removing**, on redundant graphs | **+0.35** |
| its top pick is the true best / in the true top-3 | 0/6, 1/6 |
| nonzero on a rigid graph | 100% of pairs |
| concentration, max/mean over pairs | 6.6x |

Strong in aggregate, useless as an argmax: adding an edge is not an infinitesimal perturbation and
eigenvalues cross. That is the **right** position for a learned method. `add_independence` is an exact rank
oracle, which makes the informed arm close to constructive-greedy-with-learned-ordering (§14.3);
`add_stiffness` gives a prior that still has to be refined.

**The add/remove asymmetry matters and should be stated.** 0.93 for placing an edge against 0.35 for
removing one, because removal is not a small perturbation of a redundant graph and the softest mode
itself changes. Pruning dominates the late episode, so expect the channel to help where edges are
placed, not where they are cut.

### 16.4 Invariance

| | translate | rotate | scale |
|---|---|---|---|
| `add_stiffness` | 4.8e-14 | **6.6e-14** | 2.2e-04 |

Exactly rotation-invariant in every domain, including `R^d`, where the raw bearings are **not**
(§11). The residual under scaling is the same position-versus-attitude column effect as §15.4 and is
negligible against a channel of order 1.

### 16.5 The information is irreducibly pairwise

Predicting the same true `dlambda` of adding `i -> j`:

| predictor | log-log correlation |
|---|---|
| pair, `\|\|b_ij v\|\|^2` | **+0.93** |
| node, `\|\|v_i\|\| · \|\|v_j\|\|` | +0.40 |
| node, `\|\|v_i\|\| + \|\|v_j\|\|` | +0.44 |

Per-node magnitudes capture under half the signal, and (16.1) says why:

```
b_ij v  =  Dp (S_j v_j - S_i v_i)  -  Da P_i v_i,       Dp = (I - p_hat p_hat^T)/||p_ij||
```

`Dp` projects **orthogonal to the bearing**. Two very soft nodes whose relative motion runs along
their mutual bearing produce no bearing change and contribute nothing. Whether a mode is visible to
a measurement depends on the direction between the two agents, which no per-node magnitude encodes.

This is worth stating precisely because it looks like it should reduce the way §12 did. It does not:
in `B = [Dp Ē^T S̄ | Da Ē_o^T P̄]` the DOF projectors `S` and `P` are per node, which is §12's
result, but `Dp` and `Da` are per edge and built from the bearing. The construction is both, and
only the restriction half is nodewise.

## 17. What removing an edge costs

§16 gives the policy a signal for *placing* an edge. This section is the other direction, which is
where the harder half of the episode is: measured on typical mid-episode `mixed` graphs, **70% of
existing edges are safely removable** and 30% are load-bearing, and until these channels nothing in
the observation separated them.

Both quantities below are **exact**, and both are read off matrices that already exist. The
rigidity matrix carries one 3-row block per directed edge, in `np.nonzero(edges)` order, so the
block of an existing edge is the slice `B[3k:3k+3]` rather than something to rebuild - verified
bit-identical to `candidate_block`, max abs difference 0.

### 17.1 Rank: block leverage

For a row block `b` of `B`, the **leverage block** is

```
H = b (BᵀB)⁺ bᵀ,          eigenvalues in [0, 1]                                     (17.1)
```

`(BᵀB)⁺ = V diag(1/w) Vᵀ` over the nonzero `w`, from the very `eigh` §16 already performs. `H` is
the block generalisation of the statistical leverage `h_i = r_i (BᵀB)⁺ r_iᵀ`: an eigenvalue of 1
marks a direction that **only this edge constrains**, so

```
rank(B) - rank(B without this edge) = #{ eigenvalues of H equal to 1 }               (17.2)
```

*Measured:* exact on 118/118 existing edges against rebuilt-matrix ground truth, and again per
domain in `tests/test_flex.py`. The 1e-6 cut sits in the same kind of empty band `candidate_gain`'s
does.

`remove_rank[i,j]` is (17.2) divided by `c_max`, zero on non-edges.

### 17.2 Stiffness: a rank-3 downdate

Dropping an edge subtracts its block from the Gram matrix, so the new spectrum is that of
`BᵀB - bᵀb`, and one `eigvalsh` gives the new stiffness at index `6n - rank_K` **exactly**. No
rebuilding and no first-order approximation.

```
remove_stiffness[i,j] = 1 - lambda(B without i->j) / lambda(B)        in [0, 1]      (17.3)
```

1 when removal breaks rigidity, 0 when the edge is free. It is worth computing exactly because
**nothing approximates it**: `add_stiffness` predicts the drop at 0.37 log-log correlation and
leverage at 0.369, against 0.93 for the addition direction (§16.3). The distribution is skewed and
that is the useful part - on removable edges the surviving ratio has median 0.99 and p10 0.29, so
most redundant edges really are free and a minority are expensive, and flagging that minority is
what a pruning policy needs.

### 17.3 Properties

- **Complementary support.** `remove_*` is nonzero only on existing edges, `add_*` only on
  non-edges (an existing edge already lies in the row space, so `b_ij Z = 0`). Together they cover
  every action the policy can take.
- **`remove_rank` is informative in both regimes**, unlike every other rigidity channel:
  `add_independence` dies once rigid (§16.1) and `add_stiffness` is zero while flexible, but
  removing an edge can drop the rank either way.
- **Exactly similarity invariant, scaling included.** `H = b (BᵀB)⁺ bᵀ` is unchanged by any
  invertible column scaling `b → bS`, since `(SᵀBᵀBS)⁺ = S⁻¹(BᵀB)⁺S⁻¹`. Measured 6.7e-14 under a
  2.7x rescale and 1.3e-13 under rotation, so it has none of the 1e-4 scale residual §15.4 and
  §16.4 carry.
- **Cost.** 3.46 → 5.76 ms per step at n=10 with ~35 edges, pinned to one BLAS thread, so about
  +66%. It grows as `m · (6n)³` and will need revisiting before n=32. Two skips keep it down: the
  `eigvalsh` is not run when the rank drops (the answer is 1) or when the framework is flexible (no
  stiffness to lose). Unpinned these timings are dominated by BLAS contention and are not
  meaningful; §13.6's warning applies here too.

### 17.4 What this does to the informed arm

With `add_independence` / `add_rank` for additions and `remove_rank` / `remove_stiffness` for
removals, the informed arm now has an **exact one-step oracle in both directions and for both
objectives**. Its results therefore say what a learned policy adds *on top of* perfect greedy
lookahead, not whether it can learn rigidity from geometry. The uninformed arm remains the headline,
and `rigidity_removal` is a separate flag precisely so that off-against-on prices this.

## 18. Estimation error: A-, D- and E-optimality, and what it measures

§15 calls λ the conditioning of the bearing → shape inverse problem, and states that shape error
scales as `1/sqrt(λ)`. Until this section that was an assertion: **no code in the repository had
ever perturbed a bearing.** This section derives the quantity properly, measures it, and reports
which spectral criterion actually predicts it.

### 18.1 The three criteria

A bearing is a unit vector, so measurement noise lives in its tangent plane. With independent
isotropic tangent noise of variance `σ²` on each of the `m` bearings, the log-likelihood is
`−‖B δχ − δz‖²/2σ²` to first order, so the Fisher information is `BᵀB/σ²` and the Cramér-Rao
covariance of the shape estimate is `σ²(BᵀB)⁺` on the identifiable subspace. Three scalars
summarise it, and they are the classical experiment-design criteria:

```
a_opt = tr((BᵀB)⁺) = Σ_k 1/w_k      A-optimality: total mean squared error      (18.1)
e_opt = 1/λ        = 1/w_min        E-optimality: the worst mode alone
d_opt = −Σ_k log w_k                D-optimality: log-volume of the ellipsoid
```

`w_k` are the `rank_K` nonzero eigenvalues of `BᵀB`, i.e. `s_k²` over the `rank_K` largest singular
values — so **all three are free**, read off the SVD `rigidity_decomposition` already performs and
discards (`rigidity.estimation_error`). Larger is worse for all three; all three are `+inf` on a
flexible framework, where the shape is not identifiable at all.

**Units.** `B`'s position columns carry `1/length` and its attitude columns are dimensionless
(§13.4), so a spectrum read off the raw matrix mixes units and tracks the pose range rather than the
topology. `scaled_rigidity_matrix` multiplies the position columns by `characteristic_length` first,
the same length unit `nullspace_in_scaled_units` uses. Position error is then in **formation radii**
and attitude error in **radians**, both dimensionless, and (18.1) is a well-defined sum over them.
λ has the same defect today and escapes it only because `λ_ref` shares the scale (§15.4).

The environment logs `shape_err = sqrt(a_opt / n)`: **RMS state error per radian of bearing noise**.
Unlike λ it has an absolute meaning and is comparable across `n`, domain and pose range — `8.0`
means one degree of bearing error (0.017 rad) displaces the shape by about 14% of its own size.

### 18.2 Measuring it (`estimation.py`)

`perturb_bearings` draws `z = normalize(b + σ (I − bbᵀ) ε)`, the small-angle limit of von
Mises-Fisher, so **σ is an angle in radians**. The noise is full 2-DOF tangent in *every* domain: a
planar agent's motion is restricted, its camera is not, and the component the restriction makes
unobservable is exactly the one `B` already zeroes as a zero row.

`solve_shape` runs damped Gauss-Newton on `Σ ‖b_ij(χ) − z_ij‖²` with `B` as the Jacobian — verified
against central differences of the estimator's *own* bearing map in all five domains
(`tests/test_estimation.py`), so the step really is a Newton step. `lstsq`'s minimum-norm solution
puts zero in `ker(B)`, which is the inadmissible DOFs together with the gauge, so the iterate
neither leaves an agent's domain nor drifts along a direction no bearing can see.

`shape_error` projects the error off `ker(B_K)`, the trivial variation space. **This quotient is
linearised**, exact to first order — which is the same linearisation the Cramér-Rao prediction
makes, so the two are comparable by construction. Measured: exact to 1e-16 in `R^d`, where
translation and scaling are linear in position, and `O(δ²)` in the oriented domains, where the
rotational gauge is curved. The experiment initialises at the true poses and so stays at `O(σ)`,
where the `O(σ²)` remainder is negligible.

**Initialising at the truth measures *local* accuracy**, which is what the Cramér-Rao bound
describes. Global convergence from a cold start is a different property and would need a different
experiment.

### 18.3 The prediction holds (`tools/crlb_validation.py`)

Cramér-Rao bounds `E[‖x‖²]`, so the **RMS** over trials is what it predicts; the mean sits about
`1/(4k)` below that for `k` identifiable modes, which is the ~10% gap a mean-based comparison shows.
200 noise draws per cell, ratio = measured / predicted:

| domain | 0.006° | 0.06° | 0.6° | 1.7° | 5.7° |
|---|---|---|---|---|---|
| `R^3` | 0.960 | 0.961 | 0.968 | **0.989** | 1.128 |
| `R^2xS^1` | 1.046 | 1.187 | 0.639 | 0.696 | 0.305 |
| `R^3xS^1` | 0.962 | 0.962 | 1.051 | 0.832 | 0.570 |
| `SE(3)` | 0.973 | 0.973 | 0.980 | 1.035 | 1.059 |
| `mixed` | 1.012 | 1.010 | 1.176 | 0.831 | 0.579 |

**The bound is confirmed**: agreement within a few percent at small σ in every domain, and the error
is exactly linear in σ over four decades. §15.0's claim is now measured rather than asserted.

**Where it stops.** The agreement degrades as the predicted *relative* error approaches order 1, and
it fails in both directions — above 1 the nonlinearity amplifies the error, below 1 the estimator
saturates on a wrong-but-bounded configuration rather than running off. The transition is not sharp
and is instance-dependent: `R^3` and `SE(3)` hold to 5.7°, while the poorly conditioned `R^2xS^1`
instance departs already at 0.06°. Treat the analytic metric as valid at the percent level of
relative error and check `tools/crlb_validation.py` before quoting it outside that.

### 18.4 A ≈ E, and D measures something else (`tools/spectral_criteria.py`)

288 rigid graphs per configuration:

| config | corr(logE, logA) | corr(logE, D) | p10–p90 logA | p10–p90 logE | median `a_opt·λ` |
|---|---|---|---|---|---|
| n8 `R^3` | **0.9923** | 0.5977 | 1.64 | 2.02 | 1.77 |
| n8 `SE(3)` | **0.9888** | 0.7974 | 1.53 | 2.10 | 1.93 |
| n8 `R^2xS^1` | **0.9976** | 0.7908 | 2.86 | 3.11 | 1.21 |
| `mixed` n=10 | **0.9915** | 0.6206 | 1.65 | 2.08 | 1.68 |

**A-optimality is a monotone restatement of E-optimality.** `a_opt·λ` has median 1.21–1.93, so the
softest mode alone carries 52–83% of the entire trace: the spectrum is bottom-dominated and the two
criteria rank graphs the same way. Swapping the trace for the min eigenvalue in the state score
cannot change what a policy learns.

**D-optimality is decorrelated** (0.60–0.80), because `Σ log w_k` weights every mode equally in log
space instead of being dragged by the bottom one. §18.5 is what decides whether that is signal.

### 18.5 Which criterion to put in the state score

At small noise the measured error *is* `sigma * sqrt(a_pos/n)` (18.3 verifies it), so ranking
graphs by A-optimality there is circular. The question that is not circular is what it costs
to *select* a topology by one criterion rather than another.

Measured two ways, and they agree:

- **A and E rank graphs identically** (18.4), and D ranks them differently but worse -- across
  domains and noise levels it agrees with the measured error less often than either.
- **Selecting by lambda costs at most 8%.** Picking the graph that maximises lambda rather than
  the one with the lowest measured error leaves 1.01x-1.08x of measured error on the table,
  in every domain tested and at bearing noise up to 5 degrees.

**So no spectral functional is a meaningfully better training signal than lambda.**
`WeightedNormalizedSpectral` exists to make that reproducible from a config, not because an arm
is expected to win.

**A rank correlation is the wrong statistic to judge this by**, and it was misleading here
before the cost was measured: it collapses when near-equal candidates are reshuffled, which
looks catastrophic and costs nothing. Quote the cost, not the correlation.

What the 8% ceiling sits against is 19.4, where choosing a repair by a criterion that carries
*no* conditioning information costs 1.7x-3.1x. Using a conditioning criterion at all is worth
far more than which one is used.

## 19. Repairing a broken framework

§5 asks how few edges could make a set of poses rigid, starting from nothing. This section
asks the question that arises *after* a formation exists and something goes wrong: an agent
leaves, a link fails, and the survivors have to restore rigidity. `required_edge_count` cannot
answer it, because it starts from the empty graph and so throws away everything the survivors
still have.

### 19.1 The bound

Let `G` be the current (flexible) graph, `Z` an orthonormal basis of `ker(B_G)`, and write the
deficit as

```
deficit = rank_K − rank(B_G)                                                         (19.1)
```

By §13.1 the exact rank an absent pair `i -> j` would add *right now* is `rank(b_ij Z)`. Sort
those marginals over the absent pairs, descending, and accumulate:

```
m_repair  =  smallest k with  Σ_{first k marginals} ≥ deficit                        (19.2)
```

**(19.2) is a lower bound.** `S ↦ rank(B_S)` is monotone submodular (§14.2), so an edge's
marginal contribution can only shrink as other edges are added. No set of `k` edges can
therefore close more of the deficit than the `k` largest marginals available now. ∎

Two details decide whether it is sound rather than merely plausible:

- **The marginals must be the per-pair gains `rank(b_ij Z)`, not the complete graph's block
  ranks `c_k`.** An edge whose own block has rank 2 may contribute only 1 to *this* graph,
  because part of its block already lies in the row space. Using `c_k` would overstate what
  each edge can do and so *understate* the count — an unsound bound, not a loose one.
- **Only absent pairs count.** An existing edge already lies in the row space, so `b_ij Z = 0`
  and its marginal is zero anyway; excluding them explicitly is what makes (19.2) read "how
  many *more*".

`repair_edge_count` returns 0 on a rigid graph, and from the empty graph it reproduces
`required_edge_count` exactly in every domain and on the `mixed` mixture — the two are one
construction seen from different starting points, which makes that agreement a check rather
than a coincidence.

### 19.2 Relation to Karimian and Tron

Karimian and Tron (CDC 2017) settle the homogeneous **2-D** case exactly. They decompose the
framework into its maximal rigid components, which meet at shared vertices they call *pins*,
and count

```
m_r = 2n − 3 − Σ_{X ∈ X_r} (2|X| − 3)                                                (19.3)
```

over the cover `X_r` of rigid components, proving a greedy algorithm attains it (their
Theorem 5).

**(19.3) is the `c_max = 1` case of (19.2).** In `R^2` every edge block has rank 1, so every
useful marginal is exactly 1 and (19.2) collapses to `deficit`. Components meeting only at
pins contribute independent constraints, so `rank(B_G) = Σ_X (2|X| − 3)`, and
`deficit = (2n − 3) − Σ_X (2|X| − 3)`, which is (19.3). `tests/test_rigidity_derived.py`
asserts the collapse directly.

What (19.2) adds is everything outside their hypotheses: `c_max = 2` domains, agents carrying
their own frames, heterogeneous mixtures, and directed edges. Their conclusions list the 3-D
extension as open, and this is it — at the cost of being a bound rather than a proven minimum,
which is the same standing §5's `m_req` has.

Note also what they used their construction *for*: their combinatorial search returns both the
first valid repair and the **best**, ranked by the second-smallest singular value of a
normalised bearing matrix — essentially λ, used as a tie-break among equally sparse repairs.
That is the same design as §15's stiffness term, arrived at independently, and it leaves open
the question of which criterion the tie-break should use, which is what §18 measures.

### 19.3 Validation

Exhaustive search over broken graphs at n=5 (n=4 for the mixes), against every edge set up to
the search cap. **Soundness is the property the bound must have**; attainment is evidence, and
is proved only in 2-D.

| config | n | cases | sound | attained | mean gap |
|---|---|---|---|---|---|
| `R^2` | 5 | 14 | **14/14** | 14/14 | 0.00 |
| `R^3` | 5 | 14 | **14/14** | 14/14 | 0.00 |
| `R^2xS^1` | 5 | 10 | **10/10** | 10/10 | 0.00 |
| `R^3xS^1` | 5 | 13 | **13/13** | 13/13 | 0.00 |
| `SE(3)` | 5 | 11 | **11/11** | 11/11 | 0.00 |
| mix `R^2/R^3/SE(3)/R^3xS^1` | 4 | 19 | **19/19** | 19/19 | 0.00 |
| mix `R^2/R^2/R^2xS^1/R^3` | 4 | 16 | **16/16** | 15/16 | 0.06 |

Sound in **97/97** cases, and attained in **96/97**. The single miss is heterogeneous,
which is where it was expected: the greedy sum over the largest marginals need not be
jointly realisable when the marginals differ between pairs, and that is the same caveat
§5 records for `required_edge_count`. In the homogeneous domains the bound was the true
minimum every time, which is what Karimian and Tron prove for `R^2` and what §14.3's
matroid argument predicts wherever `c_max = 1`.

Reproduce with `tools/repair_bound.py`.

The bound is also checked against a construction that cannot fail: removing `k` edges from a
rigid graph leaves one repairable in at most `k`, since putting those `k` back will do, so any
bound above `k` would be unsound. That holds for `k = 1, 2, 3` in all five domains.

### 19.4 Which repair, not how many

§19.1-19.3 settle the count. The count is not where the difficulty is.

After a break, several different edge sets of the same minimum size restore rigidity. If
they all recover the shape about equally well, the choice is free and there is nothing to
optimise past the count. Enumerating every minimum-size repair at small `n` and scoring each
by shape error (§18.1), 36-40 broken graphs per configuration, two edges dropped:

| config | cases | repairs found | worst/best shape error | greedy percentile | greedy/best |
|---|---|---|---|---|---|
| `R^2` | 40 | 80 | **13.9x** (max 2530x) | 52% | 2.53x |
| `R^3` | 38 | 31 | **4.2x** (max 100x) | 42% | 1.70x |
| `R^2xS^1` | 40 | 36 | **17.4x** (max 403x) | 55% | 3.08x |
| `SE(3)` | 39 | 16 | **6.3x** (max 137x) | 39% | 1.67x |
| `mixed` | 36 | 21 | **9.7x** (max 813x) | 44% | 2.14x |

(worst/best is a geometric mean over instances; percentile is where the marginal-gain repair
lands among the valid ones, so 50% is indistinguishable from choosing at random.)

**Two things follow.**

The choice matters: equally sparse repairs differ by 4x to 17x in the error they leave
behind, and by up to three decades on individual instances. Minimum-edge repair is therefore
badly underdetermined as a specification.

And the classical method does not make that choice. Marginal rank gain is the criterion that
gets the count right, and it carries no information about conditioning, so greedy lands at
the 39th-55th percentile of the repairs it could have picked and costs 1.7x-3.1x the best
available error. This is not a defect of that algorithm; it is optimising the count, which
it does optimally in the `c_max = 1` domains (§14.3, and Karimian and Tron Theorem 5).

That gap is the one place measured so far where the edge-count objective is provably
saturated *and* something else is provably not. Karimian and Tron list a criterion for
choosing edges as open; this says what such a criterion is worth.

Reproduce with `tools/repair_choice.py`.
