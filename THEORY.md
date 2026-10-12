# Theory

The algebra behind the environment. It covers what the rigidity matrix is, what its rank and null
space mean, and how each derived quantity in the code (`rank_K`, `c_max`, `m_req`, the state score,
the rigidity observation channels, the repair bound) follows from it. `CLAUDE.md` describes the
code and `DESIGN_NOTES.md` the implementation choices. The thesis is the full reference.

---

## 1. Notation

`n` agents. Agent `i` has position `p_i ∈ R³` and orientation `R_i ∈ SO(3)`. Its *domain* fixes
which of those degrees of freedom it actually has (`R^2`, `R^3`, `R^2xS^1`, `R^3xS^1`, `SE(3)`).

A directed edge `i -> j` means `i` measures the bearing to `j`:

```
p_ij  = p_j - p_i          p̂_ij = p_ij / ||p_ij||        b_ij = R_iᵀ p̂_ij
```

`b_ij` is the measurement in `i`'s own frame. In `R^2` and `R^3` there is no frame and `R_i = I`,
so `b_ij = p̂_ij` is a global-frame vector, which is why raw bearings in the observation are not
rotation invariant in those two domains (§12).

```
P(x) = I - x xᵀ          orthogonal projector onto x⊥, for unit x
[a]×                     skew-symmetric matrix with [a]× b = a × b
```

---

## 2. The rigidity matrix

### 2.1 The bearing derivative

```
d/dt p̂_ij = (1/||p_ij||) · P(p̂_ij) · (ṗ_j - ṗ_i)                                   (2.1)
```

Only the component of relative velocity perpendicular to `p̂_ij` rotates the bearing. The
component along it changes the distance and leaves the bearing unchanged, which is the scale
blindness of bearing measurements.

With orientation (world-frame angular velocity `ω_i`, `Ṙ_i = [ω_i]× R_i`):

```
ḃ_ij = R_iᵀ [p̂_ij]× ω_i + (1/||p_ij||) R_iᵀ P(p̂_ij) (ṗ_j - ṗ_i)                      (2.2)
```

### 2.2 Construction

`rigidity.extended_bearing_rigidity_matrix` builds, per edge `k = (i,j)`,

```
Dp_k = (1/||p_ij||) · R_iᵀ P(p̂_ij)          Da_k = -R_iᵀ [p̂_ij]×
```

and assembles

```
B = [ Dp Ēᵀ S̄ | Da Ē_oᵀ P̄ ]                                                          (2.3)
```

with incidence `E[i,k] = -1, E[j,k] = +1`, `E_o[i,k] = -1`, and `S̄ = blkdiag(S_1 … S_n)`,
`P̄ = blkdiag(P_1 … P_n)` the per-node DOF projectors of §2.3. The position half contracts `E` to
`(ṗ_j - ṗ_i)`, matching (2.1), and the orientation half picks up the `-1` of `E_o`, matching (2.2).

`B` is `(3m × 6n)`, one 3-row block per directed edge, in `np.nonzero(edges)` order. The per-edge
block structure is used throughout what follows. The construction is checked against central
differences of the bearing map in all five domains and in heterogeneous mixes.

### 2.3 The DOF restriction belongs to the node

`rigidity.node_dof_projectors(agent)` returns per node:

| domain | `S_i` (translational) | `P_i` (rotational) |
|---|---|---|
| `R^2` | `diag(1,1,0)` | `0` |
| `R^3` | `I₃` | `0` |
| `R^2xS^1` | `diag(1,1,0)` | `e₃e₃ᵀ` |
| `R^3xS^1` | `I₃` | `v vᵀ` |
| `SE(3)` | `I₃` | `I₃` |

Applied on the column side, every coordinate an agent cannot vary becomes an exactly-zero column,
which is what the nullity accounting of Michieletto et al. requires (their rank identity counts the
inadmissible variations as null columns).

The per-edge alternative of their Table III, a matrix `U_ij` multiplying `(p_j - p_i)`, applies one
restriction to both endpoints. It is correct when `S_i = S_j` and wrong in a mixed network: for a
planar agent measuring a spatial one it re-enables the planar agent's `z` coordinate. No choice of
`U_ij` is faithful on admissible variations and zero on inadmissible ones unless the two agents are
vertically aligned, so the restriction has to be per node. Proofs and a counterexample are in
`docs/dof_restriction_note.pdf`. On homogeneous networks the two constructions give the same
matrix, and `bearing_DOFs` is kept only as the Table I reference for that comparison (faithful at
`v = e₃`).

`P_i = v vᵀ` is in world coordinates, so a global rotation of the network has to rotate each agent's
`rotation_axis` as well.

---

## 3. Trivial motions, `rank_K` and rigidity

A motion is trivial when it changes no bearing. From (2.1) and (2.2):

- **Translation** of every agent by the same `v`. Every block vanishes.
- **Uniform scaling** `ṗ_i = c(p_i - p̄)`. Then `ṗ_j - ṗ_i` is parallel to `p̂_ij` and `P(p̂_ij)`
  kills it.
- **Global rotation** is trivial only when every agent carries a frame that rotates with it. In
  `R^d` the bearings are global and a rotation changes them.

So the trivial space is the translations, one scaling, and however much of the rotation group the
agents' frames absorb. For homogeneous networks:

| domain | DOF/agent | trivial dim | `rank_K` | `c_max` |
|---|---|---|---|---|
| `R^2` | 2 | 3 | `2n - 3` | 1 |
| `R^3` | 3 | 4 | `3n - 4` | 2 |
| `R^2xS^1` | 3 | 4 | `3n - 4` | 1 |
| `R^3xS^1` | 4 | 5 | `4n - 5` | 2 |
| `SE(3)` | 6 | 7 | `6n - 7` | 2 |

`rank_K` is the rank of the rigidity matrix `B_K` of the complete graph on the same poses. **A
framework is infinitesimally bearing rigid (IBR) iff `rank(B) = rank_K`.** The code computes `rank_K`
numerically from `B_K`, so heterogeneous networks need no closed form. For a mix,

```
rank_K  ≤  Σ_i dim D_i  −  (3 if any agent is planar else 4)
```

and the exact trivial space is `ker(B_K)`, which by Michieletto Theorem 1 is the trivial variation
set in every domain and mix.

**Rank is generic.** Every entry of `B(χ)` is rational in the configuration `χ`, so `rank(B(χ))`
drops only on a proper algebraic subset of measure zero. For poses drawn from a continuous
distribution the rank is almost surely its generic value, fixed by the graph and the domain
assignment. A rank-based objective therefore contains no geometry, and the geometry enters only
through the conditioning term of §6.

---

## 4. Edge counts

### 4.1 Per-edge block rank `c_k` and `c_max`

Edge `k`'s block touches only its endpoints' columns, so

```
c_k = rank [ P(p̂_ij) S_i | P(p̂_ij) S_j | [p̂_ij]× P_i ]
```

In homogeneous `R^d`, `c_k = d - 1` for every edge, independent of geometry: 2 in `R^3`, 1 in `R^2`.
A bearing is one angle in the plane and two in space. In a mix `c_k` varies with the endpoints'
domains. `c_max = max_k c_k` over the complete graph is exact and is the unit the state score
normalizes by (`rigidity.max_edge_rank`).

### 4.2 The minimum edge count `m_req`

Rank is subadditive over the block rows, `rank(B) ≤ Σ_e c_e`, so a rigid graph needs

```
m ≥ m_req,   m_req = fewest edges whose largest c_e sum to rank_K                  (4.1)
```

which is `⌈rank_K / c_max⌉` when all blocks are equal and coincides with the closed form of Trinh et
al. for homogeneous `R^d` (`MBR_required_Rd`). `rigidity.required_edge_count` accumulates the
complete graph's block ranks in descending order. **It is a lower bound.** Subadditivity says no
smaller set is rigid, not that one of this size exists, and in a heterogeneous network the largest
blocks may not be jointly realizable. It is used for reporting and for the minimality metric, and
it stays out of the reward.

### 4.3 `is_MBR`

The minimality test sorts the current graph's `c_e`, accumulates until `rank_K`, and calls the
graph minimal iff it is IBR and `m` equals that count. It is exact in homogeneous `R^d`. On
heterogeneous networks it can report a false negative, and because it uses the current graph's
blocks rather than the complete graph's it can also report a false positive. "At bound" (rigid on
exactly `env.m_req` edges) is reported beside it for that reason.

---

## 5. Submodularity and the greedy guarantee

A set function `f` is **submodular** when, for `S ⊆ T` and `e ∉ T`,

```
f(S ∪ {e}) − f(S)   ≥   f(T ∪ {e}) − f(T)                                            (5.1)
```

**`rank(B_S)` is monotone submodular.** Each edge contributes the row space `V_e` of its block, and
`f(S) = dim(Σ_{e ∈ S} V_e)`. The marginal value of `e` is `dim V_e − dim(V_e ∩ V_S)`, and
`V_S ⊆ V_T` makes it shrink with `S`.

Fewest edges making the framework rigid is therefore **minimum submodular cover**. Wolsey (1982)
proves greedy (take the largest marginal gain) is an `H(d)` approximation, with `d = max_e f({e})`
and `H(k) = 1 + 1/2 + … + 1/k`. Here `d = c_max`, so greedy is exact in `R^2` and `R^2xS^1` (the
independent sets form a matroid) and within 1.5 of optimal in the spatial domains. The
`constructive` baseline is this algorithm.

**The rigidity eigenvalue is monotone but not submodular.** Adding an edge adds a PSD term to `BᵀB`,
so by Weyl's inequality no eigenvalue decreases. Diminishing returns fails, since two edges can
brace a direction that neither braces alone. Counterexamples are easy to find numerically
(`tools/submodularity.py`), so greedy carries no guarantee on the conditioning.

---

## 6. Conditioning and estimation error

### 6.1 What the rigidity eigenvalue means

Let `λ` be the smallest nonzero eigenvalue of `BᵀB` (index `6n - rank_K` in the ascending spectrum).
For a variation `δχ` orthogonal to the trivial motions,

```
||B δχ||  ≥  √λ · ||δχ||                                                             (6.1)
```

so `λ` bounds how much the bearings move when the shape deforms. It is **not** a degree of rigidity.
Rigidity is binary, and exact bearings recover the shape at any `λ > 0`. `λ` is the conditioning of
the bearing to shape inverse problem, and it has three readings.

1. **Estimation.** Shape error is at most `1/√λ` times the bearing error.
2. **Control.** A bearing formation controller linearizes to `ẋ = -BᵀB x`, so the slowest
   non-trivial mode decays at rate `λ`.
3. **Distance to degeneracy.** `√λ` is the spectral-norm distance from `B` to the nearest
   rank-deficient matrix (Eckart-Young). That matrix need not be the rigidity matrix of a
   realizable configuration, so this is a proxy.

`λ` grows with every added edge, so maximizing it alone favors the complete graph.

### 6.2 The three criteria

With independent isotropic tangent noise of per-axis variance `σ²` on each bearing, the Fisher
information is `BᵀB/σ²` and the Cramér-Rao covariance of the shape estimate is `σ²(BᵀB)⁺` on the
identifiable subspace. With `w_k` the `rank_K` nonzero eigenvalues of `BᵀB`:

```
a_opt = tr((BᵀB)⁺) = Σ_k 1/w_k      A-optimality: total mean squared error          (6.2)
e_opt = 1/λ        = 1/w_min        E-optimality: the worst mode
d_opt = −Σ_k log w_k                D-optimality: log-volume of the error ellipsoid
```

All three are read off the singular values `rigidity_decomposition` already computes
(`rigidity.estimation_error`), and all three are `+inf` on a flexible framework.

### 6.3 Units and `shape_err`

`B`'s position columns carry `1/length` and its attitude columns are dimensionless, so a spectrum of
the raw matrix tracks the pose range. `scaled_rigidity_matrix` multiplies the position columns by
the formation's RMS radius (`characteristic_length`) first. Position error is then in formation
radii and attitude error in radians.

```
shape_err = sqrt(a_opt / n)                                                          (6.3)
```

on the scaled matrix is the RMS state error per radian of bearing noise. It is dimensionless and
comparable across `n`, domain and pose range. It is a slope `d(error)/dσ` of a linearization, so it
holds while `σ · shape_err` is small and should be quoted at an operating point. `σ` is the per-axis
standard deviation, and the RMS angle between true and noisy bearing is `σ√2`.

### 6.4 Measuring it

`estimation.py` checks the prediction by simulation.

- `perturb_bearings` adds full 2-DOF tangent noise to every bearing in every domain. A planar
  agent's motion is restricted, its camera is not.
- `solve_shape` runs damped Gauss-Newton on `Σ ||b_ij(χ) − z_ij||²` with `B` as the Jacobian, steps
  restricted to each agent's admissible DOFs. It is the maximum-likelihood estimator the Cramér-Rao
  bound describes, initialized at the truth, so it measures local accuracy.
- `shape_error` projects the error off `ker(B_K)`. This gauge quotient is linear, the same
  linearization the bound makes, so it is a small-error metric. `max_scale_ratio` catches a
  formation collapsed to a point, which the linear quotient would otherwise score as error-free.

`rigidity.measurement_sensitivity` splits `a_opt` exactly over measurements. Noise on bearing `k`
reaches the estimate through `B⁺`, so its share is the squared norm of the matching columns of
`B⁺`. The shares sum to the total, per edge or per measuring agent.

---

## 7. The state score

### 7.1 Rank and edge terms

```
φ = ( w_r · rank(B)  -  w_e · m · c_max ) / rank_K,        (w_r, w_e) = (100, 25)   (7.1)
```

`m · c_max` is the most rank `m` edges could carry, so both terms are ranks over `rank_K` and `φ` is
dimensionless. An edge adding `Δr` rank changes `φ` by `(w_r Δr − w_e c_max)/rank_K`. At
`Δr = c_max` this is positive iff `w_r > w_e`, in every domain and at every `n`, because the same
`c_max/rank_K` factor appears on both sides. Removing a redundant edge gains `w_e c_max / rank_K`, so
a full-rank addition is worth `(w_r − w_e)/w_e = 3` prunes. The optimum is
`w_r − w_e · m_min · c_max / rank_K ≤ 75`, with equality when the poses admit a perfectly packed rigid
graph. `m_req` is deliberately not in (7.1), since an understated bound would over-penalize edges.

`Weighted` (`20 · rank − 10 · m`) is the older un-normalized score. Its exchange rate between rank
and edges depends on `c_max`, so it does not transfer across domains, and it is kept only so old
runs replay.

### 7.2 The conditioning term

`WeightedNormalizedShapeError` adds

```
φ = (7.1) + κ · one_edge · 1[IBR] · q,           one_edge = w_e · c_max / rank_K
q = sigmoid( −( log10(shape_err) − c₀ ) / s ),   c₀ = 1.35,  s = 0.70 decades      (7.2)
```

with `c₀` and `s` fixed constants (`SHAPE_ERR_CENTRE`, `SHAPE_ERR_SIGMOID_DECADES`, re-derived by
`tools/shape_error_scale.py`). `q ∈ (0,1)` rises as the error falls and is 0.5 for a typical minimal
graph. The term is gated on IBR, so becoming rigid is never punished, and it is bounded by
`κ · one_edge`, so it is denominated in edges. It is exactly similarity invariant because
`shape_err` is. `κ = 0` reproduces (7.1).

Over rigid graphs (7.2) is `φ = w_r − one_edge · (m − κ q)`, so the optimum moves from `m` to `m + 1`
edges only when `κ · (q*(m+1) − q*(m)) > 1`. `κ` is an upper bound on the edges traded for
conditioning, not the trade itself. Below that threshold `κ` selects among graphs of equal edge
count, which differ widely in shape error.

`WeightedNormalized` and `WeightedNormalizedSpectral` use the same form with `q` a sigmoid of
`log10(λ/λ_ref)`, where `λ_ref` is the median `λ` of a few greedy constructions on the episode's own
poses. `WeightedNormalizedSpectral` reads `λ`, `a_opt` or `d_opt` according to
`spectral_functional`. Both are kept so earlier runs replay. All scores agree at `κ = 0`.

### 7.3 Potential-based shaping and the discount factor

The step reward is `r_t = γ φ(s_{t+1}) − φ(s_t)` plus optional penalties, potential-based shaping
with `γ = 0.99` set in `step()`. The discounted return telescopes,

```
Σ_{t=0}^{T-1} γᵗ r_t = γ^T φ_T − φ_0                                                   (7.3)
```

so with the same `γ` the agent discounts by, the shaping changes no optimal policy (Ng et al.). At
`γ = 1` the return would be `φ_T − φ_0`, and under a near-uniform policy the advantage
`E[φ_T | s'] − E[φ_T | s]` tends to zero because the walk over edge sets forgets `s`, which is why
`γ < 1`. The `γ` in `step()` is a literal, so it matches the trainers' `discount_factor` only while
that stays at its default 0.99.

---

## 8. Rigidity observation channels

All channels are computed on the scaled matrix of §6.3, which makes them similarity invariant. With
`Z` an orthonormal basis of `ker(B)` and `b_ij` the 3-row block edge `i -> j` would have:

### 8.1 Adding an edge

```
rank([B; b_ij]) − rank(B) = rank(b_ij Z)                                             (8.1)
```

since `b_ij` raises the rank by the dimension of its component outside `rowspace(B) = ker(B)^⊥`.
This is exact in every domain and mix. Two channels follow, over all ordered pairs:

```
add_independence[i,j] = ||b_ij Z||_F / ||b_ij||_F        ∈ [0, 1]
add_rank[i,j]         = rank(b_ij Z) / c_max             ∈ [0, 1]
```

`candidate_gain` forms

```
b_ij Z = D_p (S_j Z_j − S_i Z_i) − D_a P_i Z_i
```

for all pairs in batched products and reads norm and rank off the 3×3 Gram matrix
`(b_ij Z)(b_ij Z)ᵀ`, with the rank cut at `add_independence > 1e-6`. `candidate_gain_reference` is the
per-pair form it is tested against. `node_freedom[i]` is node `i`'s share of the non-trivial part of
`ker(B)`.

### 8.2 The softest mode

On a rigid framework `ker(B)` is the trivial space and every channel of §8.1 is zero. With `v` the
eigenvector of `BᵀB` at `λ`, first-order perturbation gives `Δλ ≈ ||b_ij v||²` for adding `i -> j`:

```
add_stiffness[i,j] = ||b_ij v||                   (pair channel)
node_slack[i]      = (||v_i^pos||, ||v_i^att||)   (node channel)
```

Both are zero unless the framework is rigid, so they are complementary to §8.1. `add_stiffness` is
a ranking prior rather than an exact gain, since adding an edge is not an infinitesimal change. The
information is pairwise. `Dp` projects orthogonal to the bearing, so two soft agents moving along
their mutual bearing are invisible to that measurement, which no per-node magnitude can encode.

### 8.3 Removing an edge

An existing edge's block is the slice `b = B[3k:3k+3]`. Its leverage block

```
H = b (BᵀB)⁺ bᵀ,       eigenvalues in [0, 1]                                         (8.2)
```

has an eigenvalue 1 for each direction only this edge constrains, so

```
rank(B) − rank(B without the edge) = #{eigenvalues of H equal to 1}
remove_rank[i,j]      = that count / c_max
remove_stiffness[i,j] = 1 − λ(BᵀB − bᵀb) / λ(BᵀB)                                   (8.3)
```

`remove_stiffness` is exact via the rank-3 downdate. Both are zero on non-edges, so `add_*` and
`remove_*` have complementary support. `remove_rank` is invariant to any invertible column scaling,
since `(SᵀBᵀBS)⁺ = S⁻¹(BᵀB)⁺S⁻¹`.

### 8.4 State quality

`state_quality` is the `q` of (7.2) for the current graph, 0 while flexible. It is the one channel
that says how good the current graph is, where the others describe differences.

---

## 9. The closed-form landscape

At `κ = 0`, `φ` is affine in rank, so a toggle changes it by exactly

```
Δφ(add i->j)    = ( w_r · rank(b_ij Z)   − w_e · c_max ) / rank_K
Δφ(remove i->j) = (−w_r · rank lost      + w_e · c_max ) / rank_K                    (9.1)
```

which `add_rank` and `remove_rank` carry. `baselines.phi_landscape` evaluates (9.1) for all pairs at
once. The `spectral` baseline ranks toggles by it and verifies a shortlist with the real `φ`, which at
`κ = 0` reproduces greedy's choices at a fraction of the rigidity computations. At `κ > 0` the
removal side stays exact (§8.3) while the addition side uses `add_stiffness`, which is why the
shortlist is rescored.

---

## 10. Repairing a broken framework

After an agent leaves or a link fails, the question is how few edges restore rigidity from the
graph in hand. With deficit `rank_K − rank(B_G)` and the exact marginals `rank(b_ij Z)` of §8.1
over the absent pairs, sorted descending:

```
m_repair = smallest k with Σ_{first k marginals} ≥ deficit                           (10.1)
```

This is a lower bound, because submodularity (§5) means marginals can only shrink as edges are
added. The marginals must be the per-pair gains, not the complete graph's `c_k`, since a block of
rank 2 may add only 1 to this graph, and using `c_k` would make the bound unsound.
`repair_edge_count` returns 0 on a rigid graph and reproduces `required_edge_count` from the empty
graph.

Karimian and Tron (CDC 2017) settle homogeneous 2-D repair exactly through rigid components,
`m_r = 2n − 3 − Σ_X (2|X| − 3)`. That is the `c_max = 1` case of (10.1). (10.1) extends it to the
spatial domains, oriented agents, heterogeneous mixes and directed edges, as a bound rather than a
proven minimum.

Several repairs of the same minimum size usually exist and they differ in the error they leave.
`repair.py` therefore scores a method by the percentile of its repair among all repairs of that
size, by shape error, with 100 the best.

---

## 11. Formation control

`convergence.py` drives a formation to a goal shape over a chosen edge set with the gradient law

```
ẋ = diag(g) Bᵀ p*                                                                    (11.1)
```

where `p*` stacks the goal bearings and `B` is built from the current formation over the same
edges. Each block of `B` carries `P(p̂_ij)/||p_ij||` and `P(p̂) p̂ = 0`, so `Bᵀ p = 0` for the current
bearings and `Bᵀ p* = Bᵀ (p* − p)`. Since `B = dp/dx`, (11.1) is gradient descent on the bearing
error. Linearized about the goal, the error dynamics are `ẋ = −BᵀB x`, so the slowest observable
mode decays at `λ`. Bearing control cannot observe translation and scale (and rotation, where every
agent has a frame), so the error is measured against the goal shape with those directions projected
out, and the bearing error is taken per edge so it does not depend on `m`.

---

## 12. Invariance

The observation is invariant to translation and uniform scaling in every domain. Bearings are unit
vectors, `coord_features` are centered and RMS-normalized, and the rigidity channels are computed on
the scaled matrix. Rotation invariance holds exactly in the domains with frames (`R^2xS^1`,
`R^3xS^1`, `SE(3)`), where `b_ij = R_iᵀ p̂_ij` rotates with the world. In `R^2` and `R^3` the raw
bearings are global-frame vectors, so a global rotation changes the observation, and any mix
containing an `R^d` agent inherits this. `rotation_augmentation` applies a random global rotation at
reset (about `z` only when a planar agent is present). The rigidity channels themselves are
rotation invariant in every domain. Rigidity is also invariant under reflection.
