# What's next

Everything durable lives elsewhere: results in `README.md`, mathematics in `THEORY.md`,
implementation reasoning in `DESIGN_NOTES.md`, the code map in `CLAUDE.md`. This file is only the
plan, and can be deleted without losing anything.

The programme it describes: **the edge-count objective is saturated, so the remaining work is a
metric that discriminates, opponents worth beating on it, and an evaluation regime where a
sequential policy can earn its place.**

---

## Why the objective moved

Measured on frozen benchmarks (2026-08-23 `stiff_dqn_gine` sweep, trained on `mixed` n=10), the
policy matches or beats greedy on edge count and minimality in every configuration tested, and
three of six configurations tie at the proven lower bound for all three methods. `THEORY.md` §14.3
says why: greedy sits 0-5% above a lower bound, so there was never much to win. **Edge count no
longer separates methods.**

`stiffness_kappa = 2` did not fix that — parity with greedy in four of six configurations, a loss
in one. And §18 now shows why replacing λ with another spectral functional will not either: the
trace is a monotone restatement of λ (rank correlation 0.99), and log-det is decorrelated from λ but
*worse* at predicting the error that actually matters. **No spectral functional is a meaningfully
better training signal than λ.**

What §18 did produce is a metric with an absolute meaning — RMS shape error per radian of bearing
noise, validated against Monte-Carlo to a few percent — and a finding that reframes the objective:
at realistic bearing noise the spectral criteria stop predicting measured error at all (rank
correlation 0.96 → 0.07 in `R^2xS^1` between 0.006° and 5°).

---

## Where this sits in the literature

Three papers set the frame, and they are worth stating because two of them hand us
ground truth and the third hands us the gap.

- **Karimian and Tron, CDC 2017, *Theory and Methods for Bearing Rigidity Recovery*.**
  Exactly the recovery problem below: an agent leaves, rigidity breaks, which edges
  restore it. They decompose into rigid components and prove an exact minimum for
  homogeneous **2-D** (`2n - 3 - sum_X (2|X| - 3)`), attained by a greedy algorithm.
  Their combinatorial variant returns both the *first* valid repair and the *best*,
  ranked by the second-smallest singular value of a normalised bearing matrix -- which
  is essentially our λ, used as a tie-break among equally sparse repairs. Their stated
  open problems are the **3-D extension**, **a criterion for choosing which edges to
  add**, and a distributed version. The first is `repair_edge_count` below; the second
  is what phi and the shape-error metric are for; the third is
  `DESIGN_NOTES.md#distributed-feasibility`.
- **Schiano and Tron, ICRA 2018, *The Dynamic Bearing Observability Matrix*.** The
  rigidity matrix is the zeroth-order term of a nonlinear observability analysis, and it
  describes a formation standing still. If the agents move with known velocities the
  global scale becomes observable and the rank rises from `6N-7` to `6N-6`. Two
  consequences here: our code reproduces their `6N-7` exactly (11/17/23 at N=3/4/5 in
  `SE(3)`), an independent check on the rigidity matrix; and everything this project
  optimises is **static** observability, which should be said rather than assumed. Their
  EKF on Lie groups is the dynamic sibling of `estimation.py`'s batch solver.
- **Su et al., Annual Reviews in Control 2026, *Bearing-based formation control: a
  survey and taxonomy*.** Reviews the whole field of bearing-based control under a
  two-level taxonomy: rigidity is the *static* route to observability, persistence of
  excitation (PE) the *temporal* one -- if bearings vary enough over time, far fewer
  edges suffice. Two things matter to us. It says "topology" six times in the whole
  paper and **never discusses choosing one**, which is this thesis's gap stated by the
  field's own review; and §9.6 names **reinforcement learning** as a promising direction
  for bearing-based control, which is a better motivation citation than the graph-RL
  survey alone because it comes from the application side.

## Done

- **Where the error comes from.** `measurement_sensitivity` decomposes the total shape error
  into each bearing's share and each agent's share, exactly and without sampling, drawn as the
  `sensitivity` figure. First thing it shows: sparsifying *concentrates* the error, so a minimal
  graph can lean half of it on a single measurement -- fragility the edge count does not report.
- **Whether the policy's edits are good ones.** Every legal edit is scored at every step and the
  policy's choice ranked among them, by phi *and* by shape error. The gap between the two
  rankings separates "the policy chooses badly" from "the objective does not ask for this".

- **`evaluation.py` is the results script.** Renamed from `baselines.py`, and now writes every
  figure the thesis needs from one run: the comparison table, per-episode and per-step CSVs,
  trajectories, outcomes, the summary, per-episode detail, measured error under bearing noise,
  predicted against measured error, per-agent uncertainty ellipses, the softest deformation mode,
  and how much the choice of repair matters.
- **`rigidity_quality`**, the observation channel saying how good the current state is. Every
  other channel is local or a difference; this is the only one that says whether the conditioning
  of the graph in hand is good. Off by default, its own flag so the ablation can price it.

- **Workstream 1 is complete.** Analytic metrics, the Monte-Carlo estimator and its
  validation, `shape_err` through the evaluation path, the functional bake-off,
  `evaluation.py --noise-sweep` with its figure, and `WeightedNormalizedSpectral` as a
  config key carrying the null result. No training runs were spent on the spectral arm,
  because §18.5 says none of the functionals beats λ as a signal.

- **Repair, and the finding that reframes it.** `repair_edge_count` bounds how few edges
  could restore rigidity; `greedy_rigid_repair` is the marginal-gain method that achieves it.
  Then `tools/repair_choice.py` asked whether the count is the hard part, and it is not:
  among *equally sized* minimum repairs the shape error spans **4x-17x** (up to three decades
  on single instances), and marginal-gain greedy sits at the **39th-55th percentile** of the
  repairs available to it, costing 1.7x-3.1x the best. **This is the clearest room for a
  learned method measured so far** -- the classical algorithm solves the count optimally and
  then chooses essentially arbitrarily. See `THEORY.md` §19.4.
- **The repair bound.** `rigidity.repair_edge_count`: fewest edges that could restore
  rigidity to a *broken* graph, by the same subadditivity argument as
  `required_edge_count` but starting from the graph in hand. Returns 0 when rigid,
  reproduces `required_edge_count` exactly from the empty graph, and collapses to
  Karimian and Tron's 2-D formula at `c_max = 1`. This is the ground truth the recovery
  experiment needs: without it, "the policy repaired it in 3 edges" cannot be scored.
- **Analytic estimation error.** `rigidity.estimation_error` / `estimation_error_blocks` /
  `scaled_rigidity_matrix`: A-, D- and E-optimality off the SVD `rigidity_decomposition` already
  performs, on a length-normalised `B`. Free — cheaper than building `B`.
- **The Monte-Carlo estimator.** `estimation.py`: tangent-plane bearing noise in radians,
  Gauss-Newton shape recovery restricted to each agent's admissible DOFs, gauge-quotiented error.
  `tests/test_estimation.py` pins the Jacobian convention, the DOF restriction, the gauge and the
  agreement with the bound.
- **`shape_err` through the evaluation path.** `Environment.shape_error_now`, `last_stats`,
  `Episode/ {Final,Best,Mean} shape err`, the `evaluation.py` result rows, `results.csv`,
  `trajectories.csv`, and a `shape err` column in both the text and figure tables.
- **Three measurement tools**, one question each: `tools/crlb_validation.py` (does the prediction
  hold, and where does it stop), `tools/spectral_criteria.py` (do A/D/E rank graphs differently),
  `tools/functional_vs_error.py` (which of them predicts the measured error).

## Now

**Baselines from the survey** (Darvariu et al. 2024 §3.2 names all three). Beating greedy on a
near-solved objective means little; these are the opponents that make a claim mean something.

- `spectral` — first-order spectral hill-climb (Wang & Van Mieghem 2008 analogue). Add
  `argmax ‖b_ij v‖²`, prune `argmin remove_stiffness`; `candidate_gain` and `removal_costs` already
  compute both. One eigendecomposition per edit against greedy's `n(n−1)` φ-evaluations. The
  "honest opponent" `THEORY.md` §14.4 asks for.
- `anneal` — simulated annealing (Schneider et al. 2011), single-edge toggles on the configured φ,
  **budget-matched to greedy's φ-evaluation count** and reported as such. Own RNG, like
  `constructive`, so enabling it does not shift the instances. Assumes no submodularity, which is
  the right property on a non-submodular objective.
- `degree` — lowest-degree-product addition, highest-degree redundant pruning (Beygelzimer et al.
  2005). No rigidity algebra beyond the rigidity test, so it doubles as the tier-1
  distributed-plausible reference from `DESIGN_NOTES.md#distributed-feasibility`.
- **A compute column.** Greedy spends ~900 φ-evaluations per instance at n=10; the policy spends
  ~10 forward passes. Nothing reports this, and amortised training against per-instance search is
  the survey's central argument for RL.

MCTS/UCT stays deferred.

## Next

**Robustness, recovery and membership change** (`robustness.py`). The differentiating experiment:
greedy and constructive are one-shot, and repair is a sequential editor's native operation. Per
frozen instance, every method faces an identical schedule of `converge → event → recover`, with
events `edge_drop_random`, `edge_drop_worst` (chosen exactly via `removal_costs`),
`bearing_corrupt` (outlier measurements — implementable only now that an estimator exists),
`node_leave`, `node_join`.

Scored against `repair_edge_count`, so "repaired in `k` edges" reads as a gap above a bound
rather than as a bare number, and against **best-of-combinatorial repair** at small `n`, which
§19.4 shows is 1.7x-3.1x better than greedy on shape error. Baselines: `greedy_rigid_repair`
(the count-optimal one), the spectral repair, and greedy hill-climbing on phi.

Karimian and Tron's factor-graph algorithm is **not** being implemented. Its contribution is
the minimum-edge guarantee in homogeneous 2-D, and `greedy_rigid_repair` was measured to attain
that bound 16/16 in `R^2` and 24/24 in `R^2xS^1` -- which §14.3's matroid argument predicts for
every `c_max = 1` domain. It would be a 2-D-only subsystem reproducing a number already in
hand, and it cannot help in `R^3`/`SE(3)`, where greedy misses the bound (15/16, 22/28) but
their theory does not reach.

`node_join` / `node_leave` change `n`. The models are already `n`-agnostic — both backbones read
`n` from the tensor, the heads take it as an unused kwarg, and every action decoder recomputes it —
so the harness bypasses skrl and calls `model.compute` directly, rebuilding the observation space
at the new `n`. **Verify that path in a test before building on it.** It also makes size
generalisation measurable without the deferred variable-`n` training work.

Primary metric is **repair cost**, not survival: a minimal graph has zero redundancy by
construction, so `survived` is 0 for every method. Robustness-by-design needs a redundancy-aware
objective and is a separate arm.

**Randomised domain composition.** A `domain_sampler` env key drawing the per-agent mix in
`reset()`; `n` stays fixed. Dirichlet-multinomial rather than per-agent uniform, which at n=10 would
draw a homogeneous corner with probability 5e-7 — the exact coverage gap this tests. Fix the
ordering while there: draw the mix, compute `m_req` for it, then sample the edge count.
`sample_initial_edge_count` currently uses the *previous* episode's cached `m_req`, correct only
while the composition is constant. Set `max_steps` from the worst case over the sampler's support
rather than per episode, so the horizon stays constant. `benchmark.py` needs per-instance `domains`.

**Pre-registered criterion.** The old one ("≥80% minimal on homogeneous SE(3) and R^3xS^1 at n=8")
is already met without the change and no longer discriminates. Replacement: train on Dirichlet
draws, evaluate on held-out composition families — the five homogeneous corners and two-domain
mixes never seen as such — and require parity with greedy on edges *and* minimality on every
family, reporting the spread across compositions.

**Statistical protocol**, threaded through the above rather than done last. Three seeds per arm;
`report.paired_stats(rows, reference="greedy")` for per-instance paired differences with a bootstrap
CI, since methods already run on identical instances and `aggregate()` throws the pairing away; 50
frozen instances rather than 20. `DESIGN_NOTES.md#horizon` measured a 35-point minimality swing
between seeds, and greedy's own numbers moved across commits on the same benchmark digest, so no
single-seed or cross-date row currently means anything.

## Later

- **Noise-aware evaluation as the headline.** §18.5's finding — spectral criteria stop predicting
  measured error at realistic noise — argues for scoring topologies on measured error directly. As
  a *reward* that is expensive (one Monte-Carlo per step) and would need a cheap surrogate; as an
  *evaluation* it is already available.
- **Spectral observation channels** for whichever functional the reward ends up carrying. Kept
  separate from any reward change: `THEORY.md` §16 explains why moving both at once makes the
  ablation uninterpretable.
- **Variable `n` in training.** Pad to `n_max`, thread a node-validity mask through both backbones,
  `global_mean_pool` and the action masks. The one open failure it addresses is size generalisation
  (n=16: 23.20 edges, 0% minimal against `m_req` 22). The robustness harness measures it first.
- **Persistence of excitation, i.e. dropping the static assumption.** Su et al. (2026) §4.2: if
  the agents move and the bearings vary enough over time, observability needs far fewer edges --
  in some settings a single pair. Schiano and Tron (2018) is the same point from the observability
  side, and shows the global scale stops being unobservable. Both say the question this thesis asks
  has a *different answer for a moving formation*. That is a redirection rather than an increment,
  so it belongs in the thesis as the honest answer to "why assume the formation is static?", not in
  the code. Worth stating precisely enough that someone could pick it up.
- **Global convergence / basin of attraction.** `solve_shape` from a cold start rather than the
  truth: how many restarts reach the true shape. No spectral functional predicts it, and a graph can
  be rigid with good λ and still riddled with spurious minima.
- **Conditioning the policy on κ**, so one run traces a front instead of one run per κ.

## Not being pursued

Deferred by decision, recorded so they are not rediscovered as ideas: UCT / model-based planning,
possibly as a baseline once everything else is done; sensing range, degree budgets and other
geometric limits; a genuinely directional architecture (GVP, e3nn, vector neurons), which is
thesis-scale.
