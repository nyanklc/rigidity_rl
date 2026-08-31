# What's next

Everything durable lives elsewhere: results in `README.md`, mathematics in `THEORY.md`,
implementation reasoning in `DESIGN_NOTES.md`, the code map in `CLAUDE.md`. This file is only
the plan, and can be deleted without losing anything.

---

## Where things stand

**The edge-count objective is solved and saturated.** On `bench_mixed` (20 frozen instances,
n=10, all five domains) the trained policy reaches 17.10 edges / 90% minimal against greedy's
17.35 / 80% and constructive's 17.75 / 30%. The proven lower bound is 17. `THEORY.md` §14.3
explains why nothing can gain much here: greedy sits 0-5% above a lower bound because the
problem is minimum submodular cover.

**Estimation error is now the metric, and it is measured rather than asserted.** `shape_err`
is the RMS state error per radian of bearing noise, comparable across `n`, domain and pose
range in a way λ is not. `estimation.py` validates it by actually perturbing bearings and
recovering the formation; agreement is a few percent at small noise (`THEORY.md` §18).

**The spectral functional does not matter.** Two 250k-step runs identical but for
`spectral_functional` (`eigenvalue` vs `trace`) are indistinguishable in training curves, in
evaluation, and in what the ablation says they read. Settled from three directions; see
§18.4-18.5. Do not spend more runs on it.

**What the policy actually reads** (ablation, both runs, three modes): `add_rank`, the exact
rank oracle, dominates. `remove_stiffness` is the first geometric channel in this project to
cost anything across all three modes -- it is the pruning signal. Raw `bearings` and
`coord_features` still cost nothing, so the spectral reward term did not make geometry matter.
`rigidity_quality` reads as a null and is a candidate for removal.

**The open failure is size.** At n=16 the policy reaches 23.20 edges and 0% minimal against
`m_req` 22. Cross-*domain* transfer is no longer the problem; cross-*size* is.

---

## Now

**Baselines from the survey** (Darvariu et al. 2024 §3.2). Beating greedy on a near-solved
objective means little; these make a claim mean something. All enter `evaluation.py` on the
existing `run_*(env, ...) -> result(...)` contract.

- `spectral` -- first-order spectral hill-climb, using `candidate_gain` and `removal_costs`.
  One eigendecomposition per edit against greedy's `n(n-1)` phi-evaluations.
- `anneal` -- simulated annealing on the configured phi, **budget-matched to greedy's
  phi-evaluation count** and reported as such. Assumes no submodularity, which is the right
  property on a non-submodular objective. Own RNG, like `constructive`.
- `degree` -- lowest-degree-product addition, highest-degree redundant pruning. No rigidity
  algebra beyond the rigidity test, so it doubles as the tier-1 distributed-plausible
  reference from `DESIGN_NOTES.md#distributed-feasibility`.
- **A compute column.** Greedy spends ~900 phi-evaluations per instance at n=10; the policy
  spends ~10 forward passes. Nothing reports this, and amortised training against per-instance
  search is the survey's central argument for RL.

## Next

**Robustness, recovery and membership change** (`robustness.py`, new). The differentiating
experiment: greedy and constructive are one-shot, and repair is a sequential editor's native
operation. Per frozen instance every method faces the same schedule of
`converge -> event -> recover`, with events `edge_drop_random`, `edge_drop_worst` (chosen
exactly via `removal_costs`), `bearing_corrupt`, `node_leave`, `node_join`.

Scored against `repair_edge_count`, so "repaired in k edges" reads as a gap above a bound, and
against best-of-combinatorial repair, which §19.4 shows is 1.7x-3.1x better than greedy on
shape error. `node_join` / `node_leave` change `n`: the models are already `n`-agnostic, so the
harness bypasses skrl and calls `model.compute` directly, rebuilding the observation space at
the new `n`. **Verify that path in a test first.** It also makes size generalisation measurable
without the deferred variable-`n` training work.

Karimian and Tron's factor-graph algorithm is deliberately **not** implemented: its contribution
is the minimum-edge guarantee in homogeneous 2-D, and `greedy_rigid_repair` was measured to
attain that bound wherever `c_max = 1`, which §14.3's matroid argument predicts.

**Randomised domain composition.** A `domain_sampler` env key drawing the per-agent mix in
`reset()`; `n` fixed. Dirichlet-multinomial rather than per-agent uniform, which at n=10 would
draw a homogeneous corner with probability 5e-7. Fix the ordering while there: draw the mix,
compute `m_req` for it, then sample the edge count -- `sample_initial_edge_count` currently uses
the *previous* episode's cached `m_req`. Set `max_steps` from the worst case over the sampler's
support so the horizon stays constant. `benchmark.py` needs per-instance `domains`.

Criterion, fixed before the run: train on Dirichlet draws, evaluate on held-out composition
families (the five homogeneous corners, and two-domain mixes never seen as such), and require
parity with greedy on edges *and* minimality on every family.

**Statistical protocol**, threaded through the above rather than done last. Three seeds per
arm; `report.paired_stats(rows, reference="greedy")` for per-instance paired differences with a
bootstrap CI, since methods already run on identical instances and `aggregate()` throws the
pairing away; 50 frozen instances rather than 20. `DESIGN_NOTES.md#horizon` measured a 35-point
minimality swing between seeds, so no single-seed row means anything on its own.

## Later

- **Variable `n` in training.** Pad to `n_max`, thread a node-validity mask through both
  backbones, `global_mean_pool` and the action masks. This is what the n=16 failure needs. The
  robustness harness measures the gap first.
- **The stop action.** `Best is min rigid` is 0.994 but `Final is min rigid` is 0.51 and
  `Edit efficiency` 0.086: the policy finds a minimal graph in ~12 of 78 steps and then
  oscillates, because `skip_enabled = False` forces an edit every step. `DESIGN_NOTES.md#horizon`
  has the arms and the measurement; it is unresolved and needs seeds.
- **Drop `rigidity_quality`, or find out why it is unread.** It is on by default in the current
  configs and the ablation says the policy ignores it.
- **Spectral observation channels** for the functional the reward carries. Kept separate from
  any reward change: `THEORY.md` §16 explains why moving both at once makes the ablation
  unreadable.
- **Persistence of excitation, i.e. dropping the static assumption.** Su et al. (2026) §4.2 and
  Schiano and Tron (2018): if the agents move, observability needs far fewer edges and the
  global scale stops being unobservable. The question this thesis asks has a different answer
  for a moving formation. A redirection rather than an increment, so it belongs in the thesis
  as the honest answer to "why assume the formation is static?".
- **Global convergence / basin of attraction.** `solve_shape` from a cold start rather than the
  truth: how many restarts reach the true shape. No spectral functional predicts it.
- **Conditioning the policy on kappa**, so one run traces a front instead of one run per kappa.

## Not being pursued

Deferred by decision, recorded so they are not rediscovered as ideas: UCT / model-based
planning, possibly as a baseline once everything else is done; sensing range, degree budgets
and other geometric limits; a genuinely directional architecture (GVP, e3nn, vector neurons),
which is thesis-scale; a measured-error reward, which costs 450-1160x lambda per evaluation
and buys at most 8% by §18.5.
