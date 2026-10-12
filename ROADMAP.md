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

**The survey's baselines and a compute axis are in** (`spectral`, `anneal`, `degree`, `cost.py`;
see `DESIGN_NOTES.md#spectral-baseline` and `#cost-counters`). What they said, and what is left:

- **Compute is not settled, and the measurement that said so is mis-metered.** The 1424 against
  greedy's 2252 charged the policy for a `reset()` no other method pays for, roughly 60% of its
  row (`DESIGN_NOTES.md#cost-counters`). Fix the metering and re-run before quoting it either way.
  Removing the reward's greedy reference cut a reset from 2974 ms to 56 ms at n=20, which moves
  this further in the policy's favour.
- **`spectral` is greedy at a fraction of the price.** Identical output at `stiffness_kappa = 0`
  (60/60 instances, five domains) for 3.7-6.9x fewer computations, growing as `n^0.81` against
  `n^2.95` (`tools/cost_scaling.py`). It replaces `greedy` as the cost-fair opponent at large `n`,
  where greedy's `O(n^6)` per network is the reason evaluations are capped where they are.
- **`degree` is the surprise and needs more instances.** 17.67 edges / 50% minimal at 79
  computations against greedy's 17.17 / 100% at 2252, on six instances. If that survives 50
  frozen instances it is a result worth stating -- a rule reading nothing but degrees lands within
  half an edge for 3% of the compute -- and it sharpens the question of what the policy is for.
  Its shape error is 20x greedy's, which is the obvious place to look for where it pays.

**The experiments compare models against each other**, and the presentation figures are
reproducible from the repository rather than from gitignored scripts. On `bench_mixed`, 20
instances, four models against eight baselines: every model reaches the proven bound more often
than any classical method (`gine` and `equi` 90%, `greedy` 65%), and `k10_gine`, trained at
`stiffness_kappa = 10`, has the best shape error (6.9 against greedy's 12.2) while placing last of
the four on edges. That is what a stiffness-weighted objective should do, and it is the first
side-by-side measurement of it.

One seed, one benchmark, and three of those four models differ from the reference in one axis
each, so read the ordering rather than the digits until the statistical protocol below is done.

Immediate follow-ups, in order:

- Re-run the above on `--benchmark` sets of 50 rather than 6, with three seeds. Nothing in this
  section is currently more than suggestive.
- `anneal` vs `greedy` at `stiffness_kappa > 0` specifically, read on `shape err` rather than on
  edges. That is the experiment §14.4 argues for and it now has both arms.

## Next

**Repair and rewiring. Landed, as `repair.py`** plus `experiments/repair.py`, not
`robustness.py` -- `ablation.py --mode noise` is already the robustness measurement. Nine
events over two classes, scored on where each method's answer sits among *every* repair of the
same size (100 = best). Design in `DESIGN_NOTES.md#repair-harness`, results in `THEORY.md`
§19.5. `bearing_corrupt` is dropped with its reason recorded there: it cannot be made well
posed without a per-edge weight in `B^T W B` and an `edge_reliability` channel, and without
those no method can beat chance.

The variable-`n` path is verified in `tests/test_repair.py` as ROADMAP asked. The load-bearing
step turned out to be `model.observation_space`, not the action head: left at the training `n`
the model returns a wrongly-sized Q-vector over misaligned inputs and raises nothing.

**What it already says.** phi hill-climbing reaches the 78th-95th percentile against
marginal-rank greedy's 38-64, so the conditioning term is most of the criterion Karimian and
Tron leave open (§19.5). `rebuild` from empty churns 13-20 edges against a repair's 2. And
`edge_drop_redundant` is a null for every classical method, which is the arm a high-κ policy
exists to answer.

**The κ question is answered, and negatively** (`THEORY.md` §19.6). Three policies on
`bench_mixed`, 20 instances: none reaches the phi hill climb on any breaking event, each sits
between marginal-rank greedy and it, and **κ does not order them** -- `k50` is not better than
`k2` on any event and worse on two. One caveat that has to travel with the numbers: the hill
climb spent exactly the minimum on only 13, 8 and 7 of 20 instances against the policies' 19-20,
and a percentile is undefined off the bound, so its 90-93 is measured on a subset. `repair_count`
is the figure that shows it. So κ selecting among equally sparse graphs (§15.6) and phi
being most of the open criterion (§19.5) both hold, while *training* at a higher κ transferring
that choice into the policy does not. Still one seed per arm.

What is left, in order:

- **Three seeds per arm.** Everything above is one seed. `report.paired_stats` now exists and
  the drivers print it, so the pairing is free; what is missing is the seeds.
- **`edge_drop_redundant` against a high-κ policy**, which is the one event where the reward
  term is the only thing that could motivate an edit. Currently every policy churns 5-7 edges
  there and the percentile is undefined, because none of them spends the minimum.
- **`experiments/repair.py --events all` on `bench_n16_R3`**, where only the bound tier is
  available.

Karimian and Tron's factor-graph algorithm is deliberately **not** implemented: its contribution
is the minimum-edge guarantee in homogeneous 2-D, and `greedy_rigid_repair` was measured to
attain that bound wherever `c_max = 1`, which §14.3's matroid argument predicts.

**Formation control as a downstream task. Landed, as `experiments/control.py` on
`convergence.py`.** A goal shape, a start displaced from it, one edge set per method used
for both the goal bearings and the controller's `B`, and the trajectory recorded: RMS
bearing error per edge and position error against the goal shape with the gauge projected
out. `THEORY.md` §20.

Settled on the way: `B^T . goal` and `B^T . (goal - current)` are the **same vector** to
1e-15, since every 3-row block of `B` projects orthogonal to its own bearing, so the
controller's form was never a missing subtraction.

**The earlier numbers here are withdrawn.** The first harness used a tolerance on the
*summed* squared bearing error, which makes a denser graph drive each edge lower to reach
the same total and so penalises the graphs it was meant to show converging fastest. It
reported the complete graph at `lambda = 2.6` as non-convergent. Everything measured under
it, including the lambda exponent and the per-method converged counts, is not restated.
The replacement measures per edge.

Two traps that stay: `GradientBasedController`'s gain mask is tiled per agent while `B`'s
columns are `[3n positions | 3n attitudes]`, so at `lin != ang` the gains land on the
wrong coordinates (priced behind `--check-gain-mask`; `control.py` is obsolete so it is
not patched); and the start must be displaced only within each agent's DOFs, or a planar
agent's `z` makes every mixed-domain run read as non-convergent.

What is left: seeds, and enough spread in `lambda` to fit the exponent rather than only
the ordering. The topologies these methods build are all minimal and therefore all at the
soft end.

**Randomised domain composition. Done, as `random_domains`**, uniform per agent rather than the
Dirichlet-multinomial planned here. The ordering fix, the worst-case horizon and per-instance
benchmark domains all landed with it; see `DESIGN_NOTES.md#random-domains`.

Uniform was chosen over Dirichlet because the argument recorded here (a homogeneous corner at
5e-7) is an argument for putting the corners *in* the training distribution, while the evaluation
plan below holds them out, and because the measured worry -- that the reward scale would move
between episodes -- did not survive contact: the quantum's p90/p10 narrows with `n` and the
composition adds 17% at n=10 and nothing at n=20 to a between-episode spread that already exists.
What uniform does not do is reach the high-DOF-majority compositions where transfer is measured to
degrade, since the multinomial concentrates near the centroid. Dirichlet-multinomial with a
concentration `alpha` is the one-parameter family that spans both (`alpha -> infinity` recovers
uniform), and is the arm to add if the held-out corners come out badly.

Criterion, fixed before the run: train on the uniform draw, evaluate on held-out composition
families (the five homogeneous corners, and two-domain mixes never seen as such), and require
parity with greedy on edges *and* minimality on every family. The corners are held out for free
here, since uniform draws never reach them.

**Statistical protocol. `report.paired_stats` landed** and `experiments/comparison.py` prints
it under the table: per-instance differences against a reference, with a percentile bootstrap
over those differences. What is left is the seeds and the instance count -- three seeds per arm
and 50 frozen instances rather than 20. `DESIGN_NOTES.md#horizon` measured a 35-point minimality
swing between seeds, so no single-seed row means anything on its own.

It already changes a reading. On `bench_mixed`, 20 instances, **the policies' edge advantage
over greedy is not supported by the instances**: `k2` is −0.20 edges [−0.45, +0.05], +0.38 phi
[−0.04, +0.82] and +0.20 at bound [−0.05, +0.45], every interval straddling zero, even though
the unpaired table reads 17.15 against 17.35 and 85% against 65%. What *is* supported is the κ
trade: `k50` is −0.334 dex of shape error [−0.472, −0.198], 16 networks better and 4 worse, and
pays +0.40 edges [+0.10, +0.75] and −0.30 at bound [−0.55, −0.05] for it. Both CIs are clear of
zero. The table's `+-` is how much the *networks* differ, which is far larger than the gap
between two methods on one network, and that is what was hiding both facts.

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
