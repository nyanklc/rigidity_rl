# What's next

Everything durable lives elsewhere: results in `README.md`, mathematics in `THEORY.md`,
implementation reasoning in `DESIGN_NOTES.md`, the code map in `CLAUDE.md`. This file is only the
plan, and can be deleted without losing anything.

---

## Now

**Train with the stiffness and removal observations on.** `stiffness_kappa = 2` alone was measured to change nothing:
the policy matched greedy on edges and minimality but carried *worse* stiffness (7.0e-04 against
3.0e-03), landing at q = 0.47, i.e. exactly the reference. The reason was that every rigidity
channel is identically zero once the graph is rigid, which is the only regime where the stiffness
exists. `rigidity_stiffness = True` adds `add_stiffness` / `node_slack`, which are nonzero exactly there,
`rigidity_removal = True` adds `remove_rank` / `remove_stiffness` so the policy can finally tell the
70% of edges that are safe to prune from the 30% that are load-bearing, and the `AddRemoveEdge` heads
now take `e_ij` so none of it is destroyed by mean aggregation. **Configs need regenerating**: the
`margin_*` keys are now `stiffness_*` and `load()` raises on the old ones.
Retrain at `stiffness_kappa = 2` with the flag on and compare against the current `margin_dqn_gine`.
Two things decide it: the stiffness column should stop sitting below greedy's, and `ablation.py` in at
least two modes should charge for `add_stiffness` and `remove_rank`. If it costs nothing even routed straight to the
head, that is a real negative result about this formulation, not a bug to chase.

**Then the bearings arm.** `include_candidate_bearings = False` is already a config switch, so
stiffness-informed with and without raw bearings needs no code. If bearings cost nothing on top of
`add_stiffness`, that prices raw geometry against derived invariants, and the no-bearings arm is
rotation-invariant by construction, which removes the `R^d` rotation-dependence rather than
augmenting around it.

**Sweep kappa.** `stiffness_kappa` is implemented and off by default. The measurement that
matters is a κ sweep - {0, 0.9, 2, 4} - on `mixed` and on n=8/`R^3`, reported as a front of edge
count against stiffness rather than as a single number, since there is no principled κ.
`tools/kappa_sweep.py` already does this for `greedy` without training, and gives the shape to
expect: stiffness ×2.0 at κ=0.9 and ×12.4 at κ=2, at a flat edge count.

**One control run is owed:** the EGNN input embedder, which makes both backbones 128 wide. Report
that comparison as *equal width, unequal capacity* - the EGNN is 10.9x GINE's parameter count at
matched width (`tools/backbone_capacity.py`), and matched parameters would instead put it at a
quarter of the width. The affine q-head repair rides along with it and probably does not need
isolating, being a defect repair rather than an arm.

## Next

**Resample the domain composition.** The open question behind the transfer failure is whether it is
composition coverage or a capacity limit. The training mixture is two agents of each domain, so
high-DOF agents are never in the majority and homogeneous corners are never drawn. A
`domain_sampler` env key drawing the mix in `reset()` separates the two. Fix the criterion before
running it, or the coverage hypothesis is unfalsifiable: ≥80% minimal on homogeneous `SE(3)` and
`R^3xS^1` at n=8, where both classical baselines reach 100%. Two ordering details - compute the mix
first, then `m_req`, then the edge count, and pass `env.m_req` into `is_MBR` rather than letting it
recompute from the current graph.

**A pairwise action head.** Held-out linear probes on the trained GINE backbone predicting "does
adding i→j raise the rank" give 0.955 from `[h_i, h_j]` against 1.000 from `e_ij` alone in `R^3`.
The backbone carries most of it, but a perfect pairwise signal exists in `e_ij` and reaches the head
degraded. Head input `[h_i, h_j, h_i ⊙ h_j, e_ij, adj_ij]`, plus routing the action-specific
rigidity scalars around the GNN as a skip connection - three layers of mean aggregation over `n−1`
pairs is the wrong thing to put between a near-affine signal and a Q-value. Note the affine-head
repair already closed part of the gap this was scoped to close, so re-probe before fixing a width.

**Evaluation protocol.** Three seeds minimum and paired statistics - a single seed at n=8/`R^3` was
measured to span at least 35 points of minimality. Report final state as the headline with
best-visited as a second column. Add a spectral stiffness baseline, since greedy is a weak opponent on
a non-submodular objective (`THEORY.md` §14.4), and extend brute force at n ≤ 5 to return the
stiffness-optimal minimal graph.

## Later

- **Margin-aware observation.** With `v` the singular vector at the rigidity eigenvalue,
  `Δλ ≈ ‖b_ij v‖²` for adding `(i,j)`, and the same number is what removing an existing edge costs.
  Measured log-log correlation 0.87-0.96 across domains, with the predictor's top pick in the true
  top-3 only 30-50% of the time - a strong ranking signal, not an oracle. Behind its own flag, and
  deliberately *not* added at the same time as the stiffness reward, or the ablation above cannot
  distinguish "the reward made geometry matter" from "we handed it the answer".
- **Multi-n training.** Pad every sub-env to `n_max` and thread a boolean node mask through both
  backbones (`EGNN` already takes a `mask` the backbone never passes; GINE's complete-digraph builder
  needs to respect it), and mask padded nodes in the action masks.
- **Conditioning the policy on κ**, so one run traces the whole front instead of one run per κ. Needs
  κ as an observation channel and validation against specialists at the endpoints. Only worth it if
  the run count becomes the bottleneck.

## Not being pursued

Deferred by decision, recorded so they are not rediscovered as ideas: UCT / model-based planning,
possibly as a baseline once everything else is done; sensing range, degree budgets and other
geometric limits; a genuinely directional architecture (GVP, e3nn, vector neurons), which is
thesis-scale and should not start before the stiffness objective shows the geometry matters at all.
