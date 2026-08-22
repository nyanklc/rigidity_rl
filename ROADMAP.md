# What's next

Everything durable lives elsewhere: results in `README.md`, mathematics in `THEORY.md`,
implementation reasoning in `DESIGN_NOTES.md`, the code map in `CLAUDE.md`. This file is only the
plan, and can be deleted without losing anything.

---

## Now

**Train against the margin.** `margin_kappa` is implemented and off by default. The measurement that
matters is a κ sweep - {0, 0.9, 2, 4} - on `mixed` and on n=8/`R^3`, reported as a front of edge
count against margin rather than as a single number, since there is no principled κ.
`tools/kappa_sweep.py` already does this for `greedy` without training, and gives the shape to
expect: margin ×2.0 at κ=0.9 and ×12.4 at κ=2, at a flat edge count.

The test that decides whether the geometric half of the thesis stands: **re-run `ablation.py` in at
least two modes and check the geometric channels now cost something.** Today, destroying `bearings`,
`coord_features`, `add_gain` and `flex_mag` costs a trained policy nothing, which is correct under a
rank-only objective (`THEORY.md` §14.0). If it is still nothing at κ > 0, the geometric machinery -
the EGNN, the all-pairs bearings, the equivariance - is not earning its place and should be reported
as such.

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
best-visited as a second column. Add a spectral margin baseline, since greedy is a weak opponent on
a non-submodular objective (`THEORY.md` §14.4), and extend brute force at n ≤ 5 to return the
margin-optimal minimal graph.

## Later

- **Margin-aware observation.** With `v` the singular vector at the rigidity eigenvalue,
  `Δλ ≈ ‖b_ij v‖²` for adding `(i,j)`, and the same number is what removing an existing edge costs.
  Measured log-log correlation 0.87-0.96 across domains, with the predictor's top pick in the true
  top-3 only 30-50% of the time - a strong ranking signal, not an oracle. Behind its own flag, and
  deliberately *not* added at the same time as the margin reward, or the ablation above cannot
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
thesis-scale and should not start before the margin objective shows the geometry matters at all.
