#!/usr/bin/env bash
# The size and domain sweep, in the order it has to happen.
#
#   ./tools/scale_runs.sh sizes      one comparison run per formation size
#   ./tools/scale_runs.sh domains    the five homogeneous corners, at the training size
#   ./tools/scale_runs.sh kappa      one run per kappa environment, classical arms included
#   ./tools/scale_runs.sh aggregate  the curves and the figures, from the runs above
#   ./tools/scale_runs.sh all        all of the above
#
# Stages run cheapest and most informative first, so that being cut off part way still
# leaves a complete curve up to whatever size was reached. Nothing here is idempotent
# and nothing overwrites an existing environment config or instance set.
#
# What it measures, and why in this order:
#
#   n=16, 24   beyond the training size, and still affordable for every classical
#              method, so the policy and greedy can be compared without either being
#              excused. These carry the argument on their own
#   n=8        below the training size, the other direction
#   corners    the five homogeneous domains at the training size. R^2 and R^2xS^1 have
#              never been evaluated and are the two the uniform training draw never
#              reaches, so they are the held-out test
#   n=48       where greedy costs minutes per network. Split in two runs: every cheap
#              method on the full set, and greedy on a subset, since one greedy number
#              per size is enough to price it
#   n=32       fills the curve between 24 and 48
#   n=96       the far end, and the only stage that is expendable

set -u
cd "$(dirname "$0")/.."

CPU=${CPU:-"--threads 1 --nice 10"}
BUDGET=(uv run tools/cpu_budget.py $CPU --)

LOGDIR=runs_outputs/logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/scale-${1:-none}-$(date +%Y%m%d-%H%M%S).log"

RUN() {
  echo "+ $*" | tee -a "$LOG"
  date +"  started %H:%M:%S" | tee -a "$LOG"
  if [ "${DRY:-0}" != "1" ]; then
    "${BUDGET[@]}" "$@" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "!! FAILED: $*" | tee -a "$LOG"
  fi
}

# ---------------------------------------------------------------- settings

P=env_actionAddRemoveEdgeDiscreteNoSelfLoops_rewardWeightedNormalizedShapeError_termMaxSteps_rigGFE_lean

# The policies carried through every size, chosen by the screening run that scored every
# checkpoint sharing this observation on 200 instances at the training size. MODELS_FEW
# and MODELS_FAR drop the set where one rollout costs minutes rather than seconds.
# A controlled ladder: same algorithm, backbone, training distribution and step count,
# kappa the only difference, plus the two arms that move a different axis.
MODELS=${MODELS:-"ref_k0_dqn_gine=k0,ref_k10_s1_dqn_gine=k10,estimation_randdom_k50_dqn_gine=k50,ref_k200_dqn_gine=k200,estimation_randdom_dqn_gine=n20,estimation_randdom_k50_ppo_gine=ppo50"}
MODELS_FEW=${MODELS_FEW:-"ref_k0_dqn_gine=k0,ref_k10_s1_dqn_gine=k10,ref_k200_dqn_gine=k200"}
MODELS_FAR=${MODELS_FAR:-"ref_k0_dqn_gine=k0,ref_k10_s1_dqn_gine=k10"}

EVAL="--seed 0 --device cuda --policy-mode sample"
SEARCH="--restarts 5 --spectral-shortlist 5"

# Every classical method that is affordable at that size. greedy evaluates all n(n-1)
# toggles before each edit and anneal is given greedy's own budget, so the pair is what
# drops out first.
ALL=initial,degree,spectral,constructive,anneal,greedy,learned
NO_ANNEAL=initial,degree,spectral,constructive,greedy,learned
CHEAP=initial,degree,spectral,constructive,learned

# ---------------------------------------------------------------- sizes

sizes() {
  RUN uv run experiments/comparison.py "${P}_n10_randdom_k10" --benchmark b2_n10_randdom \
      --episodes 200 $EVAL $SEARCH --methods $ALL --model "$MODELS" \
      --paired-reference greedy --no-plots --tag scale

  RUN uv run experiments/comparison.py "${P}_n16_randdom_k10" --benchmark b2_n16_randdom \
      --episodes 100 $EVAL $SEARCH --methods $ALL --model "$MODELS" \
      --paired-reference greedy --no-plots --tag scale

  RUN uv run experiments/comparison.py "${P}_n24_randdom_k10" --benchmark b2_n24_randdom \
      --episodes 50 $EVAL $SEARCH --methods $ALL --model "$MODELS" \
      --paired-reference greedy --no-plots --tag scale

  RUN uv run experiments/comparison.py "${P}_n8_randdom_k10" --benchmark b2_n8_randdom \
      --episodes 200 $EVAL $SEARCH --methods $ALL --model "$MODELS" \
      --paired-reference greedy --no-plots --tag scale
}

sizes_large() {
  # every cheap method on the full set, then greedy on a subset of the same instances.
  # The aggregator takes the newest run per (method, size), so the two do not collide
  RUN uv run experiments/comparison.py "${P}_n48_randdom_k10" --benchmark b2_n48_randdom \
      --episodes 25 $EVAL $SEARCH --methods $CHEAP --model "$MODELS_FEW" \
      --paired-reference spectral --no-plots --tag scale
  RUN uv run experiments/comparison.py "${P}_n48_randdom_k10" --benchmark b2_n48_randdom \
      --episodes 8 $EVAL $SEARCH --methods greedy --no-plots --tag scale_greedy

  RUN uv run experiments/comparison.py "${P}_n32_randdom_k10" --benchmark b2_n32_randdom \
      --episodes 25 $EVAL $SEARCH --methods $NO_ANNEAL --model "$MODELS" \
      --paired-reference greedy --no-plots --tag scale
}

sizes_far() {
  RUN uv run experiments/comparison.py "${P}_n96_randdom_k10" --benchmark b2_n96_randdom \
      --episodes 8 $EVAL $SEARCH --methods $CHEAP --model "$MODELS_FAR" \
      --paired-reference spectral --no-plots --tag scale
}

# ---------------------------------------------------------------- domains

domains() {
  for D in R2 R3 R2xS1 R3xS1 SE3; do
    RUN uv run experiments/comparison.py "${P}_n10_${D}_k10" --benchmark "b2_n10_${D}" \
        --episodes 100 $EVAL $SEARCH --methods $ALL --model "$MODELS" \
        --paired-reference greedy --no-plots --tag domain
  done
}

domains_n16() {
  for D in R2 R3 R2xS1 R3xS1 SE3; do
    RUN uv run experiments/comparison.py "${P}_n16_${D}_k10" --benchmark "b2_n16_${D}" \
        --episodes 50 $EVAL $SEARCH --methods $ALL --model "$MODELS" \
        --paired-reference greedy --no-plots --tag domain
  done
}

# ---------------------------------------------------------------- kappa

# Each policy in the environment it was trained for, with greedy and the annealer hill
# climbing that same objective, so the classical arms optimize what the policy was
# trained on rather than one fixed objective.
kappa() {
  RUN uv run experiments/comparison.py "${P}_n10_randdom_k0" --benchmark b2_n10_randdom \
      --episodes 100 $EVAL $SEARCH --methods $ALL --model ref_k0_dqn_gine=k0 \
      --paired-reference greedy --no-plots --tag front_k0
  RUN uv run experiments/comparison.py "${P}_n10_randdom_k10" --benchmark b2_n10_randdom \
      --episodes 100 $EVAL $SEARCH --methods $ALL --model ref_k10_s1_dqn_gine=k10 \
      --paired-reference greedy --no-plots --tag front_k10
  # the kappa = 50 environment is the unsuffixed one
  RUN uv run experiments/comparison.py "${P}_n10_randdom" --benchmark b2_n10_randdom \
      --episodes 100 $EVAL $SEARCH --methods $ALL \
      --model estimation_randdom_k50_dqn_gine=k50 \
      --paired-reference greedy --no-plots --tag front_k50
  RUN uv run experiments/comparison.py "${P}_n10_randdom_k200" --benchmark b2_n10_randdom \
      --episodes 100 $EVAL $SEARCH --methods $ALL --model ref_k200_dqn_gine=k200 \
      --paired-reference greedy --no-plots --tag front_k200
}

# ---------------------------------------------------------------- aggregate

aggregate() {
  RUN uv run experiments/scale.py "${P}_n10_randdom_k10" --seed 0 --device cuda \
      --model "$MODELS" --methods greedy,spectral,degree,constructive,anneal \
      --prior "runs_outputs/*__scale" --tag scale
  RUN uv run experiments/kappa.py "${P}_n10_randdom_k10" --seed 0 --device cuda \
      --prior "runs_outputs/*__front_k*" --tag kappa
  RUN uv run experiments/generalisation.py "${P}_n10_randdom_k10" --seed 0 --device cuda \
      --model "$MODELS" --methods greedy,spectral,degree \
      --prior "runs_outputs/*__domain" --tag domains
}

# ----------------------------------------------------------------

case "${1:-}" in
  sizes)       sizes ;;
  sizes_large) sizes_large ;;
  sizes_far)   sizes_far ;;
  domains)     domains ;;
  domains_n16) domains_n16 ;;
  kappa)       kappa ;;
  aggregate)   aggregate ;;
  all)         sizes; domains; kappa; sizes_large; sizes_far; aggregate ;;
  *)           sed -n '2,10p' "$0"; exit 1 ;;
esac
