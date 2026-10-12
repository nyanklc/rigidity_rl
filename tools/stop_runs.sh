#!/usr/bin/env bash
# The policies that choose when to halt, across sizes.
#
#   ./tools/stop_runs.sh sizes      the stop policies at n = 10, 16, 24
#   ./tools/stop_runs.sh sizearm    the policy trained at n = 16, across sizes
#   ./tools/stop_runs.sh aggregate  fold them into the size curves
#
# A policy without a stop action has to apply an edit at every step, so what it found is
# the best graph it passed through and where it ended is an accident of the horizon. A
# policy with a stop action decides when to halt, so the graph it halted on is its answer
# and its step count is its real cost. `baselines.rollout_result_final` scores these rows
# on the final state for that reason, and carries the best-visited figures beside them,
# the gap between the two being what says whether it stopped in the right place.
#
# Only the policies are run here. The classical methods do not use the action space, so
# their rows in the matching non-stop environment at the same kappa are the same numbers,
# and re-running greedy at n = 24 to get them again would cost more than everything else
# in this script put together.

set -u
cd "$(dirname "$0")/.."

CPU=${CPU:-"--threads 1 --nice 14"}
BUDGET=(uv run tools/cpu_budget.py $CPU --)

LOGDIR=runs_outputs/logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/stop-${1:-none}-$(date +%Y%m%d-%H%M%S).log"

RUN() {
  echo "+ $*" | tee -a "$LOG"
  date +"  started %H:%M:%S" | tee -a "$LOG"
  if [ "${DRY:-0}" != "1" ]; then
    "${BUDGET[@]}" "$@" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "!! FAILED: $*" | tee -a "$LOG"
  fi
}

P=env_actionAddRemoveEdgeDiscreteNoSelfLoops_rewardWeightedNormalizedShapeError_termMaxSteps_rigGFE_lean

# The four stop policies, one per conditioning weight. The kappa = 10 arm was trained on
# the same random-domain distribution as the rest of this chapter; the other three were
# trained on the fixed five-domain mixture, which is a difference to state rather than
# one to hide, since they share the observation and the action space exactly.
STOPS="ref_stop_dqn_gine=stop_k10,estimation_stop_dqn_gine=stop_k2,estimation_stop_k50_dqn_gine=stop_k50,estimation_stop_k200_dqn_gine=stop_k200"

EVAL="--seed 0 --device cuda --policy-mode sample"

# Training size as a controlled variable. Everything else in the chapter was trained on
# ten-agent formations; this one was trained on sixteen at the same conditioning weight,
# so the difference between its curve and the kappa = 10 curve is the training size and
# nothing else, bar the two differences stated in the chapter: it ran for 150k steps
# against 400k, and it was trained after the reward's shaping was changed from
# phi(s') - phi(s) to the discounted 0.99 phi(s') - phi(s).
SIZEARM="ref_n16_k10_dqn_gine=n16"

sizearm() {
  RUN uv run experiments/comparison.py "${P}_n10_randdom_k10" \
      --benchmark b2_n10_randdom --episodes 200 $EVAL --methods learned \
      --model "$SIZEARM" --no-plots --tag scale_arm
  RUN uv run experiments/comparison.py "${P}_n16_randdom_k10" \
      --benchmark b2_n16_randdom --episodes 100 $EVAL --methods learned \
      --model "$SIZEARM" --no-plots --tag scale_arm
  RUN uv run experiments/comparison.py "${P}_n24_randdom_k10" \
      --benchmark b2_n24_randdom --episodes 50 $EVAL --methods learned \
      --model "$SIZEARM" --no-plots --tag scale_arm
  RUN uv run experiments/comparison.py "${P}_n48_randdom_k10" \
      --benchmark b2_n48_randdom --episodes 25 $EVAL --methods learned \
      --model "$SIZEARM" --no-plots --tag scale_arm
}

sizes() {
  RUN uv run experiments/comparison.py "${P}_n10_randdom_k10_stop" \
      --benchmark b2_n10_randdom --episodes 200 $EVAL --methods learned \
      --model "$STOPS" --no-plots --tag scale_stop
  RUN uv run experiments/comparison.py "${P}_n16_randdom_k10_stop" \
      --benchmark b2_n16_randdom --episodes 100 $EVAL --methods learned \
      --model "$STOPS" --no-plots --tag scale_stop
  RUN uv run experiments/comparison.py "${P}_n24_randdom_k10_stop" \
      --benchmark b2_n24_randdom --episodes 50 $EVAL --methods learned \
      --model "$STOPS" --no-plots --tag scale_stop
}

aggregate() {
  RUN uv run experiments/scale.py "${P}_n10_randdom_k10" --seed 0 --device cuda \
      --model "$STOPS" --methods greedy,spectral,degree,constructive,anneal \
      --prior "runs_outputs/*__scale*" --tag scale_with_stop
}

case "${1:-}" in
  sizes)     sizes ;;
  sizearm)   sizearm ;;
  aggregate) aggregate ;;
  all)       sizes; sizearm; aggregate ;;
  *)         sed -n '2,5p' "$0"; exit 1 ;;
esac
