#!/usr/bin/env bash
# Everything the results chapter needs, in the order it has to happen.
#
#   ./tools/thesis_runs.sh configs      write the environment configs
#   ./tools/thesis_runs.sh benchmarks   freeze the instance sets that do not exist yet
#   ./tools/thesis_runs.sh train        the training runs, core first
#   ./tools/thesis_runs.sh extra        the optional training arms
#   ./tools/thesis_runs.sh eval         every evaluation run
#   ./tools/thesis_runs.sh tools        the measurements that need no model
#   ./tools/thesis_runs.sh all          all of the above
#
# Every command's output is teed to runs_outputs/logs/<stage>-<timestamp>.log. The
# tools/ scripts write no files of their own, so that log is where their tables live.
#
# DRY=1 prints the commands without running them. Nothing here is idempotent: a
# second `train` prompts to continue or start fresh, and `configs` overwrites.
#
# The reference configuration, one axis of which every arm moves:
#
#   formation     n = 10, random_domains, so the mix is redrawn every episode
#   objective     WeightedNormalizedShapeError at kappa = 10
#   algorithm     DQN, gamma 0.99, 400k steps, seeds 0/1/2
#   backbone      GINE, 3 layers, H = 128, head 256
#   action space  AddRemoveEdgeDiscreteNoSelfLoops, no stop action
#   observation   all six rigidity flags on, graph_features off
#   evaluation    bench_n10_randdom, 20 frozen instances, sampled rollout on cuda
#
# What already exists and is not retrained here:
#   estimation_randdom_k50_{dqn,ppo}_{gine,equi}   the kappa = 50 arm, and the full
#                                                  algorithm x backbone 2x2, 400k steps
#   estimation_randdom_dqn_gine                    n = 20, kappa = 2, the size arm
# The kappa = 50 environment is the unsuffixed ..._n10_randdom config, which is why
# no _k50 config is generated below.
#
# A model only loads against an environment agreeing on every key in
# experiments/common.OBS_KEYS, which includes action_type, the six rigidity flags and
# include_candidate_bearings. So the observation and action-space arms are evaluated in
# their own environments and never beside the reference in one run.

set -u
cd "$(dirname "$0")/.."

# Everything runs under tools/cpu_budget.py, which caps every BLAS and torch thread
# pool and nices the process. These workloads are one Python thread driving matrices
# too small to pay for the fan-out, so capping frees the machine and is usually also
# faster. Override with CPU, for instance CPU="--threads 2 --cores reserve:4" to keep
# four logical CPUs free for interactive work, or CPU="" to run unwrapped.
if [ -z "${CPU+set}" ]; then CPU="--threads 1 --nice 10"; fi
BUDGET=()
if [ -n "$CPU" ]; then BUDGET=(uv run tools/cpu_budget.py $CPU --); fi

# Everything both prints and is logged. The tools/ scripts write no files of their own,
# so without this their tables exist only in terminal scrollback and a number quoted in
# the thesis has nothing behind it. One log per stage per invocation.
LOGDIR=runs_outputs/logs
LOG="$LOGDIR/${1:-none}-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$LOGDIR"

RUN() {
  echo "+ $*" | tee -a "$LOG"
  if [ "${DRY:-0}" != "1" ]; then
    "${BUDGET[@]}" "$@" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "!! FAILED: $*" | tee -a "$LOG"
  fi
}
ENVRUN() {            # RUN with a leading VAR=VALUE list, so DRY prints the assignments
  echo "+ $*" | tee -a "$LOG"
  [ "${DRY:-0}" = "1" ] && return 0
  local assigns=()
  while [ $# -gt 0 ]; do
    case "$1" in
      [A-Za-z_]*=*) assigns+=("$1"); shift ;;
      *) break ;;
    esac
  done
  env "${assigns[@]}" "${BUDGET[@]}" "$@" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || echo "!! FAILED: ${assigns[*]} $*" | tee -a "$LOG"
}

# ---------------------------------------------------------------- settings

P=env_actionAddRemoveEdgeDiscreteNoSelfLoops_rewardWeightedNormalizedShapeError_termMaxSteps
RIG=_rigGFE_lean
E_K0="${P}${RIG}_n10_randdom_k0"
E_K10="${P}${RIG}_n10_randdom_k10"
E_K50="${P}${RIG}_n10_randdom"                    # the existing one, kappa = 50
E_K200="${P}${RIG}_n10_randdom_k200"
E_NORIG="${P}_lean_n10_randdom_k10_norig"         # no rigidity flags, so no rig tag
E_TIER1="${P}_lean_n10_randdom_k10_tier1"
E_SELECT="env_actionSelectNodesSequentially_rewardWeightedNormalizedShapeError_termMaxSteps${RIG}_n10_randdom_k10"
E_STOP="${P}${RIG}_n10_randdom_k10_stop"
E_N8="${P}${RIG}_n8_randdom_k10"
E_N16="${P}${RIG}_n16_randdom_k10"
E_N20="${P}${RIG}_n20_randdom_k10"

# Generator settings shared by every config. These are the values the existing
# kappa = 50 config carries, so a new arm differs from it only on its own axis.
GEN="RANDOM_DOMAINS=true ACTION_TYPE=AddRemoveEdgeDiscreteNoSelfLoops \
     STATE_SCORE_TYPE=WeightedNormalizedShapeError TERMINATION_CONDITION_TYPE=MaxSteps \
     OBS_TYPE=Dict GRAPH_FEATURES=false ROTATION_AUGMENTATION=true \
     TRACK_DATA_ENABLE=true RANDOM_GRAPH_WITH_MEAN_MIN_EDGES=true \
     STIFFNESS_REF_SAMPLES=3 TRUNCATE_ENABLE=false ACTION_REWARDS_ENABLE=false \
     ONLY_RANDOMIZE_EDGES=false"
RIG_ON="RIGIDITY_GLOBAL=true RIGIDITY_QUALITY=true RIGIDITY_FLEX=true \
        RIGIDITY_EDGE=true RIGIDITY_STIFFNESS=true RIGIDITY_REMOVAL=true \
        INCLUDE_CANDIDATE_BEARINGS=true"
RIG_OFF="RIGIDITY_GLOBAL=false RIGIDITY_QUALITY=false RIGIDITY_FLEX=false \
         RIGIDITY_EDGE=false RIGIDITY_STIFFNESS=false RIGIDITY_REMOVAL=false"

# Trainer settings shared by every run. Everything not named here is left at the value
# the existing kappa = 50 models were trained with: lr 3e-4, gamma 0.99, batch 256,
# replay 10000, update every 4 steps, target every 200 updates, skrl's polyak 0.005.
DQN="BACKBONE=GINE TOTAL_TIMESTEPS=400000 EGREEDY_STEPS=200000 \
     EPSILON_START=0.8 EPSILON_END=0.02 GNN_HIDDEN_DIM=128 QNETWORK_HEAD_HIDDEN_DIM=256"
PPO="BACKBONE=GINE TOTAL_TIMESTEPS=400000 ROLLOUT_SIZE=256 LEARNING_EPOCHS=2 \
     MINI_BATCHES=8 ENTROPY_LOSS_SCALE=0.01 GNN_HIDDEN_DIM=128 \
     ACTOR_HEAD_HIDDEN_DIM=256 CRITIC_HEAD_HIDDEN_DIM=256"

# Evaluation flags, and which driver accepts what. Audited against every
# add_common_args / add_model_args call in experiments/:
#
#   driver              --benchmark  --episodes  --brief  --policy-mode
#   comparison.py            y           y          y          y
#   repair.py                y           y          y          y
#   control.py               y           n          y          n
#   feature_ablation.py      n           n          n          n
#   training.py              n           n          n          n
#   generalisation.py        n           n          n          n
#
# So EVAL is for comparison and repair, and SIMPLE for the rest. control.py takes
# SIMPLE plus its own --benchmark and --cases.
EVAL="--episodes 20 --seed 0 --device cuda --policy-mode sample"
SIMPLE="--seed 0 --device cuda"
SEARCH="--restarts 5 --spectral-shortlist 5"
BENCH=bench_n10_randdom

# ---------------------------------------------------------------- configs

configs() {
  # kappa arms at the reference
  ENVRUN $GEN $RIG_ON STIFFNESS_KAPPA=0   uv run environment.py 10 "R^2" k0
  ENVRUN $GEN $RIG_ON STIFFNESS_KAPPA=10  uv run environment.py 10 "R^2" k10
  ENVRUN $GEN $RIG_ON STIFFNESS_KAPPA=200 uv run environment.py 10 "R^2" k200

  # observation arms. norig prices the exact one-step oracles, tier1 additionally
  # removes the candidate bearings, which is what a distributed agent could not have
  ENVRUN $GEN $RIG_OFF INCLUDE_CANDIDATE_BEARINGS=true STIFFNESS_KAPPA=10 \
         uv run environment.py 10 "R^2" k10_norig
  ENVRUN $GEN $RIG_OFF INCLUDE_CANDIDATE_BEARINGS=false STIFFNESS_KAPPA=10 \
         uv run environment.py 10 "R^2" k10_tier1

  # action space and termination. ACTION_TYPE appears twice on the next line, once
  # from GEN and once as the override; env takes the last assignment
  ENVRUN $GEN $RIG_ON STIFFNESS_KAPPA=10 ACTION_TYPE=SelectNodesSequentially \
         uv run environment.py 10 "R^2" k10
  ENVRUN $GEN $RIG_ON STIFFNESS_KAPPA=10 SKIP_ENABLED=true SKIP_IS_STOP=true \
         TIME_PENALTY_VALUE=0.05 uv run environment.py 10 "R^2" k10_stop

  # the sizes the generalization section evaluates on. One config per n is enough:
  # a frozen benchmark carries its own domains, so an n = 8 random-domain config
  # evaluates bench_n8_SE3 correctly
  ENVRUN $GEN $RIG_ON STIFFNESS_KAPPA=10 uv run environment.py 8  "R^2" k10
  ENVRUN $GEN $RIG_ON STIFFNESS_KAPPA=10 uv run environment.py 16 "R^2" k10
  ENVRUN $GEN $RIG_ON STIFFNESS_KAPPA=10 uv run environment.py 20 "R^2" k10
}

# ---------------------------------------------------------------- benchmarks

benchmarks() {
  # bench_n10_randdom and bench_n20_randdom already exist, both 20 instances at seed 0
  RUN uv run benchmark.py "$E_N8"  bench_n8_randdom  --instances 20 --seed 0
  RUN uv run benchmark.py "$E_N16" bench_n16_randdom --instances 20 --seed 0
}

# ---------------------------------------------------------------- training

train() {
  # A1  the reference, three seeds. This is also the only measurement of the seed
  #     spread, and the one arm that must not be cut below three
  for S in 0 1 2; do
    ENVRUN $DQN SEED=$S uv run train_dqn.py "$E_K10" ref_k10_s${S}_dqn_gine
  done

  # A3  the rest of the kappa front. kappa = 50 is the existing 2x2, kappa = 10 is the
  #     reference above. kappa = 0 is the control the ablation test needs: the geometric
  #     channels must cost nothing there and something above it
  ENVRUN $DQN SEED=0 uv run train_dqn.py "$E_K0"   ref_k0_dqn_gine
  ENVRUN $DQN SEED=0 uv run train_dqn.py "$E_K200" ref_k200_dqn_gine

  # A4  rigidity channels off
  ENVRUN $DQN SEED=0 uv run train_dqn.py "$E_NORIG" ref_norig_dqn_gine

  # B2  the factored action space
  ENVRUN $DQN SEED=0 uv run train_dqn.py "$E_SELECT" ref_select_dqn_gine

  # B3  the stop action, at time_penalty 0.05. One run rather than a penalty sweep:
  #     the sweep is what one seed could not separate, and repeating it will not either
  ENVRUN $DQN SEED=0 uv run train_dqn.py "$E_STOP" ref_stop_dqn_gine
}

extra() {
  # C1  the tier-1 observation, which is what a distributed version would live with
  ENVRUN $DQN SEED=0 uv run train_dqn.py "$E_TIER1" ref_tier1_dqn_gine
  # C2  four times longer, to establish that the step budget is not the constraint
  ENVRUN $DQN SEED=0 TOTAL_TIMESTEPS=1600000 EGREEDY_STEPS=800000 \
         uv run train_dqn.py "$E_K10" ref_long_dqn_gine
  # B4  the size arm at the reference kappa. The existing n = 20 model is at kappa = 2
  ENVRUN $DQN SEED=0 uv run train_dqn.py "$E_N20" ref_n20_k10_dqn_gine
  # A2/B1 at the reference kappa, if the existing kappa = 50 2x2 is not accepted as
  #       the algorithm and backbone evidence
  ENVRUN $PPO SEED=0 uv run train_ppo.py "$E_K10" ref_k10_ppo_gine
  ENVRUN $DQN SEED=0 BACKBONE=Equivariant uv run train_dqn.py "$E_K10" ref_k10_dqn_equi
}

# ---------------------------------------------------------------- evaluation

eval_runs() {
  REF=ref_k10_s0_dqn_gine
  SEEDS="ref_k10_s0_dqn_gine=s0,ref_k10_s1_dqn_gine=s1,ref_k10_s2_dqn_gine=s2"
  KAPPAS="ref_k0_dqn_gine=k0,ref_k10_s0_dqn_gine=k10,estimation_randdom_k50_dqn_gine=k50,ref_k200_dqn_gine=k200"

  # 8.2  learning curves, read off the tensorboard events under runs/<model_name>
  RUN uv run experiments/training.py "$E_K10" $SIMPLE \
      --runs ref_k10_s0_dqn_gine,ref_k10_s1_dqn_gine,ref_k10_s2_dqn_gine,ref_k0_dqn_gine,ref_k200_dqn_gine

  # 8.4, 8.5, 8.7, 8.10  the one run where the full baseline comparison is the point.
  #     It carries the main table, the cost block, the noise sweep and the decision
  #     analysis, which is on because n = 10 is under the n <= 12 gate
  RUN uv run experiments/comparison.py "$E_K10" --benchmark $BENCH $EVAL $SEARCH \
      --methods all --model "$SEEDS" \
      --noise-sweep 0.1,0.5,1,2,5 --noise-trials 100 \
      --paired-reference greedy --plot-episodes 3 --tag main

  # 8.3  the design arms. Each moved an OBS_KEYS entry, so each is evaluated in its own
  #      environment against greedy there. Policy against policy, so the other baselines
  #      would only crowd the figures
  RUN uv run experiments/comparison.py "$E_NORIG" --benchmark $BENCH $EVAL $SEARCH \
      --methods greedy,learned --model ref_norig_dqn_gine=norig \
      --paired-reference greedy --tag arm_obs
  RUN uv run experiments/comparison.py "$E_SELECT" --benchmark $BENCH $EVAL $SEARCH \
      --methods greedy,learned --model ref_select_dqn_gine=select \
      --paired-reference greedy --tag arm_action
  RUN uv run experiments/comparison.py "$E_STOP" --benchmark $BENCH $EVAL $SEARCH \
      --methods greedy,learned --model ref_stop_dqn_gine=stop \
      --paired-reference greedy --tag arm_stop
  # the algorithm x backbone 2x2, on its own kappa
  RUN uv run experiments/comparison.py "$E_K50" --benchmark $BENCH $EVAL $SEARCH \
      --methods greedy,learned --paired-reference greedy --tag arm_2x2 \
      --model estimation_randdom_k50_dqn_gine=dqn_gine,estimation_randdom_k50_dqn_equi=dqn_equi,estimation_randdom_k50_ppo_gine=ppo_gine,estimation_randdom_k50_ppo_equi=ppo_equi

  # 8.6  one run per kappa environment with every policy in each, so greedy
  #      hill-climbs the matching phi and lands on the same axes. The curve itself is
  #      edges against shape error, neither of which depends on which phi scored the row
  for K in 0 10 50 200; do
    case $K in 0) ENVK=$E_K0;; 10) ENVK=$E_K10;; 50) ENVK=$E_K50;; 200) ENVK=$E_K200;; esac
    RUN uv run experiments/comparison.py "$ENVK" --benchmark $BENCH $EVAL $SEARCH \
        --methods greedy,learned --model "$KAPPAS" \
        --paired-reference greedy --tag "tradeoff_k$K"
  done

  # 8.6  the curve itself, read off the four runs above: each policy in its own
  #      environment, greedy at every kappa, both axes objective free
  RUN env PYTHONPATH=. uv run tools/tradeoff.py --kappas 0,10,50,200

  # 8.5  what a budget buys. One rollout per network at the full horizon; the
  #      observation carries no step counter, so truncating it gives exactly what a
  #      shorter run would have produced. Search arms only, no stop action
  RUN uv run experiments/budget.py "$E_K10" --benchmark $BENCH $EVAL $SEARCH \
      --methods greedy,learned --model "$KAPPAS" --tag budget

  # 8.8  one comparison per instance set, then the aggregator. The n = 8 and n = 16
  #      configs are random-domain, so the homogeneous benchmarks read as the
  #      cross-domain arm and the sizes as the cross-size arm. --no-plots because these
  #      exist only to fill results.csv and their own figures are never used
  for B in bench_n8_randdom bench_n8_R3 bench_n8_R3_rot bench_n8_SE3 bench_n8_R3xS1; do
    RUN uv run experiments/comparison.py "$E_N8" --benchmark "$B" $EVAL $SEARCH \
        --methods greedy,learned --model "$REF" --no-plots --tag gen
  done
  for B in bench_n16_randdom bench_n16_R3 bench_n16_SE3; do
    RUN uv run experiments/comparison.py "$E_N16" --benchmark "$B" $EVAL $SEARCH \
        --methods greedy,learned --model "$REF" --no-plots --tag gen
  done
  RUN uv run experiments/comparison.py "$E_N20" --benchmark bench_n20_randdom $EVAL \
      $SEARCH --methods greedy,learned --model "$REF" --no-plots --tag gen
  RUN uv run experiments/comparison.py "$E_K10" --benchmark $BENCH $EVAL \
      $SEARCH --methods greedy,learned --model "$REF" --no-plots --tag gen
  # only the runs tagged gen, so the kappa front and the design arms stay out of it
  RUN uv run experiments/generalisation.py "$E_K10" $SIMPLE --model "$REF" \
      --prior "runs_outputs/*__gen"

  # 8.9  the falsifiable test. Geometric channels must cost nothing at kappa = 0 and
  #      something at kappa = 200, each arm scored in its own environment. No
  #      --episodes here, the driver takes --ablation-episodes instead
  RUN uv run experiments/feature_ablation.py "$E_K0" $SIMPLE \
      --model ref_k0_dqn_gine --mode all --ablation-episodes 20
  RUN uv run experiments/feature_ablation.py "$E_K200" $SIMPLE \
      --model ref_k200_dqn_gine --mode all --ablation-episodes 20

  # 8.12  repair. Six events at k = 2: one breaking edge drop, both membership events,
  #       the redundant drop that only the conditioning term can motivate, the spurious
  #       addition, and the none control that the locality columns are read on
  RUN uv run experiments/repair.py "$E_K10" --benchmark $BENCH $EVAL --restarts 5 \
      --events none,edge_drop_random,edge_drop_redundant,edge_add_spurious,node_leave,node_join \
      --k 2 --cap 200000 --want 400 --drift 0.1 \
      --methods greedy_repair,greedy,optimal_repair,rebuild,learned --model "$KAPPAS"

  # 8.13  control arm (b): one topology per method at matched edge count, on shared goal
  #       and start formations, so the edge set is the only thing that differs
  RUN uv run experiments/control.py "$E_K10" --benchmark $BENCH $SIMPLE --restarts 5 \
      --methods greedy,complete,learned --cases 5 --sigma 0.15 --tol 1e-3 \
      --steps 200000 --dt 1e-3 --sample-every 200 --lin 1.0 --ang 1.0 \
      --model "$KAPPAS"
}

# ---------------------------------------------------------------- no model needed

tools_runs() {
  ENVRUN PYTHONPATH=. uv run tools/cost_scaling.py --ns 6,8,12,16 --domain R^3
  # the kappa values the policies are trained at, plus the small ones where the
  # selection among equally sparse graphs first bites
  ENVRUN PYTHONPATH=. uv run tools/kappa_sweep.py --kappas 0,0.9,2,10,50,200 \
      --n 8 --domain R^3 --instances 12 --samples 3
  for T in crlb_validation spectral_criteria functional_vs_error submodularity \
           shape_error_scale verify_chapter4 reverse_edge laplacian_structure \
           attitude_observability repair_bound repair_choice domain_closed_forms; do
    ENVRUN PYTHONPATH=. uv run "tools/$T.py"
  done
  ENVRUN PYTHONPATH=.:tools uv run tools/kappa_trade.py     # slow
}

# ----------------------------------------------------------------

case "${1:-}" in
  configs)    configs ;;
  benchmarks) benchmarks ;;
  train)      train ;;
  extra)      extra ;;
  eval)       eval_runs ;;
  tools)      tools_runs ;;
  all)        configs; benchmarks; train; eval_runs; tools_runs ;;
  *)          sed -n '2,36p' "$0"; exit 1 ;;
esac
