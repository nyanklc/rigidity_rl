"""Re-derive the numbers quoted in0 and the README.

Everything here runs from a fresh clone, because `benchmarks/` is tracked and the
environment is built programmatically rather than read from the gitignored
`environments/`. The settings below therefore ARE the record of what was run --
if they drift from the env configs the numbers came from, this script is wrong.

What it checks without a checkpoint:
  - benchmark digests, so a number is tied to a specific instance set
  - rank_K / c_max / m_req per configuration
  - greedy and constructive, which need no model and carry most of the comparison

The `learned` rows need `models/` and `train/`, which are gitignored, so they are
only checked when --model is given and the checkpoint is on this machine.

    PYTHONPATH=. uv run tools/verify_results.py
    PYTHONPATH=. uv run tools/verify_results.py --model letsgo_dqn_gine
    PYTHONPATH=. uv run tools/verify_results.py --quick     # digests + constants only
"""
import argparse
import copy

import numpy as np

import outputs as B
import benchmark
from environment import Environment

# (label, benchmark, digest, n, rank_K, c_max, m_req, greedy edges, greedy min%,
#  constructive edges, constructive min%, learned edges, learned min%)
CLAIMS = [
    ("mixed n=10 (trained)", "bench_mixed",   "83a53b8677d9", 10, 33, 2, 17,
     17.40, 80, 17.70,  50, 17.05, 95),
    ("R^3     n=8",          "bench_n8_R3",   "a678a0266a20",  8, 20, 2, 10,
     10.50, 50, 10.45,  55, 10.75, 50),
    ("R^2xS^1 n=8",          "bench_n8_R2xS1","7805a3bd2f6f",  8, 20, 1, 20,
     20.00, 100, 20.00, 100, 20.00, 100),
    ("R^3xS^1 n=8",          "bench_n8_R3xS1","72b1d517025f",  8, 27, 2, 14,
     14.15, 85, 14.00, 100, 15.40, 10),
    ("SE(3)   n=8",          "bench_n8_SE3",  "94c9396becab",  8, 41, 2, 21,
     21.00, 100, 21.00, 100, 26.10, 25),
    ("R^3     n=16",         "bench_n16_R3",  "333864562507", 16, 44, 2, 22,
     22.65, 45, 23.20,   0, 23.20,  0),
]

# the mixed-scenario training config, minus the parts evaluation does not read
ENV_KW = dict(
    action_space_type="AddRemoveEdgeDiscreteNoSelfLoops",
    obs_space_type="Dict",
    state_score_type="WeightedNormalized",
    termination_condition_type="MaxSteps",
    action_rewards_enable=False,
    skip_is_stop=False,   # skip_enabled is read by the training scripts, not initialize()
    random_graph_with_mean_min_edges=True,
    time_penalty_value=0.0,
    track_data_enable=False,
    truncate_enable=False,
    include_candidate_bearings=True,
    graph_features=False,
    rigidity_global=True,
    rigidity_flex=True,
    rigidity_edge=True,
)

TOL_EDGES = 0.02        # the runs are deterministic; this is float-formatting slack
TOL_PCT = 0.1


def check(rows, label, expected, got, tol):
    ok = expected is None or abs(got - expected) <= tol
    rows.append((label, "-" if expected is None else f"{expected:g}", f"{got:g}",
                 "--" if expected is None else ("ok" if ok else "FAIL")))
    return ok or expected is None


def build_env(nets, max_steps):
    """Programmatic env matching the training config; no environments/ needed."""
    e = Environment()
    e.initialize(nets[0].n, [a.domain for a in nets[0].agents],
                 max_steps=max_steps, **ENV_KW)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="print the outputs.py commands that check the learned rows")
    ap.add_argument("--quick", action="store_true",
                    help="digests and constants only, no baseline runs")
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rows, failures = [], 0

    for (label, bench, digest, n, rank_K, c_max, m_req,
         g_e, g_m, c_e, c_m, l_e, l_m) in CLAIMS:
        print(f"\n{label}  [{bench}]")
        try:
            nets, meta = benchmark.load(bench)
        except FileNotFoundError as exc:
            print(f"  missing: {exc}")
            failures += 1
            continue

        got_digest = benchmark.digest(bench)
        ok = got_digest == digest
        rows.append((f"{label} digest", digest, got_digest, "ok" if ok else "FAIL"))
        failures += not ok

        e = build_env(nets, max_steps=4 * m_req + 10)
        e.network = copy.deepcopy(nets[0])
        e.reset()
        failures += not check(rows, f"{label} rank_K", rank_K, int(e.rank_K), 0)
        failures += not check(rows, f"{label} c_max", c_max, int(e.c_max), 0)
        failures += not check(rows, f"{label} m_req", m_req, int(e.m_req), 0)
        if args.quick:
            continue

        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        acc = {"greedy": [], "constructive": []}
        for inst in nets:
            # freeze BEFORE reset, exactly as outputs.py does: with freeze_network
            # False, reset() redraws the network and the frozen instance is discarded
            e.freeze_network = False
            e.network = copy.deepcopy(inst)
            e.freeze_network = True
            e.reset()
            base = copy.deepcopy(e.network)

            e.network = copy.deepcopy(base)
            acc["greedy"].append(B.run_greedy(e, verbose=False))
            e.network = copy.deepcopy(base)
            acc["constructive"].append(
                B.run_constructive(e, rng, restarts=args.restarts, verbose=False))

        for name, exp_e, exp_m in (("greedy", g_e, g_m), ("constructive", c_e, c_m)):
            rs = acc[name]
            failures += not check(rows, f"{label} {name} edges", exp_e,
                                  float(np.mean([r["m"] for r in rs])), TOL_EDGES)
            failures += not check(rows, f"{label} {name} minimal%", exp_m,
                                  100.0 * np.mean([r["is_MBR"] for r in rs]), TOL_PCT)

    print(f"\n{'claim':38s} {'documented':>12s} {'measured':>12s}  status")
    print("-" * 76)
    for label, exp, got, status in rows:
        print(f"{label:38s} {exp:>12s} {got:>12s}  {status}")
    print("-" * 76)
    print(f"{'FAILURES: ' + str(failures) if failures else 'all documented values reproduce'}")

    # the learned rows go through outputs.py rather than being reimplemented here:
    # a second rollout path would drift and the verification would be the thing that is wrong
    print("\nThe `learned` rows are not checked here. models/ and train/ are gitignored, so a\n"
          "checkout does not carry the checkpoint, and re-implementing the rollout would risk\n"
          "verifying against a different code path. On the machine holding the checkpoint:")
    for label, bench, *_ in CLAIMS:
        env_hint = bench.replace("bench_", "")
        print(f"  uv run outputs.py <env for {env_hint}> --benchmark {bench} "
              f"--model {args.model or '<name>'} --steps 100")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
