"""Run a command with the BLAS/torch thread pools capped and, optionally, pinned
to a subset of the CPUs.

The training scripts drive a `SyncVectorEnv`, so they are one Python thread; the
only thing that spreads them over the machine is BLAS and torch fanning out over
matrices too small to pay for the synchronisation. Capping the pools frees the
machine and is usually also faster -- see the note at the top of `flag_cost.py`.

Pinning is the separate question of leaving cores free for interactive work. On a
hybrid CPU `--cores efficiency` keeps the fast cores for the browser at the cost
of the slower clock; `--cores reserve:4` just keeps four logical CPUs free.

    PYTHONPATH=. uv run tools/cpu_budget.py [--threads N] [--cores SPEC]
                                            [--nice N] [--dry-run] -- <command...>

    # what this exists for
    uv run tools/cpu_budget.py -- uv run train_dqn.py <environment_name> <model_name>
    uv run tools/cpu_budget.py --cores efficiency -- uv run train_dqn.py <env> <model>
"""
import argparse
import os
import sys

# Every thread-pool variable that numpy, torch and their BLAS backends read.
# torch has no env var of its own for intra-op threads; it follows OMP_NUM_THREADS.
THREAD_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]


def cpu_max_freq(cpu):
    """Peak kHz of one logical CPU, or None where cpufreq does not report it."""
    path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"
    try:
        with open(path) as f:
            return int(f.read().strip())
    except OSError:
        return None


def core_tiers():
    """(performance, efficiency) CPU lists, split by peak clock.

    A hybrid CPU reports two or more distinct peak frequencies; the lowest tier is
    the efficiency cores. On a uniform CPU both lists are every CPU, so the
    keywords stay meaningful (they just pin nothing useful).
    """
    cpus = sorted(os.sched_getaffinity(0))
    freqs = {c: cpu_max_freq(c) for c in cpus}
    if any(f is None for f in freqs.values()) or len(set(freqs.values())) < 2:
        return cpus, cpus
    slowest = min(freqs.values())
    efficiency = [c for c in cpus if freqs[c] == slowest]
    performance = [c for c in cpus if freqs[c] != slowest]
    return performance, efficiency


def parse_cores(spec):
    """A --cores value to an explicit CPU list, or None for no pinning."""
    if spec in (None, "all"):
        return None
    performance, efficiency = core_tiers()
    if spec == "efficiency":
        return efficiency
    if spec == "performance":
        return performance
    if spec.startswith("reserve:"):
        k = int(spec.split(":", 1)[1])
        cpus = sorted(os.sched_getaffinity(0))
        if not 0 <= k < len(cpus):
            raise SystemExit(f"cannot reserve {k} of {len(cpus)} CPUs")
        # Hand back the fast cores first: the reserved ones are for interactive work.
        given_up = set((performance + efficiency)[:k])
        return [c for c in cpus if c not in given_up]
    # Otherwise a taskset-style list: "0-7", "12-19", "0,2,4-6".
    cpus = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def compact(cpus):
    """[12,13,14,19] -> '12-14,19', for printing."""
    if not cpus:
        return "none"
    runs, start, prev = [], cpus[0], cpus[0]
    for c in cpus[1:] + [None]:
        if c != prev + 1:
            runs.append(str(start) if start == prev else f"{start}-{prev}")
            start = c
        prev = c
    return ",".join(runs)


def main():
    parser = argparse.ArgumentParser(
        description="run a command with capped thread pools and optional CPU pinning")
    parser.add_argument("--threads", type=int, default=1,
                        help="value for every BLAS/torch thread-pool variable (default 1)")
    parser.add_argument("--cores", default="all",
                        help="all (default), efficiency, performance, reserve:K, "
                             "or a taskset-style list such as 12-19")
    parser.add_argument("--nice", type=int, default=10,
                        help="niceness increment, 0 to leave it alone (default 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be set and exit without running")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="the command to run, after a --")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command and not args.dry_run:
        parser.error("no command given (put it after a --)")

    if args.threads < 1:
        parser.error("--threads must be at least 1")

    cpus = parse_cores(args.cores)

    env = dict(os.environ)
    for var in THREAD_VARS:
        env[var] = str(args.threads)

    print(f"threads {args.threads}  "
          f"cores {compact(cpus) if cpus else 'all'}  "
          f"nice +{args.nice}", file=sys.stderr)

    if args.dry_run:
        for var in THREAD_VARS:
            print(f"  {var}={env[var]}", file=sys.stderr)
        print(f"  command: {' '.join(command) if command else '(none)'}", file=sys.stderr)
        return

    if cpus is not None:
        os.sched_setaffinity(0, set(cpus))
    if args.nice:
        os.nice(args.nice)

    # exec rather than spawn, so signals and the exit code reach the caller unchanged.
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
