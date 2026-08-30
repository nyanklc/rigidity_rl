"""Compare training runs on the metrics that separate a policy from a search.

Reads runs/<name>/ directly, so it works while a run is still going. Reports the
tail average of each metric plus its trajectory in fifths, which is what shows a
collapse: a metric that starts fine and decays looks identical to a healthy one
in the tail alone.

A caution the stop-action comparison ran into: training episodes carry epsilon
exploration, so these curves understate any arm with short episodes. One random
action costs proportionally more over 8 steps than over 50, and it can be the
stop action. Judge terminations on an argmax evaluation (evaluation.py with a
frozen benchmark), not on these curves.

    PYTHONPATH=. uv run tools/compare_runs.py runA runB runC
    PYTHONPATH=. uv run tools/compare_runs.py --list
    PYTHONPATH=. uv run tools/compare_runs.py runA runB --metrics Length,Decision
"""
import argparse
import glob
import os

import numpy as np

DEFAULT = [
    "Episode/ Best is min rigid", "Episode/ Best is rigid", "Episode/ Best nr edges",
    "Episode/ Final is min rigid", "Episode/ Final nr edges",
    "Episode/ Best-final score gap", "Episode/ Length", "Episode/ Terminated",
    "Episode/ Edit efficiency", "Decision/ useful", "Decision/ wasted",
    "Actions/ skip fraction", "Probe/ argmax minimal", "Probe/ argmax-sample gap",
    "Probe/ useful (argmax)", "Probe/ max abs logit", "Loss / Q-network loss",
]


def load(run):
    """Scalars from the NEWEST run in runs/<run>, not all of them merged.

    Training twice under one name leaves both sets of event files in the same
    directory, and pointing EventAccumulator at the directory silently splices
    them into one series.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = glob.glob(f"runs/{run}/events.out.tfevents.*")
    # events.out.tfevents.<starttime>.<host>.<pid>.<n>. One training process writes
    # several of these, sometimes a second apart, so the pid is what identifies a run.
    groups = {}
    for f in files:
        parts = os.path.basename(f).split(".")
        groups.setdefault(parts[5] if len(parts) > 5 else f, []).append(f)
    if not groups:
        return {}, set()
    if len(groups) > 1:
        print(f"  note: runs/{run} holds {len(groups)} runs; reading the newest only")
    newest = groups[max(groups, key=lambda k: max(os.path.getmtime(f) for f in groups[k]))]

    out, tags = {}, set()
    for f in newest:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for t in ea.Tags()["scalars"]:
            out.setdefault(t, ea.Scalars(t))
            tags.add(t)
    return out, tags


def tail(series, frac=0.15):
    if not series:
        return float("nan")
    k = max(1, int(len(series) * frac))
    return float(np.mean([x.value for x in series[-k:]]))


def fifths(series):
    if not series:
        return []
    xs = [x.value for x in series]
    k = max(1, len(xs) // 5)
    return [float(np.mean(xs[i * k:(i + 1) * k])) for i in range(5)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--metrics", default=None,
                    help="comma-separated substrings; default is a curated set")
    ap.add_argument("--trajectory", action="store_true",
                    help="also print each metric in fifths of training")
    args = ap.parse_args()

    if args.list or not args.runs:
        names = sorted(os.path.basename(p) for p in glob.glob("runs/*") if os.path.isdir(p))
        print(f"{len(names)} runs under runs/:")
        for n in names:
            print("   ", n)
        raise SystemExit(0)

    data, tagsets = {}, {}
    for r in args.runs:
        if not os.path.isdir(f"runs/{r}"):
            print(f"  runs/{r} does not exist")
            raise SystemExit(1)
        data[r], tagsets[r] = load(r)

    if args.metrics:
        pats = args.metrics.split(",")
        metrics = sorted({t for ts in tagsets.values() for t in ts
                          if any(p.lower() in t.lower() for p in pats)})
    else:
        metrics = [m for m in DEFAULT if any(m in ts for ts in tagsets.values())]

    steps = {r: max((s[-1].step for s in data[r].values() if s), default=0) for r in args.runs}
    print("tail average over the last 15% of each run\n")
    w = max(len(r) for r in args.runs) + 2
    print(f"  {'metric':34s}" + "".join(f"{r:>{w}s}" for r in args.runs))
    print(f"  {'steps':34s}" + "".join(f"{steps[r]:>{w}d}" for r in args.runs))
    print("  " + "-" * (34 + w * len(args.runs)))
    for m in metrics:
        row = f"  {m:34s}"
        for r in args.runs:
            v = tail(data[r].get(m, []))
            row += f"{'-' if np.isnan(v) else f'{v:.3f}':>{w}s}"
        print(row)

    if args.trajectory:
        print("\nover training, mean in each fifth\n")
        for m in metrics:
            print(f"  {m}")
            for r in args.runs:
                f = fifths(data[r].get(m, []))
                print(f"    {r:{w}s} " + (" ".join(f"{v:7.3f}" for v in f) if f else "(absent)"))

    missing = {r: [m for m in metrics if m not in tagsets[r]] for r in args.runs}
    if any(missing.values()):
        print("\nmetrics absent from a run (older code, or the probe was off):")
        for r, ms in missing.items():
            if ms:
                print(f"  {r}: {', '.join(ms)}")
