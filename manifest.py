"""Training manifests: a self-contained record of everything that determined a run.

A checkpoint is only meaningful together with the code that produced it. `train/<name>.json`
therefore archives the full text of every file that affects a trained model -- the environment,
the rigidity maths, the graph features, and the policy backbone -- plus the model class sources,
the environment config, the scenario, and the versions/seed the run used.

The sources are gzipped and base64'd into one field to keep manifests around 10 KB instead of
100 KB. Use this module's CLI to read them back:

    uv run manifest.py list                 what every run carries, and whether it still verifies
    uv run manifest.py show <name> [file]   print archived source
    uv run manifest.py diff <name> [file]   archived vs the working tree
    uv run manifest.py verify <name>        rebuild the model from the archive, check the weights
    uv run manifest.py backfill [--write]   add what is missing to older manifests
"""

import base64
import difflib
import glob
import gzip
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

# Order matters: this is a valid import order, so replaying them one by one into sys.modules
# lets each file's `from x import y` resolve to the archived x rather than the installed one.
ARCHIVED_FILES = [
    "util.py",
    "rigidity.py",
    "network.py",
    "scenario.py",
    "environment.py",
    "policy/gnn_backbone.py",
]

MANIFEST_VERSION = 2


def manifest_path(model_name):
    return os.path.join("train", f"{model_name}.json")


def load(model_name):
    with open(manifest_path(model_name), "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------------------
def collect_sources(root="."):
    out = {}
    for rel in ARCHIVED_FILES:
        path = os.path.join(root, rel)
        if os.path.exists(path):
            with open(path, "r") as f:
                out[rel] = f.read()
    return out


def encode_sources(sources):
    raw = json.dumps(sources).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")


def decode_sources(blob):
    if not blob:
        return {}
    return json.loads(gzip.decompress(base64.b64decode(blob)).decode("utf-8"))


def sources_of(train_info):
    """Archived sources, tolerating the older `backbone_source` key."""
    sources = decode_sources(train_info.get("sources_b64gz"))
    if not sources and train_info.get("backbone_source"):
        sources = {"policy/gnn_backbone.py": "\n".join(train_info["backbone_source"])}
    return sources


def current_sources_differ(sources, root="."):
    """{path: (added, removed)} for archived files that differ from the working tree."""
    drift = {}
    for rel, archived in sources.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            drift[rel] = (0, len(archived.splitlines()))
            continue
        with open(path, "r") as f:
            current = f.read()
        if current == archived:
            continue
        diff = list(difflib.unified_diff(archived.splitlines(), current.splitlines(), n=0))
        added = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
        removed = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
        drift[rel] = (added, removed)
    return drift


def describe_drift(drift):
    return ", ".join(f"{rel} (+{a} -{r})" for rel, (a, r) in sorted(drift.items()))


# ---------------------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------------------
def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:
        return ""


def _versions():
    out = {"python": platform.python_version()}
    for name in ("torch", "skrl", "numpy", "torch_geometric", "gymnasium", "egnn_pytorch"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = "unknown"
    return out


def collect_provenance(seed=None, device=None, captured_at_training=True):
    return {
        "captured_at_training": captured_at_training,
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "device": device,
        "seed": seed,
        "packages": _versions(),
    }


# ---------------------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------------------
def scenario_raw(env_config):
    """Contents of scenarios/<name>.json -- gitignored, so it must live in the manifest."""
    name = env_config.get("scenario")
    if not name:
        return None
    path = os.path.join("scenarios", f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def build_manifest(base, env_config, seed=None, device=None):
    """`base` is the algorithm-specific dict the training script already assembles."""
    manifest = dict(base)
    manifest["manifest_version"] = MANIFEST_VERSION
    manifest["sources_b64gz"] = encode_sources(collect_sources())
    manifest["scenario_raw"] = scenario_raw(env_config)
    manifest["provenance"] = collect_provenance(seed=seed, device=device)
    return manifest


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def _names():
    return sorted(os.path.basename(p)[:-5] for p in glob.glob("train/*.json"))


def cmd_list(_args):
    print(f"{'run':<62}{'ver':>4}{'sources':>9}{'scen':>6}{'prov':>6}  status")
    print("-" * 104)
    for name in _names():
        try:
            info = load(name)
        except Exception as e:
            print(f"{name[:60]:<62}{'?':>4}{'':>9}{'':>6}{'':>6}  unreadable: {e}")
            continue
        srcs = sources_of(info)
        status = info.get("reconstructible")
        if status is False:
            note = f"NOT reconstructible: {info.get('reconstructible_reason', '')}"
        elif srcs:
            drift = current_sources_differ(srcs)
            note = f"drift: {describe_drift(drift)}" if drift else "matches working tree"
        else:
            note = "no archived sources"
        print(f"{name[:60]:<62}{info.get('manifest_version', 1):>4}"
              f"{len(srcs):>9}{'yes' if info.get('scenario_raw') else '-':>6}"
              f"{'yes' if info.get('provenance') else '-':>6}  {note[:60]}")


def cmd_show(args):
    info = load(args.name)
    sources = sources_of(info)
    if not sources:
        print(f"{args.name} has no archived sources")
        return 1
    if not args.file:
        print(f"archived in {args.name}:")
        for rel, text in sorted(sources.items()):
            print(f"  {rel:<28} {len(text.splitlines()):>5} lines")
        prov = info.get("provenance")
        if prov:
            print("\nprovenance:")
            for k in ("captured_at_training", "timestamp", "command", "git_commit",
                      "git_dirty", "device", "seed"):
                if k in prov:
                    print(f"  {k:<22} {prov[k]}")
            print(f"  packages               {prov.get('packages')}")
        return 0
    match = [r for r in sources if r == args.file or r.endswith("/" + args.file)]
    if not match:
        print(f"{args.file} not archived; have: {sorted(sources)}")
        return 1
    print(sources[match[0]])
    return 0


def cmd_diff(args):
    sources = sources_of(load(args.name))
    if not sources:
        print(f"{args.name} has no archived sources")
        return 1
    targets = sorted(sources) if not args.file else \
        [r for r in sources if r == args.file or r.endswith("/" + args.file)]
    any_diff = False
    for rel in targets:
        path = rel
        current = open(path).read() if os.path.exists(path) else ""
        if current == sources[rel]:
            continue
        any_diff = True
        for line in difflib.unified_diff(
            sources[rel].splitlines(), current.splitlines(),
            fromfile=f"archived/{rel}", tofile=f"working/{rel}", lineterm="",
        ):
            print(line)
    if not any_diff:
        print("archived sources are identical to the working tree")
    return 0


def cmd_verify(args):
    from agent_loader import verify_manifest
    ok, detail = verify_manifest(args.name)
    print(f"{'OK  ' if ok else 'FAIL'}  {args.name}: {detail}")
    return 0 if ok else 1


def cmd_backfill(args):
    from agent_loader import verify_manifest

    print(f"{'run':<62}  action")
    print("-" * 100)
    changed = 0
    for name in _names():
        try:
            info = load(name)
        except Exception as e:
            print(f"{name[:60]:<62}  skip (unreadable: {e})")
            continue

        missing = []
        # older manifests recorded only the env config's *name*; recover the contents from
        # environments/ while that file still exists, otherwise the run is unreplayable
        if not info.get("environment_config_raw") and info.get("environment_config"):
            env_path = os.path.join("environments", f"{info['environment_config']}.json")
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    info["environment_config_raw"] = json.load(f)
                missing.append("environment_config_raw")
        if not info.get("sources_b64gz"):
            missing.append("sources")
        if info.get("scenario_raw") is None and (
            info.get("environment_config_raw", {}).get("scenario")
        ):
            missing.append("scenario_raw")
        if not info.get("provenance"):
            missing.append("provenance")
        if not missing:
            print(f"{name[:60]:<62}  already complete")
            continue

        # only archive today's sources for a run whose weights they can actually rebuild
        ok, detail = verify_manifest(name, train_info=info)
        if not ok:
            info["reconstructible"] = False
            info["reconstructible_reason"] = detail
            print(f"{name[:60]:<62}  NOT reconstructible ({detail[:40]}) -> marked only")
        else:
            info["manifest_version"] = MANIFEST_VERSION
            info["sources_b64gz"] = encode_sources(collect_sources())
            info["scenario_raw"] = scenario_raw(info.get("environment_config_raw", {}))
            prov = collect_provenance(captured_at_training=False)
            prov["note"] = ("backfilled from the working tree; these sources rebuild the "
                            "checkpoint's parameters but were not captured when it was trained")
            info["provenance"] = prov
            info["reconstructible"] = True
            print(f"{name[:60]:<62}  backfilled {'+'.join(missing)}")

        changed += 1
        if args.write:
            path = manifest_path(name)
            if not os.path.exists(path + ".bak"):
                with open(path + ".bak", "w") as f:
                    json.dump(load(name), f, indent=4)
            with open(path, "w") as f:
                json.dump(info, f, indent=4)

    print(f"\n{changed} manifest(s) {'updated' if args.write else 'would change'}"
          f"{'' if args.write else ' -- rerun with --write to apply'}")
    return 0


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(fn=cmd_list)

    s = sub.add_parser("show"); s.add_argument("name"); s.add_argument("file", nargs="?")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("diff"); s.add_argument("name"); s.add_argument("file", nargs="?")
    s.set_defaults(fn=cmd_diff)

    s = sub.add_parser("verify"); s.add_argument("name"); s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("backfill"); s.add_argument("--write", action="store_true")
    s.set_defaults(fn=cmd_backfill)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
