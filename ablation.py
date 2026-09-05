"""What information does a trained policy actually use?

Perturbs one observation channel at a time and measures how much the policy
notices. Two independent readings, because they answer different questions:

  sensitivity  along an unperturbed reference trajectory, perturb the channel at
               each state and see whether the argmax action changes. "Does the
               decision depend on this channel *here*?"
  outcome      run the whole episode with the channel perturbed and compare the
               graph the policy ends up with. "Does the *result* depend on it?"

A channel can be sensitive but not matter (the policy reacts, but recovers), and
it can matter without being sensitive (a small nudge compounds). Read both.

Every method is scored through the same phi the agent trains on, and every
variant sees the *same* instances (freeze_network), so the columns are paired.

    uv run ablation.py <model_name> [environment_name] [options]

"""
import argparse
import copy
import csv
import os
import sys

import numpy as np
import torch
from skrl.utils.spaces.torch import flatten_tensorized_space, unflatten_tensorized_space

import agent_loader
from environment import OBS_PRESETS


# ---------------------------------------------------------------- channel layout

# Mirrors build_dict_obs()'s concatenation order. Kept as data rather than
# re-derived so a channel can be named; verified against the real observation
# width in resolve_layout(), which raises if environment.py's order moves.
NODE_BLOCKS = [
    ("domain",        5, None),
    ("degree",        2, None),
    ("closeness",     1, "graph_features"),
    ("eigenvector",   1, "graph_features"),
    ("node_between",  1, "graph_features"),
    ("rigidity_glob", 3, "rigidity_global"),
    ("quality", 1, "rigidity_quality"),
    ("node_freedom",  1, "rigidity_flex"),
    ("node_slack",    2, "rigidity_stiffness"),
]

EDGE_BLOCKS = [
    ("bearings",      3, None),
    ("edge_exists",   1, "edge_exists"),
    ("edge_between",  1, "graph_features"),
    ("reciprocity",   1, None),
    ("common_nbrs",   1, None),
    ("add_independence", 1, "rigidity_flex"),
    ("pair_max_rank", 1, "rigidity_edge"),
    ("add_rank",      1, "rigidity_edge"),
    ("add_stiffness", 1, "rigidity_stiffness"),
    ("remove_rank",   1, "rigidity_removal"),
    ("remove_stiffness", 1, "rigidity_removal"),
]


def obs_flags(env):
    """Effective build_dict_obs flags: preset defaults overridden by env flags."""
    obs_type = getattr(env, "obs_space_type", None) or getattr(env, "obs_type", "Dict")
    preset = OBS_PRESETS.get(obs_type, {})
    rig = bool(getattr(env, "last_rigidity", None))
    return {
        "node_set": preset.get("node_set", "graph"),
        "coords": preset.get("coords", True),
        "edges": preset.get("edges", True),
        "selection": preset.get("selection", True),
        "edge_exists": preset.get("edge_exists", True),
        "graph_features": getattr(env, "graph_features", True),
        "rigidity_global": rig and getattr(env, "rigidity_global", False),
        "rigidity_quality": rig and getattr(env, "rigidity_quality", False),
        "rigidity_flex": rig and getattr(env, "rigidity_flex", False),
        "rigidity_edge": rig and getattr(env, "rigidity_edge", False),
        "rigidity_stiffness": rig and getattr(env, "rigidity_stiffness", False),
        "rigidity_removal": rig and getattr(env, "rigidity_removal", False),
    }


def _blocks(spec, flags, width, key):
    """Named sub-channels, or one safe whole-tensor block if the layout disagrees.

    An archived environment can have a different build_dict_obs() than the one
    imported here, and mislabelling a channel would be worse than being coarse --
    the whole point is knowing *which* information the policy uses.
    """
    out, off = [], 0
    for name, w, flag in spec:
        if flag is not None and not flags[flag]:
            continue
        out.append((name, slice(off, off + w)))
        off += w
    if off != width:
        print(f"  note: {key} has {width} channels, this layout expects {off} -- probably an "
              f"archived observation format.\n"
              f"        falling back to one block for all of {key} rather than risk "
              f"mislabelling it.")
        return [(f"{key} (whole)", slice(0, width))]
    return out


def resolve_layout(env, obs_dict):
    """[(name, key, slice)] over every perturbable channel."""
    flags = obs_flags(env)
    layout = []

    nf = obs_dict["node_features"].shape[-1]
    if flags["node_set"] == "graph":
        layout += [(n, "node_features", s)
                   for n, s in _blocks(NODE_BLOCKS, flags, nf, "node_features")]
    else:
        # the legacy presets pack bearings into node_features; not worth naming
        layout.append((f"node_features[{flags['node_set']}]", "node_features", slice(0, nf)))

    if "edge_features" in obs_dict:
        ef = obs_dict["edge_features"].shape[-1]
        layout += [(n, "edge_features", s)
                   for n, s in _blocks(EDGE_BLOCKS, flags, ef, "edge_features")]

    for key in ("coord_features", "adj", "selection"):
        if key in obs_dict:
            w = obs_dict[key].shape[-1]
            layout.append((key, key, slice(0, w)))
    return layout


# ---------------------------------------------------------------- perturbation

def perturb(x, sl, mode, rng):
    """Destroy the information in x[..., sl], in place on a copy.

    shuffle is the honest default: it permutes values across nodes/pairs, so the
    channel keeps its marginal distribution and only its association with a
    particular node or pair is destroyed. zero and noise also change the input
    *scale*, which a network can notice for reasons that have nothing to do with
    the channel's meaning.
    """
    # randomness is drawn on the cpu and moved, so one seed gives the same
    # perturbation whichever device the model is on, and a cpu generator does not
    # have to match a cuda tensor
    x = x.clone()
    block = x[..., sl]
    if mode == "zero":
        block = torch.zeros_like(block)
    elif mode == "noise":
        noise = torch.randn(block.shape, generator=rng).to(block.device, block.dtype)
        block = block + noise * block.std().clamp(min=1e-6)
    elif mode == "shuffle":
        flat = block.reshape(-1, block.shape[-1])
        perm = torch.randperm(flat.shape[0], generator=rng).to(flat.device)
        block = flat[perm].reshape(block.shape)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    x[..., sl] = block
    return x


def ablate_obs(obs, space, key, sl, mode, rng):
    """Returns (flattened obs, whether the perturbation actually changed anything).

    Shuffling a channel that is constant along the shuffled axis is a no-op --
    true of a one-hot domain channel in a homogeneous network, and of the global
    rigidity channels, which are tiled identically across nodes. Without this
    flag such a row reads as a confident 0% and looks like "the policy ignores
    it" when in fact nothing was ablated.
    """
    d = dict(unflatten_tensorized_space(space, obs))
    before = d[key]
    d[key] = perturb(before, sl, mode, rng)
    changed = not torch.equal(before, d[key])
    return flatten_tensorized_space(d), changed


# ---------------------------------------------------------------- rollout

def scores(agent, obs):
    role = "policy" if "policy" in agent.models else "q_network"
    with torch.no_grad():
        s, _ = agent.models[role].compute({"observations": obs}, role=role)
    return s


def rollout(agent, wrapped, raw, steps, space=None, key=None, sl=None,
            mode="shuffle", rng=None, collect=None):
    """One greedy episode, optionally with a channel perturbed at every step.

    collect: if given, a list that receives (obs, reference_scores) per step, for
    the sensitivity pass to reuse the reference trajectory's states.
    """
    obs, _ = wrapped.reset()
    seen, used = set(), 0
    for used in range(1, steps + 1):
        if collect is not None:
            collect.append(obs)
        fed = obs if key is None else ablate_obs(obs, space, key, sl, mode, rng)[0]
        action = torch.argmax(scores(agent, fed), dim=-1, keepdim=True)
        obs, _, terminated, truncated, _ = wrapped.step(action)
        if terminated.any() or truncated.any():
            break
        # An unperturbed argmax policy is a function of the edge set, so a repeated
        # state is an infinite cycle and every later step is wasted. Stopping there
        # changes the reference's score by nothing and gives `used` as the budget
        # the perturbed rollouts are held to.
        if key is None:
            state = raw.network.edges.tobytes()
            if state in seen:
                break
            seen.add(state)

    best = raw.best_stats or {}
    return {
        "phi": float(raw.best_state_score),
        "m": float(best.get("m", raw.network.edges.sum())),
        "rigid": float(bool(best.get("is_IBR", False))),
        "minimal": float(bool(best.get("is_MBR", False))),
    }, used


def sensitivity(agent, states, space, key, sl, mode, rng):
    """Decision flips, score shift, and mask disturbance at fixed reference states.

    A channel the model masks with (adj) can flip the decision without moving any
    finite score, so the two are reported separately rather than averaged together.
    """
    flips, shifts, masked, live = [], [], [], []
    for obs in states:
        ref = scores(agent, obs)
        alt_obs, changed = ablate_obs(obs, space, key, sl, mode, rng)
        live.append(float(changed))
        alt = scores(agent, alt_obs)
        flips.append(float((ref.argmax(-1) != alt.argmax(-1)).float().mean()))
        finite = torch.isfinite(ref) & torch.isfinite(alt)   # masked actions are -inf
        shifts.append(float((ref - alt)[finite].abs().mean()) if finite.any() else 0.0)
        masked.append(float((torch.isfinite(ref) != torch.isfinite(alt)).any()))
    return (float(np.mean(flips)), float(np.mean(shifts)),
            float(np.mean(masked)), float(np.mean(live)))


# ---------------------------------------------------------------- reporting

# Terminal table and CSV are both rendered from order_rows() + table_rows() +
# legend(), so the file a result gets read from cannot disagree with the console
# it was watched on.

NOT_ABLATED = "constant along the shuffled axis, not ablated"

COLUMNS = ["rank", "channel", "status", "feeds_action_mask",
           "flip_pct", "abs_dscore", "d_phi", "d_edges", "d_rigid_pct", "d_minimal_pct"]


def order_rows(rows):
    """Most-depended-upon first; flip% breaks ties among the many exact zeros."""
    return sorted(rows, key=lambda r: (-abs(r["d_phi"]), -r["flip"]))


def table_rows(rows, ref):
    """One dict per printed line, values already rounded as displayed.

    Cells are left empty rather than zero for a channel that was never actually
    perturbed: a 0.0 there would be averaged and plotted as evidence of
    independence, which is the one thing it is not.
    """
    out = []
    for i, r in enumerate(order_rows(rows), start=1):
        live = r["perturbed"] >= 0.01
        out.append({
            "rank": i if live else "",
            "channel": r["channel"],
            "status": "ok" if live else "not_ablated",
            "feeds_action_mask": "yes" if r["mask_changed"] > 0.01 else "no",
            "flip_pct": round(100 * r["flip"], 1) if live else "",
            "abs_dscore": round(r["shift"], 4) if live else "",
            "d_phi": round(r["d_phi"], 2) if live else "",
            "d_edges": round(r["d_m"], 2) if live else "",
            "d_rigid_pct": round(100 * r["d_rigid"], 1) if live else "",
            "d_minimal_pct": round(100 * r["d_minimal"], 1) if live else "",
        })
    out.append({
        "rank": "", "channel": "(reference)", "status": "reference",
        "feeds_action_mask": "", "flip_pct": "", "abs_dscore": "",
        "d_phi": round(ref["phi"], 2), "d_edges": round(ref["m"], 2),
        "d_rigid_pct": round(100 * ref["rigid"], 1),
        "d_minimal_pct": round(100 * ref["minimal"], 1),
    })
    return out


def legend(rows, args, meta):
    lines = [
        "how to read this",
        f"  {meta}",
        f"  Rows are the observation channels. Each is destroyed one at a time by",
        f"  '{args.mode}'; everything else is left alone. {args.episodes} episodes, greedy actions,",
        "  every variant on the same instances.",
        "",
        "  flip%     how often destroying the channel changes the action the policy picks.",
        "            0% means the policy is not reading it at all at these states.",
        "  |dscore|  mean change in the policy's own logits / Q-values. Magnitude of the",
        "            reaction, where flip% is whether the reaction changed the decision.",
        "  d phi     reference phi MINUS ablated phi, so POSITIVE = the policy got worse",
        "            without the channel = it depends on it.",
        "  d rigid% / d minimal%",
        "            same convention: POSITIVE = the ablated policy scored lower.",
        "  d edges   also reference minus ablated, but FEWER edges is better, so here",
        "            NEGATIVE is what the ablation cost. Opposite sense to the rest.",
        "",
        "  The (reference) row is the unablated policy, and is ABSOLUTE, not a delta:",
        "  it is the phi / edges / rigid% / minimal% every other row is measured from.",
        "",
        "  A channel near zero on every column is one the policy has learned to ignore.",
        "  Sensitive but no outcome cost means it reacts and recovers.",
    ]
    if any(r["perturbed"] < 0.01 for r in rows):
        lines += [
            "",
            "  Rows marked 'not ablated' are constant along the axis being shuffled -- a",
            "  one-hot domain channel in a homogeneous network, or the global rigidity",
            "  channels, which are tiled identically across nodes. Shuffling them changes",
            "  nothing, so they carry NO evidence either way, and their cells are left",
            "  empty rather than zero. Use --mode noise or --mode zero to ablate those.",
        ]
    if any(r["mask_changed"] > 0.01 for r in rows):
        lines += [
            "",
            "  Channels marked * (feeds_action_mask) feed the model's action mask, so",
            "  perturbing one changes which actions are legal, not just what the policy",
            "  knows. Its flip% is not comparable to the others, and |dscore| understates",
            "  it (a score moving to -inf is skipped). Read it as a control: it should",
            "  flip a lot.",
        ]
    return lines


def render_table(printed, out=sys.stdout):
    """The table, from table_rows() output alone -- no access to the raw run.

    Split out so a csv written by an earlier version can be re-rendered through
    exactly this code rather than a lookalike.
    """
    w = max(len(r["channel"]) for r in printed) + 2
    p = lambda *a: print(*a, file=out)

    p(f"\n{'':{w}}{'sensitivity':>21}   {'outcome (ablated whole episode)':>44}")
    p(f"{'channel':<{w}}{'flip%':>8}{'|dscore|':>13}   "
      f"{'d phi':>9}{'d edges':>9}{'d rigid%':>10}{'d minimal%':>12}")
    p("-" * (w + 21 + 3 + 40))

    for r in printed:
        name = r["channel"] + ("*" if r["feeds_action_mask"] == "yes" else "")
        if r["status"] == "not_ablated":
            p(f"{name:<{w}}--  {NOT_ABLATED}")
            continue
        if r["status"] == "reference":
            p("-" * (w + 21 + 3 + 40))
            p(f"{'(reference)':<{w}}{'':>8}{'':>13}   "
              f"{r['d_phi']:>9.2f}{r['d_edges']:>9.2f}{r['d_rigid_pct']:>9.1f}%"
              f"{r['d_minimal_pct']:>11.1f}%")
            continue
        p(f"{name:<{w}}{r['flip_pct']:>7.1f}%{r['abs_dscore']:>13.4f}   "
          f"{r['d_phi']:>+9.2f}{r['d_edges']:>+9.2f}{r['d_rigid_pct']:>+9.1f}%"
          f"{r['d_minimal_pct']:>+11.1f}%")


def report(rows, ref, args, meta, out=sys.stdout):
    render_table(table_rows(rows, ref), out)
    print("", file=out)
    for line in legend(rows, args, meta):
        print(line, file=out)


def write_csv(path, rows, ref, args, meta):
    """Data to <path>, the readable report to <path>.txt beside it.

    Same split outputs.py uses (results.csv next to summary.txt), for the same
    reason: a legend commented into the head of the csv means every spreadsheet,
    csv viewer and `column -s,` shows 27 lines of noise before the table. The csv
    is a plain rectangle with the header on line 1; the prose lives next door,
    where it can be read without a parser.

    Same ranking and the same reference row in both, from table_rows().
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS)
        wr.writeheader()
        wr.writerows(table_rows(rows, ref))

    notes = os.path.splitext(path)[0] + ".txt"
    with open(notes, "w") as f:
        report(rows, ref, args, meta, out=f)
    return path, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("environment", nargs="?", default=None,
                    help="defaults to the environment recorded in the manifest")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=None, help="cap per episode")
    ap.add_argument("--mode", choices=["shuffle", "zero", "noise"], default="shuffle")
    ap.add_argument("--channels", default=None, help="comma-separated subset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--live-env", action="store_true",
                    help="score against the current environment instead of the archived one. "
                         "The archive reproduces what the run was trained on, which is right for "
                         "the policy but keeps environment-side measurement fixes out.")
    ap.add_argument("--csv", default=None,
                    help="write the table here; the legend goes to a .txt beside it")
    args = ap.parse_args()

    agent, wrapped, raw, info = agent_loader.load_run(
        args.model, args.environment, device=args.device,
        prefer_archived_env=not args.live_env)
    agent.enable_models_training_mode(False)

    steps = args.steps or int(min(getattr(raw, "max_steps", 100), 100))
    space = wrapped.observation_space
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw.freeze_network = False
    obs, _ = wrapped.reset()
    layout = resolve_layout(raw, unflatten_tensorized_space(space, obs))
    if args.channels:
        want = {c.strip() for c in args.channels.split(",")}
        layout = [c for c in layout if c[0] in want]
        if not layout:
            sys.exit(f"no channels matched; available: {[c[0] for c in layout]}")

    meta = (f"model {args.model}  |  env={args.environment}"
            f"  |  n={raw.network.n} {raw.network.agents[0].domain}"
            f"  |  {raw.action_space_type}  |  {len(layout)} channels"
            f"  |  mode={args.mode}  |  {args.episodes} episodes x {steps} steps"
            f"  |  seed {args.seed}  |  {'live' if args.live_env else 'archived'} env")
    print(f"\n{meta}")

    rows, ref_row, converged = measure(
        agent, wrapped, raw, layout, space, episodes=args.episodes, steps=steps,
        mode=args.mode, rng=rng)

    print(" " * 40, end="\r")
    meta += (f"  |  stopped at the reference's convergence, "
             f"median {int(np.median(converged))} of {steps} steps")
    report(rows, ref_row, args, meta)

    if args.csv:
        data, notes = write_csv(args.csv, rows, ref_row, args, meta)
        print(f"\nwrote {data}\n      {notes}  (the table and legend above)")


def measure(agent, wrapped, raw, layout, space, *, episodes, steps, mode, rng,
            progress=True):
    """Destroy each channel in turn and measure what it cost. (rows, ref_row, converged).

    The whole loop, so outputs.py can run it against an agent and an environment it has
    already built rather than loading a second copy of both.
    """
    acc = {name: [] for name, _, _ in layout}
    sens = {name: [] for name, _, _ in layout}
    ref_acc = []
    converged = []

    for ep in range(episodes):
        raw.freeze_network = False
        wrapped.reset()                 # draw a fresh instance
        instance = copy.deepcopy(raw.network)
        raw.freeze_network = True       # then hold it for every variant

        # every rollout edits the graph, so it has to be put back or variant k
        # would start from wherever variant k-1 finished
        def restore():
            raw.network = copy.deepcopy(instance)
            wrapped.reset()

        restore()
        states = []
        # `used` is where the reference converged; every variant gets the same
        # budget, or ablating a channel would buy extra exploration for free
        ref, used = rollout(agent, wrapped, raw, steps, collect=states)
        ref_acc.append(ref)
        converged.append(used)

        for name, key, sl in layout:
            sens[name].append(sensitivity(agent, states, space, key, sl, mode, rng))
            restore()
            abl, _ = rollout(agent, wrapped, raw, used, space, key, sl, mode, rng)
            acc[name].append({k: ref[k] - abl[k] for k in ref})

        if progress:
            print(f"  episode {ep + 1}/{episodes}", end="\r", flush=True)

    raw.freeze_network = False
    mean = lambda rs, k: float(np.mean([r[k] for r in rs]))
    ref_row = {k: mean(ref_acc, k) for k in ref_acc[0]}
    rows = [{
        "channel": name,
        "flip": float(np.mean([s[0] for s in sens[name]])),
        "shift": float(np.mean([s[1] for s in sens[name]])),
        "mask_changed": float(np.mean([s[2] for s in sens[name]])),
        "perturbed": float(np.mean([s[3] for s in sens[name]])),
        "d_phi": mean(acc[name], "phi"),
        "d_m": mean(acc[name], "m"),
        "d_rigid": mean(acc[name], "rigid"),
        "d_minimal": mean(acc[name], "minimal"),
    } for name, _, _ in layout]
    return rows, ref_row, converged


if __name__ == "__main__":
    main()
