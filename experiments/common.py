"""Setup every experiment shares: the arguments, the environment, the models.

`experiments/` holds one runnable script per experiment. They stay independent -- each
writes its own run directory and none needs another to have run first -- but they must
agree on what an environment is, how a model is named, and what makes a checkpoint
loadable here, or two runs will report numbers that are not comparable and nothing will
say so. That agreement lives here.

Figures and tables live in `report.py` and the methods in `baselines.py`, for the same
reason: one implementation, so a palette or a method cannot drift between scripts.
"""

import json
import os
import sys
from datetime import datetime

# the repository root, where the libraries below live. Importing this through the
# package puts it on the path already; this is for the paths that do not, so the
# module fails on a missing dependency rather than on where it was imported from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import numpy as np

import benchmark
import manifest
import report
from environment import Environment

# the widest a derived model label is allowed to get before it is trimmed from the front
LABEL_WIDTH = 14


def describe_domains(env):
    """What mix the run is over. `env.domains` is one episode's draw when the sampler
    redraws it, so reporting that as the run's composition would be wrong."""
    if getattr(env, "random_domains", False):
        return "uniformly random domains per agent"
    domains = env.domains if isinstance(env.domains, list) else [env.domains]
    return domains[0] if len(set(domains)) == 1 else f"mixed {sorted(set(domains))}"


def run_info_blocks(args, env, env_config_data, models, steps, methods, rows):
    """[(section, [(field, value)])] for the run_info figure.

    Everything a reader needs to place this run: what was evaluated, on what, and what
    each model label stands for. The per-model block records the objective a model was
    *trained* on against the one it is *scored* by, which is the one difference between
    models that the table cannot show and that changes how its rows should be read.
    """
    dom = describe_domains(env)
    kappa = env_config_data.get("stiffness_kappa")

    blocks = [("this run", [
        ("environment", args.environment_name),
        ("network", f"{env.network.n} agents in {dom}"),
        ("action space", env.action_space_type),
        ("objective", f"{env.state_score_type}, stiffness_kappa {kappa}"),
        ("instances", (f"{args.episodes} from benchmark {args.benchmark} "
                       f"({benchmark.digest(args.benchmark)})" if args.benchmark
                       else f"{args.episodes} random, seed {args.seed}")),
        ("methods", ", ".join(methods)),
        ("rollout", f"--policy-mode {args.policy_mode}, {steps} steps, every model"),
        ("device", args.device),
        ("written", datetime.now().strftime("%Y-%m-%d %H:%M")),
    ])]

    for label, m in models.items():
        man = m["manifest"] or {}
        cfg = man.get("environment_config_raw") or {}
        prov = man.get("provenance") or {}
        hp = man.get("hyperparameters") or {}
        lr = hp.get("learning_rate")
        trained_kappa = cfg.get("stiffness_kappa")
        objective = f"{cfg.get('state_score_type', '?')}, stiffness_kappa {trained_kappa}"
        if trained_kappa != kappa:
            objective += "   (scored here at %s)" % kappa
        blocks.append((f"model  {label}", [
            ("name", m["name"]),
            ("algorithm", f"{m['algorithm']}, {man.get('backbone', '?')} backbone"),
            ("width", f"gnn {man.get('gnn_hidden_dim', '?')}, "
                      f"head {man.get('head_hidden_dim', '?')}"),
            ("training", f"{man.get('timesteps_completed', '?')} of "
                         f"{man.get('total_timesteps_configured', '?')} steps, "
                         f"{man.get('status', '?')}"),
            ("learning rate", lr if not isinstance(lr, list) else ", ".join(map(str, lr))),
            ("trained on", man.get("environment_config")),
            ("its objective", objective),
            ("seed / commit", f"{prov.get('seed')} / {str(prov.get('git_commit'))[:12]}"
                              f"{' (dirty)' if prov.get('git_dirty') else ''}"),
        ]))
    return blocks


def load_manifest(name):
    """The run manifest for a trained model, or {} when there is none."""
    path = os.path.join("train", f"{name}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# The config keys that decide the observation's width and meaning. A model whose training
# config disagrees with the evaluation one on any of these has a checkpoint of the wrong
# shape, which surfaces deep inside agent_loader as "parameter shapes do not match".


OBS_KEYS = ("obs_type", "action_type", "graph_features",
            "include_candidate_bearings", "rigidity_global", "rigidity_quality",
            "rigidity_flex", "rigidity_edge", "rigidity_stiffness", "rigidity_removal")

# `n` is deliberately not in OBS_KEYS. The model weights do not depend on it (verified
# in tests/test_repair.py), and neither do the feature widths of the live presets, whose
# node channels are per-node and edge channels per-pair. So a checkpoint trained at
# another n is a cross-SIZE evaluation, which is an open question here rather than an
# error. The legacy presets below are the exception: they reshape an (n, n, 3) bearing
# tensor into node features, so their node width is 3n and the first Linear is n-locked.
N_DEPENDENT_NODE_SETS = ("domain_signbearing", "domain_bearing", "bearing")


def obs_width_depends_on_n(obs_type):
    from environment import OBS_PRESETS
    preset = OBS_PRESETS.get(obs_type, {})
    return preset.get("node_set", "graph") in N_DEPENDENT_NODE_SETS


def observation_mismatch(trained, current):
    """[(key, trained value, current value)] for the keys that change the observation."""
    keys = list(OBS_KEYS)
    if obs_width_depends_on_n(trained.get("obs_type")) or \
            obs_width_depends_on_n(current.get("obs_type")):
        keys.append("n")
    return [(k, trained.get(k), current.get(k)) for k in keys
            if trained.get(k) != current.get(k)]


def parse_models(values):
    """--model NAME[=LABEL], repeatable and comma-splittable. -> [(name, label)].

    One model keeps the label `learned`: that is the row name every run and every test
    written before --model took a list is keyed on. Two or more are labelled by the
    `_`-separated tokens that actually differ between them, so
    stiff_dqn_gine / stiff_ppo_gine read as dqn / ppo rather than as two long names.
    """
    names, given = [], {}
    for v in values or []:
        for part in v.split(","):
            part = part.strip()
            if not part:
                continue
            name, _, label = part.partition("=")
            names.append(name)
            if label:
                given[name] = label
    if not names:
        return []
    if len(names) == 1 and not given:
        return [(names[0], "learned")]

    auto = auto_labels(names)
    out, seen = [], {}
    for name in names:
        label = given.get(name, auto[name])
        if label in seen:
            raise SystemExit(
                f"--model: {name} and {seen[label]} both come out as '{label}'. "
                f"Pass an explicit label, e.g. --model {name}=<label>")
        seen[label] = name
        out.append((name, label))
    return out


def auto_labels(names):
    """{name: label} from the `_` tokens that tell `names` apart.

    Tokens every name carries say nothing, so they go. What is left is trimmed from the
    front if it still does not fit, since the distinguishing part of these names is
    usually at the end (stiff_dqn_gine vs stiff_dqn_equi).
    """
    parts = [n.split("_") for n in names]
    shared = set(parts[0]).intersection(*(set(p) for p in parts[1:]))
    out = {}
    for name, p in zip(names, parts):
        keep = [t for t in p if t not in shared] or p
        label = "_".join(keep)
        while len(label) > LABEL_WIDTH and len(keep) > 1:
            keep = keep[1:]
            label = "_".join(keep)
        out[name] = label[:LABEL_WIDTH]
    return out


# ---------------------------------------------------------------- arguments

def add_common_args(parser, benchmark=True, episodes=True, brief=True):
    """The arguments every experiment needs to name what it ran on.

    A script only takes the ones it reads: an experiment that offers --benchmark and
    then evaluates on something else is worse than one that does not offer it.
    """
    parser.add_argument("environment_name")
    if benchmark:
        parser.add_argument("--benchmark", default=None,
                            help="evaluate on a frozen instance set from benchmarks/ "
                                 "instead of sampling, which is what makes two runs "
                                 "comparable across config regenerations")
    else:
        parser.set_defaults(benchmark=None)
    if episodes:
        parser.add_argument("--episodes", type=int, default=20)
    else:
        parser.set_defaults(episodes=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="device for the --model rollout")
    parser.add_argument("--tag", default=None,
                        help="label appended to the run directory name")
    parser.add_argument("--out-dir", default=None,
                        help="write results here instead of runs_outputs/<generated name>")
    if brief:
        parser.add_argument("--brief", action="store_true",
                            help="print the table without the explanatory legend")
    return parser


def add_model_args(parser, required=False, policy_mode=True):
    parser.add_argument("--model", action="append", default=None, required=required,
                        help="trained model from train/, repeatable and comma-splittable. "
                             "NAME=LABEL names its row; otherwise a label is derived from "
                             "the parts of the names that differ. One model is labelled "
                             "`learned`")
    if policy_mode:
        parser.add_argument("--policy-mode", default="sample", choices=("sample", "greedy"),
                            help="sample (default): sampled actions, i.e. the policy used "
                                 "as a sampling search over the horizon, scored on the best "
                                 "state it finds. greedy: the single action the policy "
                                 "considers best, which is reproducible and terminates on a "
                                 "cycle. A DQN q-network has nothing to sample from and is "
                                 "argmax either way")
    else:
        parser.set_defaults(policy_mode="greedy")
    return parser


# ---------------------------------------------------------------- setup

def load_environment(args):
    """(env, env_config_data, steps) or None when the config is not there."""
    filepath = "./environments/" + args.environment_name + ".json"
    if not os.path.exists(filepath):
        print(f"file {filepath} does not exist")
        return None
    np.random.seed(args.seed)
    with open(filepath, "r") as fh:
        env_config_data = json.load(fh)
    env = Environment()
    env.load(filepath)
    # gymnasium spaces carry their own RNG, so np.random.seed does not make
    # action_space.sample() reproducible -- the random baseline needs this too
    env.action_space.seed(args.seed)
    steps = getattr(args, "steps", None)
    if steps is None:
        steps = int(env.truncate_max_steps if env.truncate_enable else env.max_steps)
    return env, env_config_data, steps


def load_models(args, env, env_config_data, model_specs, steps):
    """(models, wrapped, env, steps), or None when a checkpoint does not fit.

    A checkpoint only loads against an environment whose observation has the same
    width and meaning. The environment NAME does not predict that -- the same config
    is regenerated under different names -- so the flags are compared instead, and the
    differing key is named before anything is loaded.
    """
    wrapped, models = None, {}        # label -> {name, agent, algorithm, manifest}
    if not model_specs:
        return models, wrapped, env, steps

    from skrl.envs.wrappers.torch import wrap_env
    from agent_loader import load_agent, load_run

    replay = getattr(args, "replay_env", False)
    if replay:
        # load_run execs one model's archived rigidity.py and environment.py into
        # sys.modules, so two archives cannot both be live
        (name, label), = model_specs
        agent, wrapped, env, info = load_run(
            name, env_name=args.environment_name, device=args.device)
        models[label] = {"name": name, "agent": agent,
                         "algorithm": (info or {}).get("algorithm", "?"),
                         "manifest": info or {}}
        env.action_space.seed(args.seed)
        if getattr(args, "steps", None) is None:
            steps = int(env.truncate_max_steps if env.truncate_enable else env.max_steps)
    else:
        bad = []
        for name, _ in model_specs:
            diff = observation_mismatch(
                load_manifest(name).get("environment_config_raw") or {}, env_config_data)
            if diff:
                bad.append((name, diff))
        if bad:
            print("these models were trained on a different observation, so their "
                  "checkpoints do not fit this environment:")
            for name, diff in bad:
                for k, was, now in diff:
                    print(f"  {name}: {k} was {was}, this environment has {now}")
            return None

        # the wrapper reads env.device to decide where to put observations; without
        # this it defaults to cuda while the agent is built on cpu
        env.device = args.device
        wrapped = wrap_env(env)
        wrapped.reset()
        n = env.network.n
        for name, label in model_specs:
            agent, algorithm = load_agent(name, wrapped, env, device=args.device)
            # not `manifest`: that name is the module this file imports
            man = load_manifest(name)
            models[label] = {"name": name, "agent": agent, "algorithm": algorithm,
                             "manifest": man}
            trained_n = (man.get("environment_config_raw") or {}).get("n")
            if trained_n is not None and trained_n != n:
                print(f"  note: {name} was trained at n={trained_n} and is being "
                      f"evaluated at n={n}. The weights do not depend on n, but the "
                      f"channels normalised by m_req do, so this is a cross-size "
                      f"result and not a like-for-like one")
            trained_on = man.get("environment_config")
            # the same environment gets regenerated under different names, so the name
            # is only a warning
            if isinstance(trained_on, str) and trained_on != args.environment_name:
                print(f"  note: {name} was trained on {trained_on}, "
                      f"scoring it on {args.environment_name}")

    for label, m in models.items():
        print(f"loaded {m['algorithm']} model '{m['name']}' as '{label}' on {args.device}"
              f"{' (replaying its archived environment)' if replay else ''}")
    return models, wrapped, env, steps


def announce(models):
    """Register the models as methods, so every figure keys their colour the same way."""
    report.configure_methods([(lab, f"trained policy {m['name']} ({m['algorithm']})")
                              for lab, m in models.items()])


def run_directory(args, models, with_plots=True):
    return report.make_run_dir(
        "runs_outputs", args.environment_name,
        model_names=[m["name"] for m in models.values()],
        tag=args.tag, out_dir=args.out_dir, with_plots=with_plots)


def instances_for(args, env):
    """The networks to evaluate on, and a note for the header.

    A benchmark is a fixture: regenerating a config resamples the instance
    distribution, which has already invalidated one comparison.
    """
    if not args.benchmark:
        return None, f"{args.episodes} random, seed {args.seed}"
    frozen, meta = benchmark.load(args.benchmark)
    if len(frozen) < args.episodes:
        print(f"  benchmark {args.benchmark} has {len(frozen)} instances; "
              f"running that many instead of {args.episodes}")
        args.episodes = len(frozen)
    return frozen, (f"{args.episodes} from benchmark {args.benchmark} "
                    f"({benchmark.digest(args.benchmark)})")


def header_for(args, env, models, context_network):
    return {
        "short": report.short_env_name(args.environment_name),
        "env": args.environment_name,
        "model": ("  ".join(f"{lab} = {m['name']}" for lab, m in models.items()) or None),
        "network": context_network,
        "episodes": args.episodes,
        "seed": args.seed,
        "benchmark": args.benchmark,
    }


def provenance(args, env_config_data, models, extra=None):
    """meta.json: what was run, on what, and against which code."""
    return {
        "args": vars(args),
        "environment_config": env_config_data,
        "models": {lab: m["name"] for lab, m in models.items()},
        # the richer form beside it: `models` stays label -> name because
        # generalisation.py reads it that way
        "model_info": {lab: {
            "name": m["name"], "algorithm": m.get("algorithm"),
            "backbone": (m.get("manifest") or {}).get("backbone"),
            "trained_kappa": ((m.get("manifest") or {}).get("environment_config_raw")
                              or {}).get("stiffness_kappa"),
        } for lab, m in models.items()},
        "benchmark": args.benchmark,
        "benchmark_digest": (benchmark.digest(args.benchmark) if args.benchmark else None),
        "provenance": manifest.collect_provenance(seed=args.seed, device=args.device),
        **(extra or {}),
    }
