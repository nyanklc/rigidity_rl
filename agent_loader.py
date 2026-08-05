"""Rebuild a trained skrl agent from the run manifest written by train_ppo.py / train_dqn.py.

The manifests in train/ embed the *source* of the actor/critic/q-network classes, so the
architecture is recovered by name from policy/ rather than being hardcoded here.

Runs from before the manifests existed have no train/<name>.json. For those, load_agent()
falls back to an interactive path that reads the architecture straight out of the checkpoint's
parameter shapes and confirms the guesses with the user (see load_agent_legacy).
"""

import contextlib
import copy
import inspect
import json
import os
import re
import sys
import tempfile
import types

import torch

from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.agents.torch.dqn import DQN, DQN_CFG
import policy
import policy.gnn_backbone
from policy import *
from policy.gnn_backbone import GNNBackboneEquivariant, GNNBackboneGINE, GNNBackboneGAT

ALGORITHMS = ("PPO", "DQN", "DDQN")

# which skrl model roles each algorithm expects, and the class-name prefix that
# implementations of that role use in policy/
MODEL_ROLES = {
    "PPO": {"policy": "PPO_ActorModel_", "value": "PPO_CriticModel_"},
    "DQN": {"q_network": "DQN_QNetwork_"},
    "DDQN": {"q_network": "DQN_QNetwork_"},
}


def model_path(algorithm, model_name):
    return f"./models/complete/{algorithm}/{model_name}.pt"


def manifest_path(model_name):
    return f"./train/{model_name}.json"


def get_class_name(architecture_lines):
    for line in architecture_lines:
        line = line.strip()
        if line.startswith("class "):
            return line.split()[1].split("(")[0].rstrip(":")
    return None


def instantiate(cls, all_kwargs):
    sig = inspect.signature(cls.__init__)
    valid_kwargs = {k: v for k, v in all_kwargs.items() if k in sig.parameters}
    return cls(**valid_kwargs)


def instantiate_model(class_name, all_kwargs):
    return instantiate(globals()[class_name], all_kwargs)


# `inspect.getsource(cls)` captures only the class body, so replaying it needs the same
# imports the files in policy/ have at module level.
MODEL_SOURCE_PREAMBLE = """
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATConv, GINEConv
from egnn_pytorch import EGNN
from skrl.models.torch import Model, CategoricalMixin, DeterministicMixin, TabularMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space
from policy.gnn_backbone import *
"""


def build_class_from_source(class_source, backbone_source=None):
    """Recreate a model class exactly as it was when the run was trained.

    The class body references GNNBackbone* by name, so the archived backbone module is
    replayed first when the manifest carries it; otherwise the current one is used, which
    is only correct as long as the backbone has not changed since.
    """
    namespace = {}
    exec(MODEL_SOURCE_PREAMBLE, namespace)
    if backbone_source:
        exec(backbone_source, namespace)
    exec(class_source, namespace)

    name = get_class_name(class_source.split("\n"))
    if name is None or name not in namespace:
        raise ValueError("archived source defines no class")
    return namespace[name]


def source_of(train_info, key):
    lines = train_info.get(key)
    return "\n".join(lines) if lines else None


def resolve_model(train_info, arch_key, kwargs, checkpoint_sd, role):
    """Build `role`'s model, preferring the architecture the run was actually trained with.

    Tries the archived source first (so a checkpoint keeps loading after policy/ changes),
    falls back to the current class of the same name, and verifies against the checkpoint's
    parameter shapes either way.
    """
    import manifest as manifest_mod

    class_source = source_of(train_info, arch_key)
    # prefer the archived bundle; sources_of() falls back to the older backbone_source key
    backbone_source = manifest_mod.sources_of(train_info).get("policy/gnn_backbone.py")
    class_name = get_class_name(class_source.split("\n")) if class_source else None
    target = shapes_of(checkpoint_sd)

    attempts = []
    if class_source:
        attempts.append(("archived source", lambda: build_class_from_source(class_source, backbone_source)))
    if class_name and class_name in globals():
        attempts.append(("current policy/", lambda: globals()[class_name]))

    arch = infer_architecture(checkpoint_sd)
    errors = []
    for label, get_cls in attempts:
        try:
            model = instantiate(get_cls(), kwargs)
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")
            continue
        if shapes_of(model.state_dict()) == target:
            if label == "archived source" and class_name in globals():
                _report_source_drift(class_name, class_source, backbone_source)
            return model, label

        # backbone depth is not a constructor argument of the model class, so a run from
        # when the backbone had 2 layers needs it swapped to what the weights imply
        depth = arch.get("gnn_layers")
        if depth and depth != getattr(getattr(model, "gnn", None), "num_layers", depth):
            try:
                rebuild_backbone(model, depth)
            except Exception:
                pass
            if shapes_of(model.state_dict()) == target:
                return model, f"{label}, backbone rebuilt at {depth} layers"

        errors.append(f"{label}: parameter shapes do not match the checkpoint")

    raise ValueError(
        f"could not rebuild the {role} ({class_name}) for this checkpoint:\n    "
        + "\n    ".join(errors)
    )


def _report_source_drift(class_name, class_source, backbone_source):
    """Say so when the archived architecture differs from what is in policy/ today."""
    notes = []
    try:
        current = inspect.getsource(globals()[class_name])
        if current.strip() != class_source.strip():
            notes.append(class_name)
    except (OSError, TypeError):
        pass
    if backbone_source:
        try:
            if inspect.getsource(policy.gnn_backbone).strip() != backbone_source.strip():
                notes.append("gnn_backbone.py")
        except (OSError, TypeError):
            pass
    if notes:
        print(f"  note: {', '.join(notes)} changed since this run; "
              f"using the archived version so the weights stay valid")


# ---------------------------------------------------------------------------------------
# Legacy checkpoints (no manifest): recover the architecture from parameter shapes
# ---------------------------------------------------------------------------------------
def find_checkpoints(model_name):
    """Which algorithm directories hold a checkpoint with this name."""
    return [a for a in ALGORITHMS if os.path.exists(model_path(a, model_name))]


def list_checkpoints():
    """{algorithm: [model_name, ...]} for everything saved under models/complete/."""
    out = {}
    for algorithm in ALGORITHMS:
        d = f"./models/complete/{algorithm}"
        if not os.path.isdir(d):
            continue
        names = sorted(f[:-3] for f in os.listdir(d) if f.endswith(".pt"))
        if names:
            out[algorithm] = names
    return out


def infer_architecture(state_dict):
    """Read back the constructor arguments a state dict was produced with.

    Only the values that are *not* recoverable from the environment are needed:
    the backbone depth and the two hidden widths. The feature dims are reported
    too, but only so a mismatch with the current observation space can be
    explained instead of surfacing as an opaque shape error.
    """
    keys = list(state_dict.keys())
    depth = len({int(m.group(1)) for k in keys if (m := re.match(r"gnn\.conv(\d+)\.", k))})
    arch = {"gnn_layers": depth or None}

    if any("edge_mlp" in k for k in keys):
        # EGNN: edge_mlp input is [h_i, h_j, ||x_i - x_j||^2, e_ij]
        dim = state_dict["gnn.conv1.node_mlp.3.weight"].shape[0]
        arch["backbone"] = "Equivariant"
        arch["node_feat_dim"] = dim
        arch["edge_feat_dim"] = state_dict["gnn.conv1.edge_mlp.0.weight"].shape[1] - 2 * dim - 1
        arch["gnn_hidden_dim"] = state_dict["gnn.conv1.edge_mlp.3.weight"].shape[0]
    elif any(re.match(r"gnn\.conv1\.nn\.", k) for k in keys):
        arch["backbone"] = "GINE"
        arch["node_feat_dim"] = state_dict["gnn.conv1.nn.0.weight"].shape[1]
        arch["gnn_hidden_dim"] = state_dict["gnn.conv1.nn.2.weight"].shape[0]
        lin = [k for k in keys if k.startswith("gnn.conv1.lin")]
        arch["edge_feat_dim"] = state_dict[lin[0]].shape[1] if lin else None
    elif any("att_src" in k for k in keys):
        arch["backbone"] = "GAT"
        lin = next((k for k in keys if re.match(r"gnn\.conv1\.lin(_src)?\.weight", k)), None)
        arch["node_feat_dim"] = state_dict[lin].shape[1] if lin else None
        arch["edge_feat_dim"] = None
        arch["gnn_hidden_dim"] = state_dict["gnn.conv1.att_src"].shape[-1]
    else:
        arch["backbone"] = "unknown"

    if "head.0.weight" in state_dict:
        arch["head_hidden_dim"] = state_dict["head.0.weight"].shape[0]
        arch["head_in"] = state_dict["head.0.weight"].shape[1]
    arch["has_skip_head"] = any(k.startswith("skip_head.") for k in keys)
    return arch


def rebuild_backbone(model, num_layers):
    """Re-create model.gnn at a different depth (older runs used 2 layers, not 3)."""
    gnn = getattr(model, "gnn", None)
    if gnn is None or not hasattr(gnn, "init_args") or num_layers in (None, gnn.num_layers):
        return
    model.gnn = type(gnn)(**gnn.init_args, num_layers=num_layers).to(model.device)


def shapes_of(state_dict):
    return {k: tuple(v.shape) for k, v in state_dict.items()}


def match_model_class(prefix, checkpoint_sd, base_kwargs, arch):
    """Every class in policy/ whose parameters match the checkpoint exactly."""
    target = shapes_of(checkpoint_sd)
    matches, near = [], []

    kwargs = dict(base_kwargs)
    if arch.get("gnn_hidden_dim"):
        kwargs["gnn_hidden_dim"] = arch["gnn_hidden_dim"]
    if arch.get("head_hidden_dim"):
        kwargs["head_hidden_dim"] = arch["head_hidden_dim"]

    for class_name in sorted(n for n in policy.__all__ if n.startswith(prefix)):
        try:
            model = instantiate_model(class_name, kwargs)
            rebuild_backbone(model, arch.get("gnn_layers"))
            got = shapes_of(model.state_dict())
        except Exception:
            continue
        if got == target:
            matches.append(class_name)
        elif set(got) == set(target):
            bad = [k for k in got if got[k] != target[k]]
            near.append((class_name, bad[:3]))
    return matches, near


def algorithm_from_roles(checkpoint):
    """What the saved model keys imply. PPO stores policy+value, DQN/DDQN a q_network."""
    roles = {k for k in checkpoint if k != "optimizer"}
    if {"policy", "value"} <= roles:
        return "PPO"
    if "q_network" in roles:
        # DQN and DDQN save the same roles, so this cannot tell them apart
        return "DQN"
    return None


def resolve_algorithm(model_name):
    """(algorithm, path, checkpoint), asking only when it genuinely cannot be decided.

    The directory under models/complete/ names the algorithm, and the saved model roles
    corroborate it. Only a contradiction or the same name in several directories needs input.
    """
    found = find_checkpoints(model_name)
    if not found:
        raise FileNotFoundError(
            f"no checkpoint named '{model_name}' under models/complete/{{{','.join(ALGORITHMS)}}}"
        )

    if len(found) > 1:
        print(f"  '{model_name}' exists under {', '.join(found)} - these are different files.")
        algorithm = _ask("  Which algorithm was this trained with?", found, default=found[0])
        path = model_path(algorithm, model_name)
        return algorithm, path, torch.load(path, map_location="cpu", weights_only=False)

    algorithm = found[0]
    path = model_path(algorithm, model_name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    implied = algorithm_from_roles(checkpoint)

    if implied is None:
        print(f"  {path} stores {sorted(k for k in checkpoint if k != 'optimizer')}, "
              f"which matches no known agent.")
        algorithm = _ask("  Which algorithm was this trained with?", list(ALGORITHMS),
                         default=algorithm)
    elif implied == "PPO" and algorithm != "PPO":
        # saved under the wrong directory at some point
        print(f"  {path} is under {algorithm}/ but stores policy+value, so it is PPO.")
        algorithm = "PPO"
    elif implied == "DQN" and algorithm == "PPO":
        print(f"  {path} is under PPO/ but stores a q_network, so it is DQN.")
        algorithm = "DQN"
    else:
        print(f"  algorithm: {algorithm} (from models/complete/{algorithm}/, "
              f"stores {sorted(k for k in checkpoint if k != 'optimizer')})")

    return algorithm, path, checkpoint


def _ask(prompt, options=None, default=None):
    suffix = f" [{'/'.join(options)}]" if options else ""
    suffix += f" (default: {default})" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        if not options or raw in options:
            return raw
        print(f"  pick one of {options}")


def load_agent_legacy(model_name, env, raw_env, device="cpu"):
    """Interactive loader for checkpoints with no train/<name>.json manifest."""
    print(f"\nNo manifest at {manifest_path(model_name)} - recovering from the checkpoint.")
    algorithm, path, checkpoint = resolve_algorithm(model_name)

    roles = MODEL_ROLES[algorithm]
    missing = [r for r in roles if r not in checkpoint]
    if missing:
        raise ValueError(
            f"{path} has {sorted(k for k in checkpoint if k != 'optimizer')}, "
            f"but {algorithm} needs {sorted(roles)}."
        )

    node_feat_dim = raw_env.observation_space["node_features"].shape[1]
    edge_feat_dim = raw_env.observation_space["edge_features"].shape[-1]
    base_kwargs = {
        "n": len(raw_env.network.agents),
        "node_feat_dim": node_feat_dim,
        "edge_feat_dim": edge_feat_dim,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "device": device,
    }

    models = {}
    for role, prefix in roles.items():
        sd = checkpoint[role]
        arch = infer_architecture(sd)
        print(f"\n  [{role}] recovered from parameter shapes:")
        for k in ("backbone", "gnn_layers", "gnn_hidden_dim", "head_hidden_dim",
                  "node_feat_dim", "edge_feat_dim"):
            if arch.get(k) is not None:
                print(f"      {k:16s} {arch[k]}")

        matches, near = match_model_class(prefix, sd, base_kwargs, arch)

        if not matches:
            print(f"\n  No class in policy/ reproduces this {role}.")
            # the usual cause: the observation format changed since the run
            ck_node, ck_edge = arch.get("node_feat_dim"), arch.get("edge_feat_dim")
            if ck_node is not None and ck_node not in (node_feat_dim, node_feat_dim + 1):
                print(f"  The checkpoint expects {ck_node} node features but this environment "
                      f"produces {node_feat_dim}.")
            if ck_edge is not None and ck_edge != edge_feat_dim:
                print(f"  The checkpoint expects {ck_edge} edge features but this environment "
                      f"produces {edge_feat_dim}.")
            if (ck_node is not None and ck_node not in (node_feat_dim, node_feat_dim + 1)) or (
                ck_edge is not None and ck_edge != edge_feat_dim
            ):
                print("  The observation space changed after this run; it cannot be replayed "
                      "on the current obs type without restoring the old feature set.")
            for name, bad in near[:5]:
                print(f"    closest: {name} differs at {bad}")
            raise ValueError(f"could not reconstruct the {role} for '{model_name}'")

        if len(matches) == 1:
            class_name = matches[0]
            print(f"      -> {class_name}")
        else:
            print(f"  Several classes fit this {role} equally well:")
            for i, m in enumerate(matches):
                print(f"      {i}: {m}")
            idx = _ask("  Which one?", [str(i) for i in range(len(matches))], default="0")
            class_name = matches[int(idx)]

        kwargs = dict(base_kwargs)
        if arch.get("gnn_hidden_dim"):
            kwargs["gnn_hidden_dim"] = arch["gnn_hidden_dim"]
        if arch.get("head_hidden_dim"):
            kwargs["head_hidden_dim"] = arch["head_hidden_dim"]
        # older runs predate the skip masking option; they always had the skip head
        kwargs["allow_skip"] = arch.get("has_skip_head", True)

        model = instantiate_model(class_name, kwargs)
        rebuild_backbone(model, arch.get("gnn_layers"))
        model.load_state_dict(sd)
        models[role] = model

    if algorithm != "PPO":
        models["target_q_network"] = copy.deepcopy(models["q_network"])

    agent = build_agent(algorithm, models, env, device, mem_size=2048 * 4,
                        experiment_name=model_name)
    print(f"\nLoaded '{model_name}' as {algorithm}.\n")
    return agent, algorithm


def build_agent(algorithm, models, env, device, mem_size, experiment_name):
    memory = RandomMemory(memory_size=mem_size, num_envs=1, device=device)

    if algorithm == "PPO":
        cfg = PPO_CFG()
        cfg.rollouts = mem_size  # to ensure we don't get garbage data from memory
        cfg.experiment.directory = "runs_inference"
        cfg.experiment.experiment_name = experiment_name
        # incentivize exploration more
        cfg.entropy_loss_scale = 0.01
        return PPO(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )

    cfg = DQN_CFG()
    cfg.experiment.directory = "runs_inference"
    cfg.experiment.experiment_name = experiment_name
    cfg.batch_size = 128
    cfg.target_update_interval = 1000
    cfg.update_interval = 4
    cfg.learning_starts = mem_size + 1
    cfg.discount_factor = 0.99
    cfg.random_timesteps = mem_size
    return DQN(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )


# ---------------------------------------------------------------------------------------
# Replaying the environment a run was trained against
# ---------------------------------------------------------------------------------------
@contextlib.contextmanager
def archived_modules(sources):
    """Temporarily make `import environment`, `import network`, ... resolve to archived text.

    Each file is exec'd into a fresh module registered under its real name, in dependency
    order, so the archived environment.py's own `from network import Network` picks up the
    archived network.py. sys.modules is restored on exit. Modules that do not affect
    observations or action semantics (control, visualizer, skrl, torch) stay as installed.
    """
    import manifest as manifest_mod

    names = [rel[:-3].replace("/", ".").split(".")[-1] for rel in manifest_mod.ARCHIVED_FILES]
    saved = {name: sys.modules.get(name) for name in names}
    saved["policy.gnn_backbone"] = sys.modules.get("policy.gnn_backbone")
    created = {}
    try:
        for rel in manifest_mod.ARCHIVED_FILES:
            if rel not in sources:
                continue
            modname = "policy.gnn_backbone" if rel.endswith("gnn_backbone.py") \
                else os.path.basename(rel)[:-3]
            module = types.ModuleType(modname)
            module.__file__ = f"<archived {rel}>"
            sys.modules[modname] = module
            if modname == "policy.gnn_backbone":
                # the model classes do `from policy.gnn_backbone import *`
                setattr(sys.modules["policy"], "gnn_backbone", module)
            exec(compile(sources[rel], f"<archived {rel}>", "exec"), module.__dict__)
            created[modname] = module
        yield created
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        if saved.get("policy.gnn_backbone") is not None:
            setattr(sys.modules["policy"], "gnn_backbone", saved["policy.gnn_backbone"])


def build_archived_env(train_info, sources, device="cpu"):
    """An Environment built from the archived code and the run's own config/scenario."""
    env_config = dict(train_info.get("environment_config_raw") or {})
    if not env_config:
        raise ValueError("manifest has no environment_config_raw to rebuild the environment from")

    tmpdir = tempfile.mkdtemp(prefix="replay_env_")
    cfg_path = os.path.join(tmpdir, "env.json")
    with open(cfg_path, "w") as f:
        json.dump(env_config, f)

    with archived_modules(sources) as mods:
        env_module = mods.get("environment")
        if env_module is None:
            raise ValueError("manifest does not archive environment.py")

        # scenarios/ is gitignored, so the scenario lives in the manifest; point the
        # archived loader at a materialised copy instead of the (possibly absent) file
        raw_scenario = train_info.get("scenario_raw")
        if env_config.get("scenario") and raw_scenario is not None:
            scen_path = os.path.join(tmpdir, "scenario.json")
            with open(scen_path, "w") as f:
                json.dump(raw_scenario, f)
            original = env_module.load_scenario
            env_module.load_scenario = lambda _p, _f=original, _s=scen_path: _f(_s)
            if hasattr(env_module, "randomize_scenario"):
                orig_rand = env_module.randomize_scenario
                env_module.randomize_scenario = lambda _p, _f=orig_rand, _s=scen_path: _f(_s)

        raw_env = env_module.Environment()
        raw_env.load(cfg_path)
        raw_env.device = device
        return raw_env


def load_run(model_name, env_name=None, device="cpu", prefer_archived_env=True):
    """Rebuild a run: (agent, wrapped_env, raw_env, train_info).

    Uses the environment the run was trained against whenever the archived sources differ
    from the working tree, so a checkpoint keeps running after the observation format,
    action semantics or rigidity maths change. Reports any drift it acts on.
    """
    from skrl.envs.wrappers.torch import wrap_env
    import manifest as manifest_mod

    train_info = None
    if os.path.exists(manifest_path(model_name)):
        train_info = manifest_mod.load(model_name)

    sources = manifest_mod.sources_of(train_info) if train_info else {}
    drift = manifest_mod.current_sources_differ(sources) if sources else {}
    # gnn_backbone drift is handled by resolve_model replaying the class source; it does not
    # justify rebuilding the environment
    env_drift = {k: v for k, v in drift.items() if k != "policy/gnn_backbone.py"}

    raw_env = None
    if env_drift and prefer_archived_env:
        print(f"  archived code differs from the working tree: "
              f"{manifest_mod.describe_drift(env_drift)}")
        print(f"  -> replaying this run's own environment "
              f"(uv run manifest.py diff {model_name} to see what changed)")
        try:
            raw_env = build_archived_env(train_info, sources, device=device)
        except Exception as e:
            print(f"  could not rebuild the archived environment ({type(e).__name__}: {e});"
                  f" falling back to the live one")
            raw_env = None
    elif drift:
        print(f"  archived code differs only in {manifest_mod.describe_drift(drift)};"
              f" using the live environment")

    if raw_env is None:
        if env_name is None:
            raise ValueError("no archived environment available and no env_name given")
        from environment import Environment
        raw_env = Environment()
        raw_env.load("./environments/" + env_name + ".json")
        raw_env.device = device

    env = wrap_env(raw_env)
    env.reset()
    agent, _algorithm = load_agent(model_name, env, raw_env, device=device)
    return agent, env, raw_env, train_info


def verify_manifest(model_name, device="cpu", train_info=None):
    """(ok, detail) -- can this manifest's model be rebuilt to fit its checkpoint?

    `train_info` overrides what is on disk, so backfill can test a patched manifest
    before committing it.
    """
    import manifest as manifest_mod
    if train_info is None:
        try:
            train_info = manifest_mod.load(model_name)
        except Exception as e:
            return False, f"unreadable manifest: {e}"

    algorithm = train_info.get("algorithm", "PPO")
    path = model_path(algorithm, model_name)
    if not os.path.exists(path):
        return False, f"no checkpoint at {path}"

    env_config = train_info.get("environment_config_raw")
    if not env_config:
        return False, "no environment_config_raw"

    try:
        sources = manifest_mod.sources_of(train_info)
        raw_env = build_archived_env(train_info, sources, device=device) if sources.get(
            "environment.py") else None
        if raw_env is None:
            from environment import Environment
            tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
            json.dump(env_config, tmp); tmp.close()
            raw_env = Environment(); raw_env.load(tmp.name); raw_env.device = device
    except Exception as e:
        return False, f"environment could not be built: {type(e).__name__}: {e}"

    try:
        from skrl.envs.wrappers.torch import wrap_env
        env = wrap_env(raw_env)
        env.reset()
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        all_kwargs = {
            "n": len(raw_env.network.agents),
            "allow_skip": env_config.get("skip_enabled", True),
            "node_feat_dim": raw_env.observation_space["node_features"].shape[1],
            "edge_feat_dim": raw_env.observation_space["edge_features"].shape[-1],
            "gnn_hidden_dim": train_info.get("gnn_hidden_dim", 32),
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "device": device,
        }
        roles = ([("policy", "actor_architecture", "head_hidden_dim"),
                  ("value", "critic_architecture", "critic_head_hidden_dim")]
                 if algorithm == "PPO"
                 else [("q_network", "q_network_architecture", "head_hidden_dim")])
        used = []
        for role, arch_key, head_key in roles:
            kwargs = dict(all_kwargs, head_hidden_dim=train_info.get(head_key, 32))
            _model, label = resolve_model(train_info, arch_key, kwargs, checkpoint[role], role)
            used.append(f"{role} via {label}")
        return True, "; ".join(used)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def load_agent(model_name, env, raw_env, device="cpu"):
    """Returns (agent, algorithm). `env` is the wrapped env, `raw_env` the Environment."""
    train_json_path = manifest_path(model_name)
    if not os.path.exists(train_json_path):
        # pre-manifest run: recover what we can from the checkpoint and ask for the rest
        return load_agent_legacy(model_name, env, raw_env, device=device)

    with open(train_json_path, "r") as f:
        train_info = json.load(f)

    algorithm = train_info.get("algorithm", "PPO")
    mem_size = train_info.get("mem_size", 2048 * 4)

    if algorithm not in ("PPO", "DQN", "DDQN"):
        raise ValueError(f"Unknown algorithm {algorithm}")

    modelpath = model_path(algorithm, model_name)
    if not os.path.exists(modelpath):
        raise FileNotFoundError(f"{modelpath} does not exist")

    n = len(raw_env.network.agents)
    # the manifest embeds the env config it was trained with, so a model trained
    # without the skip action is rebuilt without it (instantiate_model drops the
    # kwarg for classes that don't take it)
    env_config = train_info.get("environment_config_raw", {})
    all_kwargs = {
        "n": n,
        "allow_skip": env_config.get("skip_enabled", True),
        "node_feat_dim": raw_env.observation_space["node_features"].shape[1],
        "edge_feat_dim": raw_env.observation_space["edge_features"].shape[-1],
        "gnn_hidden_dim": train_info.get("gnn_hidden_dim", 32),
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "device": device,
    }

    # needed to verify the rebuilt architecture actually fits the saved weights
    checkpoint = torch.load(modelpath, map_location=device, weights_only=False)

    if algorithm == "PPO":
        roles = [("policy", "actor_architecture", "head_hidden_dim"),
                 ("value", "critic_architecture", "critic_head_hidden_dim")]
    else:
        roles = [("q_network", "q_network_architecture", "head_hidden_dim")]

    models = {}
    for role, arch_key, head_key in roles:
        kwargs = all_kwargs.copy()
        kwargs["head_hidden_dim"] = train_info.get(head_key, 32)
        models[role], used = resolve_model(
            train_info, arch_key, kwargs, checkpoint[role], role
        )
        print(f"  {role}: {get_class_name(train_info[arch_key]) or '?'} ({used})")

    if algorithm != "PPO":
        models["target_q_network"] = copy.deepcopy(models["q_network"])

    agent = build_agent(algorithm, models, env, device, mem_size, model_name)
    agent.load(modelpath)
    return agent, algorithm
