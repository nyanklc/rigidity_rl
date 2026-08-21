"""Does editing policy/ change what an archived checkpoint computes?

A checkpoint is supposed to survive edits to `policy/`: the manifest archives the
model class source and `policy/gnn_backbone.py`, and `agent_loader.resolve_model`
replays the archived text rather than the class sitting in the tree today. This
prints a deterministic digest of the loaded model's output on one fixed
observation, so that promise can be checked instead of trusted -- run it before
touching a model class, run it after, and diff.

The loader also says which source it used; the line to look for is
`(archived source)` plus, once the tree has moved, a `changed since this run`
note. A digest that moves while that note is printed means the archive is not
actually insulating the checkpoint.

    PYTHONPATH=. uv run tools/checkpoint_fingerprint.py letsgo_dqn_gine
    PYTHONPATH=. uv run tools/checkpoint_fingerprint.py letsgo_dqn_gine --env env_..._n8_R3
    PYTHONPATH=. uv run tools/checkpoint_fingerprint.py letsgo_dqn_gine --seed 3

Needs the checkpoint and its manifest, so it does not run on a fresh clone --
`models/` and `train/` are gitignored.
"""
import argparse

import numpy as np
import torch
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space

import agent_loader


def fingerprint(model_name, env_name=None, seed=0, role=None):
    agent, _env, raw_env, train_info = agent_loader.load_run(model_name, env_name)

    # the observation has to be the same one every time, so seed immediately
    # before the reset that produces it
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs, _ = raw_env.reset()

    role = role or ("policy" if train_info.get("algorithm") == "PPO" else "q_network")
    model = agent.models[role]
    model.eval()
    x = flatten_tensorized_space(tensorize_space(raw_env.observation_space, obs, device="cpu"))
    with torch.no_grad():
        out, _ = model.compute({"observations": x}, role=role)

    # masked entries are -inf by design, so the digest is taken over the finite ones
    finite = out[torch.isfinite(out)]
    return model, out, finite


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_name")
    p.add_argument("--env", default=None, help="environment config, if not the manifest's")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--role", default=None, help="policy / value / q_network")
    args = p.parse_args()

    model, out, finite = fingerprint(args.model_name, args.env, args.seed, args.role)

    print("\nFINGERPRINT")
    print(f"  model     : {args.model_name}  seed {args.seed}")
    print(f"  class     : {type(model).__name__}")
    print(f"  out shape : {tuple(out.shape)}")
    print(f"  finite n  : {int(finite.numel())} of {int(out.numel())}")
    print("  sum       : %.12e" % float(finite.sum()))
    print("  min / max : %.12e  %.12e" % (float(finite.min()), float(finite.max())))
    print("  argmax    : %d" % int(torch.argmax(out)))


if __name__ == "__main__":
    main()
