#!/usr/bin/env bash

set -e

# install uv if missing
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # common install path for uv
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv already installed"
fi

# create venv
uv venv --python 3.12
source .venv/bin/activate

# install dependencies
uv pip install \
    numpy matplotlib scipy pandas viser \
    stable-baselines3 tqdm numpy-quaternion
uv pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126
uv pip install torch-geometric
uv pip install \
    pyg_lib torch_scatter torch_sparse \
    torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
uv pip install skrl egnn-pytorch

echo "Setup complete"