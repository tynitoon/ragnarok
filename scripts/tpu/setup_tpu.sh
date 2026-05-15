#!/bin/bash
# setup_tpu.sh — software setup for a fresh Ragnarok Cloud TPU VM.
#
# Run this ON the TPU VM (via `gcloud compute tpus tpu-vm ssh`) after the
# TPU VM has been provisioned. It installs PyTorch/XLA, clones the repo,
# installs Ragnarok's dependencies, and runs a hello-TPU smoke test.
#
# Designed for the spot-preemption workflow: when a spot TPU VM is
# preempted, recreate the VM and re-run this script — a working
# environment is restored in ~5-10 minutes. Nothing on the TPU VM is
# load-bearing; the code lives in git, results are pushed to git.
#
# Idempotent: safe to re-run on a partially-set-up VM.
set -e

REPO_URL="https://gitlab.com/mortier.jeremie/ragnarok.git"
VENV="$HOME/ragnarok-venv"
REPO="$HOME/ragnarok"
TORCH_VER="2.9.0"   # torch and torch_xla MUST match exactly (ABI compat)

echo "=== [1/6] System packages ==="
sudo apt-get update -qq
sudo apt-get install -y python3.10-venv git >/dev/null 2>&1

echo "=== [2/6] Virtualenv ($VENV) ==="
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --upgrade pip -q

echo "=== [3/6] PyTorch/XLA $TORCH_VER ==="
# torch and torch_xla pinned to the same version — a mismatch causes
# an "undefined symbol" ImportError in _XLAC.so (the torch_xla C++ lib
# is compiled against a specific torch ABI).
"$VENV/bin/pip" install "torch==$TORCH_VER" "torch_xla[tpu]==$TORCH_VER" \
  --progress-bar off -f https://storage.googleapis.com/libtpu-releases/index.html

echo "=== [4/6] Clone / update repo ($REPO) ==="
if [ ! -d "$REPO/.git" ]; then
  git clone "$REPO_URL" "$REPO"
else
  git -C "$REPO" pull --ff-only
fi

echo "=== [5/6] Ragnarok dependencies ==="
# Core deps required for the primary pair (cartpole -> mountaincar-continuous)
# and the analysis pipeline. NOT torch (installed above with torch_xla).
"$VENV/bin/pip" install --progress-bar off \
  "gymnasium[classic-control]>=0.29.0" tensorboard pytest lifelines
# DMControl deps for the secondary pairs (cartpole-swingup etc.) — best
# effort; the primary-pair calibration does not need them.
"$VENV/bin/pip" install --progress-bar off dm_control 2>/dev/null \
  && echo "  dm_control installed" \
  || echo "  dm_control skipped (not needed for the primary pair)"
# Install the ragnarok package itself, editable, without re-resolving deps
# (they are already installed above; --no-deps avoids pulling a second torch).
"$VENV/bin/pip" install -e "$REPO" --no-deps --progress-bar off

echo "=== [6/6] Hello-TPU test ==="
"$VENV/bin/python" -c "
import torch, torch_xla
import torch_xla.core.xla_model as xm
dev = xm.xla_device()
t = torch.randn(256, 256, device=dev)
r = (t @ t).sum().item()
print(f'torch {torch.__version__} | torch_xla {torch_xla.__version__} | device {dev} | matmul {r:.1f}')
import ragnarok
from ragnarok.infrastructure.device import DEVICE, IS_XLA
print(f'ragnarok DEVICE={DEVICE} IS_XLA={IS_XLA}')
assert IS_XLA, 'expected XLA device to be active on a TPU VM'
print('SETUP_TPU_OK')
"
echo ""
echo "Setup complete. Activate with:  source $VENV/bin/activate"
