#!/bin/bash
# populate_venv_disk.sh — one-time: install the PyTorch/XLA venv onto the
# attached persistent disk `ragnarok-venv`.
#
# Run ON a TPU VM that has the persistent disk attached read-write:
#   gcloud compute tpus tpu-vm attach-disk <node> --disk=ragnarok-venv \
#       --mode=read-write --zone=europe-west4-b
#
# After this, a fresh spot TPU VM just mounts the disk and uses the venv
# directly — setup drops from ~8 min (pip install torch_xla) to ~1 min
# (mount + git clone). Spot preemption windows become usable.
#
# Idempotent: safe to re-run if a populate was interrupted (a preempted
# spot VM mid-install) — pip resumes, the format step is skipped.
set -e

MNT=/mnt/venv
VENV="$MNT/ragnarok-venv"
TORCH_VER="2.9.0"          # torch and torch_xla MUST match exactly

echo "=== [1/5] Locate the attached data disk ==="
# GCP exposes persistent disks at /dev/disk/by-id/google-*. The boot disk
# is google-persistent-disk-0; the data disk is the other whole-disk entry
# (drop -partN partitions and the boot disk).
DEV=$(ls /dev/disk/by-id/google-* 2>/dev/null \
        | grep -v -- '-part[0-9]' | grep -v 'persistent-disk-0$' | head -1)
if [ -z "$DEV" ]; then echo "ERROR: no attached data disk found"; lsblk; exit 1; fi
echo "  data disk: $DEV"

echo "=== [2/5] Format (if blank) + mount at $MNT ==="
if ! sudo blkid "$DEV" >/dev/null 2>&1; then
  echo "  blank disk -> mkfs.ext4"
  sudo mkfs.ext4 -F "$DEV"
fi
sudo mkdir -p "$MNT"
mountpoint -q "$MNT" || sudo mount "$DEV" "$MNT"
sudo chown -R "$USER:$USER" "$MNT"

echo "=== [3/5] System python3-venv ==="
sudo apt-get update -qq
sudo apt-get install -y python3.10-venv >/dev/null 2>&1

echo "=== [4/5] venv + PyTorch/XLA $TORCH_VER + deps on the disk ==="
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install "torch==$TORCH_VER" "torch_xla[tpu]==$TORCH_VER" \
  --progress-bar off -f https://storage.googleapis.com/libtpu-releases/index.html
"$VENV/bin/pip" install --progress-bar off \
  "gymnasium[classic-control]>=0.29.0" tensorboard pytest lifelines
"$VENV/bin/pip" install --progress-bar off dm_control 2>/dev/null \
  && echo "  dm_control installed" || echo "  dm_control skipped"

echo "=== [5/5] Smoke test ==="
"$VENV/bin/python" -c "
import torch, torch_xla
print(f'venv OK | torch {torch.__version__} | torch_xla {torch_xla.__version__}')"
mkdir -p "$MNT/checkpoints"
sync
sudo umount "$MNT"
echo "POPULATE_DONE  (venv -> $VENV ; checkpoints dir -> $MNT/checkpoints)"
