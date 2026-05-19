#!/bin/bash
# mount_venv_disk.sh — mount the persistent venv disk at /mnt/venv on a
# fresh spot TPU VM (one created with --data-disk source=.../ragnarok-venv).
#
# Fast setup: the PyTorch/XLA venv is already on the disk (see
# populate_venv_disk.sh), so there is no pip install — setup is ~1 min
# (mount + git clone) instead of ~8, which makes short spot windows usable.
#
# Use:  git clone <repo> ~/ragnarok && bash ~/ragnarok/scripts/tpu/mount_venv_disk.sh \
#         && cd ~/ragnarok && /mnt/venv/ragnarok-venv/bin/python -m scripts.<X>
set -e
MNT=/mnt/venv
DEV=$(ls /dev/disk/by-id/google-* 2>/dev/null \
        | grep -v -- '-part[0-9]' | grep -v 'persistent-disk-0$' | head -1)
if [ -z "$DEV" ]; then echo "ERROR: venv disk not attached"; lsblk; exit 1; fi
sudo mkdir -p "$MNT"
mountpoint -q "$MNT" || sudo mount "$DEV" "$MNT"
test -x "$MNT/ragnarok-venv/bin/python" \
  || { echo "ERROR: venv missing on disk ($MNT/ragnarok-venv)"; exit 1; }
echo "venv disk mounted: $DEV -> $MNT"
echo "python: $MNT/ragnarok-venv/bin/python"
