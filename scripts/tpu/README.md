# TPU provisioning & setup

Scripts and recipes for running Ragnarok on Google Cloud TPU VMs via the
TPU Research Cloud (TRC) allocation.

## Why this exists

Ragnarok's TRC allocation includes on-demand v4 (stable, but queued) and
spot v5e/v6e (fast to provision, but **preemptible** — Google can reclaim
them at any time). The first spot v5e was preempted ~25 minutes after
creation. Nothing on a TPU VM is load-bearing: the code lives in git,
results are pushed to git, so a preemption just means "recreate + re-run
`setup_tpu.sh`".

## One-time prerequisites

- `gcloud` CLI authenticated (`gcloud auth login`), project set to
  `ragnarok-rl-research`.
- A VPC subnet in the target region. `us-central2` (v4 zone) is not covered
  by the default auto-mode network — a manual subnet `ragnarok-uc2` was
  created there. `europe-west4` (v5e/v6e zones) has a default subnet.

## Provisioning a TPU VM

### Option A — on-demand v4 (stable, may queue for a long time)

```bash
gcloud compute tpus queued-resources create ragnarok-qr-v4 \
  --node-id=ragnarok-tpu-v4 --zone=us-central2-b \
  --accelerator-type=v4-8 --runtime-version=tpu-vm-v4-pt-2.0 \
  --network=default --subnetwork=ragnarok-uc2 \
  --project=ragnarok-rl-research
```

### Option B — spot v5e (fast, preemptible)

```bash
gcloud compute tpus queued-resources create ragnarok-qr-v5e \
  --node-id=ragnarok-tpu-v5e --zone=europe-west4-b \
  --accelerator-type=v5litepod-1 --runtime-version=v2-alpha-tpuv5-lite \
  --network=default --subnetwork=default --spot \
  --project=ragnarok-rl-research
```

### Poll until ACTIVE

```bash
gcloud compute tpus queued-resources describe <QR_NAME> \
  --zone=<ZONE> --format="value(state.state)"
# WAITING_FOR_RESOURCES -> PROVISIONING -> ACTIVE
```

## Software setup (after ACTIVE)

```bash
gcloud compute tpus tpu-vm ssh <NODE_ID> --zone=<ZONE> \
  --command="git clone https://gitlab.com/mortier.jeremie/ragnarok.git ~/ragnarok 2>/dev/null; bash ~/ragnarok/scripts/tpu/setup_tpu.sh"
```

`setup_tpu.sh` is idempotent — installs python3-venv, creates the venv,
installs PyTorch/XLA (torch + torch_xla pinned to the same version),
clones/updates the repo, installs Ragnarok's dependencies, and runs a
hello-TPU smoke test. Expect ~5-10 minutes (mostly the torch download).

## Running work on the TPU VM

```bash
gcloud compute tpus tpu-vm ssh <NODE_ID> --zone=<ZONE> \
  --command="cd ~/ragnarok && ~/ragnarok-venv/bin/python -m scripts.pilot_run ..."
```

The Ragnarok code is portable: `infrastructure/device.py` auto-detects XLA
and the training loops call `mark_step()` (no-op off-TPU). No code change
is needed between the local CUDA machine and the TPU VM.

## Discipline: delete TPU VMs when idle

A running TPU VM (even idle) consumes TRC quota. The TRC welcome email
explicitly asks to delete unused Cloud TPUs and Queued Resources.

```bash
gcloud compute tpus queued-resources delete <QR_NAME> --zone=<ZONE> --quiet --force
```

## Windows / gcloud notes

- gcloud on Windows uses PuTTY's `plink` for SSH, not OpenSSH. Do **not**
  pass OpenSSH-style `-o` flags — plink rejects them. The first SSH prompts
  to cache the host key; answer `y` once.
- The SDK needs a working Python. If `gcloud` reports a `WindowsApps/python3`
  permission error, the bundled Python under
  `…\Cloud SDK\google-cloud-sdk\platform\bundledpython\` works, or reinstall
  a current Cloud SDK (it bundles its own Python).
