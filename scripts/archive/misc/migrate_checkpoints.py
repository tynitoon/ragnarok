"""One-shot migration for pre-Phase-5.3 checkpoints to schema v2.

Pre-5.3 ckpts had an `actor_critic` key (separate ActorCritic class) and
lacked `acting_policy_mode` / `latent_policy`. Phase 5.3 unified the
policy into a single DirectPolicyNet/ContinuousPolicyNet, and Phase 1.1
added the `acting_policy_mode` switch for latent-trunk transfer.

Usage:
    python -m scripts.migrate_checkpoints path/to/ckpt.pt
    python -m scripts.migrate_checkpoints checkpoints/  # migrate a directory

The script renames `actor_critic` → `policy` and stamps the v2 marker.
It does NOT synthesize a `latent_policy` state dict — there's no way to
recover one that didn't exist. The migrated ckpt is loadable only if the
caller's latent_policy head has the right dims, otherwise agent.load()
will error on the state-dict shape check. Migration is best-effort: the
old ckpt's architecture may no longer match the current code; in that
case delete and re-train.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


TARGET_SCHEMA_VERSION = 2


def migrate_one(path: Path, dry_run: bool = False) -> str:
    """Migrate a single .pt file in place. Returns a status line."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return f"SKIP {path}: cannot load ({e})"

    if not isinstance(ckpt, dict):
        return f"SKIP {path}: not a dict checkpoint"

    existing_version = ckpt.get("schema_version")
    if existing_version == TARGET_SCHEMA_VERSION:
        return f"OK   {path}: already v{TARGET_SCHEMA_VERSION}"

    changed = []

    if "policy" not in ckpt and "actor_critic" in ckpt:
        ckpt["policy"] = ckpt.pop("actor_critic")
        changed.append("actor_critic→policy")

    if "acting_policy_mode" not in ckpt:
        ckpt["acting_policy_mode"] = "obs"
        changed.append("acting_policy_mode=obs")

    if "latent_policy" not in ckpt:
        # No way to reconstruct a latent_policy head that never existed.
        # Leave it absent — agent.load() will surface this as a clear
        # missing-key error rather than silently skip.
        changed.append("latent_policy=MISSING")

    ckpt["schema_version"] = TARGET_SCHEMA_VERSION
    changed.append(f"schema_version=v{TARGET_SCHEMA_VERSION}")

    if dry_run:
        return f"DRY  {path}: would update ({', '.join(changed)})"

    torch.save(ckpt, path)
    return f"MIG  {path}: {', '.join(changed)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path,
                        help="Checkpoint files or directories to migrate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    args = parser.parse_args(argv)

    targets: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            targets.extend(sorted(p.rglob("*.pt")))
        elif p.exists():
            targets.append(p)
        else:
            print(f"WARN {p}: does not exist")

    if not targets:
        print("No .pt files found.")
        return 1

    for t in targets:
        print(migrate_one(t, dry_run=args.dry_run))

    return 0


if __name__ == "__main__":
    sys.exit(main())
