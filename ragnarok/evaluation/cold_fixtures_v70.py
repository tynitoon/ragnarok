"""Strict scorer-only reader for raw fixed-hardware episode fixtures."""
import hashlib
import json
from pathlib import Path

import numpy as np

from ragnarok.environments.cold_emu_pool_v70 import ROM_SPECS, CONVENTION


def cold_fixture_frames(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("kind") != "scorer_only_cold_raw_development_fixture" or
            manifest.get("agent_training_allowed") is not False or
            manifest.get("reset_convention") != CONVENTION or manifest.get("game") not in ROM_SPECS or
            manifest.get("raw_every_real_frame") is not True):
        raise ValueError("Not a declared scorer-only cold raw fixture")
    bootstrap = ROM_SPECS[manifest["game"]][1]
    seen, previous_episode, previous_charge = 0, 0, bootstrap
    paths = set()
    for shard in manifest["shards"]:
        name = shard["path"]
        if Path(name).name != name or name in paths:
            raise ValueError("Shard paths must be unique local filenames")
        paths.add(name)
        path = directory/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError("Fixture shard hash mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            rgb,ram = arrays["rgb"],arrays["ram"]
            episodes,charges,actions = arrays["episode"],arrays["charged_frame"],arrays["action"]
        n = shard["frames"]
        if (rgb.shape!=(n,210,160,3) or rgb.dtype!=np.uint8 or ram.shape!=(n,128) or ram.dtype!=np.uint8 or
                episodes.shape!=(n,) or episodes.dtype!=np.int32 or charges.shape!=(n,) or charges.dtype!=np.int64 or
                actions.shape!=(n,) or actions.dtype!=np.uint8 or np.any(actions>=18)):
            raise ValueError("Malformed fixture arrays")
        for i in range(n):
            episode,charge = int(episodes[i]),int(charges[i])
            jump = episode-previous_episode
            if jump not in (0,1) or (seen==0 and jump!=0) or charge-previous_charge!=1+bootstrap*jump:
                raise ValueError("Noncontiguous real frame or incomplete cold-start charge")
            seen += 1
            previous_episode,previous_charge = episode,charge
            yield rgb[i],ram[i],episode,charge
    ledger = manifest["accounting"]
    if (seen!=manifest["frames"] or previous_charge!=ledger["charged_frames"] or
            ledger["cold_loads"]!=previous_episode+1 or ledger["experiment_frames"] or ledger["restores"] or
            ledger["format_detection_emulated_frames"]!=60*(previous_episode+1)):
        raise ValueError("Fixture length or final ledger is inconsistent")
