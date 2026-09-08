"""Scorer-only development audit of ALE reset counters; does not enable new games.

Costs are derived from the reviewed ALE 0.12.1 source. Observed selection counts
are diagnostics, never a proof of a bound for admitting future resets.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time


# ROM MD5, reset base, extra real frames per pressSelect iteration.
DEVELOPMENT = {
    "breakout": ("f34f08e5eb96e500e851a80be3277a56", 72, 2),
    "freeway": ("8e0ab801b1705a740b476b7f588c6d16", 72, 2),
    "seaquest": ("240bfbac5163af4df5ae713985386f92", 68, 0),
    "ms_pacman": ("87e79cd41ce136fd4f72cc6e2c161bee", 72, 11),
    "asteroids": ("ccbd36746ed4525821a8083b0d6d2c2c", 72, 3),
}
SEEDS = (21000, 21001, 21002)


def audit(game):
    import ale_py
    from ale_py.roms import get_rom_path
    if ale_py.__version__ != "0.12.1":
        raise ValueError("This source-based audit requires ALE 0.12.1")
    expected, base, per_select = DEVELOPMENT[game]
    rom = get_rom_path(game)
    if hashlib.md5(rom.read_bytes()).hexdigest() != expected:
        raise ValueError("ROM differs from the source-audited development ROM")
    records = []
    for seed in SEEDS:
        ale = ale_py.ALEInterface()
        ale.setInt("random_seed", seed)
        ale.setInt("frame_skip", 1)
        ale.setInt("max_num_frames_per_episode", 256)
        ale.setFloat("repeat_action_probability", 0.0)
        for name in ("color_averaging", "display_screen", "sound"):
            ale.setBool(name, False)
        ale.loadROM(rom)
        if ale.getMode() != 0:
            raise ValueError("Audit expects default public mode 0; no implicit mode change")
        rng = random.Random(seed + 100_000)
        actions = list(ale.getLegalActionSet())
        played_before_reset = 0
        before = 0
        for reset_index in range(5):
            if reset_index:
                before = ale.getFrameNumber()
                ale.reset_game()
            selections = ale.getFrameNumber() - before
            if selections < 0 or (per_select == 0 and selections != 0):
                raise ValueError("Unexpected reset frame getter; review source formula")
            records.append(dict(seed=seed, reset_index=reset_index,
                                getter_selection_count=selections,
                                source_derived_reset_frames=base + per_select * selections,
                                played_before_reset=played_before_reset,
                                episode_frame_after_reset=ale.getEpisodeFrameNumber(),
                                lives_after_reset=ale.lives(), public_mode=ale.getMode(),
                                ram_sha256=hashlib.sha256(ale.getRAM().tobytes()).hexdigest(),
                                rgb_sha256=hashlib.sha256(ale.getScreenRGB().tobytes()).hexdigest()))
            played_before_reset = 0
            # Fixed development perturbation; includes both partial and terminal episodes.
            for i in range((32, 127, 256, 256)[reset_index % 4] if reset_index < 4 else 0):
                if ale.game_over():
                    break
                if i % 16 == 0:
                    action = rng.choice(actions)
                previous = ale.getEpisodeFrameNumber()
                ale.act(action)
                delta = ale.getEpisodeFrameNumber() - previous
                if delta != 1:
                    raise ValueError("Real frame accounting diverged")
                played_before_reset += 1
        del ale
    root = Path(__file__).resolve().parents[1]
    return dict(kind="development_ale_reset_source_counter_audit", game=game,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                rom_md5=expected, ale_version=importlib.metadata.version("ale-py"),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256((root/"docs/ARC4_RESET_AUDIT_PROTOCOL.json").read_bytes()).hexdigest(),
                records=records,
                total_source_derived_frames=sum(r["source_derived_reset_frames"] + r["played_before_reset"] for r in records),
                agent_training_allowed=False, runtime_activation_allowed=False,
                limitation="Finite samples establish observed costs only, not a worst-case bound.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=tuple(DEVELOPMENT), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Keep previous audits; choose a new output path")
    result = audit(args.game)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"game":args.game, "observed_selection_counts":sorted({r["getter_selection_count"] for r in result["records"]}),
                      "total_source_derived_frames":result["total_source_derived_frames"]}))
