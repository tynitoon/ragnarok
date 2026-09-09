"""Report raw actor coverage independently of RGB calibration agreement."""
import argparse
import hashlib
import json
from pathlib import Path

from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.seaquest_v70 import KINDS, decode_seaquest_ram


def coverage(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest["game"] != "seaquest":
        raise ValueError("Seaquest fixture required")
    names = (*KINDS, "surface_submarine")
    stats = {k: dict(frames=0, instances=0, episodes=set(), first_frame=None) for k in names}
    levels, surface_levels = set(), set()
    for index, (_, ram, episode, _) in enumerate(cold_fixture_frames(directory), 1):
        labels = decode_seaquest_ram(ram)
        levels.add(int(ram[61]))
        surface_levels.add(int(ram[60]))
        for kind in names:
            n = sum(o.slot == "surface" if kind == "surface_submarine" else o.kind == kind for o in labels)
            if n:
                s = stats[kind]
                s["frames"] += 1
                s["instances"] += n
                s["episodes"].add(episode)
                if s["first_frame"] is None:
                    s["first_frame"] = index
    for s in stats.values():
        s["episodes"] = sorted(s["episodes"])
        s["minimum_50_instances_observed"] = s["instances"] >= 50
    root = Path(__file__).resolve().parents[1]
    sources = ("scripts/check_seaquest_coverage_v70.py", "ragnarok/evaluation/seaquest_v70.py",
               "ragnarok/evaluation/cold_fixtures_v70.py", "docs/ARC4_SEAQUEST_COVERAGE_PROTOCOL.json")
    return dict(kind="seaquest_raw_ram_coverage_only", seed=manifest["seed"], frames=manifest["frames"],
                charged_frames=manifest["accounting"]["charged_frames"], per_type=stats,
                progression_ram61=sorted(levels), surface_unlock_ram60=sorted(surface_levels),
                intended_coverage_observed=all(s["minimum_50_instances_observed"] for s in stats.values()),
                source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
                fixture_manifest_sha256=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
                scope="Counts of RAM-indicated visible actors only. Instances sums objects over frames, not independent lifetimes. No RGB agreement, no geometry pass, no agent result",
                calibration_pass=None, agent_training_allowed=False, gate_result=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve prior results; choose a new output")
    result = coverage(args.fixture)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
