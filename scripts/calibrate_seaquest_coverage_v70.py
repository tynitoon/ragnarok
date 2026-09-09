"""Development-only Seaquest coverage geometry; collection scores never pooled.

The additional criteria were declared in the coverage protocol before17331.
Reserved execution remains disabled pending independent review and a complete
source/commit seal. A coverage count alone never establishes calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.seaquest_v70 import KINDS, REVISION, decode_seaquest_ram, screen_seaquest_objects
from scripts.calibrate_breakout_v70 import compare_boxes
from scripts.calibrate_seaquest_v70 import SOURCE_PATHS as RAW_SOURCES
from scripts.capture_seaquest_coverage_v70 import SOURCE_PATHS as CAPTURE_SOURCES

SOURCE_PATHS = tuple(dict.fromkeys((*RAW_SOURCES, *CAPTURE_SOURCES,
                                  "scripts/calibrate_seaquest_coverage_v70.py")))


def new_stats():
    return dict(frames=0, presence_count_agreement=0, positive_frames=0, positive_count_agreement=0,
                ram_objects=0, pixel_objects=0, joint_matches=0, boxes_within1=0, boxes_exact=0)


def accumulate(s, a, b):
    n, within, exact = compare_boxes(a, b)
    s["frames"] += 1
    s["presence_count_agreement"] += len(a) == len(b)
    if a or b:
        s["positive_frames"] += 1
        s["positive_count_agreement"] += len(a) == len(b)
    s["ram_objects"] += len(a)
    s["pixel_objects"] += len(b)
    s["joint_matches"] += n
    s["boxes_within1"] += within
    s["boxes_exact"] += exact
    return n, within


def finish(s):
    s["presence_fraction"] = s["presence_count_agreement"]/s["frames"] if s["frames"] else None
    s["positive_presence_fraction"] = s["positive_count_agreement"]/s["positive_frames"] if s["positive_frames"] else None
    s["box_fraction"] = s["boxes_within1"]/s["joint_matches"] if s["joint_matches"] else None
    # These values were declared in the pre-capture coverage protocol text.
    s["passed"] = (s["joint_matches"] >= 50 and s["presence_fraction"] >= .99 and
                   s["positive_presence_fraction"] is not None and s["positive_presence_fraction"] >= .99 and
                   s["box_fraction"] >= .99)


def calibrate(directory, original_result):
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root/"docs/ARC4_SEAQUEST_COVERAGE_PROTOCOL.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    expected = dict(game="seaquest", seed=protocol["development_seed"], frames=protocol["frames"],
                    action_repeat=protocol["action_repeat"], mode=0, raw_every_real_frame=True,
                    action_schedule="instrumentation_rgb_controller_v1", episode_limit=protocol["episode_limit"],
                    sticky_probability=.25, sticky_implementation="worker-python-random-per-frame",
                    reset_convention="fixed_hardware_cold_start_v1", agent_training_allowed=False,
                    reference_training_allowed=False, agent_entry_states_allowed=False)
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Outside the declared development coverage; reserved execution is disabled")
    hashes = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if manifest.get("source_sha256") != {p: hashes[p] for p in CAPTURE_SOURCES}:
        raise ValueError("Coverage capture sources differ from the pre-capture controller/protocol")
    original = json.loads(original_result.read_text(encoding="utf-8"))
    if (original.get("seed") != 17311 or original.get("kind") != "independent_seaquest_raw_calibration" or
            original.get("source_sha256") != {p: hashes[p] for p in RAW_SOURCES}):
        raise ValueError("Original development must be scored with this exact candidate/checker")
    stats = {k: new_stats() for k in (*KINDS, "surface_submarine")}
    examples = {k: [] for k in stats}
    for index, (rgb, ram, episode, _) in enumerate(cold_fixture_frames(directory), 1):
        truth, pixels = decode_seaquest_ram(ram), screen_seaquest_objects(rgb)
        for kind, s in stats.items():
            if kind == "surface_submarine":
                a = [o for o in truth if o.slot == "surface"]
                b = [o for o in pixels if o.kind == "submarine" and o.box[1] < 57]
            else:
                a, b = [o for o in truth if o.kind == kind], [o for o in pixels if o.kind == kind]
            n, within = accumulate(s, a, b)
            if (len(a) != len(b) or n != within) and len(examples[kind]) < 8:
                examples[kind].append(dict(frame=index, episode=episode, ram=[o.box for o in a], pixels=[o.box for o in b]))
        if index % 4096 == 0:
            print(f"geometry {index}/{manifest['frames']}", flush=True)
    for s in stats.values():
        finish(s)
    original_agreement = {
        kind: s["presence_fraction"] >= .99 and (s["joint_matches"] == 0 or s["box_fraction"] >= .99)
        for kind, s in original["per_type"].items()
    }
    if hashes != {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}:
        raise RuntimeError("Scoring sources changed during the run")
    return dict(kind="seaquest_development_coverage_geometry", seed=manifest["seed"], decoder_revision=REVISION,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), per_type=stats,
                original_collection_agreement=original_agreement,
                original_collection_calibration_pass=original["calibration_pass"],
                coverage_collection_pass=all(s["passed"] for s in stats.values()),
                combined_development_pass=all(original_agreement.values()) and all(s["passed"] for s in stats.values()),
                first_disagreements=examples, source_sha256=hashes,
                original_result_sha256=hashlib.sha256(original_result.read_bytes()).hexdigest(),
                fixture_manifest_sha256=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
                scope="Separate per-collection fractions. Empty original types remain uncalibrated there; all types and the surface subtype require50joint instance-frame matches in the additional collection. No pooled dilution.",
                agent_training_allowed=False, gate_result=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--original-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve prior results; choose a new output")
    result = calibrate(args.fixture, args.original_result)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("combined_development_pass", "per_type")}, indent=2))
