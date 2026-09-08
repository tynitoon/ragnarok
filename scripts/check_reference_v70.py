"""Reference-only development diagnostic after independent calibration; never a gate."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from ragnarok.evaluation.fixtures_v70 import fixture_frames
from ragnarok.evaluation.pong_v70 import decode_pong_ram
from ragnarok.evaluation.perception_v70 import PerceptionScorer
from scripts.ref_extract_v70 import ReferenceExtractor


def check(directory, calibration, *, oracle_reset_control=False):
    root = Path(__file__).resolve().parents[1]
    recorded = json.loads(calibration.read_text(encoding="utf-8"))
    if recorded.get("kind") != "development_pong_ram_calibration" or recorded.get("calibration_pass") is not True:
        raise ValueError("Reference scoring requires a successful independent calibration")
    expected = recorded["source_sha256"]["ragnarok/evaluation/pong_v70.py"]
    if hashlib.sha256((root / "ragnarok/evaluation/pong_v70.py").read_bytes()).hexdigest() != expected:
        raise ValueError("Decoder differs from calibrated source")
    if hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest() != recorded["fixture_manifest_sha256"]:
        raise ValueError("Calibration and reference fixture must match")
    reference, scorer = ReferenceExtractor(), PerceptionScorer()
    episode = None
    decode_seconds = reference_seconds = score_seconds = 0.0
    started = time.perf_counter()
    for frame in fixture_frames(directory):
        tic = time.perf_counter()
        truths = decode_pong_ram(frame["ram"])
        decode_seconds += time.perf_counter() - tic
        tic = time.perf_counter()
        if oracle_reset_control and frame["episode"] != episode:
            reference.reset()
            episode = frame["episode"]
        boxes = reference.observe(frame["rgb"])
        reference_seconds += time.perf_counter() - tic
        tic = time.perf_counter()
        scorer.add(truths, boxes, frame=frame["index"], episode=frame["episode"])
        score_seconds += time.perf_counter() - tic
    wall_seconds = time.perf_counter() - started
    report = scorer.report()
    measured = decode_seconds + reference_seconds + score_seconds
    sources = ["scripts/ref_extract_v70.py", "scripts/check_reference_v70.py", "ragnarok/evaluation/perception_v70.py",
               "ragnarok/evaluation/pong_v70.py", "ragnarok/evaluation/fixtures_v70.py"]
    return dict(kind="pong_raw_reference_development_diagnostic", metrics=report,
                oracle_reset_control=oracle_reset_control,
                calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
                fixture_manifest_sha256=recorded["fixture_manifest_sha256"],
                source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
                versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
                decode_seconds=decode_seconds, reference_seconds=reference_seconds, score_seconds=score_seconds,
                wall_seconds=wall_seconds, measured_raw_frames_per_second=report["frames"]/measured,
                including_fixture_io_raw_frames_per_second=report["frames"]/wall_seconds,
                proposed_reference_recall_bar_met=(report["recall_moving"] is not None and report["recall_moving"] >= .9),
                gate_result=None, agent_training=False,
                limitations=["Only calibrated Pong raw frames; not max-pooled observation evaluation.",
                             "Every observed frame counted, including cold background and transition errors.",
                             "Oracle episode reset is a separate labelled control, disabled by default.",
                             "No learned parser, agency, drive, reader, identity or transfer result.",
                             "Timing excludes emulator, IPC and branch generation; not the gate throughput."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--oracle-reset-control", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Keep the old diagnostic; choose a new path")
    result = check(args.fixtures, args.calibration, oracle_reset_control=args.oracle_reset_control)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
