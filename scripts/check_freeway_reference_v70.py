"""Pixels-only reference diagnostic against the independently calibrated Freeway judge."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.perception_v70 import PerceptionScorer
from ragnarok.evaluation.freeway_v70 import decode_freeway_ram
from scripts.ref_extract_v70 import ReferenceExtractor


def check(directory, calibration):
    root=Path(__file__).resolve().parents[1]
    recorded=json.loads(calibration.read_text())
    if recorded.get("kind")!="independent_freeway_raw_calibration" or recorded.get("calibration_pass") is not True:
        raise ValueError("Reference requires a passed independent Freeway calibration")
    from scripts.calibrate_freeway_v70 import SOURCE_PATHS
    if recorded.get("seed")!=17212 or set(recorded["source_sha256"])!=set(SOURCE_PATHS):
        raise ValueError("Expected reserved Freeway calibration with the complete source registry")
    for p,h in recorded["source_sha256"].items():
        if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h:
            raise ValueError("Sources differ from the independent calibration")
    fixture_hash=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest()
    if fixture_hash!=recorded["fixture_manifest_sha256"]:
        raise ValueError("Calibration and reference must use the same fixture")
    protocol=json.loads((root/"docs/ARC4_FREEWAY_REFERENCE_PROTOCOL.json").read_text())
    reference=ReferenceExtractor()
    scorer=PerceptionScorer(roi=tuple(protocol["roi_xyxy"]))
    decode_seconds=reference_seconds=score_seconds=0.0
    start=time.perf_counter()
    for i,(rgb,ram,episode,charged) in enumerate(cold_fixture_frames(directory)):
        tic=time.perf_counter()
        truth=decode_freeway_ram(ram)
        decode_seconds+=time.perf_counter()-tic
        tic=time.perf_counter()
        boxes=reference.observe(rgb)  # the only input; no reset signal, truth or labels
        reference_seconds+=time.perf_counter()-tic
        tic=time.perf_counter()
        scorer.add(truth,boxes,frame=i,episode=episode)
        score_seconds+=time.perf_counter()-tic
    elapsed=time.perf_counter()-start
    sources=("scripts/check_freeway_reference_v70.py","scripts/ref_extract_v70.py",
             "ragnarok/evaluation/perception_v70.py",
             "ragnarok/evaluation/freeway_v70.py","ragnarok/evaluation/cold_fixtures_v70.py",
             "docs/ARC4_FREEWAY_REFERENCE_PROTOCOL.json")
    metrics=scorer.report()
    return dict(kind="freeway_raw_reference_development_diagnostic",metrics=metrics,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),fixture_manifest_sha256=fixture_hash,
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
                versions={p:importlib.metadata.version(p) for p in ("numpy","scipy")},
                decode_seconds=decode_seconds,reference_seconds=reference_seconds,score_seconds=score_seconds,
                wall_seconds=elapsed,measured_raw_frames_per_second=metrics["frames"]/(decode_seconds+reference_seconds+score_seconds),
                including_fixture_io_raw_frames_per_second=metrics["frames"]/elapsed,
                oracle_reset_control=False,gate_result=None,agent_training=False,
                limitations=["Raw Freeway only, not pooled observations or a gate.",
                             "Reference is never a learned-parser teacher and receives only ordinary RGB.",
                             "All frames included; judge motion reset stays separate from reference history.",
                             "The second chicken and each lane car have distinct identities; stationary visible objects count for precision.",
                             "Timing excludes emulator, IPC, branch generation and learned parser."])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--calibration",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Keep prior diagnostics; choose a new output")
    result=check(args.fixture,args.calibration)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({"metrics":result["metrics"],"measured_raw_frames_per_second":result["measured_raw_frames_per_second"]},indent=2))
