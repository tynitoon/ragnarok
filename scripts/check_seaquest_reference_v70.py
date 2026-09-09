"""Causal reference diagnostic on a fresh uniform Seaquest capture, never a teacher."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.seaquest_v70 import KINDS,decode_seaquest_ram
from ragnarok.evaluation.scene_perception_v70 import SceneObject,ScenePerceptionScorer
from scripts.ref_extract_v70 import ReferenceExtractor
from scripts.seaquest_raw_prerequisite_v70 import RAW_SEALED_SOURCES,require_raw_calibration

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL="docs/ARC4_SEAQUEST_REFERENCE_PROTOCOL.json"
SOURCE_PATHS=tuple(dict.fromkeys((*RAW_SEALED_SOURCES,PROTOCOL,"scripts/check_seaquest_reference_v70.py",
    "scripts/seaquest_raw_prerequisite_v70.py","scripts/ref_extract_v70.py",
    "ragnarok/evaluation/scene_perception_v70.py","ragnarok/evaluation/perception_v70.py")))


def check(directory):
    proof=require_raw_calibration()
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding="utf-8"))
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    expected=dict(game="seaquest",seed=protocol["seed"],action_seed=protocol["seed"]+100000,
                  frames=protocol["frames"],mode=0,action_repeat=16,episode_limit=108000,
                  sticky_probability=.25,sticky_implementation="worker-python-random-per-frame",
                  reset_convention="fixed_hardware_cold_start_v1",raw_every_real_frame=True,agent_training_allowed=False)
    if any(manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError("Reference requires its fresh declared ordinary capture, never calibration coverage")
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    capture_sources=("scripts/capture_cold_v70.py","ragnarok/environments/emu_pool.py",
                     "ragnarok/environments/frame_budget_v70.py","ragnarok/environments/cold_emu_pool_v70.py")
    if manifest["source_sha256"]!={p:hashes[p] for p in capture_sources}:
        raise ValueError("Fresh capture source provenance differs")
    reference=ReferenceExtractor()
    scorer=ScenePerceptionScorer(roi=tuple(protocol["roi_xyxy"]))
    decode_seconds=reference_seconds=score_seconds=0.0
    started=time.perf_counter()
    for frame,(rgb,ram,episode,_) in enumerate(cold_fixture_frames(directory)):
        tick=time.perf_counter()
        truth=[SceneObject(f"{o.kind}:{o.slot}",o.kind,o.box,True) for o in decode_seaquest_ram(ram)]
        decode_seconds+=time.perf_counter()-tick
        tick=time.perf_counter()
        boxes=reference.observe(rgb)  # Only RGB, continuous causal history.
        reference_seconds+=time.perf_counter()-tick
        tick=time.perf_counter()
        scorer.add(truth,boxes,frame=frame,episode=episode)
        score_seconds+=time.perf_counter()-tick
    elapsed=time.perf_counter()-started
    if hashes!={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}:
        raise RuntimeError("Reference diagnostic sources changed during execution")
    metrics=scorer.report()
    for kind in KINDS:
        metrics["per_type"].setdefault(kind,dict(visible=0,eligible=0,recalled=0,recall_moving=None))
    return dict(kind="seaquest_raw_reference_development_diagnostic",metrics=metrics,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                versions={p:importlib.metadata.version(p) for p in ("numpy","scipy")},
                source_sha256=hashes,raw_calibration_proof_sha256=proof,
                fixture_manifest_sha256=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
                decode_seconds=decode_seconds,reference_seconds=reference_seconds,score_seconds=score_seconds,
                wall_seconds=elapsed,measured_raw_frames_per_second=metrics["frames"]/(decode_seconds+reference_seconds+score_seconds),
                oracle_reset_control=False,agent_training=False,gate_result=None,
                limitations=["Fresh ordinary raw capture only; no emulator/IPC/branches in timing.",
                             "RAM allocation slots and visible box-center motion are not independently tracked lifetimes.",
                             "All frames included, no reference training on prior coverage or reserved fixtures.",
                             "Reference receives no oracle inputs and is never a learned-parser teacher."])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Preserve previous diagnostics")
    result=check(args.fixture)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ("metrics","measured_raw_frames_per_second")},indent=2))
