"""Development-only Ms. Pac-Man geometry; reserved collection remains disabled."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from scripts.calibrate_breakout_v70 import compare_boxes,SOURCE_PATHS as SHARED_SOURCES
from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.mspacman_v70 import KINDS,REVISION,decode_mspacman_ram,screen_mspacman_objects

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL="docs/ARC4_MSPACMAN_RAW_PROTOCOL.json"
SOURCE_PATHS=(*SHARED_SOURCES,PROTOCOL,"ragnarok/evaluation/mspacman_v70.py","scripts/calibrate_mspacman_v70.py")


def calibrate(directory):
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding="utf-8"))
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    expected=dict(game="ms_pacman",seed=protocol["development_seed"],frames=protocol["frames"],
                  mode=0,action_repeat=16,episode_limit=108000,action_seed=protocol["development_seed"]+100000,
                  sticky_probability=.25,sticky_implementation="worker-python-random-per-frame",
                  reset_convention="fixed_hardware_cold_start_v1",raw_every_real_frame=True,
                  agent_training_allowed=False,gate_result=None)
    if any(manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError("Only the prospective development fixture is permitted; reserved is disabled")
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    capture_sources=("scripts/capture_cold_v70.py","ragnarok/environments/emu_pool.py",
                     "ragnarok/environments/frame_budget_v70.py","ragnarok/environments/cold_emu_pool_v70.py")
    if manifest.get("source_sha256")!={p:hashes[p] for p in capture_sources}:
        raise ValueError("Capture source provenance differs")
    stats={k:dict(frames=0,presence_count_agreement=0,positive_frames=0,positive_count_agreement=0,
                  joint_matches=0,boxes_within1=0,boxes_exact=0,ram_objects=0,pixel_objects=0) for k in KINDS}
    examples={k:[] for k in KINDS}
    for index,(rgb,ram,episode,_) in enumerate(cold_fixture_frames(directory),1):
        truth,pixels=decode_mspacman_ram(ram),screen_mspacman_objects(rgb)
        for kind,s in stats.items():
            a,b=[o for o in truth if o.kind==kind],[o for o in pixels if o.kind==kind]
            n,within,exact=compare_boxes(a,b)
            s["frames"]+=1
            s["presence_count_agreement"]+=len(a)==len(b)
            if a or b:
                s["positive_frames"]+=1
                s["positive_count_agreement"]+=len(a)==len(b)
            s["ram_objects"]+=len(a)
            s["pixel_objects"]+=len(b)
            s["joint_matches"]+=n
            s["boxes_within1"]+=within
            s["boxes_exact"]+=exact
            if (len(a)!=len(b) or within!=n) and len(examples[kind])<8:
                examples[kind].append(dict(frame=index,episode=episode,ram=[o.box for o in a],pixels=[o.box for o in b]))
        if index%2048==0:
            print(f"geometry {index}/{protocol['frames']}",flush=True)
    for s in stats.values():
        s["presence_fraction"]=s["presence_count_agreement"]/s["frames"] if s["frames"] else None
        s["positive_presence_fraction"]=s["positive_count_agreement"]/s["positive_frames"] if s["positive_frames"] else None
        s["box_fraction"]=s["boxes_within1"]/s["joint_matches"] if s["joint_matches"] else None
        s["passed"]=(s["joint_matches"]>=50 and s["presence_fraction"]>=.99 and
                     s["positive_presence_fraction"] is not None and s["positive_presence_fraction"]>=.99 and s["box_fraction"]>=.99)
    if hashes!={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}:
        raise RuntimeError("Preserve this run: sources changed during development scoring")
    return dict(kind="mspacman_raw_development_calibration",seed=manifest["seed"],decoder_revision=REVISION,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),per_type=stats,
                calibration_pass=all(s["passed"] for s in stats.values()),first_disagreements=examples,
                source_sha256=hashes,fixture_manifest_sha256=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
                versions={p:importlib.metadata.version(p) for p in ("ale-py","numpy","scipy")},
                agent_training_allowed=False,reference_training_allowed=False,agent_entry_states_allowed=False,gate_result=None)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Preserve previous results")
    result=calibrate(args.fixture)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ("calibration_pass","per_type")},indent=2))
