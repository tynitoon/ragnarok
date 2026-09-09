"""Sealed reserved calibration of unchanged visible Asteroids instruments."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from scripts.calibrate_breakout_v70 import compare_boxes,SOURCE_PATHS as SHARED_SOURCES
from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.asteroids_visible_v70 import KINDS,REVISION,decode_visible_asteroids_ram,screen_visible_asteroids_objects

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL="docs/ARC4_ASTEROIDS_RAW_PROTOCOL.json"
SOURCE_PATHS=(*SHARED_SOURCES,PROTOCOL,"ragnarok/evaluation/asteroids_v70.py","scripts/calibrate_asteroids_v70.py",
    "ragnarok/evaluation/asteroids_rocks_v70.py","ragnarok/evaluation/asteroids_occlusion_v70.py",
    "ragnarok/evaluation/asteroids_deblend_v70.py","ragnarok/evaluation/asteroids_deblend_edges_v70.py",
    "ragnarok/evaluation/asteroids_history_v70.py","ragnarok/evaluation/asteroids_player_roi_v70.py",
    "ragnarok/evaluation/asteroids_visible_v70.py","scripts/calibrate_asteroids_visible_v70.py")


def calibrate(directory):
    from scripts.validate_asteroids_raw_v70 import verify_seal,verify_fixture_attempt
    seal=verify_seal()
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding="utf-8"))
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    verify_fixture_attempt(directory,manifest,seal)
    expected=dict(game="asteroids",seed=protocol["reserved_validation_seed"],frames=protocol["frames"],
                  mode=0,action_repeat=16,episode_limit=108000,action_seed=protocol["reserved_validation_seed"]+100000,
                  sticky_probability=.25,sticky_implementation="worker-python-random-per-frame",
                  reset_convention="fixed_hardware_cold_start_v1",raw_every_real_frame=True,
                  agent_training_allowed=False,gate_result=None)
    if any(manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError("Only the prospectively sealed reserved fixture is permitted")
    paths=tuple(seal["source_sha256"])
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    capture_sources=("scripts/capture_cold_v70.py","ragnarok/environments/emu_pool.py",
                     "ragnarok/environments/frame_budget_v70.py","ragnarok/environments/cold_emu_pool_v70.py")
    if manifest.get("source_sha256")!={p:hashes[p] for p in capture_sources}:
        raise ValueError("Capture source provenance differs")
    stats={k:dict(frames=0,presence_count_agreement=0,positive_frames=0,positive_count_agreement=0,
                  joint_matches=0,boxes_within1=0,boxes_exact=0,ram_objects=0,pixel_objects=0) for k in KINDS}
    examples={k:[] for k in KINDS};previous=None;last_episode=None
    for index,(rgb,ram,episode,_) in enumerate(cold_fixture_frames(directory),1):
        if episode!=last_episode:previous=None
        truth,pixels=decode_visible_asteroids_ram(ram,previous),screen_visible_asteroids_objects(rgb)
        previous=ram.copy();last_episode=episode
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
    if hashes!={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}:
        raise RuntimeError("Preserve this run: sources changed during development scoring")
    return dict(kind="asteroids_raw_visible_reserved_calibration",seed=manifest["seed"],decoder_revision=REVISION,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),per_type=stats,
                calibration_pass=all(s["passed"] for s in stats.values()),validation_seal_sha256=seal["_file_sha256"],first_disagreements=examples,
                ram_history="Previous adjacent raw RAM within episode, explicit current fallback at cold start. Pixels never enter RAM decoding.",
                rgb_history="Stateless raw RGB only; pure glyph-fit cache has no time, RAM or episode input.",
                label_scope="Visible connected pieces by allocation; no cross-allocation RAM merge, no physical lifetimes certified.",
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
