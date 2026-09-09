"""Single reserved Seaquest temporal validation, after a complete committed seal."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine, ROM_SPECS, CONVENTION
from ragnarok.evaluation.cold_pooled_fixtures_v70 import cold_pooled_fixture_frames
from ragnarok.evaluation.pooled_seaquest_v70 import KINDS, REVISION, pooled_seaquest_ram_objects, pooled_seaquest_screen_objects
from scripts.seaquest_coverage_controller_v4 import CoverageController
from scripts.seaquest_raw_prerequisite_v70 import require_raw_calibration
from scripts.capture_seaquest_pooled_v70 import _PairRAMTap, SOURCE_PATHS as CAPTURE_SOURCES, PROTOCOL as DEVELOPMENT_PROTOCOL
from scripts.calibrate_seaquest_pooled_v70 import SOURCE_PATHS as DEVELOPMENT_SOURCES
from scripts.calibrate_seaquest_coverage_v70 import new_stats, accumulate, finish
from scripts.committed_seal_v70 import verify_committed_seal

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL="docs/ARC4_SEAQUEST_POOLED_RESERVED_PROTOCOL.json"
SOURCE_PATHS=(*DEVELOPMENT_SOURCES,PROTOCOL,"scripts/validate_seaquest_pooled_v70.py")


def hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}


def _write(path,value):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def _require_committed(path):
    relative=path.relative_to(ROOT).as_posix()
    current=subprocess.check_output(["git","-C",str(ROOT),"hash-object",f"--path={relative}",str(path)],text=True).strip()
    committed=subprocess.check_output(["git","-C",str(ROOT),"rev-parse",f"HEAD:{relative}"],text=True).strip()
    if current!=committed:
        raise ValueError(f"Commit required before sealing or capture: {relative}")


def development_evidence(base,source_hashes):
    report_path=base/"development_baseline.json"
    manifest_path=base/"development/manifest.json"
    report=json.loads(report_path.read_text(encoding="utf-8"))
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_proof=require_raw_calibration()
    if (report.get("kind")!="seaquest_two_raw_frame_development_calibration" or report.get("seed")!=17321 or
            report.get("calibration_pass") is not True or report.get("decoder_revision")!=REVISION or
            report.get("source_sha256")!={p:source_hashes[p] for p in DEVELOPMENT_SOURCES} or
            report.get("fixture_manifest_sha256")!=hashlib.sha256(manifest_path.read_bytes()).hexdigest() or
            report.get("raw_calibration_proof_sha256")!=raw_proof or
            manifest.get("source_sha256")!={p:source_hashes[p] for p in CAPTURE_SOURCES} or
            manifest.get("seed")!=17321 or manifest.get("decisions")!=8192 or manifest.get("partition")!="development"):
        raise ValueError("Development provenance differs from the fixed successful baseline")
    if set(report["per_type"])!=set((*KINDS,"surface_submarine")):
        raise ValueError("Every category and surface subtype must be present")
    for value in report["per_type"].values():
        recomputed=dict(value)
        finish(recomputed)
        if recomputed!=value or not recomputed["passed"] or recomputed["frames"]!=8192:
            raise ValueError("Development counts or criteria are not a complete pass")
    for shard in manifest["shards"]:
        path=manifest_path.parent/shard["path"]
        if path.parent.resolve()!=manifest_path.parent.resolve() or hashlib.sha256(path.read_bytes()).hexdigest()!=shard["sha256"]:
            raise ValueError("Preserve exact development archives")
    evidence={}
    for path in (report_path,manifest_path):
        _require_committed(path)
        evidence[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return evidence,raw_proof


def create_seal(base):
    source_hashes=hashes()
    evidence,raw_proof=development_evidence(base,source_hashes)
    for path in SOURCE_PATHS:
        _require_committed(ROOT/path)
    commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    seal=dict(kind="seaquest_pooled_reserved_seal",created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
              source_commit=commit,source_sha256=source_hashes,development_results=evidence,
              raw_calibration_proof_sha256=raw_proof,reserved_seed=17322,decisions=8192)
    _write(base/"reserved_seal.json",seal)
    return seal


def verify_seal(base):
    seal=verify_committed_seal(ROOT,(base/"reserved_seal.json").relative_to(ROOT),SOURCE_PATHS)
    evidence,raw_proof=development_evidence(base,seal["source_sha256"])
    if (seal.get("kind")!="seaquest_pooled_reserved_seal" or seal.get("reserved_seed")!=17322 or
            seal.get("decisions")!=8192 or seal.get("development_results")!=evidence or
            seal.get("raw_calibration_proof_sha256")!=raw_proof):
        raise ValueError("Reserved specification or prerequisite changed")
    return seal


def _capture_reserved(destination):
    if destination.exists():
        raise ValueError("Keep existing fixtures; choose a new path")
    raw_proof=require_raw_calibration()
    protocol=json.loads((ROOT/DEVELOPMENT_PROTOCOL).read_text(encoding="utf-8"))
    game,seed,decisions="seaquest",17322,protocol["decisions"]
    source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    engine = _ColdPixelEngine(EmulatorConfig(game=game,seed=seed,frame_skip=4,mode="live",
                                        max_episode_frames=protocol["episode_limit"], max_charged_frames=decisions*(ROM_SPECS[game][1]+4)+ROM_SPECS[game][1]))
    engine.set_native_wrapper(_PairRAMTap)
    destination.mkdir(parents=True)
    names = ("raw_previous","raw_current","pooled","ram_previous","ram_current",
             "episode","frame_in_episode","charged_frame","action")
    rows = {k:[] for k in names}
    shards, previews = [], []
    controller=CoverageController()
    episode, reset_spent = 0, engine.accounting()["reset_emulated_frames"]
    previous_local,ordinary_real_frames=0,0
    started = time.perf_counter()
    for index in range(1,decisions+1):
        action=controller.action(engine.observe(raw=True))
        pooled = engine.step(action)
        before, after = (rgb.copy() for rgb in engine._frames)
        if not np.array_equal(pooled,np.maximum(before,after)):
            raise RuntimeError("Captured pool differs from actual wrapper observation")
        report = engine.accounting()
        reset=report["reset_emulated_frames"] != reset_spent
        if reset:
            episode += 1
            reset_spent = report["reset_emulated_frames"]
        tap=engine._ale  # a new tap is installed after every cold construction
        local=engine._ale.getEpisodeFrameNumber()
        advanced=local if reset else local-previous_local
        ordinary_real_frames+=advanced
        if (not 1<=advanced<=4 or report["charged_frames"]!=ordinary_real_frames+128*(episode+1) or
                report["cold_loads"]!=episode+1 or report["experiment_frames"] or report["restores"]):
            raise RuntimeError("Pooled ordinary-frame/cold-load accounting failed")
        previous_local=local
        values = (before,after,pooled,tap.previous,tap.current,episode,
                  local,report["charged_frames"],action)
        for key,value in zip(names,values):
            rows[key].append(value)
        if index in (1,4,16,32,64,128,256,512,768,1024,1536,decisions):
            previews.append((index,pooled.copy()))
        if len(rows["pooled"])==128 or index==decisions:
            n=len(rows["pooled"])
            path=destination/f"decisions_{index-n+1:05d}_{index:05d}.npz"
            arrays={key:np.stack(rows[key]) for key in names[:5]}
            arrays.update({key:np.array(rows[key],dtype=np.int64) for key in names[5:]})
            np.savez_compressed(path,**arrays)
            shards.append(dict(path=path.name,decisions=n,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            rows={k:[] for k in names}
        if index%2048==0:
            print(f"pooled capture {index}/{decisions}; charged={report['charged_frames']}",flush=True)
    if source_hashes!={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}:
        raise RuntimeError("Pooled capture sources changed during the run")
    if raw_proof!=require_raw_calibration():
        raise RuntimeError("Raw prerequisite changed during the run")
    sheet=Image.new("RGB",(4*176,((len(previews)+3)//4)*240),"#202020")
    draw=ImageDraw.Draw(sheet)
    for i,(index,rgb) in enumerate(previews):
        x,y=(i%4)*176,(i//4)*240
        draw.text((x+8,y+5),f"decision{index}",fill="white")
        sheet.paste(Image.fromarray(rgb),(x+8,y+24))
    sheet.save(destination/"fixed_pooled_frames.png")
    manifest=dict(kind="scorer_only_cold_two_raw_frame_fixture",partition="reserved",seed=seed,action_schedule="instrumentation_rgb_controller_v4_each_decision",
                  decisions=decisions,action_repeat=4,episode_count=episode+1,game=game,mode=0,reset_convention=CONVENTION,sticky_probability=.25,
                  episode_limit=protocol["episode_limit"],sticky_implementation="worker-python-random-per-frame",ordinary_real_frames=ordinary_real_frames,
                  created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                  work_seconds=time.perf_counter()-started,accounting=engine.accounting(),shards=shards,
                  versions={p:importlib.metadata.version(p) for p in ("ale-py","numpy")},
                  source_sha256=source_hashes,raw_calibration_proof_sha256=raw_proof,
                  wrapper_pool_verified=True,agent_training_allowed=False,reference_training_allowed=False,agent_entry_states_allowed=False,gate_result=None)
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


def score_reserved(directory):
    raw_proof=require_raw_calibration()
    protocol=json.loads((ROOT/DEVELOPMENT_PROTOCOL).read_text(encoding="utf-8"))
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    expected=dict(game="seaquest",partition="reserved",seed=17322,decisions=protocol["decisions"],
                  mode=0,action_repeat=4,action_schedule="instrumentation_rgb_controller_v4_each_decision",
                  episode_limit=protocol["episode_limit"],sticky_implementation="worker-python-random-per-frame",
                  reset_convention="fixed_hardware_cold_start_v1",sticky_probability=.25,
                  wrapper_pool_verified=True,agent_training_allowed=False,reference_training_allowed=False,
                  agent_entry_states_allowed=False,raw_calibration_proof_sha256=raw_proof)
    if any(manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError("Outside the fixed pooled reserved specification")
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if manifest["source_sha256"]!=hashes:
        raise ValueError("Capture sources differ from the prospective protocol")
    stats={kind:new_stats() for kind in (*KINDS,"surface_submarine")}
    examples={kind:[] for kind in stats}
    for row in cold_pooled_fixture_frames(directory):
        truth=pooled_seaquest_ram_objects(row["ram_previous"],row["ram_current"])
        pixels=pooled_seaquest_screen_objects(row["raw_previous"],row["raw_current"])
        for kind,s in stats.items():
            select=lambda o:o.kind=="submarine" and o.box[1]<57 if kind=="surface_submarine" else o.kind==kind
            a,b=[o for o in truth if select(o)],[o for o in pixels if select(o)]
            n,within=accumulate(s,a,b)
            if (len(a)!=len(b) or n!=within) and len(examples[kind])<8:
                examples[kind].append(dict(decision=int(row["index"]),ram=[o.box for o in a],pixels=[o.box for o in b]))
        if row["index"]%2048==0:
            print(f"pooled geometry {row['index']}/{protocol['decisions']}",flush=True)
    for s in stats.values():
        finish(s)
    if hashes!={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}:
        raise RuntimeError("Pooled calibration sources changed during execution")
    return dict(kind="seaquest_two_raw_frame_reserved_calibration",seed=manifest["seed"],decoder_revision=REVISION,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),calibration_pass=all(s["passed"] for s in stats.values()),
                per_type=stats,first_disagreements=examples,source_sha256=hashes,
                fixture_manifest_sha256=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
                raw_calibration_proof_sha256=raw_proof,
                scope="Component-window counts.4-connected spatial envelopes, separate surface subtype. Two-source provenance may be ambiguous from maxRGB alone.",
                agent_training_allowed=False,gate_result=None)



def run_reserved(base):
    base=Path(base).resolve()
    destination=base/"reserved"
    output=base/"reserved_result.json"
    attempt=base/"reserved_attempt.json"
    if any(p.exists() for p in (destination,output,attempt)):
        raise ValueError("Preserve the previous reserved attempt, including incomplete construction")
    seal=verify_seal(base)
    seal_sha=hashlib.sha256((base/"reserved_seal.json").read_bytes()).hexdigest()
    # Durable marker before the first constructor, even if construction fails.
    _write(attempt,dict(kind="seaquest_pooled_reserved_attempt",seed=17322,
                       started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),seal_sha256=seal_sha))
    _capture_reserved(destination)
    verify_seal(base)
    result=score_reserved(destination)
    verify_seal(base)
    result.update(seal_sha256=seal_sha,reference_training_allowed=False,agent_entry_states_allowed=False)
    _write(output,result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("seal","run"))
    parser.add_argument("--base",type=Path,default=Path("craft_v6_out/v70_seaquest_pooled"))
    args=parser.parse_args()
    result=create_seal(args.base.resolve()) if args.mode=="seal" else run_reserved(args.base)
    print(json.dumps(result,indent=2))
