"""Development-only geometry of actual Seaquest two-frame observations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from ragnarok.evaluation.cold_pooled_fixtures_v70 import cold_pooled_fixture_frames
from ragnarok.evaluation.pooled_seaquest_v70 import KINDS,REVISION,pooled_seaquest_ram_objects,pooled_seaquest_screen_objects
from scripts.calibrate_seaquest_coverage_v70 import new_stats,accumulate,finish
from scripts.capture_seaquest_pooled_v70 import SOURCE_PATHS as CAPTURE_SOURCES,PROTOCOL
from scripts.seaquest_raw_prerequisite_v70 import require_raw_calibration

ROOT=Path(__file__).resolve().parents[1]
SOURCE_PATHS=tuple(dict.fromkeys((*CAPTURE_SOURCES,"ragnarok/evaluation/pooled_seaquest_v70.py",
    "ragnarok/evaluation/cold_pooled_fixtures_v70.py","scripts/calibrate_seaquest_pooled_v70.py")))


def calibrate(directory):
    raw_proof=require_raw_calibration()
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding="utf-8"))
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    expected=dict(game="seaquest",partition="development",seed=protocol["development_seed"],decisions=protocol["decisions"],
                  mode=0,action_repeat=4,action_schedule="instrumentation_rgb_controller_v4_each_decision",
                  episode_limit=protocol["episode_limit"],sticky_implementation="worker-python-random-per-frame",
                  reset_convention="fixed_hardware_cold_start_v1",sticky_probability=.25,
                  wrapper_pool_verified=True,agent_training_allowed=False,reference_training_allowed=False,
                  agent_entry_states_allowed=False,raw_calibration_proof_sha256=raw_proof)
    if any(manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError("Outside the declared pooled development; reserved execution disabled")
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if manifest["source_sha256"]!={p:hashes[p] for p in CAPTURE_SOURCES}:
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
    return dict(kind="seaquest_two_raw_frame_development_calibration",seed=manifest["seed"],decoder_revision=REVISION,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),calibration_pass=all(s["passed"] for s in stats.values()),
                per_type=stats,first_disagreements=examples,source_sha256=hashes,
                fixture_manifest_sha256=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
                raw_calibration_proof_sha256=raw_proof,
                scope="Component-window counts.4-connected spatial envelopes, separate surface subtype. Two-source provenance may be ambiguous from maxRGB alone.",
                agent_training_allowed=False,gate_result=None)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=calibrate(args.fixture) if not args.out.exists() else parser.error("Preserve the previous result")
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ("calibration_pass","per_type")},indent=2))
