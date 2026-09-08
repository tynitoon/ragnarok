"""Independent two-raw-source calibration of actual pooled Breakout windows."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from ragnarok.evaluation.breakout_v70 import KINDS
from ragnarok.evaluation.pooled_breakout_v70 import REVISION, pooled_breakout_ram_objects, pooled_breakout_screen_objects
from ragnarok.evaluation.cold_pooled_fixtures_v70 import cold_pooled_fixture_frames
from ragnarok.evaluation.pong_v70 import edges

SOURCE_PATHS = ("ragnarok/evaluation/breakout_v70.py","ragnarok/evaluation/cold_pooled_fixtures_v70.py","ragnarok/evaluation/pooled_breakout_v70.py",
                "scripts/calibrate_breakout_pooled_v70.py","docs/ARC4_BREAKOUT_POOLED_PROTOCOL.json",
                "scripts/capture_cold_pooled_v70.py","ragnarok/environments/cold_emu_pool_v70.py",
                "ragnarok/environments/emu_pool.py","ragnarok/environments/frame_budget_v70.py",
                "ragnarok/evaluation/pong_v70.py")


def compare_boxes(truth, pixels):
    n = min(len(truth),len(pixels))
    if not n:
        return 0,0,0
    distance = np.max(np.abs(np.array([edges(x.box) for x in truth])[:,None]-
                            np.array([edges(x.box) for x in pixels])[None,:]),axis=-1)
    valid = distance<=1
    # Invalid assignments outweigh all valid distances, maximizing valid cardinality first.
    rows,cols = linear_sum_assignment(np.where(valid,distance,2*n+1))
    return n,int(valid[rows,cols].sum()),int((distance[rows,cols]==0).sum())


def calibrate(directory):
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root/"docs/ARC4_BREAKOUT_POOLED_PROTOCOL.json").read_text())
    manifest = json.loads((directory/"manifest.json").read_text())
    if (manifest["game"]!="breakout" or manifest["decisions"]!=protocol["decisions"] or
            manifest["seed"] not in (protocol["development_seed"],protocol["reserved_validation_seed"])):
        raise ValueError("Fixture is outside the declared calibration")
    expected_parameters=dict(mode=0,sticky_probability=.25,action_repeat=4,
                             reset_convention="fixed_hardware_cold_start_v1",wrapper_pool_verified=True)
    if any(manifest.get(k)!=v for k,v in expected_parameters.items()) or manifest.get("action_seed")!=manifest["seed"]+100_000:
        raise ValueError("Fixture parameters differ from the declared calibration")
    hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if manifest["seed"]==protocol["reserved_validation_seed"]:
        seal=json.loads((directory.parent/"validation_seal.json").read_text())
        if seal.get("validation_seed")!=manifest["seed"] or seal.get("source_sha256")!=hashes:
            raise ValueError("Reserved calibration requires the unchanged sealed sources")
    stats = {kind:dict(frames=0,presence_count_agreement=0,joint_matches=0,boxes_within1=0,boxes_exact=0,
                       ram_objects=0,pixel_objects=0) for kind in KINDS}
    examples = []
    for frame in cold_pooled_fixture_frames(directory):
        index=frame["index"]
        truth=pooled_breakout_ram_objects(frame["ram_previous"],frame["ram_current"])
        pixels=pooled_breakout_screen_objects(frame["raw_previous"],frame["raw_current"])
        for kind in KINDS:
            a,b = [o for o in truth if o.kind==kind],[o for o in pixels if o.kind==kind]
            n,within,exact = compare_boxes(a,b)
            s = stats[kind]
            s["frames"] += 1
            s["presence_count_agreement"] += len(a)==len(b)
            s["joint_matches"] += n
            s["boxes_within1"] += within
            s["boxes_exact"] += exact
            s["ram_objects"] += len(a)
            s["pixel_objects"] += len(b)
            if (len(a)!=len(b) or n!=within) and len(examples)<32:
                examples.append(dict(frame=index,kind=kind,ram=[o.box for o in a],pixels=[o.box for o in b]))
    for s in stats.values():
        s["presence_fraction"] = s["presence_count_agreement"]/s["frames"]
        s["box_fraction"] = s["boxes_within1"]/s["joint_matches"] if s["joint_matches"] else None
        criteria=protocol["criteria_per_type"]
        s["passed"] = (s["joint_matches"]>=criteria["minimum_joint_matches"] and
                       s["presence_fraction"]>=criteria["presence_count_agreement_fraction"] and
                       s["box_fraction"]>=criteria["matched_boxes_all_edges_within1pixel_fraction"])
    return dict(kind="independent_breakout_pooled_calibration",created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                decoder_revision=REVISION, seed=manifest["seed"], per_type=stats,
                calibration_pass=all(s["passed"] for s in stats.values()),first_disagreements=examples,
                fixture_manifest_sha256=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
                source_sha256=hashes,versions={p:importlib.metadata.version(p) for p in ("numpy","scipy","ale-py")},
                gate_result=None,agent_training_allowed=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Preserve prior results; choose a new output")
    result=calibrate(args.fixture)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({"calibration_pass":result["calibration_pass"],"per_type":result["per_type"]},indent=2))
