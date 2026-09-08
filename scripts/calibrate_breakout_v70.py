"""Independent raw-screen/RAM calibration of the declared Breakout object set."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from ragnarok.evaluation.breakout_v70 import KINDS, REVISION, decode_breakout_ram, screen_breakout_objects
from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.pong_v70 import edges


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
    protocol = json.loads((root/"docs/ARC4_BREAKOUT_CALIBRATION_PROTOCOL.json").read_text())
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest["game"]!="breakout" or manifest["frames"]!=protocol["frames_per_sequence"]:
        raise ValueError("Fixture is outside the declared calibration")
    stats = {kind:dict(frames=0,presence_count_agreement=0,joint_matches=0,boxes_within1=0,boxes_exact=0,
                       ram_objects=0,pixel_objects=0) for kind in KINDS}
    examples = []
    for index,(rgb,ram,episode,charge) in enumerate(cold_fixture_frames(directory),1):
        truth,pixels = decode_breakout_ram(ram),screen_breakout_objects(rgb)
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
        s["passed"] = s["joint_matches"]>=50 and s["presence_fraction"]>=.99 and s["box_fraction"]>=.99
    sources = ("ragnarok/evaluation/breakout_v70.py","ragnarok/evaluation/cold_fixtures_v70.py",
               "scripts/calibrate_breakout_v70.py","docs/ARC4_BREAKOUT_CALIBRATION_PROTOCOL.json")
    return dict(kind="independent_breakout_raw_calibration",created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                decoder_revision=REVISION, seed=manifest["seed"], per_type=stats,
                calibration_pass=all(s["passed"] for s in stats.values()),first_disagreements=examples,
                fixture_manifest_sha256=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
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
