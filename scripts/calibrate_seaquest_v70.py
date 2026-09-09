"""Independent raw-screen/RAM calibration of the declared Seaquest object set."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from scripts.calibrate_breakout_v70 import compare_boxes

from ragnarok.evaluation.seaquest_v70 import KINDS, REVISION, decode_seaquest_ram, screen_seaquest_objects
from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames

SOURCE_PATHS = ("ragnarok/evaluation/seaquest_v70.py","ragnarok/evaluation/cold_fixtures_v70.py",
                "scripts/calibrate_seaquest_v70.py","docs/ARC4_SEAQUEST_CALIBRATION_V3_PROTOCOL.json",
                "scripts/capture_cold_v70.py","ragnarok/environments/cold_emu_pool_v70.py",
                "ragnarok/environments/emu_pool.py","ragnarok/environments/frame_budget_v70.py",
                "ragnarok/evaluation/pong_v70.py","scripts/calibrate_breakout_v70.py")


def calibrate(directory):
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root/"docs/ARC4_SEAQUEST_CALIBRATION_V3_PROTOCOL.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory/"manifest.json").read_text())
    if (manifest["game"]!="seaquest" or manifest["frames"]!=protocol["frames"] or
            manifest["seed"] not in (protocol["development_seed"],protocol["reserved_validation_seed"])):
        raise ValueError("Fixture is outside the declared calibration")
    expected_parameters=dict(mode=0,sticky_probability=.25,action_repeat=16,episode_limit=108_000,
                             raw_every_real_frame=True,reset_convention="fixed_hardware_cold_start_v1",
                             sticky_implementation="worker-python-random-per-frame")
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
    for index,(rgb,ram,episode,charge) in enumerate(cold_fixture_frames(directory),1):
        truth,pixels = decode_seaquest_ram(ram),screen_seaquest_objects(rgb)
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
    return dict(kind="independent_seaquest_raw_calibration",created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
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
