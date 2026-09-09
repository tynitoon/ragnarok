"""All-category development check of the separate Ms. Pac-Man visible candidate."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from scripts.calibrate_mspacman_v70 import SOURCE_PATHS as NOMINAL_SOURCES
from scripts.calibrate_breakout_v70 import compare_boxes
from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.mspacman_v70 import KINDS
from ragnarok.evaluation.mspacman_maze_visible_v70 import REVISION,decode_visible_ram,MsPacmanPixelJudge

ROOT=Path(__file__).resolve().parents[1]
SOURCE_PATHS=(*NOMINAL_SOURCES,
    "ragnarok/evaluation/mspacman_render_v70.py","ragnarok/evaluation/mspacman_support_v70.py",
    "ragnarok/evaluation/mspacman_ghosts_v70.py","ragnarok/evaluation/mspacman_pixels_v70.py",
    "ragnarok/evaluation/mspacman_visible_v70.py","scripts/calibrate_mspacman_render_v70.py",
    "ragnarok/evaluation/mspacman_player_list_v70.py","ragnarok/evaluation/mspacman_fruit_v70.py",
    "ragnarok/evaluation/mspacman_list_visible_v70.py","scripts/calibrate_mspacman_list_v70.py",
    "ragnarok/evaluation/mspacman_maze_visible_v70.py","scripts/calibrate_mspacman_maze_v70.py")


def calibrate(directory):
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("game")!="ms_pacman" or manifest.get("seed") not in (17411,17431):
        raise ValueError("Only prospective development; reserved sequences remain closed")
    if manifest["seed"]==17411:
        expected=dict(frames=8192,action_repeat=16,action_seed=117411)
    else:
        expected=dict(frames=32768,action_repeat=2,action_schedule="instrumentation_rgb_maze_walker_v1")
    expected.update(mode=0,episode_limit=108000,sticky_probability=.25,
                    sticky_implementation="worker-python-random-per-frame",agent_training_allowed=False,
                    raw_every_real_frame=True,reset_convention="fixed_hardware_cold_start_v1",gate_result=None)
    if any(manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError("Development collection differs from its prospective schedule")
    paths=tuple(dict.fromkeys((*SOURCE_PATHS,*manifest["source_sha256"])))
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    if any(hashes[p]!=v for p,v in manifest["source_sha256"].items()):
        raise ValueError("Capture source provenance differs")
    stats={k:dict(frames=0,presence_count_agreement=0,positive_frames=0,positive_count_agreement=0,
                  joint_matches=0,boxes_within1=0,boxes_exact=0,ram_objects=0,pixel_objects=0) for k in KINDS}
    examples={k:[] for k in KINDS}
    pixel_judge=MsPacmanPixelJudge();previous=None;ep=None
    for index,(rgb,ram,episode,_) in enumerate(cold_fixture_frames(directory),1):
        if ep!=episode:
            previous=None
        truth,pixels=decode_visible_ram(ram,previous),pixel_judge.step(rgb)
        for kind,s in stats.items():
            a,b=[o for o in truth if o.kind==kind],[o for o in pixels if o.kind==kind]
            n,within,exact=compare_boxes(a,b)
            s["frames"]+=1;s["presence_count_agreement"]+=len(a)==len(b)
            if a or b:
                s["positive_frames"]+=1;s["positive_count_agreement"]+=len(a)==len(b)
            s["ram_objects"]+=len(a);s["pixel_objects"]+=len(b);s["joint_matches"]+=n
            s["boxes_within1"]+=within;s["boxes_exact"]+=exact
            if (len(a)!=len(b) or within!=n) and len(examples[kind])<12:
                examples[kind].append(dict(frame=index,episode=episode,ram=[o.box for o in a],pixels=[o.box for o in b]))
        previous=ram.copy();ep=episode
        if index%2048==0:
            print(f"visible geometry {index}/{manifest['frames']}",flush=True)
    for s in stats.values():
        s["presence_fraction"]=s["presence_count_agreement"]/s["frames"] if s["frames"] else None
        s["positive_presence_fraction"]=s["positive_count_agreement"]/s["positive_frames"] if s["positive_frames"] else None
        s["box_fraction"]=s["boxes_within1"]/s["joint_matches"] if s["joint_matches"] else None
        s["passed"]=(s["joint_matches"]>=50 and s["presence_fraction"]>=.99 and
                     s["positive_presence_fraction"] is not None and s["positive_presence_fraction"]>=.99 and s["box_fraction"]>=.99)
    if hashes!={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}:
        raise RuntimeError("Sources changed during scoring; preserve this incomplete attempt")
    return dict(kind="mspacman_raw_visible_development_calibration",decoder_revision=REVISION,
                seed=manifest["seed"],created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                per_type=stats,calibration_pass=all(s["passed"] for s in stats.values()),first_disagreements=examples,
                source_sha256=hashes,fixture_manifest_sha256=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
                versions={p:importlib.metadata.version(p) for p in ("ale-py","numpy","scipy")},
                ram_history="Previous ordinary RAM within same episode; explicit current fallback at each archived cold start.",
                rgb_history="One causal pixel judge for the entire sequence; no oracle episode resets.",
                agent_training_allowed=False,reference_training_allowed=False,agent_entry_states_allowed=False,gate_result=None)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error("Preserve earlier results")
    result=calibrate(args.fixture)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ("calibration_pass","per_type")},indent=2))
