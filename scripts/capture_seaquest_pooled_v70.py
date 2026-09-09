"""Prospective Seaquest pooled development capture; no learning or reserved execution."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine, ROM_SPECS, CONVENTION
from scripts.seaquest_coverage_controller_v4 import CoverageController
from scripts.seaquest_raw_prerequisite_v70 import RAW_SEALED_SOURCES,require_raw_calibration

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL="docs/ARC4_SEAQUEST_POOLED_PROTOCOL.json"
SOURCE_PATHS=tuple(dict.fromkeys((*RAW_SEALED_SOURCES,
    "scripts/seaquest_raw_prerequisite_v70.py", "scripts/capture_seaquest_pooled_v70.py", PROTOCOL)))


class _PairRAMTap:
    def __init__(self,native):
        self.native = native
        self.previous = self.current = None

    def __getattr__(self,name):
        return getattr(self.native,name)

    def act(self,action):
        self.previous = self.native.getRAM().copy()
        value = self.native.act(action)
        self.current = self.native.getRAM().copy()
        return value


def capture(destination, *, partition="development"):
    if partition!="development":
        raise ValueError("Reserved pooled capture disabled until the full candidate and seal are committed")
    if destination.exists():
        raise ValueError("Keep existing fixtures; choose a new path")
    raw_proof=require_raw_calibration()
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding="utf-8"))
    game,seed,decisions="seaquest",protocol["development_seed"],protocol["decisions"]
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
    manifest=dict(kind="scorer_only_cold_two_raw_frame_fixture",partition="development",seed=seed,action_schedule="instrumentation_rgb_controller_v4_each_decision",
                  decisions=decisions,action_repeat=4,episode_count=episode+1,game=game,mode=0,reset_convention=CONVENTION,sticky_probability=.25,
                  episode_limit=protocol["episode_limit"],sticky_implementation="worker-python-random-per-frame",ordinary_real_frames=ordinary_real_frames,
                  created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                  work_seconds=time.perf_counter()-started,accounting=engine.accounting(),shards=shards,
                  versions={p:importlib.metadata.version(p) for p in ("ale-py","numpy")},
                  source_sha256=source_hashes,raw_calibration_proof_sha256=raw_proof,
                  wrapper_pool_verified=True,agent_training_allowed=False,reference_training_allowed=False,agent_entry_states_allowed=False,gate_result=None)
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--partition",choices=("development",),default="development")
    args=parser.parse_args()
    result=capture(args.out,partition=args.partition)
    print(json.dumps({k:result[k] for k in ("seed","decisions","episode_count","work_seconds","accounting")},indent=2))
