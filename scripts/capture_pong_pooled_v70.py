"""Scorer-only two-frame fixtures, including actual reset windows and repeat truncations."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw

from ragnarok.environments.emu_pool import EmulatorConfig, _PixelEngine


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


def capture(destination, *, seed, decisions):
    if destination.exists():
        raise ValueError("Keep existing fixtures; choose a new path")
    if not 256 <= decisions <= 4096:
        raise ValueError("Pooled development captures are bounded to256..4096 decisions")
    engine = _PixelEngine(EmulatorConfig(seed=seed,frame_skip=4,mode="live",
                                        max_charged_frames=decisions*76+72))
    tap = _PairRAMTap(engine._ale)
    engine._ale = tap
    destination.mkdir(parents=True)
    names = ("raw_previous","raw_current","pooled","ram_previous","ram_current",
             "episode","frame_in_episode","charged_frame","action")
    rows = {k:[] for k in names}
    shards, previews = [], []
    rng = random.Random(seed+100_000)
    episode, reset_spent = 0, 72
    started = time.perf_counter()
    for index in range(1,decisions+1):
        if (index-1)%4==0:
            action = rng.randrange(18)
        pooled = engine.step(action)
        before, after = (rgb.copy() for rgb in engine._frames)
        if not np.array_equal(pooled,np.maximum(before,after)):
            raise RuntimeError("Captured pool differs from actual wrapper observation")
        report = engine.accounting()
        if report["reset_emulated_frames"] != reset_spent:
            episode += 1
            reset_spent = report["reset_emulated_frames"]
        values = (before,after,pooled,tap.previous,tap.current,episode,
                  engine._ale.getEpisodeFrameNumber(),report["charged_frames"],action)
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
    sheet=Image.new("RGB",(4*176,((len(previews)+3)//4)*240),"#202020")
    draw=ImageDraw.Draw(sheet)
    for i,(index,rgb) in enumerate(previews):
        x,y=(i%4)*176,(i//4)*240
        draw.text((x+8,y+5),f"decision{index}",fill="white")
        sheet.paste(Image.fromarray(rgb),(x+8,y+24))
    sheet.save(destination/"fixed_pooled_frames.png")
    root=Path(__file__).resolve().parents[1]
    sources=["scripts/capture_pong_pooled_v70.py","ragnarok/environments/emu_pool.py",
             "ragnarok/environments/frame_budget_v70.py","docs/ARC4_PONG_POOLED_PROTOCOL.json"]
    manifest=dict(kind="scorer_only_pong_two_raw_frame_fixture",seed=seed,action_seed=seed+100_000,
                  decisions=decisions,action_repeat=4,episode_count=episode+1,game="pong",mode=0,
                  created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                  work_seconds=time.perf_counter()-started,accounting=engine.accounting(),shards=shards,
                  versions={p:importlib.metadata.version(p) for p in ("ale-py","numpy")},
                  source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
                  wrapper_pool_verified=True,agent_training_allowed=False,gate_result=None)
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--decisions",type=int,default=2048)
    args=parser.parse_args()
    result=capture(args.out,seed=args.seed,decisions=args.decisions)
    print(json.dumps({k:result[k] for k in ("seed","decisions","episode_count","work_seconds","accounting")},indent=2))
