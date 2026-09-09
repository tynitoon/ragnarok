"""Preserve raw-development palette inventory for temporal diagnostics only."""
from pathlib import Path
import hashlib
import json
import numpy as np
from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.asteroids_v70 import KINDS,screen_asteroids_objects

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'craft_v6_out/v70_asteroids_calibration'

if __name__=='__main__':
    out=BASE/'development_palette_inventory_v0.npz'
    meta=BASE/'development_palette_inventory_v0.json'
    if out.exists() or meta.exists():raise ValueError('Preserve previous inventory')
    rams=[];episodes=[];boxes=[];kinds=[];frames=[];counts=[]
    for index,(rgb,ram,episode,_) in enumerate(cold_fixture_frames(BASE/'development')):
        objects=screen_asteroids_objects(rgb)
        rams.append(ram);episodes.append(episode)
        counts.append([sum(o.kind==k for o in objects) for k in KINDS])
        for o in objects:
            boxes.append(o.box);kinds.append(KINDS.index(o.kind));frames.append(index)
        if (index+1)%2048==0:print(f'palette inventory {index+1}/8192',flush=True)
    np.savez_compressed(out,ram=np.stack(rams),episode=np.asarray(episodes,dtype=np.int32),
                        count=np.asarray(counts,dtype=np.int32),boxes=np.asarray(boxes,dtype=np.int16),
                        kind=np.asarray(kinds,dtype=np.uint8),frame=np.asarray(frames,dtype=np.int32))
    sources=(str(Path(__file__).relative_to(ROOT)).replace('\\','/'),
             'ragnarok/evaluation/asteroids_v70.py','ragnarok/evaluation/cold_fixtures_v70.py')
    result=dict(kind='asteroids_development_palette_inventory',seed=17511,frames=index+1,kinds=KINDS,
                payload_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256((BASE/'development/manifest.json').read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                note='Derived from every archived raw development image. Palette assignments are nominal, not certified labels. No training data or agent observations.',
                agent_training_allowed=False,gate_result=None)
    with meta.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(frames=index+1,objects=len(boxes))))
