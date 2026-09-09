"""Preserved development diagnosis of fruit glyphs, no ROM or reserved data."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.mspacman_v70 import PALETTES

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'craft_v6_out/v70_mspacman_calibration/coverage_development_v1'

if __name__=='__main__':
    out=FIXTURE.parent/'fruit_support_development_diagnostic.json'
    if out.exists():raise ValueError('Preserve earlier diagnosis')
    previous=None;ep=None;patterns={};first={};pointers=Counter();n=0
    for index,(rgb,current,episode,_) in enumerate(cold_fixture_frames(FIXTURE),1):
        if ep!=episode:previous=None
        r=current.astype(int) if previous is None else previous.astype(int)
        for i in range(int(r[34])+1):
            kind=int(r[45+i])>>4
            if kind==7:continue
            x,y=int(r[11])-12,int(r[17])+1
            pointers[(kind,256*int(r[53+i])+int(r[51+i])+int(r[17]))]+=1
            if 0<=x<=152 and 0<=y<=161:
                crop=rgb[y:y+10,x:x+8];mask=np.zeros((10,8),dtype=bool)
                for colour in PALETTES['fruit']:mask|=np.all(crop==colour,axis=2)
                bits=tuple(map(int,np.packbits(mask,axis=1).ravel()))
                patterns.setdefault(kind,Counter())[bits]+=1
                first.setdefault((kind,bits),index);n+=1
        previous=current.copy();ep=episode
    result=dict(kind='mspacman_fruit_support_development_diagnostic',seed=17431,frames=index,full_body_crops=n,
                patterns={str(k):[dict(rows=list(rows),count=count,first_frame=first[(k,rows)]) for rows,count in c.most_common()] for k,c in patterns.items()},
                compensated_pointer_counts={str(k):v for k,v in pointers.items()},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                fixture_sha256=hashlib.sha256((FIXTURE/'manifest.json').read_bytes()).hexdigest(),
                agent_training_allowed=False,gate_result=None)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v[:4] for k,v in result['patterns'].items()},indent=2))
