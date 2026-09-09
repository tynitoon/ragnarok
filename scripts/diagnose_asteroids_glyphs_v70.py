"""Exploratory rock glyph inventory from complete development components."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.asteroids_v70 import ASTEROID_ROWS,x_position

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'craft_v6_out/v70_asteroids_calibration'

if __name__=='__main__':
    out=BASE/'rock_glyph_development_diagnostic.json'
    if out.exists():raise ValueError('Preserve earlier diagnosis')
    inventory=BASE/'development_palette_inventory_v0.npz'
    with np.load(inventory,allow_pickle=False) as z:boxes=z['boxes'];frames=z['frame'];kinds=z['kind']
    patterns={};first={};skipped=Counter();accepted=0
    for index,(rgb,ram,ep,charge) in enumerate(cold_fixture_frames(BASE/'development')):
        r=ram.astype(int)
        if not r[57]&1:continue
        pixels=boxes[(frames==index)&(kinds==1)]
        candidates=[]
        for row in ASTEROID_ROWS:
            if not r[row+18] or r[row]==224:continue
            yy=int(r[row]) if r[row]<128 else int(r[row])-256
            code=int(r[row+36]);size=code&127
            group=0 if size<32 else 1 if size<48 else 2
            w,h=((16,28),(8,14),(4,8))[group]
            x,y=x_position(r[row+18])-int(group>0),24+2*yy+int(row>=12)-2*int(group>0)
            if x<1 or x+w>=160 or y<20 or y+h>=194:continue
            candidates.append((x,y,w,h,row,code,group))
        if not len(pixels) or not candidates:continue
        pred=np.asarray([c[:4] for c in candidates]);cost=np.max(np.abs(pred[:,None]-pixels[None]),axis=-1)
        a,b=linear_sum_assignment(cost)
        for i,j in zip(a,b):
            x,y,w,h,row,code,group=candidates[i]
            xx,yy,ww,hh=map(int,pixels[j])
            if cost[i,j]>1 or (ww,hh)!=(w,h):skipped['noncomplete_nominal_match']+=1;continue
            crop=rgb[yy:yy+h,xx:xx+w]
            colours,counts=np.unique(crop.reshape(-1,3),axis=0,return_counts=True)
            options=[(int(n),tuple(map(int,c))) for c,n in zip(colours,counts) if np.any(c)]
            if not options:continue
            _,colour=max(options)
            mask=np.all(crop==colour,axis=2)
            ys,xs=np.where(mask)
            if not len(xs) or (xs.min(),ys.min(),xs.max()+1,ys.max()+1)!=(0,0,w,h):
                skipped['dominant_colour_not_complete']+=1;continue
            rows=tuple(int(sum((1<<(w-1-dx)) for dx,q in enumerate(line) if q)) for line in mask)
            key=(group,int(bool(code&16)),int(bool(code&128)))
            patterns.setdefault(key,Counter())[rows]+=1
            first.setdefault((key,rows),dict(frame=index+1,slot=row,code=code,colour=colour,box=(xx,yy,w,h)))
            accepted+=1
    result=dict(kind='asteroids_rock_glyph_development_inventory',seed=17511,frames=index+1,accepted_complete_components=accepted,
                selection_limits=dict(skipped),patterns={str(key):[dict(rows=list(rows),count=n,example=first[(key,rows)]) for rows,n in counter.most_common()] for key,counter in patterns.items()},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
                note='Only full isolated-size components selected to infer glyph hypotheses; this is not a calibration score. Future scoring must include all8192 frames, wraps, overlaps and incomplete sprites. No RAM input to a pixel checker and no RGB input to a RAM decoder.',
                gate_result=None,agent_training_allowed=False)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:dict(variants=len(v),top=v[:2]) for k,v in result['patterns'].items()},indent=2))
