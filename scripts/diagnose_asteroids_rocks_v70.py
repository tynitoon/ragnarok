"""Retain successive raw-development hypotheses for Asteroids rock rendering."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
from ragnarok.evaluation.asteroids_v70 import ASTEROID_ROWS,x_position,decode_asteroids_ram,SlottedLabel
from scripts.calibrate_breakout_v70 import compare_boxes

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'craft_v6_out/v70_asteroids_calibration'
VARIANTS=('nominal_preserved','current_phase57','phase_signed_y','phase_signed_cohort_size')


def predict(ram,variant):
    if variant=='nominal_preserved':
        return [o for o in decode_asteroids_ram(ram) if o.kind=='asteroid']
    r=ram.astype(int)
    if not r[57]&1:return []
    if variant=='current_phase57':
        return [o for o in decode_asteroids_ram(ram) if o.kind=='asteroid']
    result=[]
    for row in ASTEROID_ROWS:
        if not r[row+18] or r[row]==224:continue
        yy=int(r[row]) if r[row]<128 else int(r[row])-256
        size=int(r[row+36])&127
        group=0 if size<32 else 1 if size<48 else 2
        w,h=((16,28),(8,15),(4,8))[group]
        x,y=x_position(r[row+18]),24+2*yy
        if variant=='phase_signed_cohort_size':
            y+=int(row>=12)
            if group:
                x-=1;y-=2
                if group==1:h=14
        left,top,right,bottom=max(0,x),max(18,y),min(160,x+w),min(194,y+h)
        if left<right and top<bottom:
            result.append(SlottedLabel('asteroid',(left,top,right-left,bottom-top),str(row)))
    return result


if __name__=='__main__':
    out=BASE/'rock_phase_geometry_development_diagnostic.json'
    if out.exists():raise ValueError('Preserve earlier diagnosis')
    inv=BASE/'development_palette_inventory_v0.npz'
    meta=json.loads((BASE/'development_palette_inventory_v0.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(inv.read_bytes()).hexdigest()==meta['payload_sha256']
    with np.load(inv,allow_pickle=False) as z:
        rams,boxes,kinds,frames=z['ram'],z['boxes'],z['kind'],z['frame']
    stats={v:Counter() for v in VARIANTS};examples={v:[] for v in VARIANTS}
    for index,ram in enumerate(rams):
        pixels=[SlottedLabel('asteroid',tuple(map(int,box)),'rgb') for box in boxes[(frames==index)&(kinds==1)]]
        for v in VARIANTS:
            truth=predict(ram,v);n,within,exact=compare_boxes(truth,pixels);s=stats[v]
            s.update(frames=1,count_equal=int(len(truth)==len(pixels)),positive=int(bool(truth or pixels)),
                     positive_equal=int(bool(truth or pixels) and len(truth)==len(pixels)),
                     ram_objects=len(truth),pixel_objects=len(pixels),matches=n,within1=within,exact=exact)
            if (len(truth)!=len(pixels) or within!=n) and len(examples[v])<16:
                examples[v].append(dict(frame=index+1,ram=[o.box for o in truth],pixels=[o.box for o in pixels]))
    result=dict(kind='asteroids_rock_phase_geometry_development_diagnostic',seed=17511,
                variants={v:dict(s) for v,s in stats.items()},first_disagreements=examples,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                inventory_sha256=meta['payload_sha256'],inventory_metadata_sha256=hashlib.sha256((BASE/'development_palette_inventory_v0.json').read_bytes()).hexdigest(),
                note='All8192 frames scored with unchanged nominal palette assignments. Variants are exploratory corrections, not reserved validation. Signed y excludes sentinel224; cohort12..19 adds1scanline; medium/small shift(-1,-2) and medium height14. Top raster boundary18 inferred from development. No criteria changed or frame excluded.',
                gate_result=None,agent_training_allowed=False)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['variants'],indent=2))
