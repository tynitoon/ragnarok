"""Separate all-frame inventory of unresolved rock raster/grouping errors."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.asteroids_v70 import ASTEROID_COLOURS
from ragnarok.evaluation.asteroids_rocks_v70 import rock_entries,rock_support
from ragnarok.evaluation.asteroids_occlusion_v70 import decode_occluded_rocks as decode_rocks
from ragnarok.evaluation.asteroids_deblend_v70 import screen_deblended_rocks as screen_rocks
from scripts.calibrate_breakout_v70 import compare_boxes

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'craft_v6_out/v70_asteroids_calibration'

if __name__=='__main__':
    out=BASE/'rock_deblend_error_inventory_v3.json';data=BASE/'rock_deblend_error_inventory_v3.npz'
    if out.exists() or data.exists():raise ValueError('Preserve prior evidence')
    stats=Counter();errors=[];all_counts=[];pixel_boxes=[];pixel_frames=[];ram_boxes=[];ram_frames=[]
    for index,(rgb,ram,episode,_) in enumerate(cold_fixture_frames(BASE/'development')):
        a,b=decode_rocks(ram),screen_rocks(rgb)
        n,within,exact=compare_boxes(a,b)
        for o in a:ram_boxes.append(o.box);ram_frames.append(index)
        for o in b:pixel_boxes.append(o.box);pixel_frames.append(index)
        all_counts.append((len(a),len(b),within,exact))
        predicted=np.zeros((210,160),dtype=bool)
        for entry in rock_entries(ram):predicted|=rock_support(entry)
        actual=np.zeros((210,160),dtype=bool)
        if int(ram[57])&1:
            for colour in ASTEROID_COLOURS:actual[16:194]|=np.all(rgb[16:194]==colour,axis=2)
        missed=int(np.count_nonzero(actual&~predicted));extra=int(np.count_nonzero(predicted&~actual))
        stats['frames']+=1;stats['raster_equal']+=missed==extra==0
        stats['missed_pixels']+=missed;stats['extra_pixels']+=extra
        if len(a)!=len(b) or within!=n:
            stats['count_error_frames']+=len(a)!=len(b);stats['geometry_error_frames']+=within!=n
            if missed==extra==0:stats['label_error_despite_exact_union']+=1
            errors.append(dict(frame=index+1,episode=episode,ram_count=len(a),pixel_count=len(b),
                               within1=within,matches=n,missing_pixels=missed,extra_pixels=extra))
        if (index+1)%2048==0:print(f'rock errors {index+1}/8192',flush=True)
    np.savez_compressed(data,counts=np.asarray(all_counts,dtype=np.int32),
                         ram_boxes=np.asarray(ram_boxes,dtype=np.int16),ram_frames=np.asarray(ram_frames,dtype=np.int32),
                         pixel_boxes=np.asarray(pixel_boxes,dtype=np.int16),pixel_frames=np.asarray(pixel_frames,dtype=np.int32))
    result=dict(kind='asteroids_rock_deblend_error_inventory',seed=17511,counts=dict(stats),errors=errors,
                payload_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('scripts/diagnose_asteroids_deblend_errors_v70.py','ragnarok/evaluation/asteroids_rocks_v70.py','ragnarok/evaluation/asteroids_occlusion_v70.py','ragnarok/evaluation/asteroids_deblend_v70.py')},
                note='Every frame retained. Union-of-coloured-pixels comparison diagnoses geometry separately from instance grouping; it is not a substitute for the frozen per-object criteria.',
                gate_result=None,agent_training_allowed=False)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['counts'],indent=2))
