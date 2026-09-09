"""Full-development diagnosis of coloured rock occlusion; all prior results preserved."""
from pathlib import Path
import hashlib
import json
from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.asteroids_rocks_v70 import screen_rocks
from ragnarok.evaluation.asteroids_occlusion_v70 import decode_occluded_rocks as decode_rocks
from scripts.calibrate_breakout_v70 import compare_boxes

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'craft_v6_out/v70_asteroids_calibration'
SOURCES=('ragnarok/evaluation/asteroids_rocks_v70.py','ragnarok/evaluation/asteroids_occlusion_v70.py','scripts/diagnose_asteroids_occlusion_v70.py',
         'ragnarok/evaluation/asteroids_v70.py','ragnarok/evaluation/cold_fixtures_v70.py',
         'scripts/calibrate_breakout_v70.py')

if __name__=='__main__':
    out=BASE/'rock_occlusion_development_v2.json'
    if out.exists():raise ValueError('Preserve earlier diagnostic')
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}
    s=dict(frames=0,count_equal=0,positive=0,positive_equal=0,ram_objects=0,pixel_objects=0,matches=0,within1=0,exact=0)
    examples=[]
    for index,(rgb,ram,episode,_) in enumerate(cold_fixture_frames(BASE/'development'),1):
        a,b=decode_rocks(ram),screen_rocks(rgb)
        n,within,exact=compare_boxes(a,b)
        s['frames']+=1;s['count_equal']+=len(a)==len(b)
        s['positive']+=bool(a or b);s['positive_equal']+=bool(a or b) and len(a)==len(b)
        s['ram_objects']+=len(a);s['pixel_objects']+=len(b);s['matches']+=n;s['within1']+=within;s['exact']+=exact
        if (len(a)!=len(b) or within!=n) and len(examples)<24:
            examples.append(dict(frame=index,episode=episode,ram=[o.box for o in a],pixels=[o.box for o in b]))
        if index%2048==0:print(f'rock raster {index}/8192',flush=True)
    assert hashes=={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}
    s.update(count_fraction=s['count_equal']/s['frames'],positive_fraction=s['positive_equal']/s['positive'] if s['positive'] else None,
             box_fraction=s['within1']/s['matches'] if s['matches'] else None)
    result=dict(kind='asteroids_rock_occlusion_development_diagnostic',seed=17511,counts=s,first_disagreements=examples,
                source_sha256=hashes,fixture_sha256=hashlib.sha256((BASE/'development/manifest.json').read_bytes()).hexdigest(),
                limits='Rock-only development diagnosis; not a complete three-category calibration. All frames and fragments retained. P0 supports occult P1; visible connected pieces are grouped through torus connectivity within each allocation. Different allocations are never merged in RAM, even when colours coincide. Pieces do not certify physical lifetimes.',gate_result=None,agent_training_allowed=False)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(s,indent=2))
