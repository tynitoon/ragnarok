"""Inspect the player entry preceding the ghost draw list on development only."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/"craft_v6_out/v70_mspacman_calibration/development"


def run():
    groups={p:defaultdict(Counter) for p in ("current","prior")}
    glyphs={p:defaultdict(Counter) for p in groups}
    colours=defaultdict(Counter)
    offsets=Counter()
    triplets=Counter()
    previous=None
    ep=None
    for index,(rgb,ram,episode,_) in enumerate(cold_fixture_frames(FIXTURE),1):
        prior=previous if previous is not None and ep==episode else ram
        triplets[tuple(int(ram[b]) for b in (34,40,46))]+=1
        yellow=np.all(rgb[:171]==(210,164,74),axis=2)
        ys,xs=np.where(yellow)
        box=(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1))
        for phase,r in (("current",ram),("prior",prior)):
            encoded=int(r[34]);hi=encoded>>4
            x=12*(encoded&15)-(hi if hi<8 else hi-16)
            y=181-int(r[46])
            code=int(r[40])
            groups[phase][code][(box[0]-x,box[1]-y,box[2]-x,box[3]-y)]+=1
            glyph=tuple(sum(int(yellow[y+dy,x+dx])<<(7-dx) for dx in range(8)
                            if 0<=x+dx<160 and 0<=y+dy<171) for dy in range(10))
            glyphs[phase][code][glyph]+=1
            if phase=="prior":
                offsets[(x-(int(r[10])-12),y-(int(r[16])+1))]+=1
                for dy in range(10):
                    for dx in range(8):
                        if 0<=x+dx<160 and 0<=y+dy<171:
                            colour=tuple(int(v) for v in rgb[y+dy,x+dx])
                            colours[code][colour]+=1
        previous=ram.copy();ep=episode
    result=dict(kind="mspacman_player_drawlist_development_diagnostic",frames=index,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                fixture_manifest_sha256=hashlib.sha256((FIXTURE/"manifest.json").read_bytes()).hexdigest(),
                coordinate_formula="x=12*(RAM34&15)-signed4(RAM34>>4), y=181-RAM46, sprite code=RAM40",
                observed_triplets=[dict(values=k,n=v) for k,v in triplets.most_common()],
                rejected_constant_zero_triplet=set(triplets)=={(0,0,0)},
                prior_offsets_from_logical_position=[dict(dxdy=k,n=v) for k,v in offsets.most_common()],
                phases={})
    for phase,bycode in groups.items():
        result["phases"][phase]=[
            dict(code=code,n=sum(c.values()),
                 boxes=[dict(edges=k,n=v) for k,v in c.most_common(10)],
                 glyphs=[dict(rows=k,n=v) for k,v in glyphs[phase][code].most_common(8)],
                 colours=[dict(rgb=k,n=v) for k,v in colours[code].most_common(12)])
            for code,c in sorted(bycode.items())]
    result["limits"]=["Development diagnostic, not reserved validation.",
                      "Graphics code and coordinates are inspected independently of logical animation hypotheses.",
                      "Glyph frequencies are in-sample; occluding ghosts, bow and maze can share crop pixels."]
    target=FIXTURE.parent/"player_drawlist_development_diagnostic.json"
    with target.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(frames=index,rejected_constant_zero_triplet=result["rejected_constant_zero_triplet"],
                         observed_triplets=result["observed_triplets"]),indent=2))


if __name__=="__main__":
    run()
