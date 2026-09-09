"""Infer compact development player sprite supports for an external RAM judge.

The mode per animation/direction is diagnostic evidence, not a held-out score.
This reads archived RGB/RAM only and cannot start an emulator or training run.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/"craft_v6_out/v70_mspacman_calibration/development"


def run():
    groups=defaultdict(Counter)
    records=[]
    previous=None
    ep=None
    for index,(rgb,ram,episode,_) in enumerate(cold_fixture_frames(FIXTURE),1):
        r=previous if previous is not None and ep==episode else ram
        x,y=int(r[10])-12,int(r[16])+1
        key=(int(r[55]),int(r[56]))
        mask=np.all(rgb==(210,164,74),axis=2)|np.all(rgb==(200,72,72),axis=2)
        rows=tuple(sum(int(mask[y+dy,x+dx])<<(7-dx) for dx in range(8)
                       if 0<=x+dx<160 and 0<=y+dy<171) for dy in range(10))
        if 0<=x and x+8<=160 and 0<=y and y+10<=171:
            groups[key][rows]+=1
        records.append((index,episode,key,x,y,rows))
        previous=ram.copy();ep=episode
    modes={key:counts.most_common(1)[0][0] for key,counts in groups.items()}
    mismatch=Counter();examples=[]
    for index,episode,key,x,y,observed in records:
        expected=tuple(sum((modes[key][dy] & (1<<(7-dx))) for dx in range(8)
                           if 0<=x+dx<160 and 0<=y+dy<171) for dy in range(10))
        d=sum((a^b).bit_count() for a,b in zip(expected,observed))
        mismatch[d]+=1
        if d and len(examples)<16:
            examples.append(dict(frame=index,episode=episode,phase_direction=key,position=(x,y),
                                 expected=expected,observed=observed,different_pixels=d))
    result=dict(kind="mspacman_player_opaque_support_development_diagnostic",frames=len(records),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                fixture_manifest_sha256=hashlib.sha256((FIXTURE/"manifest.json").read_bytes()).hexdigest(),
                candidate_basis="Previous-episode-local RAM55/56 animation and orientation; yellow plus red-bow support within8x10 at previous RAM10-12,RAM16+1. Modal patterns inferred only on wholly in-ROI examples; clipped patterns scored on every archived frame.",
                patterns=[dict(phase_direction=key,whole_frames=sum(counts.values()),
                               modes=[dict(rows=k,n=v) for k,v in counts.most_common(5)])
                          for key,counts in sorted(groups.items())],
                different_pixel_histogram=[dict(n_pixels=k,n_frames=v) for k,v in sorted(mismatch.items())],
                first_differences=examples,
                limits=["In-sample template inference; independent reserved validation remains closed.",
                        "A red ghost can contaminate the red-bow pixel measurement during overlap; the modal crop is a hypothesis, not a guaranteed exact renderer.",
                        "One-pixel box accuracy does not establish exact silhouette or occlusion accuracy.",
                        "Do not use RGB to condition the RAM judge at inference time."],
                agent_training_allowed=False,gate_result=None)
    with (FIXTURE.parent/"player_opaque_support_development_diagnostic.json").open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(patterns=[dict(key=v["phase_direction"],mode=v["modes"][0]) for v in result["patterns"]],
                         different_pixel_histogram=result["different_pixel_histogram"]),indent=2))


if __name__=="__main__":
    run()
