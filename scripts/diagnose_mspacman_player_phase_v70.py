"""Development-only renderer timing diagnosis; never a learned-agent input."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "craft_v6_out/v70_mspacman_calibration/development"


def diagnose():
    records = []
    previous = None
    previous_episode = None
    for index, (rgb, ram, episode, _) in enumerate(cold_fixture_frames(FIXTURE), 1):
        mask = np.all(rgb[:171] == (210, 164, 74), axis=2)
        ys, xs = np.where(mask)
        if not len(xs):
            raise ValueError("Previously observed player support absent")
        box = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
        prior_available = previous is not None and episode == previous_episode
        prior = previous if prior_available else ram
        origins = ((int(ram[10]) - 12, int(ram[16]) + 1),
                   (int(prior[10]) - 12, int(prior[16]) + 1))
        signatures = [(box[0]-x, box[1]-y, box[2]-x, box[3]-y) for x,y in origins]
        x,y = origins[1]
        glyph = tuple(sum(int(mask[y+dy, x+dx]) << (7-dx) for dx in range(8)
                          if 0 <= x+dx < 160 and 0 <= y+dy < 171) for dy in range(10))
        records.append(dict(frame=index, episode=episode, prior_available=prior_available,
                            signatures=signatures, glyph=glyph,
                            current=[int(v) for v in ram], prior=[int(v) for v in prior]))
        previous = ram.copy()
        previous_episode = episode
    result = dict(kind="mspacman_player_render_phase_development_diagnostic", frames=len(records),
                  fixture_manifest_sha256=hashlib.sha256((FIXTURE/"manifest.json").read_bytes()).hexdigest(),
                  diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  no_prior_frame_count=sum(not r["prior_available"] for r in records),
                  conventions="Box signatures are left,top,right,bottom relative to x=RAM10-12,y=RAM16+1; all edges exclusive.",
                  coordinate_candidates={}, phase_hypotheses=[], source_byte_cardinality={})
    for i,name in enumerate(("current", "prior_same_episode_else_current")):
        counts=Counter(r["signatures"][i] for r in records)
        result["coordinate_candidates"][name] = dict(
            fixed_8x10_within1=sum(max(abs(a-b) for a,b in zip(r["signatures"][i],(0,0,8,10)))<=1 for r in records),
            signatures=[dict(edges=k,n=v) for k,v in counts.most_common(35)])
    for byte in range(128):
        result["source_byte_cardinality"][str(byte)]=len({r["current"][byte] for r in records})
    for phase in ("current","prior"):
        for cols in ((55,56),(54,55,56),(51,55,56),(55,56,30),(55,56,39)):
            groups=defaultdict(Counter)
            for r in records:
                key=tuple(r[phase][b] for b in cols)
                groups[key][r["signatures"][1]]+=1
            modal={key:counts.most_common(1)[0][0] for key,counts in groups.items()}
            exact=within=0
            errors=[]
            for r in records:
                key=tuple(r[phase][b] for b in cols)
                distance=max(abs(a-b) for a,b in zip(r["signatures"][1],modal[key]))
                exact+=distance==0
                within+=distance<=1
                if distance>1 and len(errors)<20:
                    errors.append(dict(frame=r["frame"],key=key,observed=r["signatures"][1],modal=modal[key]))
            entry=dict(phase=phase,bytes=cols,n_groups=len(groups),modal_exact=exact,modal_within1=within,
                       evaluation="In-sample explanatory diagnostic only, not a held-out calibration",first_errors=errors)
            if len(groups)<=64:
                entry["groups"]=[dict(key=key,n=sum(counts.values()),modal=modal[key],
                                      signatures=[dict(edges=k,n=v) for k,v in counts.most_common(8)])
                                 for key,counts in sorted(groups.items())]
            result["phase_hypotheses"].append(entry)
    glyphs=Counter(r["glyph"] for r in records)
    result["common_glyphs"]=[dict(rows=k,n=v) for k,v in glyphs.most_common(35)]
    result["limits"]=["Development data only; no reserved seed opened.",
                      "Modal signatures fitted and evaluated on the same data identify possible renderer fields only.",
                      "A general phase rule must be established before independent reserved calibration.",
                      "First ordinary frame of each episode has no archived predecessor; current RAM is an explicit fallback."]
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Preserve existing diagnosis")
    result=diagnose()
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(frames=result["frames"],coordinates={k:v["fixed_8x10_within1"] for k,v in result["coordinate_candidates"].items()},
                          phases=[{k:v for k,v in x.items() if k not in ("groups","first_errors")} for x in result["phase_hypotheses"]]),indent=2))
