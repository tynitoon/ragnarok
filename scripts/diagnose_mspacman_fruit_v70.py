"""Development-only fruit visibility/geometry; all frames and failures retained."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames
from ragnarok.evaluation.mspacman_v70 import PALETTES, SlottedLabel
from scripts.calibrate_breakout_v70 import compare_boxes

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "craft_v6_out/v70_mspacman_calibration/coverage_development_v1"


def diagnose():
    previous = None
    ep = None
    stats = Counter()
    patterns = Counter()
    last_values = Counter()
    disagreements = []
    code_boxes = {}
    for index, (rgb, current, episode, _) in enumerate(cold_fixture_frames(FIXTURE), 1):
        if ep != episode:
            previous = None
        r = current.astype(int) if previous is None else previous.astype(int)
        last_values[int(r[34])] += 1
        codes = [int(r[45+i]) for i in range(min(2, int(r[34])+1))]
        fruit_codes = [c for c in codes if c >> 4 != 7]
        patterns[(tuple(codes), bool(r[11] and r[17]))] += 1
        pixels = []
        for colour in PALETTES["fruit"]:
            labels, _ = ndimage.label(np.all(rgb[:171] == colour, axis=2), np.ones((3, 3), dtype=np.uint8))
            for sy, sx in ndimage.find_objects(labels):
                pixels.append(SlottedLabel("fruit", (sx.start, sy.start, sx.stop-sx.start, sy.stop-sy.start), "rgb"))
        truth = []
        if fruit_codes:
            x, y = int(r[11])-12, int(r[17])+1
            left, top, right, bottom = max(0,x), max(0,y), min(160,x+8), min(171,y+10)
            if left < right and top < bottom:
                truth.append(SlottedLabel("fruit", (left,top,right-left,bottom-top), "fruit"))
        n, within, exact = compare_boxes(truth, pixels)
        stats.update(frames=1, ram_objects=len(truth), pixel_objects=len(pixels), matches=n, within1=within, exact=exact,
                     count_equal=int(len(truth)==len(pixels)), positive=int(bool(truth or pixels)),
                     positive_equal=int(bool(truth or pixels) and len(truth)==len(pixels)))
        if (len(truth)!=len(pixels) or n!=within) and len(disagreements)<40:
            disagreements.append(dict(frame=index,episode=episode,previous_codes=codes,ram=truth[0].box if truth else None,
                                      pixels=[o.box for o in pixels],r11=int(r[11]),r17=int(r[17]),r34=int(r[34])))
        if len(pixels)==1 and fruit_codes:
            key=str(tuple(fruit_codes))
            counts=code_boxes.setdefault(key,Counter())
            x,y,w,h=pixels[0].box
            counts[(x-int(r[11]),y-int(r[17]),w,h)]+=1
        previous=current.copy();ep=episode
        if index%8192==0:print(f"fruit diagnostic {index}/32768",flush=True)
    return dict(kind="mspacman_fruit_render_list_development_diagnostic",seed=17431,
                hypothesis="Previous RAM34+1 special entries at45/46; high nibble7 player, others fruit. Fruit logical prior xy(-12,+1),8x10. No RGB in RAM hypothesis. Independent per-colour8-connected components.",
                counts=dict(stats),last_values=dict(last_values),patterns={str(k):v for k,v in patterns.items()},
                code_offsets={k:[[list(a),n] for a,n in c.most_common()] for k,c in code_boxes.items()},
                first_disagreements=disagreements,
                fixture_sha256=hashlib.sha256((FIXTURE/'manifest.json').read_bytes()).hexdigest(),
                diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                gate_result=None,agent_training_allowed=False)


if __name__ == "__main__":
    out=FIXTURE.parent/'fruit_list_development_diagnostic.json'
    if out.exists():raise ValueError("Preserve earlier diagnosis")
    result=diagnose()
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['counts'],indent=2))
