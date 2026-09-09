"""Complete ghost-only reproduction of temporal V1, retaining every mismatch."""
import json
from collections import Counter
import numpy as np
from scripts.capture_mspacman_pooled_v70 import DESTINATION, ROOT, sha, write_new
from scripts.calibrate_mspacman_pooled_rows_v70 import new_stats,accumulate,finish
from ragnarok.evaluation.triple_pooled_fixtures_v70 import triple_pooled_fixture_frames
from ragnarok.evaluation.mspacman_maze_visible_v70 import MsPacmanPixelJudge,wrapped_ghost_supports,support_boxes
from ragnarok.evaluation.mspacman_ghosts_v70 import GHOST_COLOURS
from ragnarok.evaluation.pooled_mspacman_v70 import _masks,_add,_objects
from ragnarok.evaluation.pong_v70 import ObjectLabel


def diagnose():
    judge=MsPacmanPixelJudge();stats=new_stats();errors=[];shapes=Counter()
    for row in triple_pooled_fixture_frames(DESTINATION):
        truth_mask,pixel_mask=_masks(),_masks()
        sources=[]
        for now,before,rgbkey,otherkey in [('ram_previous','ram_before_previous','raw_previous','raw_current'),
                                          ('ram_current','ram_previous','raw_current','raw_previous')]:
            ram,prev,rgb,other=(row[k] for k in (now,before,rgbkey,otherkey))
            for e,mask in wrapped_ghost_supports(ram,prev):
                y=e.box[1]
                if 0<=y<171 and y%12==1:mask[y,:]=False
                for obj in support_boxes(mask,'ghost',f'draw:{e.index}'):_add(truth_mask,obj)
            raw=[o for o in judge.step(rgb) if o.kind=='ghost']
            details=[]
            for obj in raw:
                x,y,w,h=obj.box
                own,competing=rgb[y:y+h,x:x+w],other[y:y+h,x:x+w]
                palette=np.any(np.stack([np.all(own==c,axis=2) for c in GHOST_COLOURS]),axis=0)
                keep=palette&(np.all(own==competing,axis=2)|np.any(own>competing,axis=2))
                yy,xx=np.nonzero(keep)
                if len(xx):
                    _add(pixel_mask,ObjectLabel('ghost',(x+int(xx.min()),y+int(yy.min()),
                        int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1))))
                details.append(dict(box=obj.box,bitmap=[''.join('1' if v else '0' for v in line) for line in palette]))
            sources.append(dict(source=rgbkey,objects=details))
        a,b=_objects(truth_mask),_objects(pixel_mask)
        n,within=accumulate(stats,a,b)
        if len(a)!=len(b):
            errors.append(dict(decision=int(row['index']),ram=[o.box for o in a],pixels=[o.box for o in b],raw_sources=sources))
            for obj in b:
                if not any(max(abs(obj.box[0]-t.box[0]),abs(obj.box[1]-t.box[1]),
                    abs(obj.box[0]+obj.box[2]-t.box[0]-t.box[2]),abs(obj.box[1]+obj.box[3]-t.box[1]-t.box[3]))<=1 for t in a):
                    shapes[str(obj.box[2:])]+=1
    finish(stats)
    original=json.loads((DESTINATION.parent/'pooled_geometry_development_v1.json').read_text(encoding='utf-8'))
    if stats!=original['per_type']['ghost']:
        raise ValueError('Ghost-only reproduction differs from the full V1 result')
    return dict(kind='mspacman_pooled_v1_complete_ghost_error_inventory',seed=17421,stats=stats,
        count_error_frames=len(errors),unmatched_pixel_shapes=dict(shapes),errors=errors,
        fixture_manifest_sha256=sha(DESTINATION/'manifest.json'),original_result_sha256=sha(DESTINATION.parent/'pooled_geometry_development_v1.json'),
        diagnostic_source_sha256=sha(ROOT/'scripts/diagnose_mspacman_ghost_counts_v70.py'),
        agent_training_allowed=False,gate_result=None)


if __name__=='__main__':
    out=DESTINATION.parent/'pooled_ghost_count_inventory_v1.json'
    if out.exists():raise ValueError('Preserve the earlier inventory')
    result=diagnose();write_new(out,result)
    print(json.dumps({k:result[k] for k in ('count_error_frames','unmatched_pixel_shapes','stats')},indent=2))
