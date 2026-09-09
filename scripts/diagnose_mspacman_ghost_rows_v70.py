"""Offline development inventory of ghost top-row visibility; no new capture."""
import json
from collections import Counter,defaultdict
import numpy as np
from scripts.capture_mspacman_pooled_v70 import DESTINATION,ROOT,sha,write_new
from ragnarok.evaluation.triple_pooled_fixtures_v70 import triple_pooled_fixture_frames
from ragnarok.evaluation.mspacman_ghosts_v70 import ghost_draw_entries,GHOST_COLOURS


def diagnose():
    total=Counter();empty=Counter();hist=defaultdict(Counter);examples=[]
    for row in triple_pooled_fixture_frames(DESTINATION):
        for now,before,rgbkey in [('ram_previous','ram_before_previous','raw_previous'),
                                  ('ram_current','ram_previous','raw_current')]:
            ram,prev,rgb=row[now],row[before],row[rgbkey]
            for e in ghost_draw_entries(ram,prev):
                x,y,_,_=e.box
                if 0<=y<171:
                    top=rgb[y,[(x+j)%160 for j in (2,3,4,5)]]
                    n=int(np.all(top==GHOST_COLOURS[e.colour_code],axis=1).sum())
                    total[y%12]+=1;empty[y%12]+=n==0;hist[y%12][n]+=1
                    if y%12!=1 and n==0 and len(examples)<16:
                        examples.append(dict(decision=int(row['index']),source=rgbkey,x=x,y=y,
                                             form=e.form_code,colour=e.colour_code))
    sources=('scripts/diagnose_mspacman_ghost_rows_v70.py',
        'ragnarok/evaluation/triple_pooled_fixtures_v70.py','ragnarok/evaluation/mspacman_ghosts_v70.py')
    return dict(kind='mspacman_development_top_row_inventory',seed=17421,
        total=dict(total),empty=dict(empty),hist={k:dict(v) for k,v in hist.items()},other_examples=examples,
        fixture_manifest_sha256=sha(DESTINATION/'manifest.json'),source_sha256={p:sha(ROOT/p) for p in sources},
        scope='All ghost draw-list entries in the two source frames. Check the four nominal top pixels; no result selection.',
        agent_training_allowed=False,gate_result=None)


if __name__=='__main__':
    out=DESTINATION.parent/'pooled_ghost_top_row_inventory_v0.json'
    if out.exists():raise ValueError('Preserve previous inventory')
    result=diagnose();write_new(out,result)
    print(json.dumps({k:result[k] for k in ('total','empty','hist','other_examples')},indent=2))
