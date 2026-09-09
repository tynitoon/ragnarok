"""Inventory the returning-eye hypothesis on every source in development17421.

No emulator, held-out frames, agent or new category. Record mismatches instead
of using RGB to choose which RAM prediction will be scored.
"""
from collections import Counter
import numpy as np
from scripts.capture_mspacman_pooled_v70 import DESTINATION,ROOT,sha,write_new
from ragnarok.evaluation.triple_pooled_fixtures_v70 import triple_pooled_fixture_frames
from ragnarok.evaluation.mspacman_ghosts_v70 import ghost_draw_entries,GHOST_COLOURS
from ragnarok.evaluation.mspacman_player_list_v70 import player_list_support
from ragnarok.evaluation.mspacman_fruit_v70 import fruit_support


def diagnose():
    entries=[];states=Counter();phases=Counter()
    for row in triple_pooled_fixture_frames(DESTINATION):
        for nowkey,prevkey,rgbkey in [('ram_previous','ram_before_previous','raw_previous'),
                                       ('ram_current','ram_previous','raw_current')]:
            now,prev,rgb=(row[k] for k in (nowkey,prevkey,rgbkey));r=prev.astype(int)
            for e in ghost_draw_entries(now,prev):
                if e.colour_code!=1:continue
                x,y,_,_=e.box;y+=47
                matches=[i for i in range(4) if x==(r[6+i]-12)%160 and y==r[12+i]+2]
                states.update(int(r[1+i])>>4 for i in matches)
                eyes=[i for i in matches if int(r[1+i])>>4 in (4,6)]
                if not eyes:continue
                if len(eyes)!=1:raise ValueError('Ambiguous eye allocation')
                mask=np.zeros((171,160),dtype=bool)
                for dy,bits in enumerate((66,165,165,165,66)):
                    if 0<=y+dy<171:
                        for dx in range(8):
                            if bits&(1<<(7-dx)):mask[y+dy,(x+dx)%160]=True
                visible=mask&~(player_list_support(now,prev)|fruit_support(now,prev))
                actual=np.zeros_like(mask)
                for dx in range(8):
                    if 0<=y<y+5<=171:
                        actual[y:y+5,(x+dx)%160]=np.all(rgb[y:y+5,(x+dx)%160]==GHOST_COLOURS[1],axis=1)
                fp=int(np.count_nonzero(visible&~actual));fn=int(np.count_nonzero(actual&~visible))
                phases[str((y%12,fp,fn))]+=1
                entries.append(dict(decision=int(row['index']),source=rgbkey,slot=e.index,
                    allocation=eyes[0],state=int(r[1+eyes[0]]),form=e.form_code,
                    old_box=e.box,eye_box=(x,y,8,5),predicted_pixels=int(mask.sum()),
                    visible_pixels=int(visible.sum()),actual_pixels=int(actual.sum()),
                    false_positive_pixels=fp,false_negative_pixels=fn,
                    predicted_rows=[''.join('1' if visible[yy,(x+dx)%160] else '0' for dx in range(8))
                        for yy in range(max(0,y),min(171,y+5))],
                    actual_rows=[''.join('1' if actual[yy,(x+dx)%160] else '0' for dx in range(8))
                        for yy in range(max(0,y),min(171,y+5))]))
    return dict(kind='mspacman_returning_eye_development_hypothesis_inventory',seed=17421,
        hypothesis='Cyan draw entry with exact x and pointer-derived y matching one logical state high nibble4/6; y=228-render_y, glyph66,165,165,165,66; player/fruit occlusion.',
        all_coordinate_matching_state_nibbles=dict(states),entries=len(entries),
        exact_visible_supports=sum(e['false_positive_pixels']==e['false_negative_pixels']==0 for e in entries),
        row_phase_errors=dict(phases),evidence=entries,
        source_sha256=sha(ROOT/'scripts/diagnose_mspacman_eyes_v70.py'),
        fixture_manifest_sha256=sha(DESTINATION/'manifest.json'),
        agent_training_allowed=False,gate_result=None)


if __name__=='__main__':
    import json
    out=DESTINATION.parent/'pooled_returning_eye_inventory_v1.json'
    if out.exists():raise ValueError('Preserve the earlier eye inventory')
    result=diagnose();write_new(out,result)
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'},indent=2))
