"""Second development composition with independent special-list actor support."""
import numpy as np
from ragnarok.evaluation.mspacman_v70 import SlottedLabel,decode_mspacman_ram
from ragnarok.evaluation.mspacman_render_v70 import power_pill_render_boxes,_ram
from ragnarok.evaluation.mspacman_ghosts_v70 import decode_ghost_drawlist,ghost_draw_entries
from ragnarok.evaluation.mspacman_support_v70 import GHOST_NORMAL_ROWS,GHOST_FRIGHTENED_ROWS,MIRROR_BYTE,add_support
from ragnarok.evaluation.mspacman_player_list_v70 import player_list_render_box,player_list_support
from ragnarok.evaluation.mspacman_fruit_v70 import fruit_render_boxes,fruit_support
from ragnarok.evaluation.mspacman_visible_v70 import MsPacmanPixelJudge

REVISION='mspacman_raw_visible_special_list_development_v2'


def actor_support(current_ram,previous_ram=None):
    current=_ram(current_ram)
    canvas=player_list_support(current_ram,previous_ram)|fruit_support(current_ram,previous_ram)
    for entry in ghost_draw_entries(current_ram,previous_ram):
        rows=GHOST_FRIGHTENED_ROWS if entry.colour_code>=4 else GHOST_NORMAL_ROWS
        if not current[30]&16:
            rows=tuple(MIRROR_BYTE[row] for row in rows)
        x,y,_,_=entry.box
        add_support(canvas,rows,x,y)
    return canvas


def decode_visible_ram(current_ram,previous_ram=None):
    result=player_list_render_box(current_ram,previous_ram)+fruit_render_boxes(current_ram,previous_ram)
    result+=decode_ghost_drawlist(current_ram,previous_ram)+power_pill_render_boxes(current_ram)
    occupied=actor_support(current_ram,previous_ram)
    for o in decode_mspacman_ram(current_ram):
        if o.kind!='pill':continue
        x,y,w,h=o.box
        ys,xs=np.where(~occupied[y:y+h,x:x+w])
        if len(xs):
            result.append(SlottedLabel('pill',(x+int(xs.min()),y+int(ys.min()),
                       int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)),o.slot))
    return result
