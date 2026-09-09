"""Third development composition: maze colours, wall flashes and ghost wraps.

The pixel checker learns walls and pill anchors from past/current RGB only.
The RAM judge separately predicts visible sprite supports; no RGB is its input.
Old V0/V1/V2 sources and results remain unchanged.
"""
import numpy as np
from scipy import ndimage
from ragnarok.evaluation.mspacman_v70 import SlottedLabel,decode_mspacman_ram,screen_mspacman_objects
from ragnarok.evaluation.mspacman_render_v70 import _ram,power_pill_render_boxes
from ragnarok.evaluation.mspacman_ghosts_v70 import ghost_draw_entries,screen_ghosts
from ragnarok.evaluation.mspacman_support_v70 import GHOST_NORMAL_ROWS,GHOST_FRIGHTENED_ROWS,MIRROR_BYTE
from ragnarok.evaluation.mspacman_player_list_v70 import player_list_render_box,player_list_support
from ragnarok.evaluation.mspacman_fruit_v70 import fruit_render_boxes,fruit_support
from ragnarok.evaluation.mspacman_pixels_v70 import MsPacmanPillPixels

REVISION='mspacman_raw_visible_maze_development_v3'
MAZE_COLOURS=((228,111,111),(101,111,228))
GRAY=(214,214,214)


def wrapped_ghost_supports(current_ram,previous_ram=None):
    current=_ram(current_ram)
    for entry in ghost_draw_entries(current_ram,previous_ram):
        rows=GHOST_FRIGHTENED_ROWS if entry.colour_code>=4 else GHOST_NORMAL_ROWS
        if not current[30]&16:rows=tuple(MIRROR_BYTE[row] for row in rows)
        x,y,_,_=entry.box
        if not 0<=x<160:raise ValueError('Uncalibrated ghost horizontal draw position')
        mask=np.zeros((171,160),dtype=bool)
        for dy,row in enumerate(rows):
            if 0<=y+dy<171:
                for dx in range(8):
                    if row&(1<<(7-dx)):mask[y+dy,(x+dx)%160]=True
        yield entry,mask


def support_boxes(mask,kind,slot):
    """One envelope per visible column band; a wrapped sprite has two pieces."""
    cols=np.where(mask.any(axis=0))[0]
    if not len(cols):return []
    groups=np.split(cols,np.where(np.diff(cols)>1)[0]+1)
    result=[]
    for i,g in enumerate(groups):
        left,right=int(g[0]),int(g[-1])+1
        ys,xs=np.where(mask[:,left:right])
        result.append(SlottedLabel(kind,(left,int(ys.min()),right-left,int(ys.max()-ys.min()+1)),f'{slot}:{i}'))
    return result


def decode_visible_ram(current_ram,previous_ram=None):
    occupied=player_list_support(current_ram,previous_ram)|fruit_support(current_ram,previous_ram)
    result=player_list_render_box(current_ram,previous_ram)+fruit_render_boxes(current_ram,previous_ram)
    for entry,mask in wrapped_ghost_supports(current_ram,previous_ram):
        occupied|=mask
        result.extend(support_boxes(mask,'ghost',f'draw:{entry.index}'))
    pills=[o for o in decode_mspacman_ram(current_ram) if o.kind=='pill']
    pills+=power_pill_render_boxes(current_ram)
    for o in pills:
        x,y,w,h=o.box
        ys,xs=np.where(~occupied[y:y+h,x:x+w])
        if len(xs):
            result.append(SlottedLabel(o.kind,(x+int(xs.min()),y+int(ys.min()),
                       int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)),o.slot))
    return result


class MsPacmanPixelJudge:
    """Causal separate registries for the two observed maze palettes.

    Gray wall flashes are removed only at wall pixels learned from earlier
    large maze components, and only when at least95% of a substantial learned
    wall mask is gray. This preserves unrelated gray objects and unseen pixels.
    """
    def __init__(self):
        self.layout=None
        self.pills={colour:MsPacmanPillPixels() for colour in MAZE_COLOURS}
        self.walls={colour:np.zeros((171,160),dtype=bool) for colour in MAZE_COLOURS}

    def step(self,rgb):
        rgb=np.asarray(rgb)
        if rgb.shape!=(210,160,3) or rgb.dtype!=np.uint8:
            raise ValueError('RGB must be uint8[210,160,3]')
        masks={colour:np.all(rgb[:171]==colour,axis=2) for colour in MAZE_COLOURS}
        colour=max(MAZE_COLOURS,key=lambda c:int(masks[c].sum()))
        if masks[colour].sum()>1000:
            self.layout=colour
            labels,_=ndimage.label(masks[colour],np.ones((3,3),dtype=np.uint8))
            for i,(sy,sx) in enumerate(ndimage.find_objects(labels),1):
                if sx.stop-sx.start>4 or sy.stop-sy.start>7:
                    self.walls[colour][sy,sx]|=labels[sy,sx]==i
        cleaned=rgb.copy()
        if self.layout is not None:
            wall=self.walls[self.layout]
            gray=np.all(rgb[:171]==GRAY,axis=2)
            if wall.sum()>2000 and np.count_nonzero(wall&gray)>=.95*int(wall.sum()):
                cleaned[:171][wall&gray]=0
        ordinary=[o for o in screen_mspacman_objects(rgb) if o.kind in ('player','fruit')]
        result=ordinary+screen_ghosts(cleaned)
        if self.layout is not None:
            normalized=rgb.copy()
            normalized[:171][masks[self.layout]]=MAZE_COLOURS[0]
            result+=self.pills[self.layout].step(normalized)
        return result
