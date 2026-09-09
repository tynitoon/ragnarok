"""Development composition of independent RAM and causal RGB instruments."""
import numpy as np

from ragnarok.evaluation.mspacman_v70 import SlottedLabel, decode_mspacman_ram, screen_mspacman_objects
from ragnarok.evaluation.mspacman_render_v70 import player_render_box, power_pill_render_boxes
from ragnarok.evaluation.mspacman_ghosts_v70 import decode_ghost_drawlist, screen_ghosts
from ragnarok.evaluation.mspacman_support_v70 import actor_support
from ragnarok.evaluation.mspacman_pixels_v70 import MsPacmanPillPixels

REVISION="mspacman_raw_visible_development_v1"


def decode_visible_ram(current_ram,previous_ram=None):
    """No RGB input: predict surviving pill pixels under RAM sprite supports."""
    nominal=decode_mspacman_ram(current_ram)
    result=player_render_box(current_ram,previous_ram)+decode_ghost_drawlist(current_ram,previous_ram)
    result+=power_pill_render_boxes(current_ram)
    result.extend(o for o in nominal if o.kind=="fruit")  # Still uncalibrated until fruit is observed.
    occupied=actor_support(current_ram,previous_ram)
    for o in nominal:
        if o.kind!="pill":
            continue
        x,y,w,h=o.box
        ys,xs=np.where(~occupied[y:y+h,x:x+w])
        if len(xs):
            result.append(SlottedLabel("pill",(x+int(xs.min()),y+int(ys.min()),
                          int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)),o.slot))
    return result


class MsPacmanPixelJudge:
    """RGB current frame and past RGB only; no RAM or oracle episode resets."""
    def __init__(self):
        self.pills=MsPacmanPillPixels()

    def step(self,rgb):
        ordinary=[o for o in screen_mspacman_objects(rgb) if o.kind in ("player","fruit")]
        return ordinary+screen_ghosts(rgb)+self.pills.step(rgb)
