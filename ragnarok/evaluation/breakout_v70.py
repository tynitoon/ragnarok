"""Experimenter-only Breakout geometry; RAM projection and separate pixel checker.

RAM facts originate in OCAtari commit467accd5f3da379d83705b008094ffb787eec8a1,
ocatari/ram/breakout.py (MIT; docs/licenses/OCAtari-MIT.txt). This candidate is
uncalibrated until the independent development/validation protocol passes.
"""
import numpy as np
from scipy import ndimage

from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION = "breakout_raw_ball_render_clip_v1"
COLOURS = ((200,72,72), (198,108,58), (180,122,48), (162,162,42), (72,160,72), (66,72,200))
KINDS = ("player", "ball") + tuple(f"bricks_{i}" for i in range(6))


def _clipped(kind, x, y, w, h):
    x0,y0,x1,y1 = max(8,x),max(32,y),min(152,x+w),min(197,y+h)
    return ObjectLabel(kind,(x0,y0,x1-x0,y1-y0)) if x1>x0 and y1>y0 else None


def decode_breakout_ram(ram):
    if not isinstance(ram,np.ndarray) or ram.dtype!=np.uint8 or ram.shape!=(128,):
        raise ValueError("Expected128uint8 RAM bytes")
    r = list(map(int,ram))
    objects = [_clipped("player",r[72]-47,189,16,4)]
    # Empirical development clipping, not a universal render boundary: the one-pixel
    # band at y195 remains visible in11/28 cases. Those disagreements stay scored.
    # The RAM coordinate also persists after the rendered sprite disappears.
    if r[101] and r[101]+9<195:
        y = r[101]+9
        objects.append(_clipped("ball",r[99]-49,y,2,min(4,195-y)))
    # Unpack alternating playfield bits into screen order; adjacent cells form a run.
    order = (0,4,3,2,1,5,6,7,8,11,12,16,15,14,13,17,18,19)
    for row in range(6):
        base = 5-row
        bits = [(r[base+30]>>6)&1]
        bits += [(r[base+offset]>>shift)&1 for offset in (24,18,12,6) for shift in (0,2,4,6)]
        bits += [(r[base]>>shift)&1 for shift in (0,2,4)]
        line = [bits[j] for j in order]
        start = None
        for col,value in enumerate(line+[0]):
            if value and start is None:
                start = col
            if not value and start is not None:
                objects.append(_clipped(f"bricks_{row}",8+8*start,57+6*row,8*(col-start),6))
                start = None
    return [obj for obj in objects if obj is not None]


def _components(mask):
    labels,_ = ndimage.label(mask)
    return [(x.start,y.start,x.stop-x.start,y.stop-y.start) for y,x in ndimage.find_objects(labels)]


def screen_breakout_objects(rgb):
    """Independent pixel-only semantic calibration; no RAM projection is called."""
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.shape!=(210,160,3):
        raise ValueError("Expected raw210x160RGB")
    objects = []
    red = np.all(rgb==COLOURS[0],axis=2)
    player_mask = np.zeros(red.shape,dtype=bool)
    player_mask[183:197,8:152] = red[183:197,8:152]
    paddles = [box for box in _components(player_mask) if box[2]>=8]
    for box in paddles:
        objects.append(ObjectLabel("player",box))
    ball_mask = np.zeros(red.shape,dtype=bool)
    ball_mask[32:197,8:152] = red[32:197,8:152]
    ball_mask[57:63] = False  # indistinguishable red brick row; disagreements remain measured
    for x,y,w,h in paddles:
        ball_mask[y:y+h,x:x+w] = False
    for box in _components(ball_mask):
        if box[2]<=4 and box[3]<=8:
            objects.append(ObjectLabel("ball",box))
    for row,colour in enumerate(COLOURS):
        y = 57+6*row
        mask = np.zeros(red.shape,dtype=bool)
        mask[y:y+6,8:152] = np.all(rgb[y:y+6,8:152]==colour,axis=2)
        for box in _components(mask):
            objects.append(ObjectLabel(f"bricks_{row}",box))
    return objects
