"""Development-only RAM sprite supports for external occlusion accounting.

Twelve compact yellow player patterns observed on seed17411. Their common
silhouettes remain hypotheses until tested on occlusions and reserved data.
All inference inputs below are RAM or explicit sprite coordinates, never RGB.
"""
import numpy as np

from ragnarok.evaluation.mspacman_render_v70 import _ram
from ragnarok.evaluation.mspacman_ghosts_v70 import ghost_draw_entries

PLAYER_YELLOW_ROWS = (
    ((0,0,65,99,99,119,87,119,190,92),
     (92,184,104,112,96,96,112,120,56,28),
     (58,125,238,234,238,198,198,130,0,0),
     (58,29,22,14,6,6,14,30,28,56)),
    ((0,34,99,99,119,119,87,119,190,92),
     (92,190,110,120,96,96,120,126,62,28),
     (58,125,238,234,238,238,198,198,68,0),
     (58,125,118,30,6,6,30,126,124,56)),
    ((28,62,127,127,127,127,95,127,190,92),
     (92,190,111,127,127,127,127,127,62,28),
     (58,125,254,250,254,254,254,254,124,56),
     (58,125,246,254,254,254,254,254,124,56)),
)

GHOST_NORMAL_ROWS = (60,126,147,219,147,255,255,255,255,85)
GHOST_FRIGHTENED_ROWS = (60,126,219,147,255,255,219,165,255,85)
MIRROR_BYTE = tuple(int(f"{row:08b}"[::-1],2) for row in range(256))


def add_support(canvas, rows, x, y):
    """OR an eight-pixel-wide sprite into a boolean visible-ROI canvas."""
    if canvas.shape != (171,160) or canvas.dtype != np.bool_:
        raise ValueError("Expected bool[171,160] support canvas")
    for dy,row in enumerate(rows):
        if 0 <= y+dy < 171:
            for dx in range(8):
                if row & (1 << (7-dx)) and 0 <= x+dx < 160:
                    canvas[y+dy,x+dx] = True


def player_support(current_ram, previous_ram=None):
    current = _ram(current_ram)
    r = current if previous_ram is None else _ram(previous_ram)
    phase, direction = int(r[55]), int(r[56])
    if phase not in range(3) or direction not in range(4):
        raise ValueError("Uncalibrated player animation state")
    canvas = np.zeros((171,160),dtype=bool)
    add_support(canvas,PLAYER_YELLOW_ROWS[phase][direction],int(r[10])-12,int(r[16])+1)
    return canvas


def actor_support(current_ram, previous_ram=None):
    """Union of hypothesized actor pixels, for external pill occlusion only.

    The current RAM30 bit16 selects a shared ghost reflection. The preceding
    RAM list selects positions and normal/frightened graphics. Development
    found some partial ghost bodies; this full-shape hypothesis retains that
    uncertainty for explicit calibration rather than consulting RGB here.
    """
    current = _ram(current_ram)
    canvas = player_support(current_ram,previous_ram)
    for entry in ghost_draw_entries(current_ram,previous_ram):
        rows = GHOST_FRIGHTENED_ROWS if entry.colour_code >= 4 else GHOST_NORMAL_ROWS
        if not current[30] & 16:
            rows = tuple(MIRROR_BYTE[row] for row in rows)
        x,y,_,_ = entry.box
        add_support(canvas,rows,x,y)
    return canvas
