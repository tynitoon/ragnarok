"""Unvalidated raw Asteroids instruments, never agent observations.

Nominal RAM addresses, encoding and palettes adapted from MIT OCAtari commit
467accd5f3da379d83705b008094ffb787eec8a1, ram/asteroids.py and vision/asteroids.py.
License: docs/licenses/OCAtari-MIT.txt. No OCAtari runtime is installed or used.
Clipping, wrapping, sprite phases and shared-colour contacts are hypotheses to
check on development; the existing library is not a ground-truth certificate.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION = "asteroids_raw_nominal_v0"
KINDS = ("player", "asteroid", "player_missile")
ROI = (0, 16, 160, 194)
PLAYER_COLOUR = (240, 128, 128)
BLUE = (117, 181, 239)
ASTEROID_COLOURS = ((180, 122, 48), (104, 72, 198), (136, 146, 62),
                   (187, 187, 53), (214, 214, 214), BLUE, (184, 70, 162), (184, 50, 50))
ASTEROID_ROWS = (3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19)


@dataclass(frozen=True)
class SlottedLabel(ObjectLabel):
    slot: str


def x_position(value):
    """Integer form of the pinned nominal coarse/fine horizontal encoding."""
    value = int(value)
    if not 0 <= value <= 255:
        raise ValueError("Expected an unsigned byte")
    if value in (0, 1):
        return 64 if value == 0 else 4
    low = value & 15
    return (10 if low & 1 else 97) + 15 * (low // 2 - 1) + 8 * (value >> 7) - ((value >> 4) & 7)


def decode_asteroids_ram(ram):
    ram = np.asarray(ram)
    if ram.shape != (128,) or ram.dtype != np.uint8:
        raise ValueError("RAM must be uint8[128]")
    r = ram.astype(int)
    result = []

    def add(kind, slot, x, y, w, h):
        left, top = max(ROI[0], x), max(ROI[1], y)
        right, bottom = min(ROI[2], x+w), min(ROI[3], y+h)
        if left < right and top < bottom:
            result.append(SlottedLabel(kind, (left, top, right-left, bottom-top), str(slot)))

    if r[74] != 224:
        direction = r[60] & 15
        x = x_position(r[73]) + (-1 if direction == 4 else 1 if direction == 12 else 0)
        # The pinned source's chained inequality for width5 is impossible.
        # Preserve its effective width6 as the explicit nominal baseline.
        add("player", 0, x, 18 + 2*r[74], 6, 10)
    for row in ASTEROID_ROWS:
        if r[row+18] and not r[row] & 128:
            size = r[row+36] & 127
            w, h = (16, 28) if size < 32 else (8, 15) if size < 48 else (4, 8)
            add("asteroid", row, x_position(r[row+18]), 24 + 2*r[row], w, h)
    for slot, offset in enumerate((83, 84)):
        if r[offset] and not r[offset+3] & 128:
            add("player_missile", slot, x_position(r[offset])+1, 17 + 2*r[offset+3], 1, 2)
    return result


def screen_asteroids_objects(rgb):
    """All palette components in ROI; no RAM, history or allocation slots."""
    rgb = np.asarray(rgb)
    if rgb.shape != (210, 160, 3) or rgb.dtype != np.uint8:
        raise ValueError("RGB must be uint8[210,160,3]")
    view = rgb[ROI[1]:ROI[3], ROI[0]:ROI[2]]
    result = []
    for colour in (PLAYER_COLOUR, *ASTEROID_COLOURS):
        labels, _ = ndimage.label(np.all(view == colour, axis=2), np.ones((3, 3), dtype=np.uint8))
        for sy, sx in ndimage.find_objects(labels):
            box = (sx.start, sy.start + ROI[1], sx.stop-sx.start, sy.stop-sy.start)
            if colour in (PLAYER_COLOUR, BLUE) and box[2] <= 2 and box[3] <= 3:
                kind = "player_missile"
            else:
                kind = "player" if colour == PLAYER_COLOUR else "asteroid"
            result.append(ObjectLabel(kind, box))
    return result
