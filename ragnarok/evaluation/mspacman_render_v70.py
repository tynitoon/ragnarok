"""Development renderer-phase hypotheses, separate from preserved nominal V0.

The external RAM judge uses RAM alone. Coordinate/animation timing and the
current power-pill blink phase were inferred on development seed 17411.
Nothing here is an agent observation or an already reserved-validated label.
"""
import numpy as np

from ragnarok.evaluation.mspacman_v70 import ROI, SlottedLabel

REVISION = "mspacman_render_phase_development_v1"


def _ram(value):
    value = np.asarray(value)
    if value.shape != (128,) or value.dtype != np.uint8:
        raise ValueError("RAM must be uint8[128]")
    return value.astype(int)


def _clipped(kind, slot, left, top, right, bottom):
    left, top = max(left, ROI[0]), max(top, ROI[1])
    right, bottom = min(right, ROI[2]), min(bottom, ROI[3])
    if left >= right or top >= bottom:
        return []
    return [SlottedLabel(kind, (left, top, right-left, bottom-top), str(slot))]


def player_render_box(current_ram, previous_ram=None):
    """Use the preceding RAM snapshot from the SAME ordinary episode.

    Callers must pass None at a cold start; the explicitly declared first-frame
    fallback uses current RAM. Mouth opening clips one side of an 8x10 body.
    Unknown animation states fail explicitly instead of silently fabricating a
    known sprite. Observed one-pixel residuals remain part of calibration.
    """
    current = _ram(current_ram)
    r = current if previous_ram is None else _ram(previous_ram)
    phase, direction = r[55], r[56]
    if not 0 <= phase <= 2 or not 0 <= direction <= 3:
        raise ValueError("Uncalibrated player animation state")
    left, top = r[10]-12, r[16]+1
    right, bottom = left+8, top+10
    trim = 2-phase
    if direction == 0:
        top += trim
    elif direction == 1:
        right -= trim
    elif direction == 2:
        bottom -= trim
    else:
        left += trim
    return _clipped("player", 0, left, top, right, bottom)


def power_pill_render_boxes(current_ram):
    """Current flags and blink phase; boxes still ignore actor occlusion.

    The phase is intentionally current, unlike actor coordinates. These are
    visible-phase candidate envelopes, not yet all visible-support boxes.
    """
    r = _ram(current_ram)
    result = []
    left_phase = bool(r[30] & 8)
    for bit, x, y in ((4,148,146),(8,148,14),(16,8,147),(32,8,15)):
        if r[117] & bit and (x == 8) == left_phase:
            result.extend(_clipped("power_pill",bit,x,y,x+4,y+7))
    return result
