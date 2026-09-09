"""Development-only Ms. Pac-Man ghost render list and independent RGB checker.

The RAM formulas were inferred on immutable raw development seed 17411. They
describe rendered supports, not all four logical allocations or persistent IDs.
Reserved seed 17412 has not validated this module. No agent receives these labels.
See ghost_renderer_phase_development_diagnostic.json for every retained residual.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ragnarok.evaluation.mspacman_v70 import ROI, SlottedLabel
from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION = "mspacman_ghost_drawlist_development_v1"
GHOST_COLOURS = (
    (180, 122, 48), (84, 184, 153), (198, 89, 179), (200, 72, 72),
    (214, 214, 214), (66, 114, 194),
)


@dataclass(frozen=True)
class GhostDrawEntry:
    """One frame-local render entry, with its rectangle BEFORE ROI clipping.

    form_code names the lower nibble for inspection only: it does not by itself
    determine the visible sprite shape. The common mirror phase was associated
    with CURRENT RAM[30] bit16 in development; no pixel mask is applied here.
    """
    index: int
    colour_code: int
    form_code: int
    box: tuple[int, int, int, int]

    @property
    def render_code(self):
        return 16 * self.colour_code + self.form_code


def _ram(value):
    value = np.asarray(value)
    if value.shape != (128,) or value.dtype != np.uint8:
        raise ValueError("RAM must be uint8[128]")
    return value.astype(int)


def ghost_draw_entries(current_ram, previous_ram=None):
    """Decode the PRECEDING snapshot from the same ordinary episode.

    At cold start callers must pass None; the declared current-RAM fallback
    preserves geometry but swaps orange/cyan at the four development starts.
    List indices are temporary render positions. Never treat them as lifetimes.
    An unobserved list length or colour fails rather than hiding a possible new
    category (fruit was absent in 17411). Low-nibble animation semantics remain
    exploratory and do not change the nominal 8x10 envelope.
    """
    current = _ram(current_ram)
    r = current if previous_ram is None else _ram(previous_ram)
    last = int(r[33])
    if last == 255:
        return []
    if not 0 <= last <= 3:
        raise ValueError("Uncalibrated ghost draw-list length")
    result = []
    for index in range(last + 1):
        horizontal = int(r[35 + index])
        code = int(r[41 + index])
        colour, form = code >> 4, code & 15
        if not 0 <= colour < len(GHOST_COLOURS):
            raise ValueError("Uncalibrated ghost draw-list colour")
        coarse = horizontal & 15
        fine = horizontal >> 4
        if fine >= 8:
            fine -= 16
        x = 12 * coarse - fine + (3 if coarse == 0 else 0)
        y = (204 if colour >= 4 else 181) - int(r[47 + index])
        result.append(GhostDrawEntry(index, colour, form, (x, y, 8, 10)))
    return result


def decode_ghost_drawlist(current_ram, previous_ram=None):
    """Nominal visible envelopes; partial occlusion remains in calibration."""
    result = []
    for entry in ghost_draw_entries(current_ram, previous_ram):
        x, y, width, height = entry.box
        left, top = max(ROI[0], x), max(ROI[1], y)
        right, bottom = min(ROI[2], x + width), min(ROI[3], y + height)
        if left < right and top < bottom:
            result.append(SlottedLabel(
                "ghost", (left, top, right-left, bottom-top),
                f"draw:{entry.index}",
            ))
    return result


def _merge_fragments(boxes):
    """Retain every component; combine only envelopes fitting one 8x10 body.

    This deterministic RGB-only grouping is an empirical pixel interpretation.
    Nearby same-colour bodies can be ambiguous outside the development sample.
    No size filter silently discards large or small components.
    """
    groups = sorted(boxes, key=lambda b: (-b[2] * b[3], b[1], b[0]))
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i):
                a, b = groups[i], groups[j]
                x, y = min(a[0], b[0]), min(a[1], b[1])
                right = max(a[0]+a[2], b[0]+b[2])
                bottom = max(a[1]+a[3], b[1]+b[3])
                if right-x <= 8 and bottom-y <= 10:
                    groups[j] = (x, y, right-x, bottom-y)
                    groups.pop(i)
                    changed = True
                    break
            if changed:
                break
    return sorted(groups, key=lambda b: (b[1], b[0]))


def screen_ghosts(rgb):
    """Pixel-only six-colour components; no RAM or reference extractor access."""
    rgb = np.asarray(rgb)
    if rgb.shape != (210, 160, 3) or rgb.dtype != np.uint8:
        raise ValueError("RGB must be uint8[210,160,3]")
    view = rgb[ROI[1]:ROI[3], ROI[0]:ROI[2]]
    result = []
    for colour in GHOST_COLOURS:
        labels, _ = ndimage.label(
            np.all(view == colour, axis=2), np.ones((3, 3), dtype=np.uint8),
        )
        boxes = [
            (s[1].start+ROI[0], s[0].start+ROI[1],
             s[1].stop-s[1].start, s[0].stop-s[0].start)
            for s in ndimage.find_objects(labels)
        ]
        result.extend(ObjectLabel("ghost", box) for box in _merge_fragments(boxes))
    return result
