"""RAM-only player rendering hypothesis for development17411 and17431.

The special list multiplexes player and fruit. Scratch RAM55/56 can describe
the fruit, so it cannot select the player's silhouette. A selected player's
16-bit graphics pointer plus its logical y identifies one of twelve patterns.
The pointer arithmetic was inferred from recorded RAM; no ROM is read or
pointer dereferenced. Geometry/support residuals remain in the diagnostic.
This is an external evaluation instrument, never an agent observation.
"""
from dataclasses import dataclass

import numpy as np

from ragnarok.evaluation.mspacman_render_v70 import _ram
from ragnarok.evaluation.mspacman_support_v70 import PLAYER_YELLOW_ROWS, add_support
from ragnarok.evaluation.mspacman_v70 import SlottedLabel

REVISION = "mspacman_player_special_list_development_v1"
PATTERN_BASE = 0xDE0C
PATTERN_STRIDE = 22


@dataclass(frozen=True)
class PlayerListEntry:
    index: int
    base_address: int
    phase: int
    direction: int
    box: tuple[int, int, int, int]


def player_list_entry(current_ram, previous_ram=None):
    """Select the preceding SAME-EPISODE player entry, or explicit cold fallback.

    RAM34 is the last special-list index (255 means empty). Codes45/46 have
    high nibble7 for the player and0..6 for fruit. The low/high pointer bytes
    are51/52 and53/54. Adding RAM16 removes the vertical-address compensation:
    A = 0xDE0C + 22 * (3 * direction + phase).
    Unknown lengths, duplicate players and unknown pointers fail explicitly.
    """
    current = _ram(current_ram)
    r = current if previous_ram is None else _ram(previous_ram)
    last = int(r[34])
    if last == 255:
        return None
    if last not in (0, 1):
        raise ValueError("Uncalibrated special-list length")
    players = []
    for index in range(last + 1):
        colour = int(r[45 + index]) >> 4
        if colour > 7:
            raise ValueError("Uncalibrated special-list colour")
        if colour == 7:
            players.append(index)
    if not players:
        return None
    if len(players) != 1:
        raise ValueError("Duplicate player in special list")
    index = players[0]
    address = 256 * int(r[53 + index]) + int(r[51 + index]) + int(r[16])
    pattern, remainder = divmod(address - PATTERN_BASE, PATTERN_STRIDE)
    if remainder or not 0 <= pattern < 12:
        raise ValueError("Uncalibrated player graphics pointer")
    direction, phase = divmod(pattern, 3)
    return PlayerListEntry(index, address, phase, direction,
                           (int(r[10]) - 12, int(r[16]) + 1, 8, 10))


def player_list_support(current_ram, previous_ram=None):
    """Predicted yellow pixels only; no RGB, scratch animation or oracle reset."""
    canvas = np.zeros((171, 160), dtype=bool)
    entry = player_list_entry(current_ram, previous_ram)
    if entry is not None:
        x, y, _, _ = entry.box
        add_support(canvas, PLAYER_YELLOW_ROWS[entry.phase][entry.direction], x, y)
    return canvas


def player_list_render_box(current_ram, previous_ram=None):
    """Visible-ROI envelope of the hypothesized support, with no hidden player."""
    canvas = player_list_support(current_ram, previous_ram)
    ys, xs = np.where(canvas)
    if not len(xs):
        return []
    entry = player_list_entry(current_ram, previous_ram)
    return [SlottedLabel("player", (int(xs.min()), int(ys.min()),
                         int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
                         f"special:{entry.index}")]
