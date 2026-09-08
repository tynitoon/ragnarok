"""Scorer-side Pong decoder and an independent screen calibration check.

RAM addresses and baseline geometry: OCAtari v2.2.1/ocatari/ram/pong.py
(MIT; notice in docs/licenses/OCAtari-MIT.txt). Neither function is a policy
observation, teacher, loss or curriculum signal. The screen verifier is a
Pong-specific calibration instrument, not the generic reference extractor.
"""
from dataclasses import dataclass

import numpy as np


KINDS = ("player", "enemy", "ball")
PALETTE = {"player": (92, 186, 92), "enemy": (213, 130, 74), "ball": (236, 236, 236)}
DECODER_REVISION = "ocatari_v2.2.1_baseline"


@dataclass(frozen=True)
class ObjectLabel:
    kind: str
    box: tuple[int, int, int, int]

    @property
    def center(self):
        x, y, w, h = self.box
        return x + w / 2, y + h / 2


def decode_pong_ram(ram):
    if not isinstance(ram, np.ndarray) or ram.shape != (128,) or ram.dtype != np.uint8:
        raise ValueError("Expected exactly 128 uint8 RAM bytes")
    r = [int(x) for x in ram]
    result = []
    if r[51] > 13:
        y, h = (34, r[51] - 33) if r[51] < 47 else (r[51] - 13, 207 - r[51] if r[51] > 192 else 15)
        if h > 0:
            result.append(ObjectLabel("player", (140, y, 4, h)))
    if r[50] > 33:
        y, h = (34, r[50] - 33) if r[50] < 49 else (r[50] - 15, 209 - r[50] if r[50] > 194 else 15)
        if h > 0:
            result.append(ObjectLabel("enemy", (16, y, 4, h)))
    if r[54] != 0 and r[49] > 49:
        result.append(ObjectLabel("ball", (r[49] - 49, r[54] - 14, 2, 4)))
    return result


def check_rgb(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (210, 160, 3):
        raise ValueError("Expected raw Pong RGB uint8, shape 210x160x3")


def screen_calibration_boxes(rgb):
    """Independent palette extent, only raw pixels and the declared playfield ROI.

    Does not consult RAM or connected components/background estimation. Measures
    visible extent: a ball may occlude part of a paddle without changing its box.
    """
    check_rgb(rgb)
    result = []
    for kind, colour in PALETTE.items():
        mask = np.all(rgb[34:194] == colour, axis=2)
        ys, xs = np.nonzero(mask)
        if len(xs):
            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
            result.append(ObjectLabel(kind, (x, y + 34, w, h)))
    return result


def edges(box):
    x, y, w, h = box
    return np.array((x, y, x + w, y + h), dtype=np.int64)
