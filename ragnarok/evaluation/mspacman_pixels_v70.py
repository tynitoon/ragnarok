"""Causal RGB-only pill instrument for Ms. Pac-Man development calibration.

An empty registry learns locations only from complete pink rectangles already
seen. It then groups surviving fragments by that location and reports their
visible bounding box. RAM, episode IDs and future frames are never inputs.
The first-maze development result does not validate unseen layouts or an object
whose first appearance is already occluded. This is not an agent observation.
"""
import numpy as np
from scipy import ndimage

from ragnarok.evaluation.pong_v70 import ObjectLabel


REVISION = "mspacman_pill_pixels_causal_v1"
# Palette and playfield ROI of the preserved nominal V0 instrument.
PILL_COLOUR = (228, 111, 111)
ROI = (0, 0, 160, 171)


def _contained(small, large):
    x, y, w, h = small
    left, top, width, height = large
    return left <= x and top <= y and x + w <= left + width and y + h <= top + height


class MsPacmanPillPixels:
    """One fresh instance per evaluation sequence; step reads only current RGB.

    Geometry persists across blank frames and later play. Detection never emits
    a remembered object without surviving pink pixels in the current image.
    A later complete power pill also replaces smaller anchors contained in it.
    """

    def __init__(self):
        self._pill_anchors = set()
        self._power_anchors = set()

    def step(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.shape != (210, 160, 3) or rgb.dtype != np.uint8:
            raise ValueError("RGB must be uint8[210,160,3]")
        view = rgb[ROI[1]:ROI[3], ROI[0]:ROI[2]]
        pink = np.all(view == PILL_COLOUR, axis=2)
        labels, _ = ndimage.label(pink, np.ones((3, 3), dtype=np.uint8))
        fragments = np.zeros_like(pink)
        full_small, full_large = set(), set()
        for label, sl in enumerate(ndimage.find_objects(labels), 1):
            box = (sl[1].start, sl[0].start,
                   sl[1].stop - sl[1].start, sl[0].stop - sl[0].start)
            w, h = box[2:]
            if w <= 4 and h <= 7:
                fragments[sl] |= labels[sl] == label
            if (w, h) == (4, 7) and pink[sl].all():
                full_large.add(box)
            elif (w, h) == (4, 2) and pink[sl].all():
                full_small.add(box)

        # A partially hidden power pill can itself become a complete 4x2
        # component. Its already observed larger support keeps its category.
        self._power_anchors.update(full_large)
        self._pill_anchors = {
            box for box in self._pill_anchors | full_small
            if not any(_contained(box, large) for large in self._power_anchors)
        }
        result = []
        for kind, anchors in (("pill", self._pill_anchors),
                              ("power_pill", self._power_anchors)):
            for x, y, w, h in sorted(anchors):
                ys, xs = np.where(fragments[y:y + h, x:x + w])
                if len(xs):
                    result.append(ObjectLabel(kind, (
                        x + int(xs.min()) + ROI[0], y + int(ys.min()) + ROI[1],
                        int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1),
                    )))
        return result
