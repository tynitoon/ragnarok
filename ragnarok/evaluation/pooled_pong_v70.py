"""Separate experimenter-side temporal adapter; the sealed raw decoder stays unchanged."""
import numpy as np

from ragnarok.evaluation.pong_v70 import KINDS, ObjectLabel, PALETTE, check_rgb, decode_pong_ram


POOLED_REVISION = "pong_two_raw_box_union_v1_canonical_order"


def pooled_ram_objects(previous, current):
    boxes = {}
    for ram in (previous, current):
        for obj in decode_pong_ram(ram):
            x, y, w, h = obj.box
            if obj.kind in boxes:
                left, top, right, bottom = boxes[obj.kind]
                boxes[obj.kind] = min(x,left), min(y,top), max(x+w,right), max(y+h,bottom)
            else:
                boxes[obj.kind] = x, y, x+w, y+h
    result = []
    for kind in KINDS:  # Swapping the two source frames must preserve ordering as well as geometry.
        if kind in boxes:
            x, y, right, bottom = boxes[kind]
            result.append(ObjectLabel(kind, (x,y,right-x,bottom-y)))
    return result


def pooled_screen_objects(previous, current):
    """Palette only on source images; keep each type's surviving max-RGB contributions.

    A type in both source pixels remains attributable even with a tie. Otherwise
    require a strict winning channel. A white ball can remove paddle pixels;
    disjoint support is represented by one enclosing box, not split identities.
    """
    check_rgb(previous)
    check_rgb(current)
    a, b = previous[34:194], current[34:194]
    wins_a, wins_b = np.any(a>b,axis=2), np.any(b>a,axis=2)
    result = []
    for kind, colour in PALETTE.items():
        ma, mb = np.all(a==colour,axis=2), np.all(b==colour,axis=2)
        survives = (ma & mb) | (ma & wins_a) | (mb & wins_b)
        ys, xs = np.nonzero(survives)
        if len(xs):
            x, y = int(xs.min()), int(ys.min())
            result.append(ObjectLabel(kind,(x,y+34,int(xs.max())-x+1,int(ys.max())-y+1)))
    return result
