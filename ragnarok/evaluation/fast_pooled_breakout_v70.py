"""Exact interval implementation of the sealed raster temporal RAM geometry.

Only the experimenter uses it. The sealed pixel checker and raw decoder remain
unchanged; every brick colour occupies one fixed horizontal band of height six.
"""
from ragnarok.evaluation.breakout_v70 import KINDS, decode_breakout_ram
from ragnarok.evaluation.pong_v70 import ObjectLabel


def pooled_breakout_ram_intervals(previous,current):
    groups={kind:[] for kind in KINDS}
    for ram in (previous,current):
        for obj in decode_breakout_ram(ram):
            groups[obj.kind].append(obj.box)
    result=[]
    for kind,boxes in groups.items():
        if not boxes:
            continue
        if kind in ("player","ball"):
            left,top=min(b[0] for b in boxes),min(b[1] for b in boxes)
            right,bottom=max(b[0]+b[2] for b in boxes),max(b[1]+b[3] for b in boxes)
            result.append(ObjectLabel(kind,(left,top,right-left,bottom-top)))
            continue
        # Equal-y/equal-height spans connect under the raster judge's 4-connectivity
        # exactly when their closed-open x intervals overlap or touch.
        boxes.sort()
        left,y,width,height=boxes[0]
        right=left+width
        for x,next_y,w,next_height in boxes[1:]:
            if (next_y,next_height)!=(y,height):
                raise ValueError("The sealed decoder's fixed brick band invariant changed")
            if x<=right:
                right=max(right,x+w)
            else:
                result.append(ObjectLabel(kind,(left,y,right-left,height)))
                left,right=x,x+w
        result.append(ObjectLabel(kind,(left,y,right-left,height)))
    return result
