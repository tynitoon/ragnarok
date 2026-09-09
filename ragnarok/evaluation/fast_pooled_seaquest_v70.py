"""Exact rectangle graph for the sealed four-connected Seaquest envelopes.

Grouping tests original rectangles only: bounding a connected group may fill a
hole, which must never introduce another connection. Members are scoring slots,
not persistent individual identities.
"""
from dataclasses import dataclass

from ragnarok.evaluation.pong_v70 import ObjectLabel
from ragnarok.evaluation.seaquest_v70 import KINDS, decode_seaquest_ram


@dataclass(frozen=True)
class EnvelopeGroup:
    label: ObjectLabel
    members: frozenset[str]


def group_rectangles(objects):
    result=[]
    for kind in KINDS:
        own=[o for o in objects if o.kind==kind]
        parents=list(range(len(own)))

        def find(i):
            while parents[i]!=i:
                parents[i]=parents[parents[i]]
                i=parents[i]
            return i

        for i,a in enumerate(own):
            x,y,w,h=a.box
            for j in range(i):
                xx,yy,ww,hh=own[j].box
                dx=min(x+w,xx+ww)-max(x,xx)
                dy=min(y+h,yy+hh)-max(y,yy)
                if (dx>0 and dy>=0) or (dy>0 and dx>=0):
                    parents[find(i)]=find(j)
        groups={}
        for i,o in enumerate(own):
            groups.setdefault(find(i),[]).append(o)
        # ndimage's label order is the first actual occupied row-major pixel.
        for group in sorted(groups.values(),key=lambda g:min((o.box[1],o.box[0]) for o in g)):
            left=min(o.box[0] for o in group)
            top=min(o.box[1] for o in group)
            right=max(o.box[0]+o.box[2] for o in group)
            bottom=max(o.box[1]+o.box[3] for o in group)
            label=ObjectLabel(kind,(left,top,right-left,bottom-top))
            result.append(EnvelopeGroup(label,frozenset(f"{o.kind}:{o.slot}" for o in group)))
    return result


def pooled_seaquest_groups(previous,current):
    return group_rectangles([*decode_seaquest_ram(previous),*decode_seaquest_ram(current)])


def fast_pooled_seaquest_ram_objects(previous,current):
    return [g.label for g in pooled_seaquest_groups(previous,current)]
