"""Separate development hypothesis for coloured rock supports and visibility.

Multiple allocations remain separate. This revision boxes each connected visible
piece after predicted hardware-player occlusion, including isolated fragments.
It never merges two RAM allocations merely because their colours coincide.
"""
import numpy as np
from scipy import ndimage

from ragnarok.evaluation.asteroids_rocks_v70 import rock_entries,rock_support,_bands
from ragnarok.evaluation.asteroids_v70 import SlottedLabel

REVISION='asteroids_rock_occlusion_development_v2'
COLOUR_INDEX=(4,7,5,3,0,6,1,2)


def visible_rock_masks(ram):
    entries=rock_entries(ram)
    supports=[rock_support(e) for e in entries]
    foreground=np.zeros((210,160),dtype=bool)
    for entry,support in zip(entries,supports):
        if entry.slot<12:foreground|=support
    return [(entry,COLOUR_INDEX[int(ram[entry.slot+36])&7],
             support if entry.slot<12 else support&~foreground)
            for entry,support in zip(entries,supports)]


def visible_piece_boxes(mask):
    labels,count=ndimage.label(mask,np.ones((3,3),dtype=np.uint8))
    parents=list(range(count+1))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for y in range(len(labels)):
        left=int(labels[y,0])
        if left:
            for yy in range(max(0,y-1),min(len(labels),y+2)):
                right=int(labels[yy,-1])
                if right:parents[root(right)]=root(left)
    mapping=np.asarray([root(i) for i in range(count+1)],dtype=np.int32)
    groups=mapping[labels]
    boxes=[]
    for group in np.unique(mapping[1:]):
        ys,xs=np.where(groups==group)
        boxes.extend(_bands(xs,ys))
    return boxes


def decode_occluded_rocks(ram):
    result=[]
    for entry,colour,support in visible_rock_masks(ram):
        for i,box in enumerate(visible_piece_boxes(support)):
            result.append(SlottedLabel('asteroid',box,f'{entry.slot}:{i}'))
    return result
