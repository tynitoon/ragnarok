"""Development RGB-only exact decomposition of two overlapping rock glyphs.

Every component is retained. A split is used only if two complete, opposite-row
parity glyphs explain exactly the coloured pixels and every valid explanation
    gives the same visible supports. Otherwise the original component is retained.
"""
from functools import lru_cache
import numpy as np
from scipy import ndimage

from ragnarok.evaluation.asteroids_v70 import ASTEROID_COLOURS,ROI
from ragnarok.evaluation.asteroids_rocks_v70 import GLYPHS,_bands,_rock_phase
from ragnarok.evaluation.asteroids_occlusion_v70 import visible_piece_boxes
from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION='asteroids_rgb_exact_two_glyph_development_v3'
TEMPLATES=tuple(np.repeat(np.asarray([[bool(bits&(1<<(width-1-x))) for x in range(width)]
                                      for bits in rows],dtype=bool),2,axis=0)
                for width,rows in GLYPHS)


def torus_masks(mask):
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
    return [groups==group for group in np.unique(mapping[1:])]


@lru_cache(maxsize=2048)
def _exact_pair(height,width,parity,payload):
    mask=np.frombuffer(payload,dtype=np.bool_).reshape(height,width)
    actual=frozenset(map(int,np.flatnonzero(mask)))
    if any(mask.shape==t.shape and np.array_equal(mask,t) for t in TEMPLATES):
        return None,'single_glyph'
    candidates=[]
    for template in TEMPLATES:
        h,w=template.shape
        if h>height or w>width:continue
        matches=ndimage.binary_erosion(mask,structure=template,origin=(-h//2,-w//2))
        for y,x in np.argwhere(matches[:height-h+1,:width-w+1]):
            ys,xs=np.where(template)
            points=frozenset(map(int,(ys+y)*width+xs+x))
            candidates.append((points,(parity+int(y))%2))
    explanations=set()
    if not candidates:return None,'no_complete_pair'
    anchor=min(actual)
    for first,p1 in candidates:
        if anchor not in first:continue
        missing=actual-first
        if not missing:continue
        for second,p2 in candidates:
            if p1==p2 or not missing<=second or not first-second:continue
            if first|second!=actual:continue
            front,back=(first,second) if p1==0 else (second,first)
            # Pair parity comes from RGB vertical alignment, not a RAM slot.
            explanations.add((tuple(sorted(front)),tuple(sorted(back-front))))
            if len(explanations)>1:return None,'ambiguous_pair'
    if not explanations:return None,'no_complete_pair'
    # Return actual foreground and background point sets, rather than their
    # rectangles, so a horizontal screen wrap is clipped by real support.
    return next(iter(explanations)),'unique_pair'


def deblend_component(mask):
    ys,xs=np.where(mask)
    if not len(xs):return [],'empty'
    # Boundary-truncated glyphs are deliberately left in the ordinary checker;
    # they remain scored, and are never dropped to improve agreement.
    if ys.min()<=19 or ys.max()>=193:
        return [mask],'boundary_component'
    columns=np.unique(xs)
    gaps=np.diff(np.r_[columns,columns[0]+160])
    start=int(columns[(int(np.argmax(gaps))+1)%len(columns)])
    nx=(xs-start)%160
    width=int(nx.max())+1;top=int(ys.min());height=int(ys.max()-top)+1
    # More than two glyph extents cannot be explained by the declared model.
    if width>32 or height>56:return [mask],'large_component'
    normal=np.zeros((height,width),dtype=bool);normal[ys-top,nx]=True
    pair,reason=_exact_pair(height,width,top%2,normal.tobytes())
    if pair is None:return [mask],reason
    result=[]
    for points in pair:
        points=np.asarray(points,dtype=np.int32)
        output=np.zeros_like(mask)
        output[points//width+top,(points%width+start)%160]=True
        result.append(output)
    return result,reason


def screen_deblended_rocks(rgb,diagnostics=None):
    if rgb.shape!=(210,160,3) or rgb.dtype!=np.uint8:
        raise ValueError('RGB must be uint8[210,160,3]')
    if not _rock_phase(rgb[ROI[1]:ROI[3]]):return []
    output=[]
    for colour in ASTEROID_COLOURS:
        mask=np.all(rgb==colour,axis=2)
        mask[:ROI[1]]=False;mask[ROI[3]:]=False
        for component in torus_masks(mask):
            pieces,reason=deblend_component(component)
            if diagnostics is not None:diagnostics[reason]+=1
            if reason=='unique_pair':
                boxes=[b for piece in pieces for b in visible_piece_boxes(piece)]
            else:
                ys,xs=np.where(component);boxes=_bands(xs,ys)
            output.extend(ObjectLabel('asteroid',b) for b in boxes)
    return output
