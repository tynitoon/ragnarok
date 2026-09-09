"""Separate RGB exact-cover candidate including vertically clipped rock glyphs.

The visible part of a template is clipped to the physical P0/P1 top boundary
and frozen ROI bottom. No interior missing pixel may be explained as clipping.
All exact explanations must agree on foreground/background visible supports.
"""
from functools import lru_cache
import numpy as np

from ragnarok.evaluation.asteroids_deblend_v70 import TEMPLATES,deblend_component,torus_masks
from ragnarok.evaluation.asteroids_rocks_v70 import _bands,_rock_phase
from ragnarok.evaluation.asteroids_occlusion_v70 import visible_piece_boxes
from ragnarok.evaluation.asteroids_v70 import ASTEROID_COLOURS,ROI
from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION='asteroids_rgb_clipped_exact_pair_development_v5'


@lru_cache(maxsize=2048)
def _clipped_pair(top,height,width,payload):
    mask=np.frombuffer(payload,dtype=np.bool_).reshape(height,width)
    actual=frozenset(map(int,np.flatnonzero(mask)))
    candidates=set()
    for template in TEMPLATES:
        h,w=template.shape
        ty,tx=np.where(template)
        for y0 in range(1-h,height):
            parity=(top+y0)%2
            visible=(top+y0+ty>=18+parity)&(top+y0+ty<ROI[3])
            if not np.any(visible):continue
            ys=ty[visible]+y0;xs=tx[visible]
            if ys.min()<0 or ys.max()>=height:continue
            for x0 in range(-int(xs.min()),width-int(xs.max())):
                xx=xs+x0
                if not np.all(mask[ys,xx]):continue
                points=frozenset(map(int,ys*width+xx))
                if points==actual:return None,'single_clipped_glyph'
                candidates.add((points,parity))
    explanations=set();anchor=min(actual)
    for first,p1 in candidates:
        if anchor not in first:continue
        missing=actual-first
        for second,p2 in candidates:
            if p1==p2 or not missing<=second or not first-second or first|second!=actual:continue
            front,back=(first,second) if p1==0 else (second,first)
            explanations.add((tuple(sorted(front)),tuple(sorted(back-front))))
            if len(explanations)>1:return None,'ambiguous_clipped_pair'
    if not explanations:return None,'no_clipped_pair'
    return next(iter(explanations)),'unique_clipped_pair'


def deblend_boundary_component(mask):
    ys,xs=np.where(mask)
    if not len(xs):return [],'empty'
    if ys.min()>19 and ys.max()<193:return deblend_component(mask)
    columns=np.unique(xs);gaps=np.diff(np.r_[columns,columns[0]+160])
    start=int(columns[(int(np.argmax(gaps))+1)%len(columns)])
    nx=(xs-start)%160
    width=int(nx.max())+1;top=int(ys.min());height=int(ys.max()-top)+1
    if width>32 or height>56:return [mask],'large_component'
    normal=np.zeros((height,width),bool);normal[ys-top,nx]=True
    pair,reason=_clipped_pair(top,height,width,normal.tobytes())
    if pair is None:return [mask],reason
    result=[]
    for points in pair:
        points=np.asarray(points,dtype=np.int32)
        piece=np.zeros_like(mask)
        piece[points//width+top,(points%width+start)%160]=True
        result.append(piece)
    return result,reason


def screen_boundary_deblended_rocks(rgb,diagnostics=None):
    if rgb.shape!=(210,160,3) or rgb.dtype!=np.uint8:
        raise ValueError('RGB must be uint8[210,160,3]')
    if not _rock_phase(rgb[ROI[1]:ROI[3]]):return []
    output=[]
    for colour in ASTEROID_COLOURS:
        mask=np.all(rgb==colour,axis=2);mask[:ROI[1]]=False;mask[ROI[3]:]=False
        for component in torus_masks(mask):
            pieces,reason=deblend_boundary_component(component)
            if diagnostics is not None:diagnostics[reason]+=1
            if reason in ('unique_pair','unique_clipped_pair'):
                boxes=[b for piece in pieces for b in visible_piece_boxes(piece)]
            else:
                ys,xs=np.where(component);boxes=_bands(xs,ys)
            output.extend(ObjectLabel('asteroid',b) for b in boxes)
    return output
