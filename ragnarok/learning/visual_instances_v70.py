"""Instances and49-float descriptors from the parser's own dense predictions.

No RAM/type/ROI/reference input. Connectedness is the disclosed G3 procedure;
the foreground and appearance embeddings must come from the learned parser.
This draft has no tracker, cross-frame identity or semantic category labels.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class DensePrediction:
    embedding: np.ndarray
    foreground_logits: np.ndarray
    displacement: np.ndarray
    controllability_logits: np.ndarray
    solidity_logits: np.ndarray


@dataclass(frozen=True)
class VisualInstance:
    frame_index: int
    box: tuple[int,int,int,int]
    source_pixels: np.ndarray  # Flat row-major indices into this frame only.
    descriptor: np.ndarray
    embedding_variance: float
    split_saturated: bool


def _variance(features):
    return float(np.square(features-features.mean(axis=0)).sum(axis=1).mean())


def _two_means(pixels, embeddings, iterations):
    features=embeddings[pixels]
    first=int(np.argmax(np.square(features-features.mean(axis=0)).sum(axis=1)))
    second=int(np.argmax(np.square(features-features[first]).sum(axis=1)))
    centers=features[[first,second]].copy()
    labels=np.zeros(len(pixels),dtype=np.int8)
    for _ in range(iterations):
        updated=np.square(features[:,None]-centers[None]).sum(axis=2).argmin(axis=1)
        if not (updated==0).any() or not (updated==1).any(): return None
        changed=not np.array_equal(updated,labels)
        labels=updated
        centers=np.stack([features[labels==i].mean(axis=0) for i in range(2)])
        if not changed: break
    return [pixels[labels==i] for i in range(2)]


def _split(pixels, embeddings, threshold, max_parts, iterations):
    parts=[pixels]
    unsplittable=set()
    while len(parts)<max_parts:
        candidate=next((i for i,p in enumerate(parts) if i not in unsplittable
                        and len(p)>1 and _variance(embeddings[p])>threshold),None)
        if candidate is None: break
        split=_two_means(parts[candidate],embeddings,iterations)
        if split is None:
            unsplittable.add(candidate)
            continue
        # Keep existing indices stable so unsplittable markers stay valid.
        parts[candidate]=split[0]
        parts.append(split[1])
    return sorted(parts,key=lambda p:int(p.min()))


def extract_instances(prediction, rgb, *, split_variance=.2, max_parts=8, kmeans_iterations=12, max_instances=4096):
    if (not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3
            or min(rgb.shape[:2])==0 or not isinstance(prediction,DensePrediction)):
        raise ValueError('An RGB uint8 HWC frame and its own dense prediction are required')
    height,width=rgb.shape[:2]
    shapes={'embedding':(32,height,width),'displacement':(2,height,width),
            'foreground_logits':(1,height,width),'controllability_logits':(1,height,width),
            'solidity_logits':(1,height,width)}
    for name,shape in shapes.items():
        array=getattr(prediction,name)
        if (not isinstance(array,np.ndarray) or array.shape!=shape or not np.issubdtype(array.dtype,np.floating)
                or not np.isfinite(array).all()):
            raise ValueError(f'Finite learned {name} with shape{shape} required')
    if (np.abs(prediction.embedding)>1).any():
        raise ValueError('The parser embedding components must be normalized into[-1,1]')
    if (np.abs(prediction.displacement)>np.finfo(np.float32).max).any():
        raise ValueError('Predicted displacement must be representable as float32')
    if not np.isfinite(split_variance) or split_variance<=0:
        raise ValueError('Positive finite embedding split variance required')
    for value,name in ((max_parts,'max_parts'),(kmeans_iterations,'kmeans_iterations'),(max_instances,'max_instances')):
        if type(value) is not int or value<1: raise ValueError(f'{name} must be a positive integer')
    components,count=ndimage.label(prediction.foreground_logits[0]>0)
    if count>max_instances: raise ValueError('Instance capacity exceeded; no components silently discarded')
    # Float64 reductions prevent a finite float32 field from overflowing on sum.
    features=prediction.embedding.astype(np.float64).reshape(32,-1).T
    flow=prediction.displacement.astype(np.float64).reshape(2,-1).T
    colors=rgb.reshape(-1,3)
    controls=1/(1+np.exp(-np.clip(prediction.controllability_logits.reshape(-1),-60,60)))
    solids=1/(1+np.exp(-np.clip(prediction.solidity_logits.reshape(-1),-60,60)))
    parts=[]
    for component,region in enumerate(ndimage.find_objects(components),1):
        ys,xs=region
        y,x=np.nonzero(components[region]==component)
        pixels=((y+ys.start)*width+x+xs.start).astype(np.int64)
        parts.extend(_split(pixels,features,split_variance,max_parts,kmeans_iterations))
        if len(parts)>max_instances: raise ValueError('Instance capacity exceeded after learned embedding splits')
    parts.sort(key=lambda p:int(p.min()))
    instances=[]
    for index,pixels in enumerate(parts):
        y,x=np.divmod(pixels,width)
        left,top,right,bottom=int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1
        color_codes=((colors[pixels,0]>=128).astype(np.int8)*4
                     +(colors[pixels,1]>=128).astype(np.int8)*2+(colors[pixels,2]>=128).astype(np.int8))
        histogram=np.bincount(color_codes,minlength=8)/len(pixels)
        descriptor=np.concatenate(([(left+right)/(2*width),(top+bottom)/(2*height),
                                    (right-left)/width,(bottom-top)/height,len(pixels)/(height*width)],
                                   flow[pixels].mean(axis=0)/[width,height],features[pixels].mean(axis=0),
                                   [controls[pixels].mean(),solids[pixels].mean()],histogram)).astype(np.float32)
        pixels=pixels.copy()
        pixels.flags.writeable=descriptor.flags.writeable=False
        variance=_variance(features[pixels])
        instances.append(VisualInstance(index,(left,top,right-left,bottom-top),pixels,descriptor,
                                        variance,variance>split_variance))
    return instances
