"""Prospective bounded RGB/metadata wire format for separate Python runtimes.

No pickle, executable object reconstruction, native state or emulator import.
This codec certifies structure, not pixel provenance. Only the closed server's
reviewed command dispatch can certify where a transmitted observation came from.
"""
from dataclasses import fields
import io
import json
import math
import struct

import numpy as np

from ragnarok.environments.ordinary_rgb_trace_v70 import OrdinaryRGBTrace
from ragnarok.environments.rgb_branch_trace_v70 import RGBBranchTrace


MAGIC=b'RGB1'
PREFIX=struct.Struct('<4sIQ')
MAX_HEADER=1<<20
MAX_PIXELS=64<<20
MAX_NODES=8192
MAX_DEPTH=16
TYPES={'ordinary':OrdinaryRGBTrace,'branch':RGBBranchTrace}


def _image_shape(shape):
    if (not isinstance(shape,(list,tuple)) or len(shape) not in (3,4)
            or any(type(n) is not int for n in shape) or shape[-1]!=3
            or not 1<=shape[-3]<=1024 or not 1<=shape[-2]<=1024
            or len(shape)==4 and not 0<=shape[0]<=64):
        raise ValueError('Wire arrays must have bounded RGB image or trace shapes')
    return math.prod(shape)


def _primitive(value):
    if value is None or type(value) in (bool,str):return True
    if type(value) is int:
        if not -(2**63)<=value<2**63:raise ValueError('Wire integer exceeds signed64 domain')
        return True
    if type(value) is float:
        if not math.isfinite(value):raise ValueError('Finite wire metadata required')
        return True
    return False


def encode(value):
    """Return one complete frame, with limits checked before writing to a pipe."""
    arrays=[];payload=[];size=nodes=0
    def visit(item,depth=0):
        nonlocal size,nodes
        nodes+=1
        if depth>MAX_DEPTH or nodes>MAX_NODES:raise ValueError('Wire metadata structure exceeds bounds')
        if _primitive(item):return ['value',item]
        if isinstance(item,np.ndarray):
            count=_image_shape(item.shape)
            if item.dtype!=np.uint8 or size+count>MAX_PIXELS:
                raise ValueError('Wire requires bounded uint8 RGB payloads')
            index=len(arrays)
            arrays.append(list(item.shape));payload.append(np.ascontiguousarray(item).tobytes());size+=count
            return ['rgb',index]
        for tag,cls in TYPES.items():
            if type(item) is cls:
                return [tag,[visit(getattr(item,f.name),depth+1) for f in fields(cls)]]
        if type(item) in (tuple,list):
            if len(item)>MAX_NODES:raise ValueError('Wire sequence exceeds bounds')
            return ['tuple' if type(item) is tuple else 'list',[visit(v,depth+1) for v in item]]
        if type(item) is dict:
            if len(item)>MAX_NODES or any(type(k) is not str for k in item):
                raise ValueError('Bounded string-key wire maps required')
            return ['map',[[k,visit(v,depth+1)] for k,v in item.items()]]
        raise ValueError('Unsupported wire object type')
    tree=visit(value)
    header=json.dumps({'root':tree,'arrays':arrays},allow_nan=False,separators=(',',':')).encode('utf-8')
    if len(header)>MAX_HEADER:raise ValueError('Wire header exceeds bounds')
    return PREFIX.pack(MAGIC,len(header),size)+header+b''.join(payload)


def _read_exact(stream,count):
    result=bytearray()
    while len(result)<count:
        block=stream.read(count-len(result))
        if not block:raise EOFError('Incomplete RGB wire frame')
        result.extend(block)
    return bytes(result)


def _unique_map(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON wire key')
        result[key]=value
    return result


def read(stream):
    """Read exactly one frame. A caller must stop the connection after any error."""
    magic,header_length,payload_length=PREFIX.unpack(_read_exact(stream,PREFIX.size))
    if magic!=MAGIC or not 0<header_length<=MAX_HEADER or payload_length>MAX_PIXELS:
        raise ValueError('Invalid or oversized RGB wire frame')
    header=json.loads(_read_exact(stream,header_length),object_pairs_hook=_unique_map)
    if type(header) is not dict or set(header)!={'root','arrays'} or type(header['arrays']) is not list:
        raise ValueError('Exact RGB wire header required')
    if len(header['arrays'])>MAX_NODES:raise ValueError('Too many wire arrays')
    counts=[_image_shape(shape) for shape in header['arrays']]
    if sum(counts)!=payload_length:raise ValueError('RGB wire byte lengths disagree')
    payload=_read_exact(stream,payload_length)
    offsets=np.cumsum([0,*counts],dtype=np.int64)
    used=set();nodes=0
    def visit(item,depth=0):
        nonlocal nodes
        nodes+=1
        if depth>MAX_DEPTH or nodes>MAX_NODES:raise ValueError('Wire metadata structure exceeds bounds')
        if type(item) is not list or len(item)!=2 or type(item[0]) is not str:
            raise ValueError('Exact typed wire node required')
        tag,value=item
        if tag=='value':
            if not _primitive(value):raise ValueError('Invalid primitive wire node')
            return value
        if tag=='rgb':
            if type(value) is not int or not 0<=value<len(counts) or value in used:
                raise ValueError('Each wire RGB array must be consumed exactly once')
            used.add(value)
            return np.frombuffer(payload,dtype=np.uint8,count=counts[value],offset=int(offsets[value])).reshape(header['arrays'][value]).copy()
        if tag in ('tuple','list'):
            if type(value) is not list or len(value)>MAX_NODES:raise ValueError('Malformed wire sequence')
            result=[visit(v,depth+1) for v in value]
            return tuple(result) if tag=='tuple' else result
        if tag=='map':
            if type(value) is not list or len(value)>MAX_NODES:raise ValueError('Malformed wire map')
            result={}
            for pair in value:
                if type(pair) is not list or len(pair)!=2 or type(pair[0]) is not str or pair[0] in result:
                    raise ValueError('Unique string-key wire map required')
                result[pair[0]]=visit(pair[1],depth+1)
            return result
        if tag in TYPES:
            cls=TYPES[tag]
            if type(value) is not list or len(value)!=len(fields(cls)):raise ValueError('Exact RGB trace fields required')
            values=[visit(v,depth+1) for v in value]
            if any(not isinstance(v,np.ndarray) for v in values):raise ValueError('RGB trace fields must be image arrays')
            return cls(*values)
        raise ValueError('Unknown wire object tag')
    value=visit(header['root'])
    if len(used)!=len(counts):raise ValueError('Unclaimed RGB wire payload')
    return value


def decode(frame):
    if not isinstance(frame,bytes):raise ValueError('An exact binary wire frame is required')
    stream=io.BytesIO(frame)
    result=read(stream)
    if stream.read(1):raise ValueError('Trailing bytes after RGB wire frame')
    return result
