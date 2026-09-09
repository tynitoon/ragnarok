"""Closed wire codec tests on synthetic arrays, without a child or emulator."""
import io
import json
from dataclasses import fields

import numpy as np
import pytest

from ragnarok.environments import rgb_wire_v70 as wire
from ragnarok.environments.ordinary_rgb_trace_v70 import OrdinaryRGBTrace,validate_trace
from ragnarok.environments.rgb_branch_trace_v70 import RGBBranchTrace


def example(length=4):
    start=np.zeros((2,8,10,3),np.uint8)
    raw=np.stack([np.full((8,10,3),i+1,np.uint8) for i in range(length)]) if length else np.zeros((0,8,10,3),np.uint8)
    return OrdinaryRGBTrace(start,raw,raw.copy(),raw[-1].copy() if length else start[-1].copy())


def frame(tree,shapes=(),payload=b''):
    header=json.dumps({'root':tree,'arrays':list(shapes)}).encode()
    return wire.PREFIX.pack(wire.MAGIC,len(header),len(payload))+header+payload


@pytest.mark.parametrize('length',[0,1,3,4])
def test_ordinary_prefix_and_branch_arrays_round_trip_exactly_without_aliases(length):
    saved=example(length);branch=RGBBranchTrace(saved.raw,saved.pooled,saved.endpoint)
    encoded=wire.encode({'ordinary':[saved],'pair':(branch,branch),'accounting':{'charged_frames':136,'failure':None}})
    result=wire.decode(encoded)
    assert type(result['ordinary'][0]) is OrdinaryRGBTrace and type(result['pair']) is tuple
    assert all(type(t) is RGBBranchTrace for t in result['pair'])
    assert validate_trace(result['ordinary'][0])==length
    for f in fields(OrdinaryRGBTrace):
        np.testing.assert_array_equal(getattr(saved,f.name),getattr(result['ordinary'][0],f.name))
    result['pair'][0].endpoint[:]=255
    assert not result['pair'][1].endpoint.max()==255 and not saved.endpoint.max()==255


def test_noncontiguous_rgb_and_two_consecutive_pipe_frames():
    rgb=np.arange(8*10*3,dtype=np.uint8).reshape(8,10,3)[:,::-1]
    stream=io.BytesIO(wire.encode(rgb)+wire.encode(('report',0)))
    np.testing.assert_array_equal(wire.read(stream),rgb)
    assert wire.read(stream)==('report',0) and stream.read()==b''


def test_partial_pipe_reads_are_reassembled_and_truncation_is_fatal():
    data=wire.encode(example())
    class Chunked(io.BytesIO):
        def read(self,n=-1):return super().read(min(n,3))
    assert validate_trace(wire.read(Chunked(data)))==4
    for missing in (1,19,len(data)-1):
        with pytest.raises(EOFError):wire.read(Chunked(data[:-missing]))


def test_unsupported_objects_ram_shape_float_arrays_and_nonfinite_metadata_refused():
    for value in (object(),np.zeros(128,np.uint8),np.zeros((8,10,3),np.float32),float('nan'),float('inf'),2**63,{1:'bad'}):
        with pytest.raises(ValueError):wire.encode(value)
    with pytest.raises(ValueError,match='Unknown'):wire.decode(frame(['python_constructor','os.system']))


def test_oversize_header_is_refused_before_reading_any_payload():
    for prefix in (wire.PREFIX.pack(wire.MAGIC,wire.MAX_HEADER+1,0),
                   wire.PREFIX.pack(wire.MAGIC,2,wire.MAX_PIXELS+1),
                   wire.PREFIX.pack(b'BAD!',2,0)):
        with pytest.raises(ValueError,match='oversized'):wire.read(io.BytesIO(prefix))


def test_array_alias_unclaimed_bytes_and_duplicate_keys_are_rejected():
    for tree,shapes,payload in (
        (['tuple',[['rgb',0],['rgb',0]]],[[1,1,3]],b'abc'),
        (['value',None],[[1,1,3]],b'abc'),
        (['rgb',0],[[1,1,3]],b'ab'),
        (['map',[['x',['value',0]],['x',['value',1]]]],[],b'')):
        with pytest.raises(ValueError):wire.decode(frame(tree,shapes,payload))
    header=b'{"root":["value",0],"root":["value",1],"arrays":[]}'
    with pytest.raises(ValueError,match='Duplicate'):wire.decode(wire.PREFIX.pack(wire.MAGIC,len(header),0)+header)


def test_depth_and_trailing_frame_bytes_are_bounded():
    nested=0
    for _ in range(18):nested=[nested]
    with pytest.raises(ValueError,match='bounds'):wire.encode(nested)
    tree=['value',0]
    for _ in range(18):tree=['list',[tree]]
    with pytest.raises(ValueError,match='bounds'):wire.decode(frame(tree))
    with pytest.raises(ValueError,match='Trailing'):wire.decode(wire.encode(None)+b'ignored')
