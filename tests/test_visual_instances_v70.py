"""Synthetic own-head inputs, with no calibrated game types or object teacher."""
import numpy as np
import pytest
from ragnarok.learning.visual_instances_v70 import DensePrediction,extract_instances


def scene(height=10,width=12):
    shape=(1,height,width)
    embedding=np.zeros((32,height,width),dtype=np.float32)
    embedding[0]=1
    return DensePrediction(embedding,np.full(shape,-10,dtype=np.float32),
        np.zeros((2,height,width),dtype=np.float32),np.zeros(shape,dtype=np.float32),
        np.zeros(shape,dtype=np.float32)),np.zeros((height,width,3),dtype=np.uint8)


def test_descriptor_geometry_forward_displacement_and_color_histogram_have49_declared_values():
    prediction,rgb=scene()
    prediction.foreground_logits[:,2:4,3:7]=10
    prediction.displacement[0,2:4,3:7]=2
    prediction.displacement[1,2:4,3:7]=-1
    rgb[2:4,3:7,0]=255
    result=extract_instances(prediction,rgb)
    assert len(result)==1 and result[0].box==(3,2,4,2)
    vector=result[0].descriptor
    assert vector.shape==(49,) and vector.dtype==np.float32
    np.testing.assert_allclose(vector[:7],[5/12,3/10,4/12,2/10,8/120,2/12,-1/10])
    np.testing.assert_array_equal(vector[7:39],[1]+[0]*31)
    np.testing.assert_allclose(vector[39:41],[.5,.5])
    np.testing.assert_array_equal(vector[41:],[0,0,0,0,1,0,0,0])
    before=vector.copy()
    prediction.embedding[:]=8;rgb[:]=255
    np.testing.assert_array_equal(vector,before)
    assert not vector.flags.writeable and not result[0].source_pixels.flags.writeable


def test_touching_regions_split_only_when_own_embedding_varies_and_split_cap_is_disclosed():
    prediction,rgb=scene()
    prediction.foreground_logits[:,2:4,2:6]=10
    assert len(extract_instances(prediction,rgb))==1
    prediction.embedding[0,2:4,4:6]=-1
    result=extract_instances(prediction,rgb)
    assert [x.box for x in result]==[(2,2,2,2),(4,2,2,2)]
    assert all(x.embedding_variance==0 and not x.split_saturated for x in result)
    limited=extract_instances(prediction,rgb,max_parts=1)
    assert len(limited)==1 and limited[0].split_saturated


def test_four_connected_components_keep_small_objects_and_scan_order_without_semantic_roi():
    prediction,rgb=scene()
    prediction.foreground_logits[0,0,0]=10
    prediction.foreground_logits[0,1,1]=10
    result=extract_instances(prediction,rgb)
    assert [i.box for i in result]==[(0,0,1,1),(1,1,1,1)]
    assert [i.frame_index for i in result]==[0,1]
    with pytest.raises(ValueError,match='silently discarded'):
        extract_instances(prediction,rgb,max_instances=1)


def test_shape_and_finiteness_guards_and_empty_foreground():
    prediction,rgb=scene()
    assert extract_instances(prediction,rgb)==[]
    with pytest.raises(ValueError): extract_instances(prediction,rgb.astype(float))
    prediction.displacement[0,0,0]=np.nan
    with pytest.raises(ValueError,match='displacement'): extract_instances(prediction,rgb)


def test_normalised_geometry_survives_resolution_change_for_identical_region_fractions():
    low,low_rgb=scene(10,12)
    high,high_rgb=scene(20,24)
    low.foreground_logits[:,2:4,3:7]=10
    high.foreground_logits[:,4:8,6:14]=10
    low.displacement[0]=1;high.displacement[0]=2
    a=extract_instances(low,low_rgb)[0]
    b=extract_instances(high,high_rgb)[0]
    np.testing.assert_allclose(a.descriptor,b.descriptor,atol=1e-7)


def test_finite_large_displacement_mean_does_not_overflow_float32():
    prediction,rgb=scene(2,2)
    prediction.foreground_logits[:]=1
    prediction.displacement[:]=np.finfo(np.float32).max
    result=extract_instances(prediction,rgb)[0]
    assert np.isfinite(result.descriptor).all()
    np.testing.assert_allclose(result.descriptor[5:7],[np.finfo(np.float32).max/2]*2)


def test_unnormalized_embedding_cannot_create_nonfinite_group_statistics():
    prediction,rgb=scene(2,2)
    prediction.foreground_logits[:]=1
    prediction.embedding[:]=np.finfo(np.float32).max
    with pytest.raises(ValueError,match='normalized'): extract_instances(prediction,rgb)
