"""Synthetic exact eight-pixel pill coverage; no archives, timings or emulator."""
import ast
import inspect
import textwrap

import numpy as np
import pytest

from ragnarok.evaluation import fast_visible_mspacman_v70 as original
from ragnarok.evaluation import gather_visible_mspacman_v70 as candidate


def expected(ram,occupied):
    return original._unoccluded_pills(original._nominal_pills(ram)+original.power_pill_render_boxes(ram),occupied)


def test_complete_source_ast_changes_only_the_pill_coverage_call():
    before=ast.parse(textwrap.dedent(inspect.getsource(original.source_ram_objects)))
    after=ast.parse(textwrap.dedent(inspect.getsource(candidate.source_ram_objects)))
    count=0
    for node in ast.walk(before):
        if isinstance(node,ast.AugAssign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name):
            if node.value.func.id=='_unoccluded_pills':
                node.value=ast.parse('_nominal_and_power(current, occupied)',mode='eval').body
                count+=1
    assert count==1 and ast.dump(before)==ast.dump(after)


@pytest.mark.parametrize('grid',[0,1])
def test_all_256_occlusion_patterns_keep_exact_order_and_clipped_boxes(grid):
    ram=np.zeros(128,np.uint8);ram[0]=grid;ram[59:101]=255
    label=original._nominal_pills(ram)[5]
    x,y,w,h=label.box
    for code in range(256):
        occupied=np.zeros((171,160),bool)
        occupied[y:y+h,x:x+w]=np.array([(code>>i)&1 for i in range(8)],bool).reshape(2,4)
        before=ram.copy(),occupied.copy()
        assert candidate._nominal_and_power(ram,occupied)==expected(ram,occupied)
        np.testing.assert_array_equal(ram,before[0]);np.testing.assert_array_equal(occupied,before[1])


def test_varied_pill_bits_power_pills_and_sparse_dense_actor_support_match():
    rng=np.random.default_rng(792)
    for i in range(100):
        ram=rng.integers(0,256,128,dtype=np.uint8)
        ram[0]=i%2
        occupied=rng.random((171,160))<(.001 if i%2 else .7)
        assert candidate._nominal_and_power(ram,occupied)==expected(ram,occupied)
    ram=np.zeros(128,np.uint8)
    assert candidate._nominal_and_power(ram,np.zeros((171,160),bool))==expected(ram,np.zeros((171,160),bool))
    for wrong in (np.zeros((170,160),bool),np.zeros((171,160),np.uint8)):
        with pytest.raises(ValueError):candidate._nominal_and_power(ram,wrong)
