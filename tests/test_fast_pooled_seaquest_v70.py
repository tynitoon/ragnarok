import numpy as np

from ragnarok.evaluation.seaquest_v70 import KINDS,SlottedLabel
from ragnarok.evaluation.pooled_seaquest_v70 import _masks,_objects
from ragnarok.evaluation.fast_pooled_seaquest_v70 import group_rectangles


def label(kind,box,slot):
    return SlottedLabel(kind,box,slot)


def raster(objects):
    masks=_masks()
    for o in objects:
        x,y,w,h=o.box
        masks[o.kind][y:y+h,x:x+w]=True
    return _objects(masks)


def test_side_contacts_diagonal_and_bounding_hole():
    objects=[label("shark",(10,10,2,8),"a"),label("shark",(12,16,8,2),"b"),
             label("shark",(17,10,2,2),"hole"),label("shark",(20,18,2,2),"diagonal")]
    groups=group_rectangles(objects)
    assert [g.label for g in groups]==raster(objects)
    assert len(groups)==3
    assert groups[0].members==frozenset(("shark:a","shark:b"))


def test_random_overlapping_rectangles_match_exact_raster_order():
    rng=np.random.default_rng(28371)
    for _ in range(256):
        objects=[label(KINDS[int(rng.integers(len(KINDS)))],
                       (int(rng.integers(8,42)),int(rng.integers(33,60)),
                        int(rng.integers(1,17)),int(rng.integers(1,12))),str(i)) for i in range(50)]
        groups=group_rectangles(objects)
        assert [g.label for g in groups]==raster(objects)
        assert sum(len(g.members) for g in groups)==len(objects)
