import numpy as np

from ragnarok.evaluation.pooled_seaquest_v70 import pooled_seaquest_screen_objects
from ragnarok.evaluation.seaquest_v70 import SHARK_COLOURS,PLAYER_COLOURS,BLUE


def test_temporal_envelopes_keep_separated_formation_actors():
    previous=np.zeros((210,160,3),dtype=np.uint8)
    current=previous.copy()
    for x in (20,36):
        previous[117:124,x:x+8]=SHARK_COLOURS[0]
        current[117:124,x+1:x+9]=SHARK_COLOURS[0]
    objects=pooled_seaquest_screen_objects(previous,current)
    assert [(o.kind,o.box) for o in objects]==[("shark",(20,117,9,7)),("shark",(36,117,9,7))]


def test_different_categories_can_contribute_different_max_channels():
    previous=np.zeros((210,160,3),dtype=np.uint8)
    current=previous.copy()
    previous[120:131,80:96]=PLAYER_COLOURS[0]
    current[121:125,82:88]=BLUE
    objects={(o.kind,o.box) for o in pooled_seaquest_screen_objects(previous,current)}
    assert ("player",(80,120,16,11)) in objects
    assert ("enemy_missile",(82,121,6,4)) in objects
    assert tuple(np.maximum(previous,current)[122,84]) not in (BLUE,PLAYER_COLOURS[0])


def test_tied_sources_survive_but_fully_overwritten_source_does_not():
    blue=np.zeros((210,160,3),dtype=np.uint8)
    blue[121:125,82:88]=BLUE
    assert [(o.kind,o.box) for o in pooled_seaquest_screen_objects(blue,blue)]==[("enemy_missile",(82,121,6,4))]
    white=np.zeros_like(blue)
    white[121:125,82:88]=PLAYER_COLOURS[1]
    assert not any(o.kind=="enemy_missile" for o in pooled_seaquest_screen_objects(blue,white))


def test_spatial_union_uses_sides_not_diagonal_only_contact():
    from ragnarok.evaluation.pooled_seaquest_v70 import _masks,_objects
    masks=_masks()
    masks["submarine"][70:72,20:22]=True
    masks["submarine"][72:74,22:24]=True
    assert len(_objects(masks))==2
    masks["submarine"][71,22]=True
    assert len(_objects(masks))==1
