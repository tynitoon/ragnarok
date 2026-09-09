"""Three development exceptions become exact visible explosion pieces."""
import json
import numpy as np
import pytest
from scripts.capture_asteroids_pooled_v70 import DESTINATION
from ragnarok.evaluation import pooled_asteroids_wrap_v70 as new
from ragnarok.evaluation.asteroids_player_roi_v70 import decode_player_missile_ram,screen_player_missile_objects


def row(index):
    manifest=json.loads((DESTINATION/'manifest.json').read_text(encoding='utf-8'))
    with np.load(DESTINATION/manifest['shards'][(index-1)//128]['path'],allow_pickle=False) as archive:
        return {k:archive[k][(index-1)%128] for k in ('ram_previous','ram_current','raw_current')}


@pytest.mark.parametrize('index,boxes',[
    (5320,[(34,18,5,6),(35,192,1,2)]),
    (5321,[(32,18,7,6),(36,192,1,2)]),
    (5322,[(34,18,5,6),(34,192,1,2)]),
])
def test_negative_y_explosion_keeps_both_visible_roi_clipped_pieces(index,boxes):
    r=row(index);saved=r['ram_current'].copy()
    with pytest.raises(ValueError,match='pointer/direction'):decode_player_missile_ram(r['ram_current'])
    labels=new.source_ram_objects(r['ram_current'],r['ram_previous'])
    pixels=screen_player_missile_objects(r['raw_current'])
    assert [o.box for o in labels if o.kind=='player']==boxes
    assert [o.box for o in pixels if o.kind=='player']==boxes
    assert np.array_equal(saved,r['ram_current'])
    assert all(16<=y and y+h<=194 for x,y,w,h in boxes)


def test_unrecognized_pointer_remains_an_error():
    r=row(5320);ram=r['ram_current'].copy();ram[109]=1
    with pytest.raises(ValueError,match='pointer/direction'):new.source_ram_objects(ram,r['ram_previous'])


def test_normal_positive_y_model_and_rgb_path_are_unchanged():
    from ragnarok.evaluation.asteroids_visible_v70 import decode_visible_asteroids_ram
    from ragnarok.evaluation.pooled_asteroids_v70 import pooled_asteroids_screen_objects
    r=row(33)
    assert new.source_ram_objects(r['ram_current'],r['ram_previous'])==decode_visible_asteroids_ram(r['ram_current'],r['ram_previous'])
    assert new.pooled_asteroids_screen_objects is pooled_asteroids_screen_objects
