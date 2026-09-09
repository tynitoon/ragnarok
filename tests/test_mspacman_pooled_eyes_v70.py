"""Returning eyes, occlusion, a false state coincidence and preserved categories."""
import numpy as np
import pytest
from tests.test_mspacman_pooled_rows_v70 import row
from ragnarok.evaluation import pooled_mspacman_rows_v70 as old
from ragnarok.evaluation import pooled_mspacman_eyes_v70 as new
from ragnarok.evaluation.mspacman_ghosts_v70 import GhostDrawEntry


@pytest.mark.parametrize('index,box',[(1957,(46,28,8,5)),(1958,(46,28,3,5)),
    (1963,(46,22,8,5)),(2025,(76,63,8,5)),(7081,(106,22,8,4)),(7086,(111,19,3,4))])
def test_visible_eyes_include_two_lobes_and_actual_player_occlusion(index,box):
    r=row(index);copies={k:v.copy() for k,v in r.items()}
    original=old.source_ram_objects(r['ram_previous'],r['ram_before_previous'])
    corrected=new.source_ram_objects(r['ram_previous'],r['ram_before_previous'])
    assert box in [o.box for o in corrected if o.kind=='ghost']
    assert box not in [o.box for o in original if o.kind=='ghost']
    assert [o for o in corrected if o.kind!='ghost']==[o for o in original if o.kind!='ghost']
    assert all(np.array_equal(r[k],v) for k,v in copies.items())
    assert new.MsPacmanPooledPixelJudge is old.MsPacmanPooledPixelJudge


def test_state5_coordinate_coincidence_is_not_a_returning_eye():
    r=row(5245)
    assert new.source_ram_objects(r['ram_previous'],r['ram_before_previous'])==old.source_ram_objects(r['ram_previous'],r['ram_before_previous'])


def test_blank_normal_ghost_top_row_is_preserved():
    r=row(246)
    assert new.source_ram_objects(r['ram_current'],r['ram_previous'])==old.source_ram_objects(r['ram_current'],r['ram_previous'])


def test_duplicate_allocation_match_is_rejected():
    ram=np.zeros(128,dtype=np.uint8)
    ram[1:3]=64;ram[6:8]=58;ram[12:14]=26
    with pytest.raises(ValueError,match='Ambiguous'):
        new.returning_eye_box(GhostDrawEntry(0,1,0,(46,-19,8,10)),ram)


def test_eye_wrap_splits_at_screen_boundary_but_not_between_eyes():
    empty=np.zeros((171,160),dtype=bool)
    assert [o.box for o in new.eye_objects((100,20,8,5),empty,'eye')]==[(100,20,8,5)]
    assert [o.box for o in new.eye_objects((157,20,8,5),empty,'eye')]==[(157,20,3,5),(2,20,3,5)]
