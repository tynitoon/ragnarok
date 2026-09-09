import numpy as np
import pytest

from ragnarok.evaluation.seaquest_v70 import decode_seaquest_ram,screen_seaquest_objects,BLUE,SHARK_COLOURS


def test_formations_have_distinct_slots_and_signed_geometry():
    ram=np.zeros(128,dtype=np.uint8)
    ram[30],ram[36],ram[93]=140,7,2
    objects=decode_seaquest_ram(ram)
    sharks=[o for o in objects if o.kind=="shark"]
    assert [(o.slot,o.box) for o in sharks]==[("enemy:0:0",(140,139,8,7)),("enemy:0:1",(156,139,4,7))]
    ram[89]=4
    subs=[o for o in decode_seaquest_ram(ram) if o.kind=="submarine"]
    assert len(subs)==2 and subs[0].box==(140,141,8,11)
    ram[105]=5
    assert "player" not in {o.kind for o in decode_seaquest_ram(ram)}
    with pytest.raises(ValueError):
        decode_seaquest_ram(ram.astype(int))


def test_shared_blue_positions_change_category_without_leaking_into_rgb_checker():
    ram=np.zeros(128,dtype=np.uint8)
    ram[74]=20
    obj=[o for o in decode_seaquest_ram(ram) if o.slot=="blue:3"][0]
    assert obj.kind=="diver" and obj.box==(20,69,8,11)
    ram[92]=4
    obj=[o for o in decode_seaquest_ram(ram) if o.slot=="blue:3"][0]
    assert obj.kind=="enemy_missile" and obj.box==(23,73,6,4)
    rgb=np.zeros((210,160,3),dtype=np.uint8)
    rgb[69:80,20:28]=BLUE
    rgb[117:124,80:88]=SHARK_COLOURS[2]
    rgb[178:189,60:68]=BLUE  # HUD/logo outside actor ROI
    assert {(o.kind,o.box) for o in screen_seaquest_objects(rgb)}=={("diver",(20,69,8,11)),("shark",(80,117,8,7))}
