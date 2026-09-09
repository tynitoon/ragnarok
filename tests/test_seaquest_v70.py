import numpy as np
import pytest
from pathlib import Path

from ragnarok.evaluation.seaquest_v70 import decode_seaquest_ram,screen_seaquest_objects,BLUE,SHARK_COLOURS


def test_formations_have_distinct_slots_and_signed_geometry():
    ram=np.zeros(128,dtype=np.uint8)
    ram[30],ram[36],ram[93]=140,7,2
    objects=decode_seaquest_ram(ram)
    sharks=[o for o in objects if o.kind=="shark"]
    assert [(o.slot,o.box) for o in sharks]==[("enemy:0:0",(140,139,8,7)),("enemy:0:1",(156,141,4,5))]
    ram[89]=4
    subs=[o for o in decode_seaquest_ram(ram) if o.kind=="submarine"]
    assert len(subs)==2 and subs[0].box==(140,141,8,9)
    ram[105]=5
    assert "player" not in {o.kind for o in decode_seaquest_ram(ram)}
    with pytest.raises(ValueError):
        decode_seaquest_ram(ram.astype(int))


def test_shared_blue_positions_change_category_without_leaking_into_rgb_checker():
    ram=np.zeros(128,dtype=np.uint8)
    ram[74]=20
    obj=[o for o in decode_seaquest_ram(ram) if o.slot=="blue:3"][0]
    assert obj.kind=="diver" and obj.box==(20,70,8,9)
    ram[92]=4
    obj=[o for o in decode_seaquest_ram(ram) if o.slot=="blue:3"][0]
    assert obj.kind=="enemy_missile" and obj.box==(22,73,6,4)
    rgb=np.zeros((210,160,3),dtype=np.uint8)
    rgb[70:79,20:28]=BLUE
    rgb[117:124,80:88]=SHARK_COLOURS[2]
    rgb[178:189,60:68]=BLUE  # HUD/logo outside actor ROI
    assert {(o.kind,o.box) for o in screen_seaquest_objects(rgb)}=={("diver",(20,70,8,9)),("shark",(80,117,8,7))}


def test_internal_projectile_recovered_from_saved_rgb_without_ram():
    # Actual development frame545: the projectile fills one silhouette hole
    # without enlarging the player's connected component. Load RGB only.
    path=Path(__file__).resolve().parents[1]/"craft_v6_out/v70_seaquest_calibration/development/frames_00513_00768.npz"
    with np.load(path,allow_pickle=False) as arrays:
        rgb=arrays["rgb"][545-513].copy()
    before=rgb.copy()
    objects=screen_seaquest_objects(rgb)
    assert ("player_missile",(49,72,8,1)) in {(o.kind,o.box) for o in objects}
    assert np.array_equal(rgb,before)
