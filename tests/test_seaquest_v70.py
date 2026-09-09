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


def test_internal_projectile_requires_every_expected_player_pixel():
    from ragnarok.evaluation.seaquest_v70 import PLAYER_ANIMATION, PLAYER_COLOURS
    rgb=np.zeros((210,160,3),dtype=np.uint8)
    for dy,row in enumerate(PLAYER_ANIMATION[164]):
        for dx,value in enumerate(row):
            if value=="#":
                rgb[80+dy,40+dx]=PLAYER_COLOURS[0]
    rgb[87,48:56]=PLAYER_COLOURS[0]
    assert ("player_missile",(48,87,8,1)) in {(o.kind,o.box) for o in screen_seaquest_objects(rgb)}
    # Independent review counterexample: a missing body pixel outside the
    # hypothesised projectile contradicts the complete rendering explanation.
    rgb[87,40]=0
    assert not any(o.kind=="player_missile" for o in screen_seaquest_objects(rgb))


@pytest.mark.parametrize("frame", [1042,1158,2694,2817,3073,14875])
def test_enemy_projectile_visible_geometry_on_saved_both_direction_cases(frame):
    base=Path(__file__).resolve().parents[1]/"craft_v6_out/v70_seaquest_calibration/coverage3_development"
    first=(frame-1)//256*256+1
    with np.load(base/f"frames_{first:05}_{first+255:05}.npz",allow_pickle=False) as arrays:
        rgb,ram=arrays["rgb"][frame-first],arrays["ram"][frame-first]
    truth=[o for o in decode_seaquest_ram(ram) if o.kind=="enemy_missile"]
    pixels=[o for o in screen_seaquest_objects(rgb) if o.kind=="enemy_missile"]
    assert truth and len(truth)==len(pixels)
    assert sorted(o.box for o in truth)==sorted(o.box for o in pixels)


@pytest.mark.parametrize("frame,expected", [(6701,None),(7469,(159,50,1,3)),(15387,(8,50,1,3))])
def test_surface_submarine_hidden_edge_and_disconnected_visible_fragments(frame,expected):
    base=Path(__file__).resolve().parents[1]/"craft_v6_out/v70_seaquest_calibration/coverage4_development"
    first=(frame-1)//256*256+1
    with np.load(base/f"frames_{first:05}_{first+255:05}.npz",allow_pickle=False) as arrays:
        rgb,ram=arrays["rgb"][frame-first],arrays["ram"][frame-first]
    truth=[o.box for o in decode_seaquest_ram(ram) if o.slot=="surface"]
    pixels=[o.box for o in screen_seaquest_objects(rgb) if o.kind=="submarine" and o.box[1]<57]
    assert truth==pixels==([] if expected is None else [expected])


@pytest.mark.parametrize("frame,expected", [(1559,(79,47,16,6)),(1788,(76,57,16,6)),(8004,(72,47,16,6))])
def test_shallow_visible_player_is_not_discarded_by_height(frame,expected):
    base=Path(__file__).resolve().parents[1]/"craft_v6_out/v70_seaquest_calibration/coverage2_development"
    first=(frame-1)//256*256+1
    with np.load(base/f"frames_{first:05}_{first+255:05}.npz",allow_pickle=False) as arrays:
        rgb=arrays["rgb"][frame-first]
    assert [o.box for o in screen_seaquest_objects(rgb) if o.kind=="player"]==[expected]


def test_underwater_submarine_edge_fragments_stay_one_actor():
    base=Path(__file__).resolve().parents[1]/"craft_v6_out/v70_seaquest_calibration/coverage2_development"
    with np.load(base/"frames_01281_01536.npz",allow_pickle=False) as arrays:
        rgb=arrays["rgb"][1447-1281]
    assert [o.box for o in screen_seaquest_objects(rgb) if o.kind=="submarine"]==[(159,145,1,6)]
