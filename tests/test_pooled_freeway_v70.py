import numpy as np

from ragnarok.evaluation.freeway_v70 import decode_freeway_ram,CAR_COLOURS,CHICKEN_COLOUR
from ragnarok.evaluation.pooled_freeway_v70 import pooled_freeway_ram_objects,pooled_freeway_screen_objects


def test_identity_and_symmetry_include_wrap_disappearance():
    a=np.zeros(128,dtype=np.uint8)
    b=a.copy()
    a[117]=2
    b[117]=3
    assert pooled_freeway_ram_objects(a,a)==decode_freeway_ram(a)
    assert pooled_freeway_ram_objects(a,b)==pooled_freeway_ram_objects(b,a)
    assert {o.kind:o.box for o in pooled_freeway_ram_objects(a,b)}["car_0"]==(159,28,1,8)


def test_two_rgb_sources_retain_contributing_pixels_and_exclude_road_dashes():
    a=np.zeros((210,160,3),dtype=np.uint8)
    b=a.copy()
    a[150:158,44:51]=CHICKEN_COLOUR
    b[148:156,44:51]=CHICKEN_COLOUR
    a[102,12:16]=b[102,12:16]=CHICKEN_COLOUR
    a[28:36,50:58]=CAR_COLOURS[0]
    b[28:36,52:60]=CAR_COLOURS[0]
    objects=pooled_freeway_screen_objects(a,b)
    assert objects==pooled_freeway_screen_objects(b,a)
    assert {o.kind:o.box for o in objects}=={"player":(44,148,7,10),"car_0":(50,28,10,8)}
    # White suppresses all red-car contributions from A, regardless of RAM geometry.
    b[28:36,50:60]=255
    assert "car_0" not in {o.kind for o in pooled_freeway_screen_objects(a,b)}
