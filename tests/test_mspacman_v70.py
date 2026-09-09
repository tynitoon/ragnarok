import numpy as np
from ragnarok.evaluation.mspacman_v70 import decode_mspacman_ram,screen_mspacman_objects


def test_functional_actor_creation_has_no_first_frame_stale_object():
    ram=np.zeros(128,dtype=np.uint8)
    ram[10],ram[16]=80,90
    ram[6:10]=[30,40,50,60]
    ram[12:16]=[20,30,40,50]
    ram[19],ram[47]=3,1
    ram[11],ram[17]=100,110
    objects=decode_mspacman_ram(ram)
    assert [o.box for o in objects if o.kind=="ghost"]==[(17,21,9,10),(27,31,9,10),(37,41,9,10),(47,51,9,10)]
    assert [o.box for o in objects if o.kind=="fruit"]==[(87,111,9,10)]
    ram[47]=0
    assert not any(o.kind=="ghost" for o in decode_mspacman_ram(ram))


def test_grid_bit_mapping_edges_and_alternating_halves():
    ram=np.zeros(128,dtype=np.uint8)
    ram[59]=16|64
    ram[60]=1|128
    ram[61]=1|128
    positions={o.box for o in decode_mspacman_ram(ram) if o.kind=="pill"}
    assert positions=={(148,7,4,2),(8,7,4,2),(108,7,4,2),(16,7,4,2),(48,7,4,2),(140,7,4,2)}


def test_rgb_all_frightened_components_and_hud_exclusion():
    rgb=np.zeros((210,160,3),dtype=np.uint8)
    rgb[40:50,20:29]=(66,114,194)
    rgb[60:70,40:49]=(66,114,194)
    rgb[173:183,125:134]=(184,50,50)
    objects=screen_mspacman_objects(rgb)
    assert len([o for o in objects if o.kind=="ghost"])==2
    assert not any(o.kind=="fruit" for o in objects)
