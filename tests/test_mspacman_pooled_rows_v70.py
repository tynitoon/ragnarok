"""Development contact witnesses and a guard against changing non-ghost targets."""
import json
import numpy as np
import pytest
from scripts.capture_mspacman_pooled_v70 import DESTINATION
from ragnarok.evaluation import pooled_mspacman_v70 as old
from ragnarok.evaluation import pooled_mspacman_rows_v70 as new


def row(index):
    manifest=json.loads((DESTINATION/'manifest.json').read_text(encoding='utf-8'))
    shard=DESTINATION/manifest['shards'][(index-1)//128]['path']
    with np.load(shard,allow_pickle=False) as archive:
        return {k:archive[k][(index-1)%128] for k in ('ram_before_previous','ram_previous','ram_current')}


@pytest.mark.parametrize('index,old_count,new_count',[(246,3,4),(1250,3,4),(1275,3,4),(1311,3,4)])
def test_a_blank_row_does_not_join_two_non_touching_visible_ghosts(index,old_count,new_count):
    r=row(index);args=[r[k] for k in ('ram_before_previous','ram_previous','ram_current')]
    a=old.pooled_mspacman_ram_objects(*args);b=new.pooled_mspacman_ram_objects(*args)
    assert sum(o.kind=='ghost' for o in a)==old_count
    assert sum(o.kind=='ghost' for o in b)==new_count
    assert [o for o in a if o.kind!='ghost']==[o for o in b if o.kind!='ghost']
    assert new.MsPacmanPooledPixelJudge is old.MsPacmanPooledPixelJudge


def test_corrected_second_source_keeps_its_original_lower_edge():
    r=row(246)
    original=[o for o in old.decode_visible_ram(r['ram_current'],r['ram_previous']) if o.kind=='ghost']
    corrected=[o for o in new.source_ram_objects(r['ram_current'],r['ram_previous']) if o.kind=='ghost']
    assert (54,37,8,10) in [o.box for o in original]
    assert (54,38,8,9) in [o.box for o in corrected]
