"""Keep terminal truncation and cold-load accounting in the temporal denominator."""
import hashlib
import json

import numpy as np
import pytest

from ragnarok.evaluation.cold_pooled_fixtures_v70 import cold_pooled_fixture_frames


def fixture(path, *, charges=(136,138,274), episodes=(0,0,1), local=(4,6,4), bad_pool=False):
    rgb=np.zeros((3,210,160,3),dtype=np.uint8)
    pool=rgb.copy()
    if bad_pool:
        pool[0,0,0,0]=1
    shard=path/"part.npz"
    np.savez_compressed(shard,raw_previous=rgb,raw_current=rgb,pooled=pool,
                        ram_previous=np.zeros((3,128),dtype=np.uint8),ram_current=np.zeros((3,128),dtype=np.uint8),
                        episode=np.array(episodes),frame_in_episode=np.array(local),charged_frame=np.array(charges),
                        action=np.zeros(3,dtype=np.int32))
    manifest=dict(kind="scorer_only_cold_two_raw_frame_fixture",agent_training_allowed=False,
                  reset_convention="fixed_hardware_cold_start_v1",game="breakout",mode=0,action_repeat=4,
                  decisions=3,episode_count=2,accounting=dict(charged_frames=charges[-1],cold_loads=2),
                  shards=[dict(path=shard.name,decisions=3,sha256=hashlib.sha256(shard.read_bytes()).hexdigest())])
    (path/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")


def test_truncated_terminal_window_and_complete_next_load_are_included(tmp_path):
    fixture(tmp_path)
    windows=list(cold_pooled_fixture_frames(tmp_path))
    assert [int(x["frame_in_episode"]) for x in windows]==[4,6,4]
    assert [int(x["charged_frame"]) for x in windows]==[136,138,274]


@pytest.mark.parametrize("kwargs",[
    dict(charges=(76,78,154)),  # old load charge omits format detection
    dict(charges=(136,138,214)),  # only the restart omits it
    dict(episodes=(0,0,2)),
    dict(local=(4,9,4)),
    dict(bad_pool=True),
])
def test_undercharges_missing_episodes_and_fabricated_windows_are_rejected(tmp_path,kwargs):
    fixture(tmp_path,**kwargs)
    with pytest.raises(ValueError):
        list(cold_pooled_fixture_frames(tmp_path))


def test_tampered_shard_is_rejected(tmp_path):
    fixture(tmp_path)
    with (tmp_path/"part.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError,match="hash"):
        list(cold_pooled_fixture_frames(tmp_path))
