"""First-act RAM, cross-shard continuity, truncation/cold charges and tampering."""
import hashlib
import json
import numpy as np
import pytest
from ragnarok.evaluation.raw_history_fixtures_v70 import KIND,raw_history_fixture_frames
from ragnarok.environments.cold_emu_pool_v70 import CONVENTION


def fixture(tmp_path,change=None):
    data=dict(rgb=np.zeros((3,210,160,3),dtype=np.uint8),
        ram=np.array([[i]*128 for i in (1,2,1)],dtype=np.uint8),
        ram_previous=np.array([[i]*128 for i in (0,1,0)],dtype=np.uint8),
        episode=np.array([0,0,1],dtype=np.int32),action=np.zeros(3,dtype=np.uint8),
        charged_frame=np.array([133,134,267],dtype=np.int64),
        frame_in_episode=np.array([1,2,1],dtype=np.int64),
        ram_frame_numbers=np.array([[0,1],[1,2],[0,1]],dtype=np.int64))
    manifest=dict(kind=KIND,game='ms_pacman',reset_convention=CONVENTION,agent_training_allowed=False,
        raw_every_real_frame=True,frames=3,episode_count=2,shards=[],
        accounting=dict(charged_frames=267,cold_loads=2,experiment_frames=0,restores=0,
            format_detection_emulated_frames=120))
    if change:change(data,manifest)
    for j,(start,end) in enumerate(((0,1),(1,3))):
        path=tmp_path/f'frames{j}.npz'
        np.savez_compressed(path,**{k:v[start:end] for k,v in data.items()})
        manifest['shards'].append(dict(path=path.name,frames=end-start,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')


def test_first_cold_ram_and_cross_shard_history_are_preserved(tmp_path):
    fixture(tmp_path)
    rows=list(raw_history_fixture_frames(tmp_path))
    assert [int(r['ram_previous'][0]) for r in rows]==[0,1,0]
    assert [r['index'] for r in rows]==[1,2,3]
    assert [int(r['charged_frame']) for r in rows]==[133,134,267]


@pytest.mark.parametrize('change',[
    lambda d,m:d['ram_previous'].__setitem__((1,0),9),
    lambda d,m:d['ram_frame_numbers'].__setitem__((0,0),1),
    lambda d,m:d['frame_in_episode'].__setitem__(2,3),
    lambda d,m:d['charged_frame'].__setitem__(2,135),
    lambda d,m:d['episode'].__setitem__(0,1),
    lambda d,m:d['action'].__setitem__(1,18),
    lambda d,m:d.update(ram_previous=d['ram_previous'].astype(np.int64)),
    lambda d,m:m.update(frames=2),
    lambda d,m:m.update(episode_count=1),
    lambda d,m:m['accounting'].update(restores=1),
])
def test_inconsistent_history_or_ledger_is_rejected(tmp_path,change):
    fixture(tmp_path,change)
    with pytest.raises(ValueError):list(raw_history_fixture_frames(tmp_path))


def test_uncharged_modification_cannot_hide_behind_old_shard_hash(tmp_path):
    fixture(tmp_path)
    with (tmp_path/'frames0.npz').open('ab') as stream:stream.write(b'tampered')
    with pytest.raises(ValueError,match='hash'):list(raw_history_fixture_frames(tmp_path))
