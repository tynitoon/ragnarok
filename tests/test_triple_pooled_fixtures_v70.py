"""Terminal truncation, cross-window causal alignment and full charge checks."""
import hashlib
import json
import numpy as np
import pytest
from ragnarok.evaluation.triple_pooled_fixtures_v70 import triple_pooled_fixture_frames, KIND


def fixture(path,mutate=None,local=(4,5,4),episodes=(0,0,1),charges=(136,137,273)):
    n=len(local)
    positions=np.array([(max(0,t-2),max(0,t-1),t) for t in local],dtype=np.int64)
    old=np.zeros((n,210,160,3),dtype=np.uint8)
    new=old.copy()
    for i,(e,t) in enumerate(zip(episodes,local)):
        old[i,10,10,:]=20*e+max(0,t-1)
        new[i,10,10,:]=20*e+t
    rams=[np.repeat((positions[:,j]+20*np.array(episodes))[:,None],128,axis=1).astype(np.uint8) for j in range(3)]
    data=dict(raw_previous=old,raw_current=new,pooled=np.maximum(old,new),
        ram_before_previous=rams[0],ram_previous=rams[1],ram_current=rams[2],
        episode=np.array(episodes,dtype=np.int64),frame_in_episode=np.array(local,dtype=np.int64),
        charged_frame=np.array(charges,dtype=np.int64),action=np.zeros(n,dtype=np.int64),
        ram_frame_numbers=positions,cold_history_fallback=np.array(local)<2)
    if mutate:mutate(data)
    shard=path/'part.npz'
    np.savez_compressed(shard,**data)
    ordinary=charges[-1]-132*(episodes[-1]+1)
    manifest=dict(kind=KIND,game='ms_pacman',reset_convention='fixed_hardware_cold_start_v1',
        mode=0,action_repeat=4,agent_training_allowed=False,reference_training_allowed=False,
        agent_entry_states_allowed=False,decisions=n,episode_count=episodes[-1]+1,ordinary_real_frames=ordinary,
        accounting=dict(charged_frames=charges[-1],cold_loads=episodes[-1]+1,restores=0,experiment_frames=0),
        shards=[dict(path=shard.name,sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),decisions=n)])
    (path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return manifest


def test_truncation_and_cold_recreation_keep_all_decisions(tmp_path):
    fixture(tmp_path)
    rows=list(triple_pooled_fixture_frames(tmp_path))
    assert [int(r['charged_frame']) for r in rows]==[136,137,273]
    assert [tuple(r['ram_frame_numbers']) for r in rows]==[(2,3,4),(3,4,5),(2,3,4)]


def test_first_one_frame_window_has_only_the_declared_fallback(tmp_path):
    fixture(tmp_path,local=(1,3,1),episodes=(0,0,1),charges=(133,135,268))
    rows=list(triple_pooled_fixture_frames(tmp_path))
    assert [bool(r['cold_history_fallback']) for r in rows]==[True,False,True]


@pytest.mark.parametrize('key,index,value',[
    ('ram_frame_numbers',(1,0),2), ('cold_history_fallback',0,True),
    ('ram_previous',(1,0),99), ('ram_before_previous',(1,0),99),
    ('raw_previous',(1,10,10,0),99), ('pooled',(1,10,10,0),99),
    ('charged_frame',2,213), ('episode',2,2), ('frame_in_episode',1,9), ('action',1,18),
])
def test_misaligned_or_undercharged_windows_reject(tmp_path,key,index,value):
    fixture(tmp_path,lambda d:d[key].__setitem__(index,value))
    with pytest.raises(ValueError):list(triple_pooled_fixture_frames(tmp_path))


def test_nonduplicated_first_ram_is_rejected(tmp_path):
    fixture(tmp_path,lambda d:d['ram_before_previous'].__setitem__((0,0),88),
            local=(1,3,1),episodes=(0,0,1),charges=(133,135,268))
    with pytest.raises(ValueError,match='duplicate'):list(triple_pooled_fixture_frames(tmp_path))


@pytest.mark.parametrize('change',['hash','duplicate','outside','ordinary','experiment','dtype'])
def test_malformed_provenance_is_rejected(tmp_path,change):
    m=fixture(tmp_path,lambda d:d.update(action=d['action'].astype(np.uint8)) if change=='dtype' else None)
    if change=='hash':m['shards'][0]['sha256']='0'*64
    if change=='duplicate':m['shards']*=2
    if change=='outside':m['shards'][0]['path']='../part.npz'
    if change=='ordinary':m['ordinary_real_frames']-=1
    if change=='experiment':m['accounting']['experiment_frames']=1
    (tmp_path/'manifest.json').write_text(json.dumps(m),encoding='utf-8')
    with pytest.raises(ValueError):list(triple_pooled_fixture_frames(tmp_path))
