"""Pre-execution preservation and actual archived source chronology, no archive read."""
import numpy as np
import pytest
from scripts import check_gather_visible_v70 as audit


@pytest.mark.parametrize('existing',['OUTPUT','ATTEMPT'])
def test_existing_result_or_attempt_stops_before_proofs_or_archives(tmp_path,monkeypatch,existing):
    for field in ('OUTPUT','ATTEMPT'):monkeypatch.setattr(audit,field,tmp_path/field)
    getattr(audit,existing).write_bytes(b'preserve')
    monkeypatch.setattr(audit,'preflight',lambda:pytest.fail('Must refuse before any proof read'))
    with pytest.raises(ValueError,match='Preserve'):audit.run()
    assert getattr(audit,existing).read_bytes()==b'preserve'


def test_actual_temporal_pairs_and_only_cold_raw_zero_are_included():
    a,b,c=(np.full(128,i,np.uint8) for i in range(3))
    row=dict(ram_before_previous=a,ram_previous=b,ram_current=c,ram=c,frame_in_episode=1)
    pairs=audit.source_pairs(row,'temporal')
    assert pairs[0][0] is b and pairs[0][1] is a and pairs[1][0] is c and pairs[1][1] is b
    pairs=audit.source_pairs(row,'raw')
    assert len(pairs)==2 and pairs[0][0] is b and pairs[0][1] is b and pairs[1][0] is c
    row['frame_in_episode']=2
    pairs=audit.source_pairs(row,'raw')
    assert len(pairs)==1 and pairs[0][0] is c and pairs[0][1] is b
    with pytest.raises(ValueError):audit.source_pairs(row,'invented')


def test_missing_independent_preflight_stops_before_opening_archives(tmp_path,monkeypatch):
    for field in ('OUTPUT','ATTEMPT','PREFLIGHT'):monkeypatch.setattr(audit,field,tmp_path/field)
    monkeypatch.setattr(audit,'proofs',lambda:dict(source_sha256={'synthetic':'only'}))
    monkeypatch.setattr(audit,'triple_pooled_fixture_frames',lambda *_:pytest.fail('No archive before preflight'))
    with pytest.raises(FileNotFoundError):audit.run()
    assert not audit.ATTEMPT.exists() and not audit.OUTPUT.exists()
