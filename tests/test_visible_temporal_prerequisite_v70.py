"""No downstream capture without a complete immutable reviewed temporal result."""
import json
from types import SimpleNamespace
import pytest
from scripts import visible_temporal_prerequisite_v70 as prerequisite


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(prerequisite,'ROOT',tmp_path)
    monkeypatch.setattr(prerequisite,'triple_pooled_fixture_frames',lambda path:iter([None]*8192))
    committed=[];monkeypatch.setattr(prerequisite,'require_committed',committed.append)
    source={'model.py':'model-hash'}
    def write(name,data):
        path=tmp_path/name;path.write_text(json.dumps(data),encoding='utf-8');return path
    stats=dict(frames=8192,presence_count_agreement=8192,positive_frames=8192,positive_count_agreement=8192,
        ram_objects=8192,pixel_objects=8192,joint_matches=8192,boxes_within1=8192,boxes_exact=8192,
        presence_fraction=1.,positive_presence_fraction=1.,box_fraction=1.,passed=True)
    result=write('result.json',dict(seed=17522,decoder_revision='v1',source_sha256=source,
        validation_seal_sha256='seal-hash',calibration_pass=True,per_type={'player':stats}))
    for name in ('capture.json','manifest.json','result_attempt.json','seal.json'):write(name,{})
    evidence={p.name:prerequisite.sha(p) for p in tmp_path.iterdir()}
    review=write('pooled_reserved_result_review_v1.json',dict(passed=True,reserved_calibration_pass_confirmed=True,
        result_sha256=prerequisite.sha(result),source_sha256=source,evidence_sha256=evidence))
    runner=SimpleNamespace(RESULT=result,ATTEMPT=tmp_path/'capture.json',DESTINATION=tmp_path,
        SPEC=dict(seal='seal.json',reserved_seed=17522),REVISION='v1',KINDS=('player',),
        check=lambda:dict(source_sha256=source,seal_sha256='seal-hash'))
    monkeypatch.setattr(prerequisite,'RUNNERS',{'asteroids':runner})
    monkeypatch.setattr(prerequisite,'REVIEW_SHA256',{'asteroids':prerequisite.sha(review)})
    return result,review,committed


def test_complete_immutable_reviewed_temporal_result_is_required(tmp_path,monkeypatch):
    result,review,committed=fixture(tmp_path,monkeypatch)
    assert prerequisite.require_temporal_calibration('asteroids')==prerequisite.sha(result)
    assert {'result.json','capture.json','manifest.json','result_attempt.json','seal.json',review.name}<=set(committed)


def test_missing_final_review_fingerprint_keeps_seed_closed(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);monkeypatch.setitem(prerequisite.REVIEW_SHA256,'asteroids',None)
    with pytest.raises(ValueError,match='not yet frozen'):prerequisite.require_temporal_calibration('asteroids')


@pytest.mark.parametrize('change',[
    lambda r:r.update(passed=False),lambda r:r.update(reserved_calibration_pass_confirmed=False),
    lambda r:r.update(result_sha256='other'),lambda r:r.update(source_sha256={}),
    lambda r:r['evidence_sha256'].pop('capture.json'),
])
def test_wrong_or_incomplete_review_cannot_unlock_reference(tmp_path,monkeypatch,change):
    _,review,_=fixture(tmp_path,monkeypatch)
    data=json.loads(review.read_text());change(data);review.write_text(json.dumps(data))
    monkeypatch.setitem(prerequisite.REVIEW_SHA256,'asteroids',prerequisite.sha(review))
    with pytest.raises(ValueError):prerequisite.require_temporal_calibration('asteroids')


def test_changed_evidence_after_review_is_rejected(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);(tmp_path/'capture.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='evidence changed'):prerequisite.require_temporal_calibration('asteroids')


def test_reserved_shards_are_strictly_rechecked_even_if_review_only_lists_manifest(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    def reject(path):raise ValueError('Reserved NPZ hash changed')
    monkeypatch.setattr(prerequisite,'triple_pooled_fixture_frames',reject)
    with pytest.raises(ValueError,match='NPZ hash changed'):
        prerequisite.require_temporal_calibration('asteroids')


def test_shared_sources_wait_for_both_final_review_fingerprints(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    monkeypatch.setitem(prerequisite.RUNNERS,'ms_pacman',object())
    monkeypatch.setitem(prerequisite.REVIEW_SHA256,'ms_pacman',None)
    with pytest.raises(ValueError,match='not yet frozen'):
        prerequisite.require_temporal_calibration('asteroids')
