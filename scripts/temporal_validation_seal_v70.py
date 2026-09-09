"""Read-only development proof and committed seal for a declared temporal validation.

No emulator is constructed here. The caller supplies its fixed specification
and independently reviewed raw prerequisite; both remain in its source closure.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import ale_py
from ragnarok.evaluation.triple_pooled_fixtures_v70 import triple_pooled_fixture_frames

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def runtime():
    return dict(python=sys.version,versions={p:importlib.metadata.version(p)
        for p in ('ale-py','numpy','scipy','pillow')},ale_binary_sha256=sha(ale_py._ale_py.__file__))


def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def require_committed(path,commit='HEAD'):
    if git('hash-object',f'--path={path}',str(ROOT/path))!=git('rev-parse',f'{commit}:{path}'):
        raise ValueError(f'Committed proof differs: {path}')


def check_statistics(report,kinds):
    if report.get('calibration_pass') is not True or set(report.get('per_type',{}))!=set(kinds):
        raise ValueError('Complete development PASS required for every declared category')
    for s in report['per_type'].values():
        counters=('frames','presence_count_agreement','positive_frames','positive_count_agreement',
            'ram_objects','pixel_objects','joint_matches','boxes_within1','boxes_exact')
        if (any(type(s[k]) is not int or s[k]<0 for k in counters)
                or s['frames']!=8192 or s['joint_matches']<50
                or not 0<s['positive_frames']<=s['frames']
                or not 0<=s['positive_count_agreement']<=s['positive_frames']
                or not 0<=s['presence_count_agreement']<=s['frames']
                or s['frames']-s['presence_count_agreement']!=s['positive_frames']-s['positive_count_agreement']
                or not 0<=s['boxes_exact']<=s['boxes_within1']<=s['joint_matches']
                or s['joint_matches']>min(s['ram_objects'],s['pixel_objects'])):
            raise ValueError('Invalid complete-development denominators')
        fractions=(s['presence_count_agreement']/s['frames'],
            s['positive_count_agreement']/s['positive_frames'],s['boxes_within1']/s['joint_matches'])
        recorded=(s['presence_fraction'],s['positive_presence_fraction'],s['box_fraction'])
        if fractions!=recorded or not all(.99<=v<=1 for v in fractions) or s['passed'] is not True:
            raise ValueError('Development does not meet its unchanged count/geometry criteria')


def development_evidence(spec,require_raw):
    source_hashes={p:sha(ROOT/p) for p in spec['source_paths']}
    raw_proof=require_raw()
    result=json.loads((ROOT/spec['development_result']).read_text(encoding='utf-8'))
    directory=ROOT/spec['development_fixture']
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    capture_attempt=json.loads((ROOT/spec['development_capture_attempt']).read_text(encoding='utf-8'))
    score_attempt=json.loads((ROOT/spec['development_score_attempt']).read_text(encoding='utf-8'))
    if (result.get('seed')!=spec['development_seed'] or result.get('decoder_revision')!=spec['revision']
            or result.get('source_sha256')!={p:source_hashes[p] for p in spec['development_sources']}
            or result.get('raw_calibration_proof_sha256')!=raw_proof
            or result.get('fixture_manifest_sha256')!=sha(directory/'manifest.json')
            or manifest.get('game')!=spec['game'] or manifest.get('seed')!=spec['development_seed']
            or manifest.get('partition')!='development' or manifest.get('decisions')!=8192
            or manifest.get('source_sha256')!={p:source_hashes[p] for p in spec['capture_sources']}
            or manifest.get('runtime')!=runtime() or capture_attempt.get('status')!='complete'
            or capture_attempt.get('seed')!=spec['development_seed']
            or capture_attempt.get('source_sha256')!=manifest.get('source_sha256')
            or capture_attempt.get('source_commit')!=manifest.get('source_commit')
            or capture_attempt.get('runtime')!=manifest.get('runtime')
            or capture_attempt.get('accounting')!=manifest.get('accounting')
            or capture_attempt.get('raw_calibration_proof_sha256')!=raw_proof
            or capture_attempt.get('manifest_sha256')!=sha(directory/'manifest.json')
            or score_attempt.get('status')!='complete' or score_attempt.get('calibration_pass') is not True
            or score_attempt.get('seed')!=spec['development_seed']
            or score_attempt.get('fixture_manifest_sha256')!=sha(directory/'manifest.json')
            or score_attempt.get('result_sha256')!=sha(ROOT/spec['development_result'])):
        raise ValueError('Complete development, capture, scoring and raw proof must agree')
    check_statistics(result,spec['kinds'])
    if sum(1 for _ in triple_pooled_fixture_frames(directory))!=8192:
        raise ValueError('Every development window and its full ledger must remain intact')
    review=json.loads((ROOT/spec['development_review']).read_text(encoding='utf-8'))
    if (review.get('passed') is not True or review.get('development_calibration_pass_confirmed') is not True
            or review.get('result_sha256')!=sha(ROOT/spec['development_result'])):
        raise ValueError('Independent review of the complete successful development required')
    preflight=json.loads((ROOT/spec['preflight_review']).read_text(encoding='utf-8'))
    if (preflight.get('passed') is not True or preflight.get('source_sha256')!=source_hashes
            or preflight.get('runtime')!=runtime()
            or preflight.get('development_result_sha256')!=sha(ROOT/spec['development_result'])):
        raise ValueError('Independent reserved preflight review must bind all current sources')
    metadata_hashes={p:sha(ROOT/p) for p in spec['metadata_paths']}
    for p in (*spec['source_paths'],*spec['metadata_paths']):require_committed(p)
    return dict(source_sha256=source_hashes,metadata_sha256=metadata_hashes,
                raw_calibration_proof_sha256=raw_proof,runtime=runtime())


def create_seal(spec,require_raw):
    path=ROOT/spec['seal']
    if path.exists():raise ValueError('Preserve the existing validation seal')
    evidence=development_evidence(spec,require_raw)
    seal=dict(kind='visible_temporal_reserved_seal',game=spec['game'],seed=spec['reserved_seed'],
        decisions=8192,created_utc=utc(),source_commit=git('rev-parse','HEAD'),**evidence)
    write_new(path,seal)
    return seal


def verify_seal(spec,require_raw):
    seal=json.loads((ROOT/spec['seal']).read_text(encoding='utf-8'))
    evidence=development_evidence(spec,require_raw)
    if (seal.get('kind')!='visible_temporal_reserved_seal' or seal.get('game')!=spec['game']
            or seal.get('seed')!=spec['reserved_seed'] or seal.get('decisions')!=8192
            or any(seal.get(k)!=v for k,v in evidence.items())):
        raise ValueError('Sealed validation source, evidence, runtime or seed differs')
    subprocess.check_call(['git','-C',str(ROOT),'merge-base','--is-ancestor',seal['source_commit'],'HEAD'])
    for p in (*spec['source_paths'],*spec['metadata_paths']):
        require_committed(p,seal['source_commit'])
    require_committed(spec['seal'])
    return dict(protocol=json.loads((ROOT/spec['protocol']).read_text(encoding='utf-8')),
        source_sha256=evidence['source_sha256'],runtime=evidence['runtime'],
        raw_calibration_proof_sha256=evidence['raw_calibration_proof_sha256'],
        review_sha256=sha(ROOT/spec['preflight_review']),seal_sha256=sha(ROOT/spec['seal']))
