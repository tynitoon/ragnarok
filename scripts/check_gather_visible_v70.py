"""Prospective exhaustive, offline eight-pixel decoder equivalence and timing."""
import hashlib
import json
import os
from time import perf_counter

import numpy as np

from ragnarok.evaluation.fast_visible_mspacman_v70 import source_ram_objects as original
from ragnarok.evaluation.gather_visible_mspacman_v70 import source_ram_objects as candidate
from ragnarok.evaluation.raw_history_fixtures_v70 import raw_history_fixture_frames
from ragnarok.evaluation.triple_pooled_fixtures_v70 import triple_pooled_fixture_frames
from scripts import check_fast_visible_mspacman_v70 as previous_audit
from scripts.temporal_validation_seal_v70 import ROOT,sha,utc,runtime,write_new,require_committed

PROTOCOL='docs/ARC4_GATHER_DECODER_PROTOCOL.json'
PRIOR_REVIEW='craft_v6_out/v70_visible_cpu/fast_mspacman_equivalence_review_v0.json'
PRIOR_SHA='945c47fc3e4b83d8063339f02c9968a30b3fc53bed404e3b4f6e92ca633578e4'
FOLDER=ROOT/'craft_v6_out/v70_visible_cpu'
PREFLIGHT=FOLDER/'gather_decoder_preflight_v0.json'
OUTPUT=FOLDER/'gather_decoder_equivalence_v0.json'
ATTEMPT=FOLDER/'gather_decoder_equivalence_v0_attempt.json'
SOURCE_PATHS=tuple(dict.fromkeys((*previous_audit.SOURCE_PATHS,
    'scripts/temporal_validation_seal_v70.py','ragnarok/evaluation/gather_visible_mspacman_v70.py',
    'tests/test_gather_visible_mspacman_v70.py','scripts/check_gather_visible_v70.py',
    'tests/test_gather_decoder_protocol_v70.py','docs/ARC4_GATHER_DECODER_DRAFT.md',PROTOCOL)))
POPULATION=dict(collections=['development','reserved','fresh_raw'],windows_each=8192,
    source_pairs=40965,ordered_labels=3613727)


def proofs():
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    if (protocol.get('kind')!='prospective_offline_gather_decoder_v0'
            or protocol.get('population')!=POPULATION
            or protocol.get('call_order')!='global_pair_even_original_first_odd_candidate_first'
            or protocol.get('new_emulated_frames')!=0
            or protocol.get('agent_training_allowed') is not False
            or protocol.get('gate_result') is not None):
        raise ValueError('Complete prospective gather decoder protocol required')
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('One CPU thread per numerical runtime required')
    if sha(ROOT/PRIOR_REVIEW)!=PRIOR_SHA:raise ValueError('Pinned earlier equivalence review changed')
    review=json.loads((ROOT/PRIOR_REVIEW).read_text(encoding='utf-8'))
    previous=previous_audit.hashes()
    if (review.get('passed') is not True or review.get('exhaustive_archived_equivalence_confirmed') is not True
            or review.get('source_sha256')!=previous['source_sha256']
            or review.get('result_sha256')!=sha(previous_audit.OUTPUT)):
        raise ValueError('Reviewed original decoder and complete archived population required')
    evidence={PRIOR_REVIEW:PRIOR_SHA,previous_audit.OUTPUT.relative_to(ROOT).as_posix():sha(previous_audit.OUTPUT)}
    for mapping in (review['source_sha256'],review['evidence_sha256'],previous['evidence_sha256']):
        for path,value in mapping.items():
            if sha(ROOT/path)!=value or path in evidence and evidence[path]!=value:
                raise ValueError('Reviewed evidence changed or disagrees')
            evidence[path]=value
    sources={p:sha(ROOT/p) for p in SOURCE_PATHS}
    for p in {*sources,*evidence}:require_committed(p)
    return dict(source_sha256=sources,evidence_sha256=evidence,runtime=runtime())


def preflight():
    proof=proofs()
    review=json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (review.get('kind')!='gather_decoder_independent_preflight' or review.get('passed') is not True
            or any(review.get(k)!=v for k,v in proof.items())):
        raise ValueError('Exact independent preflight required before opening archive arrays')
    require_committed(PREFLIGHT.relative_to(ROOT).as_posix())
    return dict(**proof,preflight_review_sha256=sha(PREFLIGHT))


def source_pairs(row,kind):
    if kind=='temporal':
        return [(row['ram_previous'],row['ram_before_previous']),
                (row['ram_current'],row['ram_previous'])]
    if kind!='raw':raise ValueError('Only the fixed temporal and raw archive kinds are admitted')
    pairs=[(row['ram'],row['ram_previous'])]
    if int(row['frame_in_episode'])==1:
        pairs.insert(0,(row['ram_previous'],row['ram_previous']))
    return pairs


def run():
    if OUTPUT.exists() or ATTEMPT.exists():raise ValueError('Preserve every gather result and attempt; no retry')
    proof=preflight()
    prior=json.loads(previous_audit.OUTPUT.read_text(encoding='utf-8'))
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    record=dict(status='started',started_utc=utc(),**proof,new_emulated_frames=0)
    write_new(ATTEMPT,record)
    collections={};pair_number=0
    try:
        for name,directory,kind in previous_audit.COLLECTIONS:
            counts=dict(windows=0,source_pairs=0,labels=0)
            seconds=dict(original_decoder=0.,candidate_decoder=0.)
            first_calls=dict(original_decoder=0,candidate_decoder=0)
            digest=hashlib.sha256()
            loader=triple_pooled_fixture_frames if kind=='temporal' else raw_history_fixture_frames
            for row in loader(directory):
                for index,(current,previous) in enumerate(source_pairs(row,kind)):
                    copies=current.copy(),previous.copy()
                    calls=[('original_decoder',original),('candidate_decoder',candidate)]
                    if pair_number%2:calls.reverse()
                    first_calls[calls[0][0]]+=1
                    decoded={}
                    for label,function in calls:
                        tick=perf_counter();objects=function(current,previous);seconds[label]+=perf_counter()-tick
                        if not np.array_equal(current,copies[0]) or not np.array_equal(previous,copies[1]):
                            raise ValueError('Decoder mutated archived RAM')
                        decoded[label]=objects
                    actual=decoded['candidate_decoder'];expected=decoded['original_decoder']
                    if actual!=expected:
                        record['first_mismatch']=dict(collection=name,index=row['index'],source=index,
                            expected=[(o.kind,o.box,o.slot) for o in expected],
                            actual=[(o.kind,o.box,o.slot) for o in actual])
                        raise ValueError('Complete ordered source labels differ')
                    digest.update(json.dumps([row['index'],index,[(o.kind,o.box,o.slot) for o in actual]],
                        separators=(',',':')).encode()+b'\n')
                    counts['source_pairs']+=1;counts['labels']+=len(actual);pair_number+=1
                counts['windows']+=1
            earlier=prior['collections'][name]
            if (counts['windows']!=8192 or any(counts[k]!=earlier[k] for k in ('source_pairs','labels'))
                    or digest.hexdigest()!=earlier['ordered_label_sha256']):
                raise ValueError('Previous complete archived population or ordered digest changed')
            collections[name]=dict(**counts,ordered_label_sha256=digest.hexdigest(),
                component_seconds=seconds,first_call_counts=first_calls)
            print(json.dumps(dict(collection=name,**collections[name])),flush=True)
        if (pair_number!=40965 or sum(x['labels'] for x in collections.values())!=3613727
                or preflight()!=proof):
            raise ValueError('Complete population, sources and evidence must remain unchanged')
        result=dict(kind='offline_ordered_gather_decoder_equivalence',passed=True,created_utc=utc(),
            **proof,collections=collections,new_emulated_frames=0,agent_training_allowed=False,
            gate_result=None,limitations=protocol['limitations'])
        write_new(OUTPUT,result)
        record.update(status='complete',completed_utc=utc(),result_sha256=sha(OUTPUT))
        return result
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error),
            completed_collections=collections,completed_source_pairs=pair_number,no_retry=True)
        raise
    finally:ATTEMPT.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    result=run()
    print(json.dumps(dict(passed=result['passed'],collections=list(result['collections']))))
