"""Complete offline ordered equivalence of the new external judge adapters."""
from collections import deque
import hashlib
import json
from time import perf_counter

import numpy as np

from ragnarok.evaluation.fast_visible_mspacman_v70 import source_ram_objects as original
from ragnarok.evaluation.vector_visible_mspacman_v70 import source_ram_objects as candidate
from ragnarok.evaluation.cached_visible_groups_v70 import CachedVisibleGroups
from ragnarok.evaluation.scene_perception_v70 import SceneObject
from ragnarok.evaluation.visible_temporal_groups_v70 import group_visible_sources
from ragnarok.evaluation.raw_history_fixtures_v70 import raw_history_fixture_frames
from ragnarok.evaluation.triple_pooled_fixtures_v70 import triple_pooled_fixture_frames
from scripts import bench_visible_reference_v70 as bench
from scripts import check_fast_visible_mspacman_v70 as previous_audit
from scripts.temporal_validation_seal_v70 import ROOT,sha,utc,write_new,require_committed

PROTOCOL='docs/ARC4_VISIBLE_ACCELERATION_PROTOCOL.json'
NEW_SOURCES=('ragnarok/evaluation/vector_visible_mspacman_v70.py',
             'ragnarok/evaluation/cached_visible_groups_v70.py',
             'tests/test_visible_acceleration_v70.py','docs/ARC4_VISIBLE_ACCELERATION_DRAFT.md',
             'scripts/check_visible_acceleration_v70.py',PROTOCOL)
OUTPUT=ROOT/'craft_v6_out/v70_visible_cpu/vector_cached_equivalence_v0.json'
ATTEMPT=OUTPUT.with_name(OUTPUT.stem+'_attempt.json')


def hashes():
    sources={p:sha(ROOT/p) for p in (*bench.SOURCE_PATHS,*NEW_SOURCES)}
    evidence=previous_audit.hashes()['evidence_sha256']
    for path in (bench.PREFLIGHT_REVIEW,bench.FAST_REVIEW,
                 previous_audit.OUTPUT.relative_to(ROOT).as_posix(),
                 'craft_v6_out/v70_mspacman_cpu/offline_scoring_profile_v0.json'):
        evidence[path]=sha(ROOT/path)
    return dict(source_sha256=sources,evidence_sha256=evidence)


def run():
    if OUTPUT.exists() or ATTEMPT.exists():raise ValueError('Preserve every equivalence result and attempt')
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    if (protocol['collections'],protocol['windows_each'],protocol['source_pairs'],protocol['groups'])!=(
            ['development','reserved','fresh_raw'],8192,40965,24576):
        raise ValueError('Unexpected predeclared complete population')
    proof=hashes()
    cpu_review=json.loads((ROOT/bench.PREFLIGHT_REVIEW).read_text(encoding='utf-8'))
    if cpu_review['source_sha256']!=bench.source_hashes():raise ValueError('Frozen115 CPU sources changed')
    if sha(ROOT/bench.FAST_REVIEW)!=bench.FAST_REVIEW_SHA256:raise ValueError('Previous equivalence review changed')
    prior=json.loads(previous_audit.OUTPUT.read_text(encoding='utf-8'))
    for mapping in proof.values():
        for path in mapping:require_committed(path)
    record=dict(status='started',started_utc=utc(),**proof,new_emulated_frames=0,agent_training_allowed=False)
    write_new(ATTEMPT,record)
    collections={}
    try:
        for name,directory,kind in previous_audit.COLLECTIONS:
            count=dict(windows=0,source_pairs=0,labels=0,groups=0)
            times=dict(original_decoder=0.,candidate_decoder=0.,original_grouping=0.,cached_grouping=0.)
            labels_digest,groups_digest=hashlib.sha256(),hashlib.sha256()
            cache=CachedVisibleGroups('ms_pacman')
            recent=deque(maxlen=2)
            loader=triple_pooled_fixture_frames if kind=='temporal' else raw_history_fixture_frames
            for row in loader(directory):
                if kind=='temporal':
                    pairs=[(row['ram_previous'],row['ram_before_previous']),
                           (row['ram_current'],row['ram_previous'])]
                    recent.clear()
                else:
                    pairs=[(row['ram'],row['ram_previous'])]
                    if int(row['frame_in_episode'])==1:
                        recent.clear();pairs.insert(0,(row['ram_previous'],row['ram_previous']))
                for source,(current,previous) in enumerate(pairs):
                    copies=current.copy(),previous.copy()
                    tick=perf_counter();expected=original(current,previous);times['original_decoder']+=perf_counter()-tick
                    tick=perf_counter();actual=candidate(current,previous);times['candidate_decoder']+=perf_counter()-tick
                    if not np.array_equal(current,copies[0]) or not np.array_equal(previous,copies[1]):
                        raise ValueError('A decoder mutated archived RAM')
                    if actual!=expected:
                        record['first_mismatch']=dict(collection=name,index=row['index'],source=source,kind='ordered_source')
                        raise ValueError('Ordered source objects differ')
                    labels_digest.update(json.dumps([row['index'],source,[(o.kind,o.box,o.slot) for o in actual]],
                        separators=(',',':')).encode()+b'\n')
                    recent.append([SceneObject(f'{o.kind}:{o.slot}',o.kind,o.box,o.kind in ('player','ghost','fruit')) for o in actual])
                    count['source_pairs']+=1;count['labels']+=len(actual)
                own=[o for scene in recent for o in scene]
                tick=perf_counter();expected=group_visible_sources(own,game='ms_pacman');times['original_grouping']+=perf_counter()-tick
                tick=perf_counter();actual=cache.group(own);times['cached_grouping']+=perf_counter()-tick
                if actual!=expected:
                    record['first_mismatch']=dict(collection=name,index=row['index'],kind='ordered_group_membership')
                    raise ValueError('Ordered groups or source membership differ')
                groups_digest.update(json.dumps([row['index'],[(g.label.kind,g.label.box,sorted(g.members)) for g in actual]],
                    separators=(',',':')).encode()+b'\n')
                count['groups']+=1;count['windows']+=1
            earlier=prior['collections'][name]
            if (count['windows']!=8192 or any(count[k]!=earlier[k] for k in ('source_pairs','labels'))
                    or labels_digest.hexdigest()!=earlier['ordered_label_sha256']):
                raise ValueError('Previous complete archived source population/digest changed')
            collections[name]=dict(**count,ordered_label_sha256=labels_digest.hexdigest(),
                ordered_groups_sha256=groups_digest.hexdigest(),component_seconds=times)
            print(json.dumps(dict(collection=name,**collections[name])),flush=True)
        if hashes()!=proof:raise ValueError('Sources or evidence changed during equivalence audit')
        if sum(c['source_pairs'] for c in collections.values())!=40965 or sum(c['groups'] for c in collections.values())!=24576:
            raise ValueError('Incomplete combined population')
        result=dict(kind='offline_ordered_vector_cached_visible_equivalence',passed=True,created_utc=utc(),
            **proof,collections=collections,new_emulated_frames=0,agent_training_allowed=False,gate_result=None,
            limitations=protocol['limitations'])
        write_new(OUTPUT,result)
        record.update(status='complete',completed_utc=utc(),result_sha256=sha(OUTPUT))
        return result
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error),completed_collections=collections)
        raise
    finally:ATTEMPT.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    result=run()
    print(json.dumps(dict(passed=result['passed'],collections=list(result['collections']))))
