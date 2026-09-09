"""Closed until both corresponding immutable temporal result proofs are reviewed.

Read-only prerequisite for fresh reference work. No emulator is created here.
Review fingerprints must be filled from actual completed independent reviews
before this new source and the downstream reference protocol are frozen.
"""
import json
from scripts import validate_mspacman_pooled_v70 as ms
from scripts import validate_asteroids_pooled_v70 as ast
from scripts.temporal_validation_seal_v70 import ROOT,sha,require_committed,check_statistics
from ragnarok.evaluation.triple_pooled_fixtures_v70 import triple_pooled_fixture_frames

SEALED_SOURCES=tuple(dict.fromkeys((*ms.SOURCE_PATHS,*ast.SOURCE_PATHS)))
REVIEW_SHA256={'ms_pacman':None,'asteroids':'15b8931c94de593cc3738727e0d0de34ce93220ee6640da87bf2dcabb590b503'}
RUNNERS={'ms_pacman':ms,'asteroids':ast}


def require_temporal_calibration(game):
    if game not in RUNNERS:raise ValueError('Only the declared training games are enabled')
    runner=RUNNERS[game]
    expected=REVIEW_SHA256[game]
    if any(not isinstance(REVIEW_SHA256[g],str) or len(REVIEW_SHA256[g])!=64 for g in RUNNERS):
        raise ValueError('Completed independent temporal reserved review is not yet frozen')
    proof=runner.check()
    review_path=runner.RESULT.parent/('pooled_reserved_result_review_v2.json' if game=='ms_pacman'
                                    else 'pooled_reserved_result_review_v1.json')
    if sha(review_path)!=expected:raise ValueError('Independent temporal review changed')
    review=json.loads(review_path.read_text(encoding='utf-8'))
    if (review.get('passed') is not True or review.get('reserved_calibration_pass_confirmed') is not True
            or review.get('result_sha256')!=sha(runner.RESULT)
            or review.get('source_sha256')!=proof['source_sha256']):
        raise ValueError('Complete successful temporal validation must be independently reviewed')
    evidence=review.get('evidence_sha256',{})
    required={p.relative_to(ROOT).as_posix() for p in (
        runner.RESULT,runner.ATTEMPT,runner.DESTINATION/'manifest.json',
        runner.RESULT.with_name(runner.RESULT.stem+'_attempt.json'),ROOT/runner.SPEC['seal'])}
    if not required<=set(evidence):raise ValueError('Temporal result review lacks required evidence')
    for p,h in evidence.items():
        if sha(ROOT/p)!=h:raise ValueError(f'Temporal evidence changed: {p}')
        require_committed(p)
    require_committed(review_path.relative_to(ROOT).as_posix())
    if sum(1 for _ in triple_pooled_fixture_frames(runner.DESTINATION))!=8192:
        raise ValueError('Complete reserved windows and ledger must remain intact')
    result=json.loads(runner.RESULT.read_text(encoding='utf-8'))
    if (result.get('seed')!=runner.SPEC['reserved_seed'] or result.get('decoder_revision')!=runner.REVISION
            or result.get('source_sha256')!=proof['source_sha256']
            or result.get('validation_seal_sha256')!=proof['seal_sha256']):
        raise ValueError('Reserved result differs from its final source/seed/seal')
    check_statistics(result,runner.KINDS)
    return sha(runner.RESULT)
