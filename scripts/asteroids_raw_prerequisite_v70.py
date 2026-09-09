"""Read-only reviewed raw proof required before subsequent Asteroids work."""
import json
from scripts.validate_asteroids_raw_v70 import (
    ROOT, BASE, SOURCE_PATHS as RAW_SEALED_SOURCES, verify_seal, sha, git,
)

REVIEW='craft_v6_out/v70_asteroids_calibration/visible_reserved_review.json'
REVIEW_SHA256='d019a8992d59ef9b050ebe0bcf489d2783da3ddd39485d4de1465d825b827dc9'


def require_raw_calibration():
    seal=verify_seal()
    if sha(ROOT/REVIEW)!=REVIEW_SHA256:
        raise ValueError('The independent raw calibration review changed')
    review=json.loads((ROOT/REVIEW).read_text(encoding='utf-8'))
    if (review.get('passed') is not True or review.get('reserved_calibration_pass_confirmed') is not True
            or review.get('source_sha256')!=seal['source_sha256']):
        raise ValueError('Passed independently reviewed raw calibration required')
    for p,expected in review['evidence_sha256'].items():
        if sha(ROOT/p)!=expected:
            raise ValueError(f'Raw calibration evidence changed: {p}')
    for p in (*review['evidence_sha256'],REVIEW):
        if git('hash-object',f'--path={p}',str(ROOT/p))!=git('rev-parse',f'HEAD:{p}'):
            raise ValueError('Commit raw calibration evidence before the next capture')
    result=json.loads((BASE/'reserved_visible_v1.json').read_text(encoding='utf-8'))
    if result.get('calibration_pass') is not True or set(result['per_type'])!={'player','asteroid','player_missile'}:
        raise ValueError('Original reserved raw result must pass all three categories')
    for stats in result['per_type'].values():
        if not (stats['passed'] and stats['joint_matches']>=50
                and stats['presence_count_agreement']/stats['frames']>=.99
                and stats['positive_count_agreement']/stats['positive_frames']>=.99
                and stats['boxes_within1']/stats['joint_matches']>=.99):
            raise ValueError('Original raw result fails its frozen criteria')
    return sha(BASE/'reserved_visible_v1.json')
