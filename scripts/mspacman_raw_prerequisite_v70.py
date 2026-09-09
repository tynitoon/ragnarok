"""Read-only proof required before any subsequent Ms. Pac-Man capture."""
import json
from scripts.validate_mspacman_raw_v70 import ROOT,BASE,SOURCE_PATHS as RAW_SEALED_SOURCES,verify_seal,sha,git,pair_verdict

REVIEW='craft_v6_out/v70_mspacman_calibration/reserved_pair_review_v3.json'
REVIEW_SHA256='a53c16554af0bc91a57ea3876aabb2a26e2323e78793b96b10e29a91fe99ef89'


def require_raw_calibration():
    seal=verify_seal()
    if sha(ROOT/REVIEW)!=REVIEW_SHA256:
        raise ValueError('The independently reviewed raw calibration receipt changed')
    review=json.loads((ROOT/REVIEW).read_text(encoding='utf-8'))
    if (review.get('passed') is not True or review.get('pair_acceptance_confirmed') is not True
            or review.get('source_sha256')!=seal['source_sha256']):
        raise ValueError('A passed independently reviewed raw calibration is required')
    for p,expected in review['evidence_sha256'].items():
        if sha(ROOT/p)!=expected:
            raise ValueError(f'Raw validation evidence changed: {p}')
    paths=(*review['evidence_sha256'],REVIEW)
    for p in paths:
        if git('hash-object',f'--path={p}',str(ROOT/p))!=git('rev-parse',f'HEAD:{p}'):
            raise ValueError('Commit all raw calibration evidence before the next capture')
    pair=json.loads((BASE/'raw_reserved_pair_v3.json').read_text(encoding='utf-8'))
    uniform=json.loads((BASE/'uniform_reserved_visible_v3.json').read_text(encoding='utf-8'))
    coverage=json.loads((BASE/'coverage_reserved_visible_v3.json').read_text(encoding='utf-8'))
    if (pair.get('pair_pass') is not True or not pair_verdict(uniform,coverage)['pair_pass']
            or pair.get('validation_seal_sha256')!=seal['_file_sha256']):
        raise ValueError('The original separate-sequence raw acceptance is required')
    return sha(BASE/'raw_reserved_pair_v3.json')
