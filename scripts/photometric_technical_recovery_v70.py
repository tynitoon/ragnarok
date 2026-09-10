"""Prepared exception to the stopped trial; run only with explicit owner approval.

No edits of original frozen sources, no new generator invocation. The wrapper
uses the original RGB-only fitting function with a verified -1 CUDA mask.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import photometric_common_v70 as common

ROOT = common.ROOT
OLD = common.OUT
DIAGNOSTIC = ROOT / 'craft_v6_out/v70_cpu_cuda_mask_diagnostic'
OUT = ROOT / 'craft_v6_out/v70_photometric_technical_recovery'
EXTRA_SOURCES = (
    'scripts/photometric_technical_recovery_v70.py',
    'docs/ARC4_PHOTOMETRIC_TECHNICAL_RECOVERY_PROPOSAL.md',
    'docs/ARC4_PHOTOMETRIC_CORRESPONDENCE_ABORT_REVIEW.md',
    'scripts/probe_cpu_cuda_mask_v70.py',
)
OLD_FIT_SHA = '0dd96f31d2730fdd514d83cb204f6af2d40c79b411ad8a2fc69a3d646729f9f1'
GENERATION_SHA = 'fe255e8871ff0de2aebc0ebb66ce6c50a2745ec6b3ab8d363991c8259d81e6bf'
DIAGNOSTIC_SHA = 'c4a18ac39ec68a098f09cf9eb629e94cb7f0bcc679be11cc0ad707d1d8f56645'


def combined_bytes():
    return sum(p.stat().st_size for folder in (OLD, DIAGNOSTIC, OUT)
               if folder.exists() for p in folder.iterdir() if p.is_file())


def ensure_space(additional_bytes, *, emergency=False):
    if combined_bytes()+additional_bytes+(0 if emergency else 65536) > 64000000:
        raise OSError('Combined original/diagnostic/recovery artifact budget exceeded')


def cpu_environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1')


def extra_hashes():
    return {name: common.sha(ROOT / name) for name in EXTRA_SOURCES}


def proofs():
    if (common.sha(OLD / 'fit_v0.json') != OLD_FIT_SHA
            or common.sha(OLD / 'generation_v0.json') != GENERATION_SHA
            or common.sha(DIAGNOSTIC / 'probe_v0.json') != DIAGNOSTIC_SHA):
        raise ValueError('Original stop, generation or diagnostic proof changed')
    stopped = common.read_json(OLD / 'fit_v0.json')
    if stopped['execution_status'] != 'failed' or (OLD / 'fit_v0_completion.json').exists():
        raise ValueError('Expected the preserved failed original attempt')
    for name in ('initial_state.pt', 'final_state.pt', 'fit_worker_v0.json', 'updates_v0.jsonl', 'score_v0.json'):
        if (OLD / name).exists():
            raise ValueError('Unexpected original model/learning/scoring artifact')
    for name, proof in stopped['artifacts'].items():
        if common.sha(OLD / name) != proof['sha256']:
            raise ValueError('Original artifact changed')
    common.verify_sources(stopped['source_sha256'])
    if common.runtime() != stopped['runtime']:
        raise ValueError('Recovery runtime differs')
    diagnostic = common.read_json(DIAGNOSTIC / 'probe_v0.json')
    if diagnostic['passed'] is not True or diagnostic['elapsed_seconds'] >= 7:
        raise ValueError('Verified diagnostic exceeds its conservative debit or failed')
    subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *EXTRA_SOURCES],
                   cwd=ROOT, check=True, capture_output=True)
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *EXTRA_SOURCES],
                   cwd=ROOT, check=True, capture_output=True)
    return dict(source_sha256=stopped['source_sha256'], recovery_source_sha256=extra_hashes(),
                original_fit_sha256=OLD_FIT_SHA, generation_sha256=GENERATION_SHA,
                diagnostic_sha256=DIAGNOSTIC_SHA, runtime=common.runtime())


def preflight():
    proof = proofs()
    common.write_json(OUT / 'recovery_preflight_v0.json', dict(
        proof, completed_utc=common.utc(), passed=True, owner_approval_still_required=True,
        new_rgb_frames=0, decoded_arrays=0, model_calls=0, optimizer_updates=0))
    print('Readiness verified; explicit owner approval is still required before run.')


def verified_readiness():
    readiness = common.read_json(OUT / 'recovery_preflight_v0.json')
    current = proofs()
    if any(readiness[key] != value for key, value in current.items()) or readiness['passed'] is not True:
        raise ValueError('Recovery readiness changed')
    return current


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise ValueError('Explicit verified CUDA mask required before importing Torch')
    intent = common.read_json(OUT / 'fit_v0_attempt.json')
    if intent.get('owner_approved_technical_recovery') is not True:
        raise ValueError('No owner-approved recovery intention')
    if extra_hashes() != intent['recovery_source_sha256']:
        raise ValueError('Recovery source bytes changed')
    common.verify_sources(intent['source_sha256'])
    common.check_time(intent)
    from scripts.photometric_worker_v70 import fit
    fit(intent)  # Original source unchanged; no generate() call.


def run(*, approved):
    if not approved:
        raise ValueError('STOP: explicit owner approval is required for this technical exception')
    proof = verified_readiness()
    intent = dict(proof, started_utc=common.utc(), started_perf_counter_ns=time.perf_counter_ns(),
                  wall_limit_seconds=280, cleanup_reserve_seconds=15, max_artifact_bytes=64000000,
                  owner_approved_technical_recovery=True, conservative_prior_debits_seconds=[8, 7],
                  total_active_time_cap_seconds=295, regenerated_scenes=0, retry=False)
    common.write_json(OUT / 'fit_v0_attempt.json', intent)
    from scripts import photometric_correspondence_v70 as supervisor
    supervisor.OUT, supervisor.environment = OUT, cpu_environment
    status, failure, files = 'failed', None, {}
    try:
        generation = common.read_json(OLD / 'generation_v0.json')
        for name, expected in generation['file_sha256'].items():
            payload = (OLD / name).read_bytes()
            common.write_bytes(OUT / name, payload)
            if common.sha(OUT / name) != expected:
                raise ValueError('Copied archive differs; no regeneration allowed')
        common.write_bytes(OUT / 'generation_v0.json', (OLD / 'generation_v0.json').read_bytes())
        if common.sha(OUT / 'generation_v0.json') != GENERATION_SHA:
            raise ValueError('Copied generation manifest changed before fitting')
        supervisor.child('scripts.photometric_technical_recovery_v70', ['worker'], 'fit_v0_output.txt', intent)
        result = common.read_json(OUT / 'fit_worker_v0.json')
        if result['execution_status'] != 'complete':
            raise ValueError('Original fitting function did not finish')
        if proofs() != proof:
            raise ValueError('Recovery proof changed during fitting')
        if common.sha(OUT / 'generation_v0.json') != GENERATION_SHA:
            raise ValueError('Copied generation manifest changed during fitting')
        for name, expected in generation['file_sha256'].items():
            if common.sha(OUT / name) != expected:
                raise ValueError('Copied data changed during fitting')
        files, size = common.artifact_inventory()
        common.check_time(intent)
        ensure_space(65536, emergency=True)
        status = 'complete'
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        try:
            files, _size = common.artifact_inventory()
        except BaseException as closure_error:
            failure += f'; inventory failure: {closure_error}'
    receipt = dict(intent, execution_status=status, failure=failure, completed_utc=common.utc(),
                   elapsed_seconds=common.elapsed(intent), artifacts=files,
                   combined_artifact_bytes_before_receipt=combined_bytes(),
                   proposals_sealed_before_judge=status == 'complete',
                   m_c_supervision_authorized=False, me_identified=False)
    common.write_json(OUT / 'fit_v0.json', receipt, emergency=True)
    if status == 'complete':
        common.publish_completion('fit_v0', intent, 'fit_v0.json')
    print(dict(status=status, failure=failure, elapsed_seconds=receipt['elapsed_seconds']))
    if status != 'complete':
        raise RuntimeError('Recovery stopped; no retry')


def judge_worker():
    verified_readiness()
    from scripts.judge_photometric_v70 import main as score_once
    score_once()  # The original judge, with the recovery output root selected before import.


def judge():
    verified_readiness()
    from scripts import photometric_correspondence_v70 as supervisor
    supervisor.OUT, supervisor.environment = OUT, cpu_environment
    original_child = supervisor.child

    def routed_child(module, arguments, log_name, intent):
        if module != 'scripts.judge_photometric_v70' or arguments:
            raise ValueError('Only the declared judge may be routed here')
        original_child('scripts.photometric_technical_recovery_v70', ['judge_worker'], log_name, intent)
        verified_readiness()  # Before the original supervisor can publish completion.

    supervisor.child = routed_child
    supervisor.judge()  # Requires independent fit review and a positive fit completion.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('preflight', 'run', 'worker', 'judge', 'judge_worker'))
    parser.add_argument('--owner-approved-technical-recovery', action='store_true')
    args = parser.parse_args()
    # Only output routing and storage admission are replaced; frozen file bytes
    # and scientific source verification remain unchanged.
    OUT.mkdir(exist_ok=True)
    common.OUT, common.ensure_space = OUT, ensure_space
    if args.operation == 'run':
        run(approved=args.owner_approved_technical_recovery)
    else:
        globals()[args.operation]()


if __name__ == '__main__':
    main()
