"""One-shot, standard-library-only supervisor: fixtures, preflight, fit, then judge."""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

from scripts.photometric_common_v70 import (
    ROOT, OUT, PROTOCOL, SOURCES, artifact_inventory, committed_sources, elapsed,
    environment, read_json, read_json_with_sha, runtime, sha, source_hashes, utc,
    ensure_space, publish_completion, verify_sources, write_bytes, write_json,
)


def source_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def fixtures():
    sources = committed_sources()
    intent = dict(started_utc=utc(), started_perf_counter_ns=time.perf_counter_ns(),
                  source_head=source_head(), source_sha256=sources, wall_limit_seconds=60,
                  scope='candidate numerical and generator fixtures only; no production seeds or optimizer',
                  maximum_backward_calls=1, optimizer_updates=0)
    write_json(OUT / 'check_v0_attempt.json', intent)
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               'tests/test_photometric_correspondence_v70.py']
    code, status, output, failure = None, 'failed', '', None
    try:
        result = subprocess.run(command, cwd=ROOT, env=environment(), capture_output=True,
                                text=True, timeout=max(0.001, 60-elapsed(intent)))
        code, output = result.returncode, result.stdout + result.stderr
        verify_sources(sources)
        status = 'passed' if code == 0 and elapsed(intent) < 60 else 'failed'
    except subprocess.TimeoutExpired as error:
        output = ((error.stdout or b'')+(error.stderr or b'')).decode('utf-8', errors='replace')
        status = 'budget_exhausted'
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
    try:
        write_bytes(OUT / 'check_v0_output.txt', output.encode('utf-8'), emergency=True)
        output_sha = sha(OUT / 'check_v0_output.txt')
    except BaseException as error:
        output_sha, status = None, 'failed'
        failure = (failure+'; ' if failure else '')+f'output: {type(error).__name__}: {error}'
    seconds = elapsed(intent)
    if seconds >= 60:
        status = 'budget_exhausted'
    write_json(OUT / 'check_v0.json', dict(intent, status=status, returncode=code, failure=failure,
               cumulative_test_wall_seconds=seconds, remaining_test_wall_seconds=max(0, 60-seconds),
               output_sha256=output_sha), emergency=True)
    if status == 'passed':
        publish_completion('check_v0', intent, 'check_v0.json')
    print(output, end='')
    print(json.dumps(dict(status=status, seconds=seconds)))
    if status != 'passed':
        raise RuntimeError('Fixture attempt failed; no production allowed')


def preflight():
    sources = committed_sources()
    check = read_json(OUT / 'check_v0.json')
    if (check['status'] != 'passed' or check['source_sha256'] != sources
            or check['output_sha256'] != sha(OUT / 'check_v0_output.txt')
            or not 0 <= check['cumulative_test_wall_seconds'] < 60
            or read_json(OUT / 'check_v0_completion.json')['receipt_sha256'] != sha(OUT / 'check_v0.json')):
        raise ValueError('Matching successful prospective fixture receipt required')
    if (OUT / 'fit_v0_attempt.json').exists():
        raise ValueError('The production attempt already exists')
    spec = read_json(ROOT / PROTOCOL)
    if (spec['train_clips'], spec['eval_clips'], spec['updates'], spec['batch_size'],
            spec['eval_batch_size'], spec['cpu_threads']) != (128, 1280, 512, 4, 16, 1):
        raise ValueError('Declared fixed loop geometry changed')
    write_json(OUT / 'preflight_v0.json', dict(
        passed=True, completed_utc=utc(), source_head=source_head(), source_sha256=sources,
        source_count=len(sources), fixture_sha256=sha(OUT / 'check_v0.json'),
        runtime=runtime(), protocol_sha256=sha(ROOT / PROTOCOL),
        scope='source bytes and fixture receipts only; no arrays, model or generation',
        new_rgb_frames=0, model_calls=0, optimizer_updates=0))
    print(json.dumps(dict(preflight='passed', sources=len(sources))))


def child(module, arguments, log_name, intent):
    errors = []
    with (OUT / log_name).open('xb') as log:
        process = subprocess.Popen([sys.executable, '-m', module, *arguments], cwd=ROOT,
                                   env=environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        def drain():
            written = 0
            try:
                while chunk := process.stdout.read1(4096):
                    allowed = min(len(chunk), 16384-written)
                    if allowed:
                        ensure_space(allowed, emergency=True)
                        log.write(chunk[:allowed])
                        log.flush()
                        written += allowed
                    if allowed != len(chunk):
                        raise OSError('Child stdout/stderr limit reached; retained prefix is 16384 bytes')
            except BaseException as error:
                errors.append(f'{type(error).__name__}: {error}')
                process.kill()

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            process.wait(timeout=max(0.001, intent['wall_limit_seconds']-elapsed(intent)))
        except BaseException:
            process.kill()
            process.wait(timeout=2)
            raise
        finally:
            reader.join(timeout=2)
            log.flush()
            os.fsync(log.fileno())
    if reader.is_alive() or errors or process.returncode:
        raise RuntimeError(f'{module} {arguments} exited {process.returncode}; log errors={errors}; see {log_name}')


def run():
    spec = read_json(ROOT / PROTOCOL)
    proof, preflight_sha = read_json_with_sha(OUT / 'preflight_v0.json')
    if (proof.get('passed') is not True or proof['source_sha256'] != committed_sources()
            or proof['runtime'] != runtime() or proof['fixture_sha256'] != sha(OUT / 'check_v0.json')):
        raise ValueError('Frozen preflight no longer matches sources, runtime or fixtures')
    intent = dict(started_utc=utc(), started_perf_counter_ns=time.perf_counter_ns(),
                  wall_limit_seconds=spec['max_work_wall_seconds'],
                  cleanup_reserve_seconds=spec['cleanup_reserve_seconds'],
                  max_artifact_bytes=spec['max_artifact_bytes'], runtime=runtime(),
                  source_head=source_head(), source_sha256=proof['source_sha256'],
                  preflight_sha256=preflight_sha, retry=False)
    # Exclusive durable marker exists before either generation or Torch imports.
    write_json(OUT / 'fit_v0_attempt.json', intent)
    attempt_sha = sha(OUT / 'fit_v0_attempt.json')
    status, failure, files, size, generation_sha = 'failed', None, {}, None, None
    try:
        child('scripts.photometric_worker_v70', ['generate'], 'generation_v0_output.txt', intent)
        generation, generation_sha = read_json_with_sha(OUT / 'generation_v0.json')
        child('scripts.photometric_worker_v70', ['fit'], 'fit_v0_output.txt', intent)
        if read_json(OUT / 'fit_worker_v0.json')['execution_status'] != 'complete':
            raise ValueError('The learner did not complete the declared work')
        verify_sources(intent['source_sha256'])
        if (sha(OUT / 'preflight_v0.json') != preflight_sha
                or sha(OUT / 'fit_v0_attempt.json') != attempt_sha
                or sha(OUT / 'check_v0.json') != proof['fixture_sha256']):
            raise ValueError('Preflight, fixture or attempt proof changed during work')
        fixture = read_json(OUT / 'check_v0.json')
        if sha(OUT / 'check_v0_output.txt') != fixture['output_sha256']:
            raise ValueError('Fixture output proof changed')
        if sha(OUT / 'generation_v0.json') != generation_sha:
            raise ValueError('The generation manifest changed after generation completed')
        for name, expected in generation['file_sha256'].items():
            if sha(OUT / name) != expected:
                raise ValueError('Generated observation or judge bytes changed during fitting')
        status = 'complete'
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
    try:
        # Exactly one final inventory; its time and size affect the final status.
        files, size = artifact_inventory()
        if size > spec['max_artifact_bytes']-64000 or elapsed(intent) >= spec['max_work_wall_seconds']-1:
            raise TimeoutError('Production artifact or wall budget has no room for its receipt')
    except BaseException as error:
        status = 'failed'
        failure = (failure+'; ' if failure else '')+f'closure: {type(error).__name__}: {error}'
    receipt = dict(intent, execution_status=status, failure=failure, completed_utc=utc(),
                   elapsed_seconds=elapsed(intent), artifacts=files, artifact_bytes=size,
                   generation_manifest_sha256=generation_sha,
                   proposals_sealed_before_judge=True if status == 'complete' else False,
                   m_c_supervision_authorized=False, me_identified=False)
    write_json(OUT / 'fit_v0.json', receipt, emergency=True)
    if status == 'complete':
        publish_completion('fit_v0', intent, 'fit_v0.json')
    # A slow final write cannot silently retain a valid complete status.
    if elapsed(intent) >= spec['max_work_wall_seconds']:
        write_json(OUT / 'fit_v0_closure_failure.json', dict(reason='final receipt exceeded wall budget',
                                                           elapsed_seconds=elapsed(intent)), emergency=True)
        raise TimeoutError('Final receipt exceeded production deadline; fit is invalid')
    print(json.dumps(dict(status=status, failure=failure, elapsed_seconds=receipt['elapsed_seconds'])))
    if status != 'complete':
        raise RuntimeError('Production closed with failure; no retry is authorized')


def judge():
    proof, review_sha = read_json_with_sha(OUT / 'fit_review_v0.json')
    fit_result, fit_sha = read_json_with_sha(OUT / 'fit_v0.json')
    completion, completion_sha = read_json_with_sha(OUT / 'fit_v0_completion.json')
    if (proof.get('passed') is not True or proof.get('fit_result_sha256') != fit_sha
            or completion['receipt_sha256'] != fit_sha
            or fit_result['execution_status'] != 'complete'
            or (OUT / 'fit_v0_closure_failure.json').exists() or runtime() != fit_result['runtime']):
        raise ValueError('Independent matching technical review required before judging')
    verify_sources(fit_result['source_sha256'])
    for name, expected in fit_result['artifacts'].items():
        if sha(OUT / name) != expected['sha256']:
            raise ValueError(f'Sealed artifact changed: {name}')
    intent = dict(started_utc=utc(), started_perf_counter_ns=time.perf_counter_ns(),
                  wall_limit_seconds=60, cleanup_reserve_seconds=5,
                  source_sha256=fit_result['source_sha256'],
                  fit_result_sha256=fit_sha,
                  fit_review_sha256=review_sha,
                  fit_completion_sha256=completion_sha,
                  runtime=runtime(),
                  scope='one judge extraction then one primary score; no model or generation', retry=False)
    write_json(OUT / 'judge_v0_attempt.json', intent)
    status, failure, hashes = 'failed', None, {}
    try:
        child('scripts.judge_photometric_v70', [], 'judge_v0_output.txt', intent)
        verify_sources(intent['source_sha256'])
        if (sha(OUT / 'fit_v0.json') != intent['fit_result_sha256']
                or sha(OUT / 'fit_review_v0.json') != intent['fit_review_sha256']
                or sha(OUT / 'fit_v0_completion.json') != completion_sha
                or runtime() != intent['runtime'] or (OUT / 'fit_v0_closure_failure.json').exists()):
            raise ValueError('Fit or independent review seal changed during judging')
        for name, expected in fit_result['artifacts'].items():
            if sha(OUT / name) != expected['sha256']:
                raise ValueError(f'Sealed input changed during judging: {name}')
        scored = read_json(OUT / 'score_v0.json')
        if (scored['score_calls'] != 1 or scored['fit_result_sha256'] != intent['fit_result_sha256']
                or scored['score_inputs_sha256'] != sha(OUT / 'score_inputs_v0.json')
                or scored['judge_details_sha256'] != sha(OUT / 'judge_details_v0.json')):
            raise ValueError('Score output seals or invocation count differ')
        names = ('score_inputs_v0.json', 'judge_details_v0.json', 'score_v0.json', 'judge_v0_output.txt')
        hashes = {n: sha(OUT / n) for n in names}
        if elapsed(intent) >= 59:
            raise TimeoutError('Judge budget exceeded')
        status = 'complete'
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
    write_json(OUT / 'judge_v0.json', dict(intent, execution_status=status, failure=failure,
        completed_utc=utc(), elapsed_seconds=elapsed(intent),
        file_sha256=hashes), emergency=True)
    if status == 'complete':
        publish_completion('judge_v0', intent, 'judge_v0.json')
    if elapsed(intent) >= 60:
        write_json(OUT / 'judge_v0_closure_failure.json', dict(reason='final receipt exceeded judge budget',
                                                             elapsed_seconds=elapsed(intent)), emergency=True)
        raise TimeoutError('Final receipt exceeded judge deadline; score is invalid')
    print(json.dumps(dict(status=status, failure=failure, elapsed_seconds=elapsed(intent))))
    if status != 'complete':
        raise RuntimeError('Judge attempt failed; preserve outputs and do not retry')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('fixtures', 'preflight', 'run', 'judge'))
    operation = parser.parse_args().operation
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[operation]()


if __name__ == '__main__':
    main()
