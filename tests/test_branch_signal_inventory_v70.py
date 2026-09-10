"""Small generated RGB only. No real archive, Torch, model or emulator."""
import json

import numpy as np
import pytest

from scripts import check_branch_signal_inventory_v70 as check

TINY = (2, 3, 3)


def fixture(lengths=(8, 8)):
    arrays = {'branch_start_pair': np.zeros((2, *TINY), np.uint8)}
    for side, length in zip(('intervention', 'control'), lengths, strict=True):
        arrays[f'branch_{side}_raw'] = np.zeros((length, *TINY), np.uint8)
    return repool(arrays)


def repool(arrays):
    for side in ('intervention', 'control'):
        raw = arrays[f'branch_{side}_raw']
        pooled = np.empty_like(raw)
        previous = arrays['branch_start_pair'][-1]
        for i, image in enumerate(raw):
            pooled[i] = np.maximum(previous, image)
            previous = image
        arrays[f'branch_{side}_pooled'] = pooled
        arrays[f'branch_{side}_endpoint'] = (pooled[-1] if len(raw) else
                                           np.maximum(*arrays['branch_start_pair'])).copy()
    return arrays


def population():
    native = dict(kind='owned_ordinary_native_wiring', passed=True, accounting_complete=True,
                  known_charged_frames=14908, constructor_count=18, cases=[])
    diagnostic = dict(kind='archived_pixel_cpu_numerical_diagnostics', passed=True,
                      ordinary_records=72, branch_pairs=6, optimizer_updates=0,
                      new_emulated_frames=0, cases=[])
    for i, name in enumerate(check.NAMES):
        branch = check.expected_branch(name) if i < 6 else None
        native['cases'].append(dict(name=name, passed=True, branch=branch,
            selected_trace_archive=f'craft_v6_out/v70_owned_ordinary_native/{name}_rgb_v0.npz'))
        diagnostic['cases'].append(dict(case=name, branch=None if branch is None else
            dict(operation_id=branch['operation_id'], common_prefix_frames=8, supervised_pixels=0,
                 positive_pixels=0, own_instances=(12, 12, 42, 14, 50, 20)[i])))
    return native, diagnostic


def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(check, 'FOLDER', tmp_path / 'result')
    monkeypatch.setattr(check, 'SHAPE', TINY)
    for field in ('OUTPUT', 'ATTEMPT', 'PREFLIGHT'):
        monkeypatch.setattr(check, field, check.FOLDER / (field + '.json'))
    monkeypatch.setattr(check, 'NATIVE', 'native.json')
    monkeypatch.setattr(check, 'DIAGNOSTIC', 'diagnostic.json')
    monkeypatch.setattr(check, 'PROTOCOL', 'protocol.json')
    native, diagnostic = population()
    evidence = {}
    for i, case in enumerate(native['cases']):
        path = tmp_path / case['selected_trace_archive']
        path.parent.mkdir(parents=True, exist_ok=True)
        # If an ordinary member is accidentally accessed, allow_pickle=False
        # refuses this object sentinel. The seven branch arrays are uint8 RGB.
        arrays = {key: np.array(['ordinary-must-not-be-read'], dtype=object) for key in check.ORDINARY_KEYS}
        if i < 6:
            arrays.update(fixture())
        np.savez(path, **arrays)
        case['selected_trace_sha256'] = check.sha(path)
        evidence[case['selected_trace_archive']] = case['selected_trace_sha256']
    (tmp_path / 'native.json').write_text(json.dumps(native))
    (tmp_path / 'diagnostic.json').write_text(json.dumps(diagnostic))
    (tmp_path / 'protocol.json').write_text(json.dumps({'limitations': ['Synthetic test only']}))
    proof = dict(source_sha256={'synthetic': 'a' * 64}, evidence_sha256=evidence,
                 runtime={'synthetic': True}, preflight_review_sha256='b' * 64)
    monkeypatch.setattr(check, 'preflight', lambda: proof)
    return native, proof


def test_transient_real_adjacent_pool_and_true_pixel_counts():
    arrays = fixture()
    arrays['branch_intervention_raw'][0, 0, 0] = [1, 2, 3]
    arrays['branch_intervention_raw'][2, 1, 2, 1] = 9
    repool(arrays)
    before = {key: value.copy() for key, value in arrays.items()}
    assert check.validate_arrays(arrays, shape=TINY) == [8, 8]
    raw = check.summarize(arrays['branch_intervention_raw'], arrays['branch_control_raw'], shape=TINY)
    pooled = check.summarize(arrays['branch_intervention_pooled'], arrays['branch_control_pooled'], shape=TINY)
    assert [r['changed_pixels'] for r in raw['horizons']] == [1, 0, 1, 0, 0, 0, 0, 0]
    assert [r['changed_pixels'] for r in pooled['horizons']] == [1, 1, 1, 1, 0, 0, 0, 0]
    assert raw['horizons'][0]['changed_fraction'] == 1 / 6  # three channels, one changed pixel
    assert [r['cumulative_union_pixels'] for r in raw['horizons']] == [1, 1, 2, 2, 2, 2, 2, 2]
    assert [r['pixels'] for r in raw['first_changed_histogram']] == [4, 1, 0, 1, 0, 0, 0, 0, 0]
    assert not raw['all_zero_over_observed_prefix']
    assert all(np.array_equal(value, before[key]) for key, value in arrays.items())


def test_pool_uses_anchor_last_frame_and_can_hide_raw_difference():
    arrays = fixture()
    arrays['branch_start_pair'][-1, 0, 0] = 255
    arrays['branch_intervention_raw'][0, 0, 0] = 255
    repool(arrays)
    assert check.validate_arrays(arrays, shape=TINY) == [8, 8]
    raw = check.summarize(arrays['branch_intervention_raw'], arrays['branch_control_raw'], shape=TINY)
    pooled = check.summarize(arrays['branch_intervention_pooled'], arrays['branch_control_pooled'], shape=TINY)
    assert raw['horizons'][0]['changed_pixels'] == 1
    assert pooled['horizons'][0]['changed_pixels'] == 0
    assert pooled['horizons'][1]['changed_pixels'] == 1


def test_prefix_missing_horizons_are_unknown_not_zero():
    arrays = fixture((3, 1))
    assert check.validate_arrays(arrays, shape=TINY) == [3, 1]
    result = check.summarize(arrays['branch_intervention_raw'], arrays['branch_control_raw'], shape=TINY)
    assert (result['common_prefix_frames'], result['unpaired_intervention_frames'],
            result['unpaired_control_frames']) == (1, 2, 0)
    assert result['horizons'][0]['changed_pixels'] == 0
    assert result['all_zero_over_observed_prefix'] is True
    assert result['complete_eight_frame_pair'] is False
    assert result['first_changed_histogram'][0]['pixels'] == 6
    for row in result['horizons'][1:]:
        assert not row['observed'] and row['changed_pixels'] is None
        assert row['observed_pixel_denominator'] is row['cumulative_union_pixels'] is None
    assert result['first_changed_histogram'][2]['pixels'] is None
    empty = fixture((0, 0))
    check.validate_arrays(empty, shape=TINY)
    result = check.summarize(empty['branch_intervention_raw'], empty['branch_control_raw'], shape=TINY)
    assert result['all_zero_over_observed_prefix'] is None
    assert all(row['pixels'] is None for row in result['first_changed_histogram'])


@pytest.mark.parametrize('fault', ['dtype', 'geometry', 'pooled', 'endpoint', 'extra'])
def test_rgb_contract_faults_rejected(fault):
    arrays = fixture()
    if fault == 'dtype':
        arrays['branch_start_pair'] = arrays['branch_start_pair'].astype(np.float32)
    elif fault == 'geometry':
        arrays['branch_control_raw'] = np.zeros((8, 1, 1, 3), np.uint8)
    elif fault == 'pooled':
        arrays['branch_control_pooled'][0, 0, 0, 0] = 1
    elif fault == 'endpoint':
        arrays['branch_control_endpoint'][0, 0, 0] = 1
    else:
        arrays['ram'] = np.zeros(128, np.uint8)
    with pytest.raises(ValueError):
        check.validate_arrays(arrays, shape=TINY)


def test_metadata_exact_scope_and_preserved_zero_C():
    native, diagnostic = population()
    assert len(check.validate_population(native, diagnostic)) == 9
    native['cases'][0]['branch']['anchor_frame'] = 575
    with pytest.raises(ValueError, match='metadata'):
        check.validate_population(native, diagnostic)
    native, diagnostic = population()
    diagnostic['cases'][0]['branch']['supervised_pixels'] = 1
    with pytest.raises(ValueError, match='counts'):
        check.validate_population(native, diagnostic)


def test_no_preflight_means_no_array_or_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(check, 'ATTEMPT', tmp_path / 'attempt.json')
    monkeypatch.setattr(check, 'OUTPUT', tmp_path / 'out.json')
    def reject():
        raise ValueError('independent preflight missing')
    monkeypatch.setattr(check, 'preflight', reject)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('NPZ opened before preflight'))
    with pytest.raises(ValueError, match='preflight missing'):
        check.run()
    assert not check.ATTEMPT.exists()


def test_complete_synthetic_population_never_reads_ordinary_arrays(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    result = check.run()
    assert result['passed'] and result['materialized_arrays'] == 42 and result['ordinary_arrays_read'] == 0
    assert [row['case'] for row in result['cases']] == list(check.NAMES)
    assert result['all_zero_pairs']['raw'] == list(check.NAMES[:6])
    assert all(row['raw']['all_zero_over_observed_prefix'] is None for row in result['cases'][6:])
    attempt = check.read_json(check.ATTEMPT)
    assert attempt['status'] == 'complete' and attempt['result_sha256'] == check.sha(check.OUTPUT)
    with pytest.raises(ValueError, match='no retry'):
        check.run()


def test_failure_persisted_without_reusing_previous_case(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    read = check.read_case
    def fail(case, proof):
        assert check.read_json(check.ATTEMPT)['status'] == 'started'
        if case['name'] == 'breakout':
            raise OSError('synthetic next archive failure')
        return read(case, proof)
    monkeypatch.setattr(check, 'read_case', fail)
    with pytest.raises(OSError, match='next archive'):
        check.run()
    attempt = check.read_json(check.ATTEMPT)
    assert attempt['status'] == 'failed' and attempt['current_case'] == 'breakout'
    assert [row['case'] for row in attempt['completed_cases']] == ['pong']
    assert not check.OUTPUT.exists()
    with pytest.raises(ValueError, match='no retry'):
        check.run()


def test_final_proof_change_retains_failure_not_result(monkeypatch, tmp_path):
    _, proof = setup(monkeypatch, tmp_path)
    calls = []
    def changing():
        calls.append(1)
        return proof if len(calls) == 1 else dict(proof, preflight_review_sha256='c' * 64)
    monkeypatch.setattr(check, 'preflight', changing)
    with pytest.raises(ValueError, match='changed during'):
        check.run()
    attempt = check.read_json(check.ATTEMPT)
    assert attempt['status'] == 'failed' and len(attempt['completed_cases']) == 9
    assert not check.OUTPUT.exists()


def test_archive_mutation_before_load_refused(monkeypatch, tmp_path):
    native, proof = setup(monkeypatch, tmp_path)
    path = tmp_path / native['cases'][0]['selected_trace_archive']
    with path.open('ab') as stream:
        stream.write(b'altered')
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('Changed archive opened'))
    with pytest.raises(ValueError, match='before reading'):
        check.read_case(native['cases'][0], proof)


def test_deadline_failure_is_final(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    values = iter([0., 301., 302.])
    monkeypatch.setattr(check.time, 'perf_counter', lambda: next(values))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('Archive after deadline'))
    with pytest.raises(TimeoutError):
        check.run()
    assert check.read_json(check.ATTEMPT)['status'] == 'failed'
    assert not check.OUTPUT.exists()


def test_readonly_proofs_separate_review_maps_and_scalar_result_hash(monkeypatch, tmp_path):
    native, _ = setup(monkeypatch, tmp_path)
    # Everything below is generated in tmp_path; no real proof/NPZ is touched.
    sources = {}
    for path in ('scripts/__init__.py', 'candidate.py'):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# synthetic\n')
        sources[path] = check.sha(target)
    evidence = {case['selected_trace_archive']: case['selected_trace_sha256'] for case in native['cases']}
    pins = {path: check.sha(tmp_path / path) for path in ('native.json', 'diagnostic.json')}
    evidence.update(pins)
    reviews = {}
    for path, result in (('owned_ordinary_native/review.json', 'native.json'), ('cpu_review.json', 'diagnostic.json')):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(dict(passed=True, result_sha256=pins[result],
                                         source_sha256=sources, evidence_sha256=evidence)))
        reviews[path] = check.sha(target)
    monkeypatch.setattr(check, 'PINNED_RESULTS', pins)
    monkeypatch.setattr(check, 'PINNED_REVIEWS', reviews)
    monkeypatch.setattr(check, 'NEW_SOURCES', tuple(sources))
    (tmp_path / 'protocol.json').write_text(json.dumps(dict(kind='prospective_branch_signal_inventory_v0',
        spec=check.SPEC, runtime=check.runtime(), agent_training_allowed=False, gate_result=None)))
    committed = []
    monkeypatch.setattr(check, 'require_committed', committed.append)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('Readonly proofs loaded arrays'))
    proof = check.proofs()
    assert proof['source_sha256'] == sources
    assert set(committed) == set(sources) | set(evidence) | set(reviews)
    # A string result hash is checked against the correct result, not iterated
    # as a mapping or accepted merely because the evidence map is intact.
    path = 'cpu_review.json'
    value = check.read_json(tmp_path / path)
    value['result_sha256'] = 'f' * 64
    (tmp_path / path).write_text(json.dumps(value))
    reviews[path] = check.sha(tmp_path / path)
    with pytest.raises(ValueError, match='attest'):
        check.proofs()


def test_exact_independent_preflight_fields_required(monkeypatch, tmp_path):
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(check, 'PREFLIGHT', tmp_path / 'preflight.json')
    proof = dict(source_sha256={'a': 'a' * 64}, evidence_sha256={'b': 'b' * 64}, runtime={'cpu': 1})
    monkeypatch.setattr(check, 'proofs', lambda: proof)
    committed = []
    monkeypatch.setattr(check, 'require_committed', committed.append)
    review = dict(kind='branch_signal_inventory_independent_preflight', passed=True, **proof)
    check.PREFLIGHT.write_text(json.dumps(review))
    assert check.preflight()['preflight_review_sha256'] == check.sha(check.PREFLIGHT)
    assert committed == ['preflight.json']
    review['runtime'] = {'cpu': 2}
    check.PREFLIGHT.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='preflight required'):
        check.preflight()
