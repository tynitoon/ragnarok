"""Synthetic temporary arrays only; never open the real survey archives."""
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from scripts import build_response_envelopes_v70 as build


def arrays(shape=(3, 4, 3)):
    return {key: np.zeros(dims, dtype=np.uint8) for key, dims in build.shapes(shape).items()}


def summary(value):
    diff = np.any(value['branch_intervention_raw'] != value['branch_control_raw'], axis=-1)
    union = np.zeros(diff.shape[1:], bool)
    rows = []
    for h, changed in enumerate(diff):
        union |= changed
        rows.append(dict(horizon=h + 1, observed=True, changed_pixels=int(changed.sum()),
                         cumulative_union_pixels=int(union.sum())))
    return dict(complete_eight_frame_pair=True, common_prefix_frames=8, horizons=rows)


def encoded(value):
    stream = io.BytesIO()
    np.savez(stream, **value)
    data = stream.getvalue()
    return data, dict(bytes=len(data), sha256=build.sha_bytes(data),
                      arrays={key: build.signature(item) for key, item in value.items()})


def setup_run(monkeypatch, tmp_path, mutate=None):
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    monkeypatch.setattr(build, 'FOLDER', tmp_path / 'derived')
    build.FOLDER.mkdir()
    monkeypatch.setattr(build, 'PREFLIGHT', build.FOLDER / 'preflight_v0.json')
    monkeypatch.setattr(build, 'SURVEY', tmp_path / 'survey.json')
    spec = dict(build.SPEC, shape=[3, 4, 3])
    monkeypatch.setattr(build, 'SPEC', spec)
    proof = dict(source_sha256={}, evidence_sha256={}, runtime={'synthetic': True})
    build.write_json(build.PREFLIGHT, dict(kind='response_envelope_independent_preflight', passed=True, **proof))
    monkeypatch.setattr(build, 'proofs', lambda: proof)
    monkeypatch.setattr(build, 'committed', lambda path: None)
    monkeypatch.setattr(build, 'git', lambda *args: 'sha1')
    cases, tree = [], []
    for game in build.GAMES:
        anchors = []
        for index in range(2):
            anchor = dict(target_delivered_frames=spec['anchors'][index], delivered_frames=spec['anchors'][index] + 1,
                          instrument_valid=True, own_anchor=dict(token=[game, index, 0],
                          pair=build.signature(arrays()['branch_start_pair'])), pairs=[])
            for command in range(18):
                value = arrays()
                # Ten nonempty anchors and two deliberately empty Seaquest anchors.
                if game != 'seaquest' and command:
                    value['branch_intervention_raw'][command % 8, index, command % 4, 1] = 23
                row = dict(command=command, status='acquired', own_anchor_token=anchor['own_anchor']['token'],
                           raw=summary(value))
                if mutate:
                    mutate(game, index, command, value, row)
                data, metadata = encoded(value)
                rel = f'craft_v6_out/v70_action_signal_survey/rgb/{game}_anchor{index}_action{command:02d}.npz'
                path = tmp_path / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                row['archive'] = dict(metadata, path=rel)
                anchor['pairs'].append(row)
                blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
                tree.append(f'100644 blob {blob}\t{rel}\0'.encode())
            anchors.append(anchor)
        cases.append(dict(game=game, anchors=anchors))
    build.write_json(build.SURVEY, dict(cases=cases, known_charged_frames=12345))
    monkeypatch.setattr(build.subprocess, 'check_output', lambda *args, **kwargs: b''.join(tree))


def test_union_uses_every_future_coordinate_once_without_projecting_onto_past():
    value = arrays()
    value['branch_intervention_raw'][0, 0, 0, 0] = 9
    value['branch_intervention_raw'][7, 0, 0, 0] = 9
    value['branch_intervention_raw'][6, 2, 3, 2] = 8
    mask = build.observed_union(value, summary(value))
    assert mask.dtype == bool and mask.sum() == 2
    assert mask[0, 0] and mask[2, 3]
    assert not value['branch_start_pair'].any()


@pytest.mark.parametrize('reason', ['truncated', 'unobserved', 'wrong_count', 'missing_horizon'])
def test_unknown_and_inconsistent_future_cannot_become_negative_label(reason):
    value = arrays()
    data = summary(value)
    if reason == 'truncated':
        data['complete_eight_frame_pair'] = False
    elif reason == 'unobserved':
        data['horizons'][3]['observed'] = False
    elif reason == 'wrong_count':
        data['horizons'][7]['cumulative_union_pixels'] = 1
    else:
        data['horizons'].pop()
    with pytest.raises(ValueError):
        build.observed_union(value, data)


def test_decoder_checks_bytes_and_signatures_not_just_shape():
    data, metadata = encoded(arrays())
    assert len(build.decode(data, metadata, (3, 4, 3))) == 7
    bad = dict(metadata, sha256='0' * 64)
    with pytest.raises(ValueError, match='bytes changed'):
        build.decode(data, bad, (3, 4, 3))
    metadata['arrays']['branch_start_pair']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='signature'):
        build.decode(data, metadata, (3, 4, 3))


def test_duplicate_zip_array_is_rejected():
    value = arrays()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for key, item in value.items():
            npy = io.BytesIO()
            np.save(npy, item)
            archive.writestr(key + '.npy', npy.getvalue())
        with pytest.warns(UserWarning):
            archive.writestr('branch_start_pair.npy', b'not a second valid member')
    data = stream.getvalue()
    metadata = dict(bytes=len(data), sha256=build.sha_bytes(data))
    with pytest.raises(ValueError, match='seven'):
        build.decode(data, metadata, (3, 4, 3))


def test_full_derivation_keeps_empty_anchors_and_only_two_dataset_fields(monkeypatch, tmp_path):
    setup_run(monkeypatch, tmp_path)
    assert build.run() == 0
    result = json.loads((build.FOLDER / 'targets_v0.json').read_text())
    assert result['passed'] is True and result['archives_read'] == 216
    assert len(result['cases']) == 12 and len(result['archive_sha256']) == 216
    assert result['inherited_collection_charged_frames'] == 12345
    assert result['new_emulated_frames'] == result['optimizer_updates'] == 0
    # Only this test's freshly written temporary dataset is decoded here.
    with np.load(build.FOLDER / 'targets_v0.npz', allow_pickle=False) as dataset:
        assert set(dataset.files) == {'past_pairs', 'observed_response_envelopes'}
        assert dataset['past_pairs'].shape == (12, 2, 3, 4, 3)
        targets = dataset['observed_response_envelopes']
        assert targets.dtype == bool and targets.shape == (12, 3, 4)
        assert np.count_nonzero(targets.sum(axis=(1, 2))) == 10
        assert not targets[6:8].any()
    before = (build.FOLDER / 'targets_v0.json').read_bytes()
    with pytest.raises(FileExistsError):
        build.run()
    assert (build.FOLDER / 'targets_v0.json').read_bytes() == before


def test_changed_control_stops_without_silently_dropping_pair(monkeypatch, tmp_path):
    def mutate(game, index, command, value, row):
        if game == 'pong' and index == 0 and command == 1:
            value['branch_control_raw'][0, 1, 1, 1] = 1
            row['raw'] = summary(value)
    setup_run(monkeypatch, tmp_path, mutate)
    assert build.run() == 1
    result = json.loads((build.FOLDER / 'targets_v0.json').read_text())
    assert result['archives_read'] == 2 and result['cases'] == [] and result['output'] is None
    assert 'NOOP' in result['failure']['message']
    assert (build.FOLDER / 'targets_v0_attempt.json').exists()
    assert len(list((tmp_path / 'craft_v6_out/v70_action_signal_survey/rgb').glob('*.npz'))) == 216


def test_time_stop_records_failure_before_first_array_read(monkeypatch, tmp_path):
    setup_run(monkeypatch, tmp_path)
    build.SPEC['max_work_seconds'] = -1
    assert build.run() == 1
    result = json.loads((build.FOLDER / 'targets_v0.json').read_text())
    assert result['archives_read'] == 0 and result['failure']['type'] == 'TimeoutError'
    assert result['passed'] is False and not (build.FOLDER / 'targets_v0.npz').exists()


def test_changed_past_cannot_be_accepted_as_the_same_anchor():
    value = arrays()
    anchor = dict(own_anchor=dict(pair=build.signature(value['branch_start_pair'])))
    value['branch_start_pair'][0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match='Past images'):
        build.validate_controls(value, None, anchor, 0)


def test_git_capability_uses_empty_stdin_not_an_unsupported_rev_parse_flag(monkeypatch):
    calls = []
    def output(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] == '--version':
            return b'git version 2.24.1.windows.2\n'
        assert command[-2:] == ['hash-object', '--stdin'] and kwargs == {'input': b''}
        return b'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391\n'
    monkeypatch.setattr(build.subprocess, 'check_output', output)
    assert build.git_blob_runtime()['version'] == 'git version 2.24.1.windows.2'
    assert len(calls) == 2


def test_wrong_git_object_algorithm_is_rejected_before_archive_work(monkeypatch):
    monkeypatch.setattr(build.subprocess, 'check_output', lambda *args, **kwargs: b'0' * 64)
    with pytest.raises(ValueError, match='before any attempt'):
        build.git_blob_runtime()
