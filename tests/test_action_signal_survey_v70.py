"""Only generated RGB, fake transport and temporary artifacts. No real proofs.

The real OwnedRGBCollector enforces causal trace ownership in these tests. The
IsolatedRGBPool constructor is replaced; no child, ALE, Torch or real NPZ opens.
"""
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import survey_action_signal_v70 as survey
from ragnarok.environments.frame_budget_v70 import FrameBudget
from ragnarok.environments.ordinary_rgb_trace_v70 import OrdinaryRGBTrace
from ragnarok.environments.rgb_branch_trace_v70 import RGBBranchTrace
from ragnarok.environments.isolated_rgb_pool_v70 import IsolatedWorkerError


TINY = (2, 3, 3)


def trace(start, length, value):
    raw = np.full((length, *TINY), value, np.uint8)
    pooled = np.empty_like(raw)
    previous = start[-1]
    for index, image in enumerate(raw):
        pooled[index] = np.maximum(previous, image)
        previous = image
    endpoint = pooled[-1] if length else np.maximum(*start)
    return RGBBranchTrace(raw, pooled, endpoint.copy())


def prepare(monkeypatch, tmp_path, *, games=('pong',), initial_charge=20000,
            reset_each_step=False, lengths=(4,), branch_length=8, fault=None):
    monkeypatch.setattr(survey, 'ROOT', tmp_path)
    monkeypatch.setattr(survey, 'FOLDER', tmp_path / 'out')
    monkeypatch.setattr(survey, 'GAMES', games)
    monkeypatch.setattr(survey, 'SHAPE', TINY)
    spec = copy.deepcopy(survey.SPEC)
    spec.update(games=list(games), anchors=[4, 8], planned_pairs_total=36 * len(games))
    monkeypatch.setattr(survey, 'SPEC', spec)
    for key, filename in dict(PREFLIGHT='preflight.json', ATTEMPT='attempt.json',
                              PROGRESS='progress.jsonl', OUTPUT='result.json', ARCHIVES='rgb').items():
        monkeypatch.setattr(survey, key, survey.FOLDER / filename)
    monkeypatch.setattr(survey, 'PROTOCOL', 'protocol.json')
    (tmp_path / 'protocol.json').write_text(json.dumps({'limitations': ['Synthetic fixtures only']}))
    proof = dict(source_sha256={'synthetic': 'a' * 64}, evidence_sha256={}, runtime={'synthetic': True},
                 child_runtime={'synthetic': True}, local_source_closure=[], preflight_review_sha256='b' * 64)
    monkeypatch.setattr(survey, 'preflight', lambda: copy.deepcopy(proof))
    created, calls = [], []

    class FakePool:
        def __init__(self, configs, *, python_executable, expected_runtime, timeout):
            assert len(configs) == 1 and configs[0].game in games
            assert configs[0].seed == 19390 and configs[0].frame_skip == 4
            assert python_executable == survey.CHILD['executable'] and expected_runtime == survey.CHILD
            assert timeout == 30
            self.game = configs[0].game
            self.closed = False
            self.frames = np.zeros((2, *TINY), np.uint8)
            self.budget = FrameBudget(max_charged_frames=30000)
            self.budget.record_play(initial_charge, agent_steps=0)
            self.steps = self.branches = 0
            self.failure = None
            self.lengths = itertools.cycle(lengths)
            created.append(self)
            calls.append(('construct', self.game))
            if fault == 'startup':
                self.closed = True
                raise IsolatedWorkerError('untyped startup', reports={0: dict(accounting=None,
                    accounting_complete=False, failure={'error_type': 'SyntheticStartup'})})

        def observe_pair(self, index=0):
            assert index == 0 and not self.closed
            return self.frames.copy()

        def step(self, actions):
            assert len(actions) == 1 and not self.closed
            calls.append(('step', self.game, actions[0]))
            self.steps += 1
            if fault == 'transport':
                self.budget.record_play(4)
                self.failure = dict(error_type='LostIsolatedTransport', accounting_complete=False)
                raise IsolatedWorkerError('BudgetExceeded-looking string is not a typed budget stop',
                                          reports={0: self.report()})
            if reset_each_step:
                self.frames[:] = 100 + self.steps
            start = self.frames.copy()
            length = next(self.lengths)
            value = self.steps % 90
            result = trace(start, length, value)
            self.budget.record_play(length)
            if length >= 2:
                self.frames = result.raw[-2:].copy()
            elif length:
                self.frames = np.stack((start[-1], result.raw[-1]))
            return [OrdinaryRGBTrace(start, result.raw, result.pooled, result.endpoint)]

        def branch_pair_trace(self, action, *, control, frames, index):
            assert not self.closed and control == index == 0 and frames == 8
            calls.append(('pair', self.game, action))
            self.branches += 1
            count = branch_length
            baseline = 30
            left = trace(self.frames, count, baseline + action)
            right = trace(self.frames, count, baseline)
            if fault == 'noop' and self.branches == 1:
                left = trace(self.frames, count, baseline + 1)
            if fault == 'control_drift' and action == 1:
                right = trace(self.frames, count, baseline + 2)
            if fault == 'control_length' and action == 1:
                right = trace(self.frames, count - 1, baseline)
            self.budget.record_experiment(len(left.raw) + len(right.raw), restores=2)
            return left, right

        def report(self, index=0):
            return dict(accounting=self.budget.report(), failure=self.failure,
                        accounting_complete=self.failure is None)

        def close(self):
            self.closed = True
            calls.append(('close', self.game))

    monkeypatch.setattr(survey, 'IsolatedRGBPool', FakePool)
    return created, calls, proof


def test_fixed_schedule_holds_four_decisions_and_is_identical_per_game():
    first = list(itertools.islice(survey.command_stream(), 40))
    assert first == list(itertools.islice(survey.command_stream(), 40))
    assert len(set(first)) > 2
    assert all(len(set(first[i:i + 4])) == 1 for i in range(0, 40, 4))
    assert all(0 <= action < 18 for action in first)


def test_all_six_cases_all_commands_both_anchors_and_exact_archive_signatures(monkeypatch, tmp_path):
    created, calls, _ = prepare(monkeypatch, tmp_path, games=survey.GAMES)
    result = survey.run()
    assert result['passed'] and result['execution_status'] == 'complete'
    assert result['planned_pairs'] == result['acquired_pairs'] == 216
    assert result['constructor_attempts'] == 6 and all(pool.closed for pool in created)
    assert result['accounting_complete'] and not result['known_charge_is_lower_bound']
    assert [case['game'] for case in result['cases']] == list(survey.GAMES)
    for case in result['cases']:
        assert case['ordinary_delivered_frames'] == 8 and case['ordinary_decisions'] == 2
        assert [anchor['delivered_frames'] for anchor in case['anchors']] == [4, 8]
        assert all(anchor['overshoot'] == 0 for anchor in case['anchors'])
        for anchor in case['anchors']:
            assert [row['command'] for row in anchor['pairs']] == list(range(18))
            assert all(row['status'] == 'acquired' for row in anchor['pairs'])
            assert anchor['pairs'][0]['raw']['all_zero_over_observed_prefix'] is True
            for row in anchor['pairs']:
                archive = row['archive']
                path = tmp_path / archive['path']
                assert path.is_file() and survey.sha(path) == archive['sha256']
                # This is a newly generated temporary fixture, never a real archive.
                with np.load(path, allow_pickle=False) as loaded:
                    assert set(loaded.files) == set(archive['arrays'])
                    for key in loaded.files:
                        assert survey.array_signature(loaded[key]) == archive['arrays'][key]
    assert result['archive_bytes'] == sum(path.stat().st_size for path in survey.ARCHIVES.glob('*.npz'))
    assert result['progress_sha256'] == survey.sha(survey.PROGRESS)
    assert survey.read_json(survey.ATTEMPT)['result_sha256'] == survey.sha(survey.OUTPUT)
    for index in range(5):
        assert calls.index(('close', survey.GAMES[index])) < calls.index(('construct', survey.GAMES[index + 1]))
    before = list(calls)
    with pytest.raises(ValueError, match='no retry'):
        survey.run()
    assert calls == before


def test_delivered_counter_survives_epoch_reset_and_short_repeat_overshoot(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path, reset_each_step=True, lengths=(3, 3, 3))
    result = survey.run()
    case = result['cases'][0]
    assert result['passed'] and case['ordinary_delivered_frames'] == 9
    assert [a['delivered_frames'] for a in case['anchors']] == [6, 9]
    assert [a['overshoot'] for a in case['anchors']] == [2, 1]
    assert [a['own_anchor']['frame'] for a in case['anchors']] == [3, 3]
    assert [a['own_anchor']['epoch'] for a in case['anchors']] == [2, 3]


@pytest.mark.parametrize('length', [0, 1, 3])
def test_equal_short_control_prefixes_are_preserved_and_missing_horizons_unknown(monkeypatch, tmp_path, length):
    prepare(monkeypatch, tmp_path, branch_length=length)
    result = survey.run()
    assert result['passed'] and result['acquired_pairs'] == 36
    row = result['cases'][0]['anchors'][0]['pairs'][0]
    assert row['lengths'] == [length, length]
    assert row['raw']['common_prefix_frames'] == length
    for cell in row['raw']['horizons'][length:]:
        assert cell['observed'] is False and cell['changed_pixels'] is None
    if length == 0:
        assert row['raw']['all_zero_over_observed_prefix'] is None


@pytest.mark.parametrize('fault,paid', [('noop', 1), ('control_drift', 2), ('control_length', 2)])
def test_instrument_disagreement_preserves_paid_rgb_and_stops_every_later_game(monkeypatch, tmp_path, fault, paid):
    created, calls, _ = prepare(monkeypatch, tmp_path, games=('pong', 'breakout'), fault=fault)
    result = survey.run()
    assert result['passed'] is False and result['execution_status'] == 'failed'
    assert len(created) == 1 and created[0].closed
    assert sum(call[0] == 'pair' for call in calls) == paid
    anchor = result['cases'][0]['anchors'][0]
    assert anchor['instrument_valid'] is False
    assert anchor['pairs'][paid - 1]['status'] == 'failed'
    assert (tmp_path / anchor['pairs'][paid - 1]['archive']['path']).is_file()
    assert not result['cases'][1]['started']
    assert all(row['raw'] is None for row in result['cases'][1]['anchors'][0]['pairs'])


def test_conservative_ordinary_budget_censors_each_game_without_paid_steps(monkeypatch, tmp_path):
    created, calls, _ = prepare(monkeypatch, tmp_path, games=('pong', 'seaquest'), initial_charge=29900)
    result = survey.run()
    assert result['passed'] and result['execution_status'] == 'censored'
    assert len(created) == 2 and all(pool.closed for pool in created)
    assert not any(call[0] in ('step', 'pair') for call in calls)
    assert [case['budget_stop']['reserved_maximum_charge'] for case in result['cases']] == [136, 132]
    assert all(case['budget_stop']['remaining_charged_capacity'] == 100 for case in result['cases'])
    assert all(case['budget_stop']['actual_ceiling_exhaustion_claimed'] is False for case in result['cases'])


def test_branch_quota_refusal_does_not_play_more_to_make_anchor_convenient(monkeypatch, tmp_path):
    _, calls, _ = prepare(monkeypatch, tmp_path, initial_charge=500)
    result = survey.run()
    case = result['cases'][0]
    assert result['passed'] and case['stop_reason'] == 'budget_reservation'
    assert case['budget_stop']['reason'] == 'experimental_quota_reservation'
    assert case['ordinary_delivered_frames'] == 4
    assert not any(call[0] == 'pair' for call in calls)


@pytest.mark.parametrize('fault', ['startup', 'transport'])
def test_untyped_isolated_failures_never_become_budget_censor_and_keep_unknown_costs(monkeypatch, tmp_path, fault):
    created, calls, _ = prepare(monkeypatch, tmp_path, games=('pong', 'breakout'), fault=fault)
    result = survey.run()
    assert not result['passed'] and result['execution_status'] == 'failed'
    assert len(created) == 1 and created[0].closed
    assert result['accounting_complete'] is False and result['known_charge_is_lower_bound'] is True
    assert result['uncertain_cost_games'] == ['pong']
    assert not result['cases'][1]['started'] and result['cases'][0]['stop_reason'] == 'instrument_failure'
    assert sum(call[0] == 'step' for call in calls) == (fault == 'transport')


def test_regressed_or_unknown_ledger_cannot_erase_an_acknowledged_charge():
    case = dict(worker_report={'accounting': dict(charged_frames=123, ordinary_frames=123)}, accounting_uncertain=False)
    survey.remember(case, dict(accounting=None, failure={'error_type': 'LostReply'}), uncertain=True)
    assert case['worker_report']['accounting']['charged_frames'] == 123
    assert case['accounting_uncertain'] and case['worker_report']['accounting_complete'] is False


def test_received_paid_pair_is_archived_before_a_failed_report_rpc(monkeypatch, tmp_path):
    created, calls, _ = prepare(monkeypatch, tmp_path, games=('pong', 'breakout'))
    factory = survey.IsolatedRGBPool

    class LostReportPool(factory):
        def report(self, index=0):
            if self.branches == 0:
                self.last_acknowledged = super().report(index)
                return self.last_acknowledged
            # The report fails after successful branch delivery. The archive
            # must already exist before entering this RPC, not during cleanup.
            path = survey.ARCHIVES / 'pong_anchor0_action00.npz'
            assert path.is_file()
            with np.load(path, allow_pickle=False) as arrays:
                assert len(arrays.files) == 7
                assert arrays['branch_intervention_raw'].shape == (8, *TINY)
                assert arrays['branch_control_raw'].shape == (8, *TINY)
            raise IsolatedWorkerError('synthetic lost post-pair report',
                                      reports={0: self.last_acknowledged})

    monkeypatch.setattr(survey, 'IsolatedRGBPool', LostReportPool)
    result = survey.run()
    assert result['passed'] is False and result['execution_status'] == 'failed'
    assert result['accounting_complete'] is False and result['known_charge_is_lower_bound'] is True
    assert result['known_charged_frames'] == 20004
    assert len(created) == 1 and created[0].closed
    assert sum(call[0] == 'step' for call in calls) == 1
    assert sum(call[0] == 'pair' for call in calls) == 1
    row = result['cases'][0]['anchors'][0]['pairs'][0]
    assert row['status'] == 'failed' and row['archive'] is not None
    assert survey.sha(tmp_path / row['archive']['path']) == row['archive']['sha256']
    assert result['archive_bytes'] == row['archive']['bytes']
    assert not result['cases'][1]['started']


def test_missing_preflight_cannot_create_attempt_or_constructor(monkeypatch, tmp_path):
    created, _, _ = prepare(monkeypatch, tmp_path)
    def no_preflight():
        raise ValueError('No independent preflight')
    monkeypatch.setattr(survey, 'preflight', no_preflight)
    with pytest.raises(ValueError, match='independent'):
        survey.run()
    assert not created and not survey.ATTEMPT.exists()


@pytest.mark.parametrize('resource', ['time', 'storage'])
def test_global_resource_censor_keeps_all_unreached_cases_and_never_retries(monkeypatch, tmp_path, resource):
    created, calls, _ = prepare(monkeypatch, tmp_path, games=('pong', 'breakout'))
    survey.SPEC['max_work_seconds' if resource == 'time' else 'max_archive_bytes'] = -1 if resource == 'time' else 1
    result = survey.run()
    assert result['passed'] and result['execution_status'] == 'censored'
    assert len(result['cases']) == 2 and all(pool.closed for pool in created)
    assert not any(call[0] == 'pair' for call in calls)
    assert result['acquired_pairs'] == 0
    assert all(row['raw'] is None for case in result['cases'] for anchor in case['anchors'] for row in anchor['pairs'])


def test_final_proof_change_is_failed_result_with_closed_worker_and_preserved_archives(monkeypatch, tmp_path):
    created, _, proof = prepare(monkeypatch, tmp_path)
    count = 0
    def changed():
        nonlocal count
        count += 1
        return proof if count == 1 else dict(proof, preflight_review_sha256='c' * 64)
    monkeypatch.setattr(survey, 'preflight', changed)
    result = survey.run()
    assert not result['passed'] and result['acquired_pairs'] == 36 and created[0].closed
    assert survey.read_json(survey.ATTEMPT)['final_proof_failure']['error_type'] == 'ValueError'


def test_progress_is_durable_before_paid_operation_and_prior_rows_are_retained(monkeypatch, tmp_path):
    created, _, _ = prepare(monkeypatch, tmp_path)
    factory = survey.IsolatedRGBPool
    class InspectingPool(factory):
        def step(self, actions):
            rows = [json.loads(line) for line in survey.PROGRESS.read_text().splitlines()]
            assert rows[-1]['phase'] == 'before_step'
            if self.steps == 1:
                raise OSError('synthetic second ordinary command failure')
            return super().step(actions)
    monkeypatch.setattr(survey, 'IsolatedRGBPool', InspectingPool)
    result = survey.run()
    assert not result['passed'] and result['acquired_pairs'] == 18 and created[0].closed
    assert len(list(survey.ARCHIVES.glob('*.npz'))) == 18


def test_ast_closure_includes_initializers_alias_modules_and_explicit_child(monkeypatch, tmp_path):
    monkeypatch.setattr(survey, 'ROOT', tmp_path)
    files = {'scripts/__init__.py': '', 'scripts/parent.py': 'from ragnarok import helper\n',
             'scripts/child.py': 'import ragnarok.extra\n', 'ragnarok/__init__.py': '',
             'ragnarok/helper.py': 'from . import extra\n', 'ragnarok/extra.py': ''}
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert survey.source_closure(['scripts/parent.py', 'scripts/child.py']) == sorted(files)


def test_proofs_refuse_uncommitted_new_source_before_any_archive_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(survey, 'ROOT', tmp_path)
    def uncommitted(name):
        raise ValueError('Uncommitted source')
    monkeypatch.setattr(survey, 'require_committed', uncommitted)
    monkeypatch.setattr(survey, 'sha', lambda *a: pytest.fail('Proof hash before commit check'))
    with pytest.raises(ValueError, match='Uncommitted'):
        survey.proofs()


def test_exact_preflight_binds_every_proof_field_without_real_proofs(monkeypatch, tmp_path):
    monkeypatch.setattr(survey, 'ROOT', tmp_path)
    monkeypatch.setattr(survey, 'PREFLIGHT', tmp_path / 'preflight.json')
    proof = dict(source_sha256={}, evidence_sha256={}, runtime={'synthetic': True},
                 child_runtime={'synthetic': True}, local_source_closure=['scripts/child.py'])
    monkeypatch.setattr(survey, 'proofs', lambda: proof)
    monkeypatch.setattr(survey, 'require_committed', lambda name: None)
    review = dict(kind='action_signal_survey_independent_preflight', passed=True, **proof)
    survey.PREFLIGHT.write_text(json.dumps(review))
    assert survey.preflight()['preflight_review_sha256'] == survey.sha(survey.PREFLIGHT)
    review['local_source_closure'] = []
    survey.PREFLIGHT.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='Exact independent'):
        survey.preflight()


def test_archive_partial_write_is_preserved_and_counted_after_flush(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    archives = survey.Archives()
    def fail(stream, **arrays):
        stream.write(b'partial')
        raise OSError('synthetic disk fault')
    monkeypatch.setattr(np, 'savez', fail)
    with pytest.raises(OSError, match='disk fault'):
        archives.save('partial.npz', {'rgb': np.zeros(TINY, np.uint8)})
    assert (survey.ARCHIVES / 'partial.npz').read_bytes() == b'partial'
    assert archives.bytes == 7
    with pytest.raises(FileExistsError):
        archives.save('partial.npz', {'rgb': np.zeros(TINY, np.uint8)})
    assert archives.bytes == 7


def test_checked_in_proposal_matches_runner_constants_without_proofs():
    # Read the new prospective JSON only; never open an archive or a worker.
    protocol = survey.read_json(survey.ROOT / survey.PROTOCOL)
    assert protocol['spec'] == survey.SPEC
    assert protocol['child_runtime'] == survey.CHILD
    assert protocol['parent_runtime'] == survey.runtime()
    assert survey.SPEC['cold_costs'] == {game: survey.ROM_SPECS[game][1] for game in survey.GAMES}
    assert survey.SPEC['max_experiment_charge_total'] == 6 * 36 * 136


def test_proof_maps_and_scalar_attestation_on_temporary_metadata_only(monkeypatch, tmp_path):
    monkeypatch.setattr(survey, 'ROOT', tmp_path)
    monkeypatch.setattr(survey, 'PROTOCOL', 'protocol.json')
    new = ('scripts/candidate.py', 'tests/candidate.py', 'protocol.json', 'draft.md')
    for name in new:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('' if name.endswith('.py') else '{}')
    protocol = dict(kind='prospective_action_signal_survey_v0', spec=survey.SPEC,
                    parent_runtime=survey.runtime(), child_runtime=survey.CHILD,
                    agent_training_allowed=False, gate_result=None)
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    (tmp_path / 'result.json').write_text(json.dumps({'passed': True}))
    (tmp_path / 'fixture.npz').write_bytes(b'synthetic metadata-only proof fixture')
    result_sha = survey.sha(tmp_path / 'result.json')
    evidence = {name: survey.sha(tmp_path / name) for name in ('result.json', 'fixture.npz')}
    review = dict(passed=True, result_sha256=result_sha, source_sha256={}, evidence_sha256=evidence)
    (tmp_path / 'review.json').write_text(json.dumps(review))
    monkeypatch.setattr(survey, 'NEW_SOURCES', new)
    monkeypatch.setattr(survey, 'source_closure', lambda entries: list(new[:2]))
    committed = []
    monkeypatch.setattr(survey, 'require_committed', committed.append)
    monkeypatch.setattr(survey, 'PREREQUISITES', [('review.json', survey.sha(tmp_path / 'review.json'), 'result.json', result_sha)])
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('Array load in proof'))
    proof = survey.proofs()
    assert set(proof['source_sha256']) == set(new)
    assert set(proof['evidence_sha256']) == {'review.json', 'result.json', 'fixture.npz'}
    assert set(committed) == set(new) | {'review.json', 'result.json', 'fixture.npz'}
    review['result_sha256'] = 'f' * 64
    (tmp_path / 'review.json').write_text(json.dumps(review))
    monkeypatch.setattr(survey, 'PREREQUISITES', [('review.json', survey.sha(tmp_path / 'review.json'), 'result.json', result_sha)])
    with pytest.raises(ValueError, match='historical review'):
        survey.proofs()
