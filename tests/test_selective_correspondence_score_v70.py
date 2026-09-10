"""Invented judge flags only: no generated images or scientific candidate run."""
from dataclasses import replace
import math

import pytest

from ragnarok.evaluation import selective_correspondence_score_v70 as scoring


def examples(*, correct=192, admitted=None, initial_correct=128):
    if admitted is None:
        admitted = correct
    rows = []
    for name in scoring.STRATA:
        for index in range(256):
            visible = name in scoring.VISIBLE
            rows.append(scoring.SceneVerdict(
                f'{name}:{index}', name,
                visible and index < admitted, visible and index < correct,
                visible and index < initial_correct, visible and index < initial_correct,
            ))
    return rows


def witnesses():
    return [scoring.WitnessVerdict(name, False) for name in scoring.WITNESSES]


def test_exact_bound_endpoints_match_closed_forms_and_zero_trials_are_unknown():
    alpha, count = scoring.ALPHA, 17
    assert scoring.upper_bound(0, 0) == 1
    assert scoring.lower_bound(0, 0) == 0
    assert scoring.upper_bound(count, count) == 1
    assert scoring.lower_bound(0, count) == 0
    assert scoring.upper_bound(0, count) == pytest.approx(1-alpha**(1/count), abs=1e-12)
    assert scoring.lower_bound(count, count) == pytest.approx(alpha**(1/count), abs=1e-12)


def test_paired_tail_matches_direct_binomial_sum_and_handles_no_discordance():
    assert scoring.paired_improvement(0, 0) == 1
    expected = sum(math.comb(10, index) for index in range(8, 11)) / 2**10
    assert scoring.paired_improvement(8, 2) == pytest.approx(expected, abs=1e-14)
    assert scoring.paired_improvement(20, 0) == pytest.approx(2**-20, abs=1e-14)


def test_all_unknown_fails_despite_no_observed_error_or_improper_admission():
    result = scoring.score(examples(correct=0, initial_correct=0), witnesses())
    assert result['verdict'] == 'FAIL'
    for row in result['visible'].values():
        assert row['admitted'] == row['correct'] == row['errors'] == 0
        assert row['error_upper'] == 1 and row['correct_coverage_lower'] == 0
        assert row['paired_pvalue'] == 1
        assert not any(row['conditions'].values())
    assert all(row['passed'] for row in result['unattributable'].values())


def test_invented_reliable_improving_candidate_passes_all_nine_requirements():
    result = scoring.score(examples(), witnesses())
    assert result['verdict'] == 'PASS' and result['scenes'] == 1280
    for row in result['visible'].values():
        assert row['paired_wins'] == 64 and row['paired_losses'] == 0
        assert row['observed_gain'] == 0.25 and row['correct_coverage'] == 0.75
        assert row['paired_pvalue'] == pytest.approx(2**-64)
        assert row['error_upper'] == pytest.approx(1-scoring.ALPHA**(1/192), abs=1e-12)
        assert all(row['conditions'].values())
    assert result['association_reliable_in_declared_family']
    assert result['learned_improvement_over_initial']
    assert result['m_c_supervision_authorized'] is False and result['me_identified'] is False


def test_identical_initial_and_final_can_be_reliable_without_learning_evidence():
    result = scoring.score(examples(initial_correct=192), witnesses())
    assert result['association_reliable_in_declared_family']
    assert not result['learned_improvement_over_initial'] and result['verdict'] == 'FAIL'
    assert all(row['paired_pvalue'] == 1 for row in result['visible'].values())


def test_significant_but_small_observed_gain_does_not_pass():
    result = scoring.score(examples(correct=256, initial_correct=236), witnesses())
    for row in result['visible'].values():
        assert row['paired_pvalue'] < scoring.ALPHA
        assert row['observed_gain'] == 20/256 < 0.10
        assert not row['conditions']['learned_gain']
    assert result['verdict'] == 'FAIL'


def test_high_admission_error_cannot_be_hidden_by_coverage():
    result = scoring.score(examples(correct=192, admitted=256), witnesses())
    assert result['verdict'] == 'FAIL'
    for row in result['visible'].values():
        assert row['errors'] == 64 and row['error_upper'] > 0.25
        assert row['conditions']['correct_coverage'] and not row['conditions']['error_bound']


def test_each_unattributable_stratum_and_each_witness_can_independently_fail():
    for name in scoring.UNATTRIBUTABLE:
        rows = [replace(row, admitted=True) if row.stratum == name else row for row in examples()]
        result = scoring.score(rows, witnesses())
        assert result['verdict'] == 'FAIL'
        assert result['unattributable'][name]['improper_admission_upper'] == 1
        assert all(row['passed'] for row in result['visible'].values())
    for name in scoring.WITNESSES:
        fixed = [replace(item, attributed=True) if item.witness_id == name else item for item in witnesses()]
        result = scoring.score(examples(), fixed)
        assert result['verdict'] == 'FAIL'
        assert not result['fixed_witnesses'][name]['passed']


def test_missing_duplicate_or_extra_scenes_and_missing_witnesses_are_rejected():
    rows = examples()
    for bad in (rows[:-1], rows+[rows[0]], [rows[0]]+rows[:-1]):
        with pytest.raises(ValueError):
            scoring.score(bad, witnesses())
    for fixed in (witnesses()[:-1], witnesses()+[witnesses()[0]]):
        with pytest.raises(ValueError):
            scoring.score(rows, fixed)


def test_judge_schema_refuses_implicit_booleans_and_impossible_correctness():
    rows = examples()
    for bad in (replace(rows[0], admitted=1), replace(rows[0], correct=None),
                replace(rows[0], admitted=False), replace(rows[0], stratum='unknown'),
                replace(rows[0], scene_id=''), replace(rows[0], initial_admitted=False)):
        with pytest.raises(ValueError):
            scoring.score([bad]+rows[1:], witnesses())
    bad = replace(rows[-1], admitted=True, correct=True)
    with pytest.raises(ValueError):
        scoring.score(rows[:-1]+[bad], witnesses())


@pytest.mark.parametrize('k,n', [(True, 1), (0, True), (-1, 1), (2, 1), (0, -1), (0.0, 1)])
def test_interval_counts_refuse_invalid_or_ambiguous_values(k, n):
    for function in (scoring.upper_bound, scoring.lower_bound):
        with pytest.raises(ValueError):
            function(k, n)
