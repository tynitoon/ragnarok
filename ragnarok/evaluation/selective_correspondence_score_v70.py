"""Prospective scene-level score of selective correspondence, judge-side only.

No images, network, fitting, candidate admission rule or pseudo-labels here.
IID sampling and prediction provenance must be verified outside this scorer.
"""
from dataclasses import dataclass

from scipy.stats import binomtest


SCENES_PER_STRATUM = 256
VISIBLE = ('visible_moving', 'visible_stationary')
UNATTRIBUTABLE = ('occluded', 'out_of_frame', 'ambiguous')
STRATA = VISIBLE + UNATTRIBUTABLE
WITNESSES = ('changed_occluder_raw', 'changed_occluder_pooled',
             'hidden_subject_raw', 'hidden_subject_pooled')
ALPHA = 0.05 / 9
MAX_ERROR = 0.05
MIN_CORRECT_COVERAGE = 0.50
MIN_OBSERVED_GAIN = 0.10


@dataclass(frozen=True, slots=True)
class SceneVerdict:
    scene_id: str
    stratum: str
    admitted: bool
    correct: bool  # False on abstention, and always false in UNATTRIBUTABLE.
    initial_admitted: bool
    initial_correct: bool


@dataclass(frozen=True, slots=True)
class WitnessVerdict:
    witness_id: str
    attributed: bool


def _counts(k, n):
    if type(k) is not int or type(n) is not int or not 0 <= k <= n:
        raise ValueError('Integer counts with 0 <= k <= n required')


def upper_bound(k, n):
    """One-sided exact upper bound at 1-ALPHA; no trials gives no assurance."""
    _counts(k, n)
    if n == 0:
        return 1.0
    return float(binomtest(k, n, alternative='less').proportion_ci(
        confidence_level=1 - ALPHA, method='exact').high)


def lower_bound(k, n):
    """One-sided exact lower bound at 1-ALPHA; no trials gives no coverage."""
    _counts(k, n)
    if n == 0:
        return 0.0
    return float(binomtest(k, n, alternative='greater').proportion_ci(
        confidence_level=1 - ALPHA, method='exact').low)


def paired_improvement(wins, losses):
    _counts(wins, wins)
    _counts(losses, losses)
    discordant = wins + losses
    return 1.0 if discordant == 0 else float(binomtest(
        wins, discordant, p=0.5, alternative='greater').pvalue)


def _validated_rows(rows):
    groups = {name: [] for name in STRATA}
    seen = set()
    for row in rows:
        if (not isinstance(row, SceneVerdict) or not isinstance(row.scene_id, str)
                or not row.scene_id or row.scene_id in seen
                or not isinstance(row.stratum, str) or row.stratum not in groups):
            raise ValueError('One unique scene ID and known stratum per SceneVerdict required')
        flags = (row.admitted, row.correct, row.initial_admitted, row.initial_correct)
        if any(type(value) is not bool for value in flags):
            raise ValueError('Judge admission and correctness flags must be actual booleans')
        if ((row.correct and not row.admitted) or (row.initial_correct and not row.initial_admitted)
                or (row.stratum in UNATTRIBUTABLE and (row.correct or row.initial_correct))):
            raise ValueError('Abstention and unobservable identity cannot be scored as correct')
        seen.add(row.scene_id)
        groups[row.stratum].append(row)
        if len(groups[row.stratum]) > SCENES_PER_STRATUM:
            raise ValueError('More than the declared number of scenes in a stratum')
    if any(len(group) != SCENES_PER_STRATUM for group in groups.values()):
        raise ValueError('All five complete strata of 256 scenes are required')
    return groups


def _validated_witnesses(witnesses):
    seen = {}
    for item in witnesses:
        if (not isinstance(item, WitnessVerdict) or item.witness_id not in WITNESSES
                or item.witness_id in seen or type(item.attributed) is not bool):
            raise ValueError('Each named occlusion witness must occur once with a boolean decision')
        seen[item.witness_id] = item.attributed
    if set(seen) != set(WITNESSES):
        raise ValueError('All four fixed occlusion witnesses are required')
    return {key: {'attributed': seen[key], 'passed': not seen[key]} for key in WITNESSES}


def score(rows, witnesses):
    """Score a frozen candidate and its initial state on the same judge queries.

    Exactly one predeclared query per IID scene, chosen without model output and
    hidden from inference. The function checks data shape, not those causal facts.
    No aggregate treats correlated pixels within a scene as independent trials.
    """
    groups = _validated_rows(rows)
    witness_results = _validated_witnesses(witnesses)
    visible, unsafe = {}, {}
    for name in VISIBLE:
        group = groups[name]
        count = len(group)
        admitted = sum(row.admitted for row in group)
        correct = sum(row.correct for row in group)
        initial_admitted = sum(row.initial_admitted for row in group)
        initial_correct = sum(row.initial_correct for row in group)
        errors = admitted - correct
        wins = sum(row.correct and not row.initial_correct for row in group)
        losses = sum(row.initial_correct and not row.correct for row in group)
        risk_upper = upper_bound(errors, admitted)
        coverage_lower = lower_bound(correct, count)
        gain = (correct - initial_correct) / count
        pvalue = paired_improvement(wins, losses)
        conditions = dict(error_bound=risk_upper <= MAX_ERROR,
                          correct_coverage=coverage_lower >= MIN_CORRECT_COVERAGE,
                          learned_gain=gain >= MIN_OBSERVED_GAIN and pvalue <= ALPHA)
        visible[name] = dict(scenes=count, admitted=admitted, errors=errors, correct=correct,
                             abstained=count-admitted, initial_admitted=initial_admitted,
                             initial_correct=initial_correct, initial_errors=initial_admitted-initial_correct,
                             error_upper=risk_upper, correct_coverage=correct/count,
                             correct_coverage_lower=coverage_lower, observed_gain=gain,
                             paired_wins=wins, paired_losses=losses, paired_pvalue=pvalue,
                             conditions=conditions, passed=all(conditions.values()))
    for name in UNATTRIBUTABLE:
        group = groups[name]
        admitted = sum(row.admitted for row in group)
        upper = upper_bound(admitted, len(group))
        unsafe[name] = dict(scenes=len(group), improper_admissions=admitted,
                            abstained=len(group)-admitted,
                            initial_improper_admissions=sum(row.initial_admitted for row in group),
                            improper_admission_upper=upper, passed=upper <= MAX_ERROR)
    controls_pass = all(item['passed'] for item in witness_results.values())
    reliable = (all(item['conditions']['error_bound'] and item['conditions']['correct_coverage']
                    for item in visible.values()) and all(item['passed'] for item in unsafe.values())
                and controls_pass)
    improved = all(item['conditions']['learned_gain'] for item in visible.values())
    return dict(kind='selective_correspondence_scene_score',
                verdict='PASS' if reliable and improved else 'FAIL',
                scenes=len(STRATA)*SCENES_PER_STRATUM, alpha_per_requirement=ALPHA,
                simultaneous_requirements=9, association_reliable_in_declared_family=reliable,
                learned_improvement_over_initial=improved, visible=visible,
                unattributable=unsafe, fixed_witnesses=witness_results,
                m_c_supervision_authorized=False, me_identified=False)
