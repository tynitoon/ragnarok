"""Pure budget arithmetic and fixed trial coverage, no emulation or model."""
from collections import Counter
from dataclasses import replace
import random

import pytest

from ragnarok.environments.frame_budget_v70 import FrameBudget
from ragnarok.learning.entry_trial_plan_v70 import entry_trial_plan,entry_point_fits


def test_local_seed_fixes_29_trials_without_disturbing_global_rng_or_claiming_full_sweep():
    before=random.getstate()
    points=entry_trial_plan(714)
    assert points==entry_trial_plan(714) and before==random.getstate()
    actions=[a for p in points for a in p.actions]
    assert len(points)==16 and len(actions)==29 and set(actions[:18])==set(range(18))
    assert len(set(actions[18:]))==11 and set(Counter(actions).values())=={1,2}
    assert sum(p.max_charge for p in points)==3944
    assert all(p.frames_per_branch==8 and p.control_action==0 for p in points)


def test_complete_lots_reproduce_existing_pure_budget_draft_exactly():
    budget=FrameBudget(max_charged_frames=20000)
    budget.record_play(132,agent_steps=0)
    points=entry_trial_plan(714)
    for point in points:
        while not entry_point_fits(point,budget):
            assert budget.can_play(4)
            budget.record_play(4)
        assert budget.charged_frames==point.before_charged_threshold
        before=budget.report()
        assert entry_point_fits(point,budget) and budget.report()==before
        for _ in point.actions:budget.record_experiment(16,restores=2,agent_steps=2)
    report=budget.report()
    assert report['charged_frames']==19720 and report['ordinary_frames']==15776
    assert report['experiment_frames']==464 and report['restores']==58 and report['restore_charges']==3480


def test_other_experiments_share_the_allowance_and_full_lot_cannot_cross_entry_cap():
    point=entry_trial_plan(714)[0]
    budget=FrameBudget(max_charged_frames=200000)
    budget.record_play(1088)
    assert entry_point_fits(point,budget)
    budget.record_experiment(8,restores=2,agent_steps=2)  # A different paid C/E2/E3 operation.
    assert not entry_point_fits(point,budget)
    before=budget.report()
    assert not entry_point_fits(point,budget) and budget.report()==before
    budget.record_play(20000-budget.charged_frames)
    assert not entry_point_fits(point,budget)  # Larger later-game cap never creates an entry exception.


def test_shorter_pairs_never_increase_the_fixed_inventory_or_create_effect_labels():
    points=entry_trial_plan(714)
    budget=FrameBudget(max_charged_frames=20000)
    budget.record_play(132,agent_steps=0)
    for point in points:
        while not entry_point_fits(point,budget):budget.record_play(4)
        for _ in point.actions:budget.record_experiment(0,restores=2,agent_steps=2)
    assert sum(len(p.actions) for p in points)==29 and budget.report()['experiment_frames']==0
    assert budget.report()['restores']==58
    assert not hasattr(points[0],'effect')


def test_bad_seed_or_modified_cost_inventory_is_refused():
    for seed in (None,True,-1,2**32,1.5):
        with pytest.raises(ValueError):entry_trial_plan(seed)
    point=entry_trial_plan(0)[0]
    budget=FrameBudget(max_charged_frames=20000)
    for modified in (replace(point,frames_per_branch=4),replace(point,control_action=1),
                     replace(point,actions=(0,)),replace(point,before_charged_threshold=0),
                     replace(point,actions=(0,0)),replace(point,actions=[0,1]),
                     replace(point,control_action=False),replace(point,frames_per_branch=8.0)):
        with pytest.raises(ValueError):entry_point_fits(modified,budget)
    with pytest.raises(ValueError):entry_point_fits(point,FrameBudget(experiment_fraction=.05))
