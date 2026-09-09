"""Pure prospective entry inventory; no emulator, observations or effect labels.

Twenty-nine paid eight-frame pairs can fit the existing20k/20% entry draft.
This inventory does not claim sufficient controller discovery or distinct states.
"""
from dataclasses import dataclass
import random

from ragnarok.environments.frame_budget_v70 import FrameBudget


ENTRY_LIMIT=20000
POINT_THRESHOLDS=(1088,2448,3808,5168,6528,7888,9248,10608,11968,13328,14688,16048,17408,18224,18904,19584)


@dataclass(frozen=True)
class EntryPoint:
    index: int
    before_charged_threshold: int
    actions: tuple[int,...]
    frames_per_branch: int=8
    control_action: int=0

    @property
    def branch_frames(self):return 2*self.frames_per_branch*len(self.actions)

    @property
    def restores(self):return 2*len(self.actions)

    @property
    def max_charge(self):return self.branch_frames+60*self.restores


def entry_trial_plan(seed):
    """Own local RNG: one full action permutation then11distinct second draws.

The caller must predeclare this seed in a later collection protocol. There is
no default, unseen table truth, semantic effect assignment or result-dependent
replacement trial. Each point is the then-current anchor, never a free reload.
"""
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Declare an own entry seed in0..2**32-1')
    rng=random.Random(seed)
    first=list(range(18));second=list(range(18))
    rng.shuffle(first);rng.shuffle(second)
    actions=first+second[:11]
    points=[];offset=0
    for index,threshold in enumerate(POINT_THRESHOLDS):
        count=2 if index<13 else 1
        points.append(EntryPoint(index,threshold,tuple(actions[offset:offset+count])))
        offset+=count
    return tuple(points)


def entry_point_fits(point,budget):
    """Read-only admission of a complete proposed lot against shared known costs.

Use an authoritative own FrameBudget, never subtract other E1/E2/E3/E4/C costs.
This check does not execute/record/consume a trial or replace the paid engine's
admission. A collector/controller must still enforce provenance and once-only
execution, including after a truncated or failed pair.
"""
    if (not isinstance(point,EntryPoint) or type(point.index) is not int or not 0<=point.index<16
            or point.before_charged_threshold!=POINT_THRESHOLDS[point.index]
            or type(point.frames_per_branch) is not int or point.frames_per_branch!=8
            or type(point.control_action) is not int or point.control_action!=0
            or not isinstance(point.actions,tuple)
            or len(point.actions)!=(2 if point.index<13 else 1)
            or any(type(a) is not int or not 0<=a<18 for a in point.actions)
            or len(set(point.actions))!=len(point.actions)):
        raise ValueError('Use the declared entry-point shape, distinct actions and cost domain')
    if not isinstance(budget,FrameBudget) or budget.report()['experiment_fraction']!=.2:
        raise ValueError('The entry draft requires its authoritative20% shared budget')
    return (budget.charged_frames>=point.before_charged_threshold
            and budget.charged_frames+point.max_charge<=ENTRY_LIMIT
            and budget.can_experiment(point.branch_frames,restores=point.restores))
