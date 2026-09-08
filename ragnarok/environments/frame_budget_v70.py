"""CPU-only, monotone accounting for the ARC 4 draft's first instrument milestone.

The allocation is explicit and fixed per instance. A later decay schedule is not
silently invented here. No observation or snapshot can reset this ledger.
"""
from fractions import Fraction


class BudgetExceeded(RuntimeError):
    """The complete requested operation does not fit the declared budget."""


def nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class FrameBudget:
    def __init__(self, *, experiment_fraction=0.2, hard_cap=0.3,
                 restore_cost=60, max_charged_frames=500_000):
        self._fraction = Fraction(str(experiment_fraction))
        cap = Fraction(str(hard_cap))
        if not 0 < self._fraction <= cap <= Fraction(3, 10):
            raise ValueError("Require 0 < experiment_fraction <= hard_cap <= 0.3")
        self._restore_cost = nonnegative_int(restore_cost, "restore_cost")
        if self._restore_cost != 60:
            raise ValueError("The ARC 4 draft charges 60 frames per restoration")
        self._limit = nonnegative_int(max_charged_frames, "max_charged_frames")
        self._ordinary = self._experimental = self._restores = 0
        self._ordinary_steps = self._experimental_steps = 0

    @property
    def charged_frames(self):
        return self._ordinary + self._experimental + self._restores * self._restore_cost

    def can_play(self, frames):
        frames = nonnegative_int(frames, "frames")
        return self.charged_frames + frames <= self._limit

    def can_experiment(self, frames, *, restores=0):
        frames = nonnegative_int(frames, "frames")
        restores = nonnegative_int(restores, "restores")
        added = frames + self._restore_cost * restores
        total = self.charged_frames + added
        experiments = self._experimental + self._restores * self._restore_cost + added
        return (total <= self._limit and
                experiments * self._fraction.denominator <= total * self._fraction.numerator)

    def record_play(self, frames, *, agent_steps=1):
        frames = nonnegative_int(frames, "frames")
        steps = nonnegative_int(agent_steps, "agent_steps")
        if not self.can_play(frames):
            raise BudgetExceeded("Ordinary play exceeds the charged-frame ceiling")
        self._ordinary += frames
        self._ordinary_steps += steps

    def record_experiment(self, frames, *, restores=0, agent_steps=0):
        frames = nonnegative_int(frames, "frames")
        restores = nonnegative_int(restores, "restores")
        steps = nonnegative_int(agent_steps, "agent_steps")
        if not self.can_experiment(frames, restores=restores):
            raise BudgetExceeded("Experiment and restore charges exceed the allocation or ceiling")
        self._experimental += frames
        self._restores += restores
        self._experimental_steps += steps

    def report(self):
        """Detached experimenter-side counters, never part of an agent observation."""
        return dict(ordinary_frames=self._ordinary, experiment_frames=self._experimental,
                    restores=self._restores, restore_charges=self._restores * self._restore_cost,
                    emulated_frames=self._ordinary + self._experimental,
                    charged_frames=self.charged_frames, ordinary_agent_steps=self._ordinary_steps,
                    experiment_agent_steps=self._experimental_steps,
                    experiment_fraction=float(self._fraction), max_charged_frames=self._limit)
