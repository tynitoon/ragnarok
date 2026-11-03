"""P_on, family M's placebo (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8.2, §9): windows from non-advancing rollouts of X's
sources, matched to X's macros on length and slot frequency ([9803, 1]), n0 = 10 per macro, plus online mining."""
from ragnarok.evaluation import objects_m_v85 as OM


def build(ctx):
    return OM.arm_hooks(ctx, "P", mining=True)


def captions(hooks, run):
    return OM.captions(hooks, run)
