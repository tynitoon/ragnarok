"""DM_on, a tuned default of family M (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8.2, §9): the designer library DM
(agent_v85.macros.DM, frozen in §4.4), n0 = 10 per macro, plus online mining."""
from ragnarok.evaluation import objects_m_v85 as OM


def build(ctx):
    return OM.arm_hooks(ctx, "DM", mining=True)


def captions(hooks, run):
    return OM.captions(hooks, run)
