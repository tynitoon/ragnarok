"""X_frozen (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8.2, §9; M3): the carried library X with no mining (the X_frozen
switch of agent_v85.macros). With the same object it is identical to X_on until expansion 250."""
from ragnarok.evaluation import objects_m_v85 as OM


def build(ctx):
    return OM.arm_hooks(ctx, "X", mining=False)


def captions(hooks, run):
    return OM.captions(hooks, run)
