"""F_M, family M's learner from scratch (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8): an empty library plus online
mining. The bandit keeps k = 4 until a re-mine keeps a macro, so F_M is byte-identical to V2s for its first 250
expansions (G-build test 6)."""
from ragnarok.evaluation import objects_m_v85 as OM


def build(ctx):
    return OM.arm_hooks(ctx, None, mining=True)


def captions(hooks, run):
    return OM.captions(hooks, run)
