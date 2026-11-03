"""X_on, family M's carried arm (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8.2, §9): the carried library X (the miner on
the unit's sources: leave-one-game-out in development, the 41 burned games minus near-siblings when sealed), each
macro with the pseudo-count n0 = 10, played by Q5, plus online re-mining every 250 expansions
(objects_m_v85, agent_v85.macros)."""
from ragnarok.evaluation import objects_m_v85 as OM


def build(ctx):
    return OM.arm_hooks(ctx, "X", mining=True)


def captions(hooks, run):
    return OM.captions(hooks, run)
