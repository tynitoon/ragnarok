"""K_mine, family M's same-game anchor (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8.1): the miner run on this game's own
G1b rows (one source), n0 = 10 per macro, plus online mining (choice: like X_on, so that K:F and X:F differ only by
the library's source, as phi = log R(X:F) / log R(K:F) requires; recorded in objects_m_v85)."""
from ragnarok.evaluation import objects_m_v85 as OM


def build(ctx):
    return OM.arm_hooks(ctx, "K_mine", mining=True)


def captions(hooks, run):
    return OM.captions(hooks, run)
