"""X12_on, family M's sealed dose arm (docs/ARC4_V85_PROTOCOL_EN.md §9, §12): the miner on 12 of the unit's source
games ([9804, 3], one draw) plus online mining. Sealed unit: B1 seeds 9853/9856 only; development unit: seed 9826
only, the F1 closure build, whose explorer refuses to run; smoke 9800 (objects_m_v85.dose_kind)."""
from ragnarok.evaluation import objects_m_v85 as OM


def build(ctx):
    kind, draw = OM.dose_kind(ctx, "X12s")
    return OM.arm_hooks(ctx, kind, mining=True, draw=draw)


def captions(hooks, run):
    return OM.captions(hooks, run)
