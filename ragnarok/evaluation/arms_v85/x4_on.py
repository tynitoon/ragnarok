"""X4_on, family M's dose arm (docs/ARC4_V85_PROTOCOL_EN.md §8.2, §9, §12): the miner on 4 of the unit's source games
plus online mining. Development: draw 1 on seeds 9826 (A) / 9829 (B), draw 2 on 9827 (A) / 9830 (B) ([9804, 1]);
sealed: one draw ([9804, 2]) at the B1 seeds 9853/9856 only; smoke: draw 1 at 9800; other seeds refused
(objects_m_v85.dose_kind)."""
from ragnarok.evaluation import objects_m_v85 as OM


def build(ctx):
    kind, draw = OM.dose_kind(ctx, "X4s")
    return OM.arm_hooks(ctx, kind, mining=True, draw=draw)


def captions(hooks, run):
    return OM.captions(hooks, run)
