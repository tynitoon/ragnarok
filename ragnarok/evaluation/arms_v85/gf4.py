"""GF4, O_4's placebo in G1-P (docs/ARC4_V85_PROTOCOL_EN.md §7.2): among V2s's own 4 proposals propose(e, 0..3),
the one whose cell has the highest S, then more lives, then shallower, then lowest j (oracle_v85.GF4); no
look-ahead, no oracle frames; the winner is executed on X_e. Same units and seeds as the O_4 row.

Captions: GF4.caption() (expansions, share of picks other than proposal 0, share of expansions whose 4
proposals share one cell).
"""
from ragnarok.evaluation.oracle_v85 import GF4


def build(ctx):
    return dict(chooser=GF4(4))


def captions(hooks, run):
    return dict(hooks["chooser"].caption(), oracle_frames=int(run["env"].oracle_frames))
