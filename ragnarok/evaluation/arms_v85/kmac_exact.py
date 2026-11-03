"""K*mac-exact, a caption row (docs/ARC4_V85_PROTOCOL_EN.md A.7, §4.3): K*mac's windows on the start's OWN lineage
above s1, every step's slot held 1 step (oracle_v85.kmac_library(s1, adapter, hold=1, keep_priv=True): priv slots
of this start's own adapter are kept, since they are playable and exact; only unmapped actions -> NOOP), played
by Q5. Seed 9818 at every kept start. Never in labels or routing.

Captions: as K*mac (library report, bandit k, executed expansions on Q5).
"""
from ragnarok.evaluation.arms_v85.kmac import q5_captions
from ragnarok.evaluation.oracle_v85 import Q5, kmac_library


def build(ctx):
    lib, rep = kmac_library(ctx.s1, ctx.body.adapter, hold=1, keep_priv=True)
    return dict(q5=Q5(lib), report=rep)


def captions(hooks, run):
    return dict(hooks["report"], **q5_captions(hooks, run))
