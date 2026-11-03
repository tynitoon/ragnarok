"""K~mac, K*mac's placebo in G1-P (docs/ARC4_V85_PROTOCOL_EN.md §7.2, §12): ANOTHER unit's cross-start library.
Partners are a derangement of the stage's units (ctx.units, in draw order) drawn with default_rng([9801, 2])
(oracle_v85.tilde_partners); the partner's K*mac library at the same start label, else at the partner's other
start, else empty, which plays as V2s (oracle_v85.tilde_kmac through the job's hash-checked loaders). The
partner's K*mac library at start s is its other start's lineage mapped through its start-s cached adapter;
Q5 resolves the canonical slots on this run's own adapter.

Smoke runs (seed 9800, never a measurement) on a single-unit start have no derangement: the library is empty
(V2s) and the caption says so (choice). A gate run with fewer than 2 units is refused by tilde_partners.

Captions: partner unit and start, the partner's library report (as K*mac; size 0 and no report if empty),
bandit k and executed expansions on Q5.
"""
from ragnarok.evaluation.arms_v85.kmac import q5_captions
from ragnarok.evaluation.oracle_v85 import Q5, lib_sha256, tilde_kmac


def build(ctx):
    if ctx.smoke and len(ctx.units) < 2:
        h, s, lib, rep = None, None, [], None
    else:
        h, s, lib, rep = tilde_kmac(ctx.game, ctx.start, ctx.units, ctx.load_s1, ctx.load_body)
    return dict(q5=Q5(lib), partner=h, partner_start=s, report=rep, lib=lib)


def captions(hooks, run):
    rep = hooks["report"] or dict(size=len(hooks["lib"]), sha256=lib_sha256(hooks["lib"]))
    return dict(rep, partner=hooks["partner"], partner_start=hooks["partner_start"],
                no_derangement=hooks["partner"] is None, **q5_captions(hooks, run))
