"""K*mac, the how-to-move gate row (docs/ARC4_V85_PROTOCOL_EN.md A.7, §4.3): a macro library from the OTHER start's
lineage above its s1 (32-step windows ending at each sensed rise; actions at window steps 0, 4, ..., 28 mapped to
canonical slots through THIS start's cached adapter, held 4 steps; unmapped and priv -> NOOP; run-length
encoded; unique sequences weighted by count: oracle_v85.kmac_library), played by Q5 as V2s's fifth process
(agent_v85.q5_v85, §4.4; in-rollout draws from X_e). If the other start is invalid the unit has no K*mac row:
`build` refuses (the queue never asks for it).

Captions (A.7, A.13): the library report (round-trip fidelity, fidelity_held, fidelity_slot, share of priv- or
NOOP-mapped sampled steps, priv and unmapped shares, library size, rises, windows, SHA-256), the bandit's k at
the end and the number of executed expansions on process 4 (Q5).
"""
from ragnarok.evaluation.oracle_v85 import Q5, kmac_library


def q5_captions(hooks, run):
    """Q5 usage of a run: bandit k at the end, executed expansions on process 4."""
    ex = run["ex"]
    return dict(bandit_k=int(ex.bandit.k), q5_expansions=int(sum(1 for r in ex.exp_meta if int(r[2]) == 4)))


def build(ctx):
    if ctx.other_s1 is None:
        raise ValueError(f"{ctx.game} {ctx.start}: the other start is invalid, so the unit has no K*mac row (A.7)")
    lib, rep = kmac_library(ctx.other_s1, ctx.body.adapter)
    return dict(q5=Q5(lib), report=rep)


def captions(hooks, run):
    return dict(hooks["report"], **q5_captions(hooks, run))
