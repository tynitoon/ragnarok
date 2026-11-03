"""P_S, family W's label placebo (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8.2): the route S critic
(ragnarok/agent_v85/critic_s.py) with Sigma from source heads whose ADV labels are permuted within each
source run ([9803, 2]), spectrum-matched to X (v84 A.8 P), and kappa_P.

The object is R/w/p_s/<unit>.npz (R = <v85 output root>/objects), hash-checked on load; captions: the object's
and kappa.json's SHA-256, kappa, and the critic's refits, activation point and steered picks
(objects_w_v85.captions_w). Identical to V2s until the critic's first two-class refit.
"""
from ragnarok.evaluation.objects_w_v85 import captions_w, hooks_w


def build(ctx):
    return hooks_w(ctx, "p_s")


def captions(hooks, run):
    return captions_w(hooks, run)
