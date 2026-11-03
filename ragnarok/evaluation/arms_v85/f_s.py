"""F_S, family W's learner from scratch (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8.1): the route S critic
(ragnarok/agent_v85/critic_s.py) with Sigma = I in the unit's shared geometry and kappa_F.

The object is R/w/f_s/<unit>.npz (R = <v85 output root>/objects), hash-checked on load; captions: the object's
and kappa.json's SHA-256, kappa, and the critic's refits, activation point and steered picks
(objects_w_v85.captions_w). Identical to V2s until the critic's first two-class refit.
"""
from ragnarok.evaluation.objects_w_v85 import captions_w, hooks_w


def build(ctx):
    return hooks_w(ctx, "f_s")


def captions(hooks, run):
    return captions_w(hooks, run)
