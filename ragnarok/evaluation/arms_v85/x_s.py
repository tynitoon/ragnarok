"""X_S, family W's carried object (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8.2): the route S critic
(ragnarok/agent_v85/critic_s.py) with the zero-mean slope prior Sigma_X of v84 A.5-A.8 (null-space threshold
1e-2), from the G1b expansion logs of the other 40 burned games (leave-one-game-out), and kappa_X.

The object is R/w/x_s/<unit>.npz (R = <v85 output root>/objects), hash-checked on load; captions: the object's
and kappa.json's SHA-256, kappa, and the critic's refits, activation point and steered picks
(objects_w_v85.captions_w). Identical to V2s until the critic's first two-class refit.
"""
from ragnarok.evaluation.objects_w_v85 import captions_w, hooks_w


def build(ctx):
    return hooks_w(ctx, "x_s")


def captions(hooks, run):
    return captions_w(hooks, run)
