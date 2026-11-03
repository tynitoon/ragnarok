"""K_S, family W's same-game anchor (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8.1 G2a): the route S critic
(ragnarok/agent_v85/critic_s.py) with Sigma from this game's own G1b expansions (v84 A.8 K*) and kappa_K.

The object is R/w/k_s/<unit>.npz (R = <v85 output root>/objects), hash-checked on load; captions: the object's
and kappa.json's SHA-256, kappa, and the critic's refits, activation point and steered picks
(objects_w_v85.captions_w). Identical to V2s until the critic's first two-class refit.
"""
from ragnarok.evaluation.objects_w_v85 import captions_w, hooks_w


def build(ctx):
    return hooks_w(ctx, "k_s")


def captions(hooks, run):
    return captions_w(hooks, run)
