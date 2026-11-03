"""T, family W's tuned default (docs/ARC4_V85_PROTOCOL_EN.md §4.4, §8.2): the route S critic
(ragnarok/agent_v85/critic_s.py) with the T_clock prior (v84 §6.2, build note 12: source heads on the 7
run-context features only) and kappa_T.

The object is R/w/t/<unit>.npz (R = <v85 output root>/objects), hash-checked on load; captions: the object's
and kappa.json's SHA-256, kappa, and the critic's refits, activation point and steered picks
(objects_w_v85.captions_w). Identical to V2s until the critic's first two-class refit.
"""
from ragnarok.evaluation.objects_w_v85 import captions_w, hooks_w


def build(ctx):
    return hooks_w(ctx, "t")


def captions(hooks, run):
    return captions_w(hooks, run)
