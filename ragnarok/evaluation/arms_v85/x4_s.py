"""X4_S, family W's dose arm (docs/ARC4_V85_PROTOCOL_EN.md §8.2, §9, §12): X_S's recipe over 4 drawn source
heads, kappa_X. The draw comes from the run's unit and seed (objects_w_v85.dose_draw: a development unit G2 draw 1
on 9826/9829, draw 2 on 9827/9830, [9804, 1, i_unit]; a sealed unit B1 9853/9856, the single sealed draw
[9804, 2, i_unit]; smoke 9800 draw 1; other seeds refused).

The object is R/w/x4_s/<unit>_d<k>.npz (R = <v85 output root>/objects), hash-checked on load; captions: the object's
and kappa.json's SHA-256, kappa, and the critic's refits, activation point and steered picks
(objects_w_v85.captions_w). Identical to V2s until the critic's first two-class refit.
"""
from ragnarok.evaluation.objects_w_v85 import captions_w, hooks_w


def build(ctx):
    return hooks_w(ctx, "x4_s")


def captions(hooks, run):
    return captions_w(hooks, run)
