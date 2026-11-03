"""X12_S, family W's sealed dose arm (docs/ARC4_V85_PROTOCOL_EN.md §9, §12): X_S's recipe over 12 drawn
source heads ([9804, 3, i_unit], one draw per unit), kappa_X; on a sealed unit B1 seeds 9853/9856 only, smoke
9800 draw 1; on a development unit only seed 9826, the F1 closure build (v85_calib.arm_closures; scripts/v85_objects.py
writes that unit's closure object), whose explorer refuses to run (objects_w_v85.dose_draw, closure_only_explorer).

The object is R/w/x12_s/<unit>_d<k>.npz (R = <v85 output root>/objects), hash-checked on load; captions: the object's
and kappa.json's SHA-256, kappa, and the critic's refits, activation point and steered picks
(objects_w_v85.captions_w). Identical to V2s until the critic's first two-class refit.
"""
from ragnarok.evaluation.objects_w_v85 import captions_w, hooks_w


def build(ctx):
    return hooks_w(ctx, "x12_s")


def captions(hooks, run):
    return captions_w(hooks, run)
