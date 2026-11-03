"""O_4-dice, a caption row (docs/ARC4_V85_PROTOCOL_EN.md A.7, §4.3, G-build test 8): as O_4, but the winner w is
re-executed on its own look-ahead stream X_{e,w} (oracle_v85.ODice, wired as Explorer2s(chooser=o, after=o.after));
its ALE actions, emulated frames after the restore and max S must equal its look-ahead's. A mismatch raises
DiceMismatch, listed in ABORTS: the job writes the run with `aborted` = the message (counted; one dated fix).
Seed 9818, one start per unit (queue). Never in labels or routing; the realised R beside S0's G_4 is a verdict
report (A.7, A.10.1(d)).

Captions: ODice.caption() (O-TIES, rank correlation, dice_checked), oracle frames, winner counts, and
`dice_ok`: every executed winner that completed its rollout matched its look-ahead (False exactly when the
run was aborted by DiceMismatch; a rollout cut by the cap or the evaluator stop is not checked).
"""
import numpy as np

from ragnarok.evaluation.oracle_v85 import DiceMismatch, ODice

ABORTS = (DiceMismatch,)


def build(ctx):
    o = ODice(4)
    return dict(chooser=o, kwargs=dict(after=o.after))


def captions(hooks, run):
    o = hooks["chooser"]
    wins = np.bincount([w for _, w, _, _ in o.log], minlength=o.m).tolist()
    return dict(o.caption(), oracle_frames=int(run["env"].oracle_frames), winner_counts=wins,
                dice_ok=bool(len(o.outcomes) == o.checked))
