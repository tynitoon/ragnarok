"""O_4, the selection-channel gate row (docs/ARC4_V85_PROTOCOL_EN.md A.7): oracle_v85.OChooser(4), wired as
Explorer2s(chooser=o, after=o.after) so that the executed outcome of each expansion is scored like a look-ahead.

Captions (A.7): oracle frames; OChooser.caption() (O-TIES = share of expansions whose m look-ahead scores
all tie, flagged above 70%; Spearman rank correlation between the winner's look-ahead score and the
executed outcome); how often each proposal j won.
"""
import numpy as np

from ragnarok.evaluation.oracle_v85 import OChooser


def build(ctx, m=4):
    o = OChooser(m)
    return dict(chooser=o, kwargs=dict(after=o.after))


def captions(hooks, run):
    o = hooks["chooser"]
    wins = np.bincount([w for _, w, _, _ in o.log], minlength=o.m).tolist()
    return dict(o.caption(), oracle_frames=int(run["env"].oracle_frames), winner_counts=wins)
