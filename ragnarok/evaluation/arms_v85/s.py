"""S, the single best process (docs/ARC4_V85_PROTOCOL_EN.md §4.4, shared by families W and M; v84 §6.1): V2s's
own cell draw, then always process q* = argmax over Q1..Q4 of the equal-game mean ADV per billed frame in the G1b
expansion logs of the other 40 burned games (leave-one-game-out; all 41 for the sealed test), with no bandit
choice (ragnarok/agent_v85/critic_s.SingleProcess).

The object is R/w/s/<unit>.npz (R = <v85 output root>/objects), hash-checked on load; captions: its SHA-256,
q* and the per-process rates.
"""
from ragnarok.evaluation.objects_w_v85 import captions_w, hooks_s


def build(ctx):
    return hooks_s(ctx)


def captions(hooks, run):
    return captions_w(hooks, run)
