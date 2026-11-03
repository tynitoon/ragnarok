"""GF, greedy frontier: K*map's placebo in G1-P (docs/ARC4_V85_PROTOCOL_EN.md §7.2) and family W's GF arm (§4.4):
each expansion, with probability 0.5 (a C_e coin) the cell with the largest S, ties to the deepest, then
uniform on C_e; otherwise V2s's own draw from P_{e,0}; the process from the bandit (oracle_v85.GF, injected as
steering). Uses no map and no oracle frames.

Captions: greedy picks (coins below 0.5), expansions and their share. GF always finds a cell, so it has no
coverage and no MAP-WEAK (K*map's captions).
"""
from ragnarok.evaluation.oracle_v85 import GF


def build(ctx):
    return dict(steer=GF())


def captions(hooks, run):
    n = int(run["ex"].exp_n)
    g = hooks["steer"].caption()
    return dict(g, expansions=n, greedy_share=g["greedy_picks"] / n if n else None)
