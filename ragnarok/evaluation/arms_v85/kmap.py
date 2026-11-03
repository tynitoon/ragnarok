"""K*map, the where-channel gate row (docs/ARC4_V85_PROTOCOL_EN.md A.7): KMap steering on the map of the
start's own lineage above s1 (oracle_v85.lineage_map, replayed on SenseEnv85(game, E_s), a separate unbilled
emulator; its frames, 4 per lineage step, are reported as map_frames, not as the run's oracle frames, since
the map is fixed per start and hashed at F1).

Captions: KMap.caption() (coverage = share of C_e coins below 0.5 that found a matching cell; MAP-WEAK if
coverage < 30%, or undefined), the map's SHA-256 and map_frames.
"""
import hashlib

import numpy as np

from ragnarok.evaluation.oracle_v85 import KMap, lineage_map


def map_sha256(items):
    h = hashlib.sha256()
    for sb, cd in items:
        h.update(np.int64(sb).tobytes())
        h.update(bytes(cd))
    return h.hexdigest()


def build(ctx):
    items = lineage_map(ctx.game, ctx.E, ctx.s1)
    return dict(steer=KMap(items), items=items, map_frames=4 * len(ctx.s1["actions"]))


def captions(hooks, run):
    return dict(hooks["steer"].caption(), map_sha256=map_sha256(hooks["items"]), map_frames=int(hooks["map_frames"]))
