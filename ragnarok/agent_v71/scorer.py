"""Experimenter-side ground truth from RAM (AtariARI address tables, Anand et al. 2019).

Never imported by agent modules. Used only to score what the agent found.
"""
# game -> {name: ram address}; 'player' entries give the controlled object's coordinate
RAM_TABLE = {
    "breakout": {"player_x": 72, "ball_x": 99, "ball_y": 101},
    "freeway": {"player_y": 14},
    "space_invaders": {"player_x": 28},
    "pong": {"player_y": 51, "enemy_y": 50, "ball_x": 49, "ball_y": 54},
}


def player_coord(game, ram):
    """(axis, value) of the player's controlled coordinate, or None."""
    t = RAM_TABLE.get(game, {})
    for k in ("player_x", "player_y"):
        if k in t:
            return k[-1], int(ram[t[k]])
    return None
