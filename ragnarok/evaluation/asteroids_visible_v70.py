"""Composition of independently developed external Asteroids instruments.

RAM labels describe visible pieces by allocation. The RGB judge receives only
the raw image. Neither instrument supplies observations or training labels to
the future learning agent.
"""
from ragnarok.evaluation.asteroids_v70 import KINDS,ROI
from ragnarok.evaluation.asteroids_player_roi_v70 import decode_player_missile_ram,screen_player_missile_objects
from ragnarok.evaluation.asteroids_history_v70 import decode_history_rocks
from ragnarok.evaluation.asteroids_deblend_edges_v70 import screen_boundary_deblended_rocks

REVISION='asteroids_raw_visible_composition_v1'


def decode_visible_asteroids_ram(ram,previous=None):
    return decode_player_missile_ram(ram)+decode_history_rocks(ram,previous)


def screen_visible_asteroids_objects(rgb):
    return screen_player_missile_objects(rgb)+screen_boundary_deblended_rocks(rgb)
