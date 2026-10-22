"""v83 game pools and seeds (docs/ARC4_V83_PROTOCOL.md §2; pools from the v82 judge). Evaluator side:
agent code never imports this module (its names must not reach the agent)."""

RESERVED = ("boxing", "frostbite", "riverraid", "kangaroo")          # never opened (ARC4_VISION 4.6)
EXCLUDED = ("enduro",)                                               # held-out candidate in ARC4_VISION

DEV = ("alien", "amidar", "assault", "asterix", "atlantis", "bank_heist", "beam_rider", "berzerk", "bowling",
       "carnival", "centipede", "chopper_command", "crazy_climber", "demon_attack", "fishing_derby", "gopher",
       "hero", "ice_hockey", "jamesbond", "krull", "kung_fu_master", "name_this_game", "phoenix", "pooyan",
       "qbert", "road_runner", "space_invaders", "star_gunner", "tennis", "time_pilot", "up_n_down",
       "video_pinball", "wizard_of_wor", "yars_revenge", "zaxxon")
CALIBRATION = ("freeway", "pong", "breakout", "seaquest", "ms_pacman", "asteroids")
SEALED = ("battle_zone", "defender", "double_dunk", "gravitar", "montezuma_revenge", "pitfall", "private_eye",
          "robotank", "skiing", "solaris", "surround", "tutankham", "venture", "air_raid", "donkey_kong",
          "frogger", "galaxian", "journey_escape", "kaboom", "keystone_kapers", "king_kong", "mario_bros",
          "mr_do", "pacman")
SECOND = ("atlantis2", "crossbow", "elevator_action", "joust", "laser_gates", "sir_lancelot", "trondead",
          "turmoil", "word_zapper")
HARD_EXPLORATION = ("montezuma_revenge", "pitfall", "venture", "private_eye", "solaris", "gravitar", "skiing")

SEEDS = dict(smoke=9130, room=(9131, 9132), dev_bars=9133, diag=9134, worth_freeze1=(9135, 9136, 9137, 9138),
             freeze2=(9141, 9142, 9143, 9144), placebo_shift=9149, reserve=(9139, 9140, 9145, 9147, 9148),
             sealed_bars=9230, sealed_eligibility=(9231, 9232), sealed_arms=(9233, 9234, 9235, 9236),
             live_diag=(9241, 9242), second_bars=9250, second_eligibility=(9251, 9252),
             second_arms=(9253, 9254, 9255, 9256))
