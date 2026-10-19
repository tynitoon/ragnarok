"""Thin ALE wrapper: frameskip 4 with a 2-frame max-pool, savestates, live/lab frame accounting.

The agent-facing API returns screen bytes only. `ram()` exists for the experimenter-side
scorer and must never be passed to the agent (the slice's tests assert the agent modules
do not import or call it).
"""
import numpy as np
from ale_py import ALEInterface, LoggerMode, roms

ALEInterface.setLoggerMode(LoggerMode.Error)
RESERVED = {"boxing", "frostbite", "riverraid", "kangaroo"}   # never opened (ARC4_VISION 4.6)


class AtariEnv:
    def __init__(self, game, seed=0, frameskip=4, sticky=0.0):
        if game in RESERVED:
            raise ValueError(f"{game} is a reserved held-out game")
        self.game = game
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", int(seed))
        self.ale.setFloat("repeat_action_probability", float(sticky))
        self.ale.setInt("frame_skip", 1)
        self.ale.loadROM(roms.get_rom_path(game))
        self.actions = list(self.ale.getMinimalActionSet())
        self.n_actions = len(self.actions)
        self.frameskip = frameskip
        self.frames = 0            # every emulated frame, branches included
        self.loads = 0
        self.reset()

    def reset(self):
        self.ale.reset_game()
        self._true_score = 0.0
        self._last = self.ale.getScreenRGB()
        self._prev = self._last
        return self.obs()

    def obs(self):
        return np.maximum(self._prev, self._last)

    def step(self, a):
        """a: index into the minimal action set. Returns (obs, game_over). No reward is returned."""
        act = self.actions[a]
        for k in range(self.frameskip):
            self._true_score += self.ale.act(act)     # experimenter-side only
            self.frames += 1
            if k == self.frameskip - 2:
                self._prev = self.ale.getScreenRGB()
        self._last = self.ale.getScreenRGB()
        over = self.ale.game_over()
        return self.obs(), over

    def clone(self):
        return (self.ale.cloneState(), self._prev.copy(), self._last.copy(), self._true_score)

    def restore(self, s):
        self.ale.restoreState(s[0])
        self._prev, self._last, self._true_score = s[1].copy(), s[2].copy(), s[3]
        self.loads += 1
        self.frames += 60          # ARC4_VISION 2.5: fixed charge per savestate load

    # ---- experimenter side only ----------------------------------------------------------
    def ram(self):
        return self.ale.getRAM()

    def lives(self):
        return self.ale.lives()

    def true_score(self):
        return self._true_score
