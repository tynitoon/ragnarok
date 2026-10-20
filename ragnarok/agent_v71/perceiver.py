"""The agent's perception stack, screen bytes only: background, me, counters.

Built online from the agent's own experience:
  background  = per-pixel mode (ReferenceExtractor, used here as the disclosed shared
                scaffold per ARC4_VISION 5 fallback; not learned perception)
  me          = savestate probes (me_buttons.probe) then per-step tracking
  counters    = CounterFinder statistics, refit periodically into HudReader regions
"""
import numpy as np

from ragnarok.agent_v71.me_buttons import probe
from ragnarok.agent_v71.tracker import MeTracker
from ragnarok.agent_v71.hud import CounterFinder, HudReader
from scripts.ref_extract_v70 import ReferenceExtractor


class Perceiver:
    def __init__(self, shape):
        self.ref = ReferenceExtractor()
        self.cf = CounterFinder(shape)
        self.hud = HudReader([])
        self.tracker = None
        self.bg = None
        self.probes = []
        self.counters = None
        self._reader = None

    def background(self):
        m = self.ref._mode
        return m.view(np.uint8).reshape(*m.shape, 4)[..., :3]

    def observe(self, obs):
        """Ordinary play only (never call on branch frames). Background every 8th frame."""
        self._n = getattr(self, "_n", 0) + 1
        if self._n % 8 == 0 or self.bg is None:
            self.ref.update(obs)
        self.cf.update(obs)

    def warm(self, env, rng, steps=1500):
        obs = env.reset()
        for _ in range(steps):
            self.observe(obs)
            obs, over = env.step(int(rng.integers(env.n_actions)))
            if over:
                obs = env.reset()
        self.bg = self.background()
        self.tracker = MeTracker(self.bg)
        return obs

    def locate(self, env, obs):
        """Probe from the current state if me is lost; then track. Returns (x, y) or None."""
        if self.tracker.lost:
            out, me = probe(env, self.bg)
            self.probes.append(out)
            if me is not None:
                self.tracker.seed(me, obs)
            return me
        return self.tracker.update(obs)

    def fit_counters(self, env, rng, steps=20000, repeat=0.95):
        """Ordinary play with action repeats (no reward), CounterFinder, then a reader pass with
        per-game tallies -> roles. The counters (score, lives) are fixed afterwards."""
        obs = env.reset()
        a = 0
        cf = CounterFinder(obs.shape[:2])
        for _ in range(steps):
            cf.update(obs); self.observe(obs)
            if rng.random() > repeat:
                a = int(rng.integers(env.n_actions))
            obs, over = env.step(a)
            if over:
                obs = env.reset(); cf.ring = []
        reader = HudReader(cf.fit())
        for _ in range(steps):
            reader.step(obs)
            if rng.random() > repeat:
                a = int(rng.integers(env.n_actions))
            obs, over = env.step(a)
            if over:
                obs = env.reset(); reader.reset_tracking()
        roles = reader.assign_roles()
        self.roles = {v: reader.regions[k] for k, v in roles.items()}
        self.counters = [self.roles[r] for r in ("score", "lives") if r in self.roles]
        self.hud = HudReader(self.counters)
        self.bg = self.background()
        self.tracker = MeTracker(self.bg)
        return self.roles

    def refit_hud(self):
        regions = self.cf.fit()
        self.hud = HudReader(regions)
        return regions

    def refit_counters(self, arch=None):
        """Refit candidate regions, replay the stored cell frames through a reader to keep only
        regions that behave like counters (monotone: changes show new states), then key on those."""
        regions = self.cf.fit()
        keep = []                                  # no counter is trusted before its statistics
        if self.counters is not None and self._reader is not None:
            mono = [self._reader.novel_in_game[i] / self._reader.events[i] if self._reader.events[i] else 0
                    for i in range(len(self._reader.regions))]
            ok = {tuple(r) for r, m, e in zip(self._reader.regions, mono, self._reader.events) if e >= 2 and m >= 0.9}
            keep = [r for r in regions if tuple(r) in ok]
        self._reader = HudReader(regions)          # statistics for the next refit
        self.hud = self._reader
        self.counters = keep
        return keep

    def hud_key(self, obs):
        regs = self.counters or []
        return tuple(self.hud._hash(obs, r) for r in regs)
