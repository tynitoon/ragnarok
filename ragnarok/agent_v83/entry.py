"""Entry, identical for every arm (docs/ARC4_V83_PROTOCOL.md §1): background learned on 12,000 steps of
uniform play (the per-pixel mode), motor adapter on the sibling emulator, seeding of me (v73/v74:
several probe sightings, median size) and a check of how often my tracker loses me in ordinary play.
No counter search and no HUD key: the score and lives arrive as senses.
"""
import numpy as np

from ragnarok.agent_v71.tracker import MeTracker
from ragnarok.agent_v83.adapter import SignatureAdapter
from ragnarok.agent_v83.probe import probe
from scripts.ref_extract_v70 import ReferenceExtractor


class _Background(ReferenceExtractor):
    def rgb(self):
        return self._mode.view(np.uint8).reshape(*self._mode.shape, 4)[..., :3].copy()


class Body:
    """Where I am: background, tracker, probes. `probe_calls` and `probe_frames` are logged per arm."""

    def __init__(self):
        self.bg = self.tracker = self.adapter = self.my_size = None
        self.probe_calls = 0
        self.probe_frames = 0

    def enter(self, env, rng, seed, warm_steps=12000, seed_steps=3000, n_sightings=5, check_steps=1500):
        b0 = env.billed
        ref = _Background(max_colours=4096)
        obs, _ = env.reset()
        for _ in range(warm_steps):
            ref.update(obs)
            obs, _, _, end = env.step(int(rng.integers(env.n_actions)))
            if end:
                obs, _ = env.reset()
        self.bg = ref.rgb()
        del ref
        b1 = env.billed
        self.adapter = SignatureAdapter(lambda: env.sibling(991), self.bg, seed)
        b2 = env.billed
        self.tracker = tr = MeTracker(self.bg)
        obs, _ = env.reset()
        self.locate(env, obs)                         # seed my colours/size once
        a, sizes = 0, ([] if tr.size is None else [tr.size])
        for t in range(seed_steps):
            if len(sizes) >= n_sightings:
                break
            if rng.random() > 0.95:
                a = int(rng.integers(env.n_actions))
            obs, _, _, end = env.step(a)
            if end:
                obs, _ = env.reset()
            if t % 10 == 9:
                tr.pos, tr.size = None, None
                self.locate(env, obs)
                if tr.pos is not None and tr.size is None:
                    tr.update(obs)                    # nearest component when the probe's centre was a hole
                if tr.size is not None:
                    sizes.append(tr.size)
        self.my_size = tr.size = float(np.median(sizes)) if sizes else None   # fixed per run: the view's size
        obs, _ = env.reset()
        tr.pos, tr.miss = None, 0
        lost = []
        for _ in range(check_steps):
            me = self.track(obs)
            lost.append(me is None or tr.miss > 0)
            if rng.random() > 0.95:
                a = int(rng.integers(env.n_actions))
            obs, _, _, end = env.step(a)
            if end:
                obs, _ = env.reset()
                tr.pos, tr.miss = None, 0
        self.report = dict(entry_frames=env.billed - b0, warm_frames=b1 - b0, adapter_frames=b2 - b1,
                           my_sizes=[float(s) for s in sizes], my_size=self.my_size, lost_frac=float(np.mean(lost)) if lost else 1.0,
                           slots=self.adapter.mapping(), n_resp=self.adapter.n_resp)
        return self.report

    def locate(self, env, obs):
        """Probe from the current state if I am lost (billed, counted); then track. (x, y) or None."""
        tr = self.tracker
        if tr.lost:
            b = env.billed
            _, me = probe(env, self.bg)
            self.probe_calls += 1
            self.probe_frames += env.billed - b
            rm, rv = probe.last_reveal
            if rm.any():                              # repair the background under where I stood
                self.bg[rm] = rv[rm]
            if me is not None and tr.colours is None and probe.last_colours:
                tr.set_colours(probe.last_colours)
            if me is not None:
                tr.seed(me, obs)
            return me
        return tr.update(obs)

    def track(self, obs):
        """Live-play perception, no probe: tracker update, else whole-screen reacquire (v74 `_track`)."""
        tr = self.tracker
        return (tr.update(obs) if tr.pos is not None else None) or tr.reacquire(obs)
