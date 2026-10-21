"""One game, end to end: perceive -> explore -> self-imitate -> play live. Every emulated frame is
counted (branches, rewinds, loads and the adapter's probes included) so learning speed can be
compared between a fresh agent and an agent carrying a policy from earlier games.

The carried object is the policy's WEIGHTS only: an identity-free net over foreground views that
acts over canonical SLOTS (ragnarok.agent_v71.adapter). Everything per game (background, counters,
slot->button table, archive, demonstration, buffers) is rebuilt at entry for every arm.

Revision after docs/ARC4_V71_DIAGNOSIS.md (2026-10-21):
  D1 masked fit;  D2/D7 SignatureAdapter (no button lost or merged, deterministic button);
  D3 the demonstration is replayed from the explorer's ROOT savestate, not a fresh reset;
  D4 the demonstration is perceived exactly as in live play (tracker, no probes, frozen background);
  D5 self-imitation by BRANCH RANKING ("bri"): from states the policy itself visits, every available
     slot is held for `hold` steps (optionally followed by `cont` policy steps), valued by the agent's
     own HUD score events minus a life-loss penalty, and the best branch is imitated. The previous
     segment rule is kept as method="segments" for comparison.
"""
import copy
import time

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.perceiver import Perceiver
from ragnarok.agent_v71.adapter import SignatureAdapter, SLOTS
from ragnarok.agent_v71.policy import Policy, ego_view
from ragnarok.agent_v71.explore import Archive, me_cell
from ragnarok.agent_v71.hud import HudReader


class Agent:
    def __init__(self, policy=None, vanish_k=None):
        self.policy = policy or Policy(len(SLOTS))
        self.frames_adapter = 0
        self.vanish_k = vanish_k            # v73: "I vanished for k steps" counts as a life lost
        self.gone = 0                       # consecutive steps my tracker found me nowhere
        self.base_slot = None               # v74: base policy of the rollout improvement
        self.live_temp = 1.0

    # ------------------------------------------------------------------ entry
    def enter(self, env, rng, seed=0, seed_steps=3000, n_sightings=5, check_steps=1500, max_lost=0.2):
        obs = env.reset()
        self.perc = perc = Perceiver(obs.shape[:2])
        perc.warm(env, rng)
        self.roles = perc.fit_counters(env, rng)
        self.adapter = SignatureAdapter(env.game, perc.bg, seed)
        self.frames_adapter = self.adapter.frames_used
        c = perc.counters or []
        self.s_idx = c.index(self.roles["score"]) if "score" in self.roles else None
        self.l_idx = c.index(self.roles["lives"]) if "lives" in self.roles else None
        perc.ref = None                     # entry statistics are not needed any more (RAM: ~0.5 GB)
        perc.cf = None
        obs = env.reset()
        perc.tracker.pos = None
        perc.locate(env, obs)               # seed my colours/size once
        # v73: where the first frames ignore every button (intros), the probe at reset finds nobody
        # and the tracker was never seeded (Name This Game: lost 100% of the time in v72). Keep
        # playing ordinarily and probing until several probes have found me; my size is the median
        # of those sightings (one sighting can merge me with a neighbour: 704 px for a ~100 px diver).
        a, tr = 0, perc.tracker
        sizes = [] if tr.size is None else [tr.size]
        for t in range(seed_steps):
            if len(sizes) >= n_sightings:
                break
            if rng.random() > 0.95:
                a = int(rng.integers(env.n_actions))
            obs, over = env.step(a)
            if over:
                obs = env.reset()
            if t % 10 == 9:
                tr.pos, tr.size = None, None
                perc.locate(env, obs)
                if tr.pos is not None and tr.size is None:
                    tr.update(obs)          # nearest component when the probe's centre was a hole
                if tr.size is not None:
                    sizes.append(tr.size)
        tr.size = float(np.median(sizes)) if sizes else None
        self.my_sizes = sizes
        # v73: can I trust "I vanished" as a death signal here? Only if, in ordinary play, my own
        # tracker (as used live: no probes) keeps me most of the time.
        obs = env.reset()
        tr.pos, tr.miss, self.gone = None, 0, 0
        lost = []
        for t in range(check_steps):
            me = self._track(obs)
            lost.append(me is None or tr.miss > 0)
            if rng.random() > 0.95:
                a = int(rng.integers(env.n_actions))
            obs, over = env.step(a)
            if over:
                obs = env.reset()
                tr.pos, tr.miss = None, 0
        self.lost_frac = float(np.mean(lost)) if lost else 1.0
        if self.vanish_k is not None and self.lost_frac > max_lost:
            self.vanish_k = None
        obs = env.reset()
        tr.pos, tr.miss, self.gone = None, 0, 0
        perc.locate(env, obs)

    @property
    def mask(self):
        return self.adapter.available

    # ------------------------------------------------------------------ acting
    def act(self, obs, me, prev, rng, temp=1.0, eps=0.02):
        """v74: with `commit` > 1 a decision is held for `commit` steps (the hold over which the
        rollout improvement measured it); `_held` is reset at the start of every trajectory."""
        img, pos, prev = ego_view(obs, self.perc.bg, me, prev)
        k, h = getattr(self, "commit", 1), getattr(self, "_held", None)
        if k > 1 and h is not None and h[1] > 0:
            e = h[0]
            self._held = (e, h[1] - 1)
        else:
            e = self.policy.act(img, pos, rng, temp=temp, eps=eps, mask=self.mask)
            self._held = (e, k - 1) if k > 1 else None
        return self.adapter.to_button(e), e, img, pos, prev

    def _track(self, obs):
        tr = self.perc.tracker
        me = (tr.update(obs) if tr.pos is not None else None) or tr.reacquire(obs)
        self.gone = self.gone + 1 if (me is None or tr.miss > 0) else 0
        return me

    def _dead(self, ev):
        """A life lost, from the screen only: an event on my lives counter, or (v73) I vanished
        for `vanish_k` consecutive steps (most games show no lives counter the finder can read)."""
        if self.l_idx is not None and self.l_idx in ev:
            return True
        return self.vanish_k is not None and self.gone >= self.vanish_k

    # ------------------------------------------------------------------ explore
    def explore(self, env, rng, iters, steps=100, repeat=0.95, guided=0.0):
        """Novelty x score drive with savestates. Returns (demo actions, events, root savestate)."""
        perc = self.perc
        arch = Archive()
        obs = env.reset()
        root = env.clone()
        perc.tracker.pos = None
        me = perc.locate(env, obs)
        arch.add(me_cell(me, perc.hud_key(obs), obs.shape), env.clone(), 0, 0)
        first = arch.cells[next(iter(arch.cells))]
        first["frame"], first["traj"] = (None, me), b""
        for _ in range(iters):
            k = arch.choose(rng)
            c = arch.cells[k]
            env.restore(c["state"])
            perc.hud.reset_tracking()
            perc.tracker.pos, perc.tracker.miss = c["frame"][1], 0
            self.gone = 0
            depth, events = c["depth"], c["events"]
            traj = bytearray(c["traj"])
            use_pol = rng.random() < guided
            obs, prev = env.obs(), None
            a = int(rng.integers(env.n_actions))
            for _ in range(steps):
                if use_pol:
                    a, _, _, _, prev = self.act(obs, perc.tracker.pos, prev, rng)
                elif rng.random() > repeat:
                    a = int(rng.integers(env.n_actions))
                obs, over = env.step(a)
                traj.append(a)
                depth += 1
                if over:
                    break
                # v74 fix: the same perception as live play (tracker update, else whole-screen
                # reacquire), so "I vanished" is the signal validated by the pre-check. v73 used the
                # probe path here (update only, no reacquire): a car over the chicken for 2 steps read
                # as a death, every Freeway trajectory was cut and the demo came out empty.
                me = self._track(obs)
                if perc.tracker.lost:
                    found = perc.locate(env, obs)
                    if found is not None:          # the probe sees me: I did not vanish
                        me, self.gone = found, 0
                ev = perc.hud.step(obs)
                if self.s_idx is not None and self.s_idx in ev:
                    events += 1
                if self._dead(ev):
                    break
                key = me_cell(me, perc.hud_key(obs), obs.shape)
                cc = arch.cells.get(key)
                if cc is None or events > cc["events"] or (events == cc["events"] and depth < cc["depth"]):
                    if cc is not None:
                        del arch.cells[key]
                    arch.add(key, env.clone(), depth, events)
                    arch.cells[key]["frame"], arch.cells[key]["traj"] = (None, me), bytes(traj)
        top = max(arch.cells.values(), key=lambda c: (c["events"], -c["depth"]))
        return list(top["traj"]), top["events"], root

    # ------------------------------------------------------------------ demonstration
    def replay_demo(self, env, demo, root, every=25):
        """D3 + D4: replay from the root savestate, perceived as in live play (tracker only)."""
        perc = self.perc
        env.restore(root)
        obs = env.obs()
        perc.tracker.pos, perc.tracker.miss = None, 0
        self.gone = 0
        D, states, prev = [], [], None
        for i, a in enumerate(demo):
            me = self._track(obs)
            p0 = prev                       # the state stores the views BEFORE this step's update
            img, pos, prev = ego_view(obs, perc.bg, me, prev)
            D.append((img.astype(np.uint8), pos, self.adapter.to_effect(a)))
            if i % every == 0:
                hud = HudReader(perc.counters or []); hud.step(obs)
                states.append((env.clone(), perc.tracker.pos, p0, obs, hud, 0))
            obs, over = env.step(a)
            if over:
                break
        if not states:                      # v74: no demonstration (no event found): start from the root
            env.restore(root)
            obs = env.obs()
            perc.tracker.pos, perc.tracker.miss = None, 0
            self._track(obs)
            hud = HudReader(perc.counters or []); hud.step(obs)
            states.append((env.clone(), perc.tracker.pos, None, obs, hud, 0))
        return D, states

    # ------------------------------------------------------------------ self-imitation
    def learn(self, env, rng, demo, root, method="bri", rounds=12, bc_steps=1500, **kw):
        D, states = self.replay_demo(env, demo, root)
        if not states:
            return dict(rounds=[])
        if D:
            self.policy.fit(*zip(*D), steps=bc_steps, mask=self.mask)
        elif method != "rpi":               # branch ranking/segments need a demonstration to start
            return dict(rounds=[])
        if method == "segments":
            return self._learn_segments(env, rng, D, states, rounds, **kw)
        if method == "rpi":
            return self._learn_rpi(env, rng, D, states, rounds, **kw)
        return self._learn_bri(env, rng, D, states, rounds, **kw)

    def _rollout_roots(self, env, rng, states, n_roll=3, length=300, every=12):
        perc, roots = self.perc, []
        for _ in range(n_roll):
            st, mpos, prev, obs, hud0, gone0 = states[int(rng.integers(len(states)))]
            env.restore(st)
            perc.tracker.pos, perc.tracker.miss = mpos, 0
            hud, self.gone = copy.deepcopy(hud0), gone0
            for k in range(length):
                if k % every == 0:          # the root carries the rollout's reader and vanish count
                    roots.append((env.clone(), perc.tracker.pos, prev, obs, copy.deepcopy(hud), self.gone))
                me = self._track(obs)
                a, _, _, _, prev = self.act(obs, me, prev, rng)
                obs, over = env.step(a)
                ev = hud.step(obs)
                if over or self._dead(ev):
                    break
        return roots

    def _branch_value(self, env, root, slot, hold, cont, rng, death_penalty=2.0):
        perc = self.perc
        st, mpos, prev, obs, hud0, gone0 = root
        env.restore(st)
        perc.tracker.pos, perc.tracker.miss = mpos, 0
        hud, self.gone = copy.deepcopy(hud0), gone0
        v, seg = 0.0, []
        for k in range(hold + cont):
            me = self._track(obs)
            if k < hold:
                img, pos, prev = ego_view(obs, perc.bg, me, prev)
                seg.append((img.astype(np.uint8), pos))
                a = self.adapter.to_button(slot)
            else:
                a, _, _, _, prev = self.act(obs, me, prev, rng)
            obs, over = env.step(a)
            ev = hud.step(obs)
            if self.s_idx is not None and self.s_idx in ev:
                v += 1.0
            if over or self._dead(ev):
                v -= death_penalty
                break
        return v, seg

    def _learn_bri(self, env, rng, D, states, rounds, hold=32, cont=0, n_roots=24, n_label=4,
                   demo_weight=0.2, buffer=20000, log=None):
        choices = [i for i in range(len(SLOTS)) if self.mask[i]]
        L, hist = [], []
        for r in range(rounds):
            roots = self._rollout_roots(env, rng, states)
            sel = [roots[int(i)] for i in rng.choice(len(roots), size=min(n_roots, len(roots)), replace=False)]
            labelled, ties, lab = 0, 0, np.zeros(len(SLOTS), int)
            for root in sel:
                vals, segs = {}, {}
                crn = int(rng.integers(2 ** 31))   # common random numbers for the continuations
                for c in choices:
                    vals[c], segs[c] = self._branch_value(env, root, c, hold, cont, np.random.default_rng(crn))
                vv = np.array([vals[c] for c in choices])
                if vv.max() == vv.min():
                    ties += 1
                    continue
                best = [c for c in choices if vals[c] == vv.max()]
                b = best[int(rng.integers(len(best)))]
                for img, pos in segs[b][:n_label]:
                    L.append((img, pos, b))
                labelled += 1; lab[b] += 1
            L = L[-buffer:]
            T = D + L
            w = np.r_[np.full(len(D), demo_weight), np.ones(len(L))]
            self.policy.fit(*zip(*T), steps=300, mask=self.mask, weights=w)
            rec = dict(round=r, roots=len(sel), labelled=labelled, ties=ties,
                       labels={SLOTS[i]: int(lab[i]) for i in np.nonzero(lab)[0]}, frames=env.frames)
            hist.append(rec)
            if log:
                log(f"   bri r{r:02d} labelled {labelled}/{len(sel)} ties {ties} labels {rec['labels']} frames {env.frames}")
        return dict(rounds=hist)

    # ------------------------------------------------------------------ v74: rollout improvement
    def _base_act(self, obs, me, prev, rng):
        """The base policy: the agent's best constant slot, or (iterated) its own greedy policy."""
        if self.base_slot is not None:
            _, _, prev = ego_view(obs, self.perc.bg, me, prev)
            return self.adapter.to_button(self.base_slot), self.base_slot, prev
        a, e, _, _, prev = self.act(obs, me, prev, rng, temp=0.05, eps=0.0)
        return a, e, prev

    def _q(self, env, root, slot, hold, horizon, rng, death_penalty):
        """Hold `slot` for `hold` steps, then follow the base policy for `horizon` steps.
        Value = my score events - death_penalty if I die (the branch ends there). Screen only."""
        perc = self.perc
        st, mpos, prev, obs, hud0, gone0 = root
        env.restore(st)
        perc.tracker.pos, perc.tracker.miss = mpos, 0
        hud, self.gone = copy.deepcopy(hud0), gone0
        self._held = None
        v, seg = 0.0, []
        for k in range(hold + horizon):
            me = self._track(obs)
            if k < hold:
                img, pos, prev = ego_view(obs, perc.bg, me, prev)
                seg.append((img.astype(np.uint8), pos))
                a = self.adapter.to_button(slot)
            else:
                a, _, prev = self._base_act(obs, me, prev, rng)
            obs, over = env.step(a)
            ev = hud.step(obs)
            if self.s_idx is not None and self.s_idx in ev:
                v += 1.0
            if over or self._dead(ev):
                v -= death_penalty
                break
        return v, seg

    def _base_roots(self, env, rng, states, n_roll, length=300, every=6, own=False, mix=0.15):
        """Roots for the improvement step: states the base policy visits (round 0) or, `own`, the
        states my current greedy policy visits (DAgger: the improved policy is the fixed expert,
        the data follow the learner's own distribution)."""
        perc, roots, seen = self.perc, [], set()
        for _ in range(n_roll):
            st, mpos, prev, obs, hud0, gone0 = states[int(rng.integers(len(states)))]
            env.restore(st)
            perc.tracker.pos, perc.tracker.miss = mpos, 0
            hud, self.gone = copy.deepcopy(hud0), gone0
            self._held = None
            for k in range(length):
                if k % every == 0:
                    key = hash(obs.tobytes())       # the emulator is deterministic: skip seen screens
                    if key not in seen:
                        seen.add(key)
                        roots.append((env.clone(), perc.tracker.pos, prev, obs, copy.deepcopy(hud), self.gone))
                me = self._track(obs)
                if own:
                    a, _, _, _, prev = self.act(obs, me, prev, rng, temp=0.05, eps=0.0)
                else:
                    a, _, prev = self._base_act(obs, me, prev, rng)
                if rng.random() < mix:             # diversity: the emulator is deterministic
                    a = self.adapter.to_button(int(rng.choice(np.nonzero(self.mask)[0])))
                obs, over = env.step(a)
                ev = hud.step(obs)
                if over or self._dead(ev):
                    break
        return roots

    def _learn_rpi(self, env, rng, D, states, rounds, hold=8, horizon=64, n_roots=96, n_label=4,
                   death_penalty=2.0, base_roots=16, iterate=False, buffer=40000, live_temp=0.05,
                   fit_mode="label", fit_steps=300, check_roots=0, commit=True, log=None):
        """Rollout policy improvement (v74). The agent first finds its own best constant slot
        (each slot held hold+horizon steps from demo states, valued by its HUD). Every round, from
        states the base policy visits, each slot is tried for `hold` steps followed by the base
        policy; the best slot is the label, the base's own slot on ties (so the distilled policy
        equals the base wherever no deviation helps). Improving on a base policy cannot, with exact
        values, do worse than it. With `iterate`, the base becomes the greedy distilled policy."""
        choices = [i for i in range(len(SLOTS)) if self.mask[i]]
        self.base_slot = None
        self.commit = hold if commit else 1   # act as measured: a decision lasts `hold` steps
        # 1. the agent's own best constant slot
        sel = [states[int(i)] for i in rng.choice(len(states), size=min(base_roots, len(states)), replace=False)]
        cv = {}
        for c in choices:
            self.base_slot = c
            cv[c] = float(np.mean([self._q(env, r, c, hold, horizon, rng, death_penalty)[0] for r in sel]))
        best_v = max(cv.values())
        self.base_slot = min(c for c in choices if cv[c] == best_v)
        const_slot = self.base_slot
        if log:
            log(f"   rpi base: constant slot {SLOTS[const_slot]} | values { {SLOTS[c]: round(v, 2) for c, v in cv.items()} }")
        L, LQ, hist = [], [], []
        for r in range(rounds):
            roots = self._base_roots(env, rng, states, n_roll=max(3, n_roots // 24), own=r > 0)
            sel = [roots[int(i)] for i in rng.choice(len(roots), size=min(n_roots, len(roots)), replace=False)]
            moved, lab, gain = 0, np.zeros(len(SLOTS), int), []
            for root in sel:
                # the base's own choice at this root (before the branches; nothing held)
                if self.base_slot is not None:
                    b0 = self.base_slot
                else:
                    self._held = None
                    self.perc.tracker.pos, self.perc.tracker.miss = root[1], 0
                    me = self._track(root[3])
                    _, b0, _ = self._base_act(root[3], me, root[2], rng)
                q, segs = {}, {}
                for c in choices:
                    q[c], segs[c] = self._q(env, root, c, hold, horizon, rng, death_penalty)
                top = max(q.values())
                b = b0 if q.get(b0, -1e9) == top else min(c for c in choices if q[c] == top)
                moved += int(b != b0); lab[b] += 1
                gain.append(top - q.get(b0, top))
                for img, pos in segs[b][:n_label]:
                    L.append((img, pos, b))
                qv = np.zeros(len(SLOTS), np.float32)
                for c in choices:
                    qv[c] = q[c]
                img0, pos0 = segs[choices[0]][0]          # the root's own view (same in every branch)
                LQ.append((img0, pos0, qv))
            L, LQ = L[-buffer:], LQ[-buffer:]
            if fit_mode == "q":
                self.policy.fit_q(*zip(*LQ), steps=fit_steps, mask=self.mask)
            else:
                self.policy.fit(*zip(*L), steps=fit_steps, mask=self.mask)
            if iterate:
                self.base_slot = None
            rec = dict(round=r, roots=len(sel), deviations=moved, mean_gain=round(float(np.mean(gain)), 3),
                       labels={SLOTS[i]: int(lab[i]) for i in np.nonzero(lab)[0]}, frames=env.frames)
            if check_roots:                                # agent-side: is my policy better than my base?
                rec["check"] = self._policy_check(env, rng, states, check_roots, hold + horizon, death_penalty, const_slot)
            hist.append(rec)
            if log:
                log(f"   rpi r{r:02d} roots {len(sel)} deviations {moved} gain {rec['mean_gain']} labels {rec['labels']} "
                    f"{rec.get('check', '')} frames {env.frames}")
        self.base_slot = None
        self.live_temp = live_temp
        return dict(rounds=hist, const_slot=SLOTS[const_slot], const_values={SLOTS[c]: v for c, v in cv.items()})

    def _policy_check(self, env, rng, states, n, length, dp, const_slot):
        """My own measure (HUD events - dp x my deaths) of my greedy policy vs my constant base, over
        `length` steps from the same `n` demo states. Charged frames; screen only."""
        sel = [states[int(i)] for i in rng.choice(len(states), size=min(n, len(states)), replace=False)]
        keep = self.base_slot
        out = {}
        for name in ("policy", "const"):
            vals = []
            for root in sel:
                self.base_slot = None if name == "policy" else const_slot
                st, mpos, prev, obs, hud0, gone0 = root
                env.restore(st)
                self.perc.tracker.pos, self.perc.tracker.miss = mpos, 0
                hud, self.gone = copy.deepcopy(hud0), gone0
                self._held = None
                v = 0.0
                for _ in range(length):
                    me = self._track(obs)
                    a, _, prev = self._base_act(obs, me, prev, rng)
                    obs, over = env.step(a)
                    ev = hud.step(obs)
                    if self.s_idx is not None and self.s_idx in ev:
                        v += 1.0
                    if over or self._dead(ev):
                        v -= dp
                        break
                vals.append(v)
            out[name] = round(float(np.mean(vals)), 2)
        self.base_slot = keep
        return out

    def _learn_segments(self, env, rng, D, states, rounds, rollouts=16, horizon=400, log=None):
        perc, S = self.perc, list(D)
        for _ in range(rounds):
            segs = []
            for _ in range(rollouts):
                st, mpos, prev, obs, hud0, gone0 = states[int(rng.integers(len(states)))]
                env.restore(st)
                perc.tracker.pos, perc.tracker.miss = mpos, 0
                hud, self.gone = copy.deepcopy(hud0), gone0
                buf, last = [], 0
                for _ in range(horizon):
                    me = self._track(obs)
                    a, e, img, pos, prev = self.act(obs, me, prev, rng)
                    buf.append((img.astype(np.uint8), pos, e))
                    obs, over = env.step(a)
                    ev = hud.step(obs)
                    if self.s_idx is not None and self.s_idx in ev:
                        segs.append(buf[last:]); last = len(buf)
                    if over or self._dead(ev):
                        break
            if segs:
                med = np.median([len(s) for s in segs])
                S.extend(x for s in segs if len(s) <= med for x in s)
            self.policy.fit(*zip(*S[-60000:]), steps=300, mask=self.mask)
        return dict(rounds=[])

    # ------------------------------------------------------------------ live
    def play_live(self, game, seed, games=10, max_steps=7000, temp=None, eps=0.02):
        """temp None: the agent's own setting (1.0 unless its learning method chose otherwise:
        rollout improvement distils argmax labels, so it plays its policy greedily)."""
        if temp is None:
            temp = getattr(self, "live_temp", 1.0)
        env = AtariEnv(game, seed=seed, sticky=0.25)
        rng = np.random.default_rng(seed)
        out, counts = [], np.zeros(env.n_actions, int)
        start = None
        for _ in range(games):
            obs = env.reset()
            tr = self.perc.tracker
            tr.pos, tr.miss = start, 0
            prev, self._held = None, None
            for _ in range(max_steps):
                me = self._track(obs)
                if start is None and me is not None:
                    start = me
                a, _, _, _, prev = self.act(obs, me, prev, rng, temp=temp, eps=eps)
                counts[a] += 1
                obs, over = env.step(a)
                if over:
                    break
            out.append(env.true_score())
        return out, (counts / max(1, counts.sum())).round(3).tolist()


def run_game(game, policy=None, seed=0, explore_iters=1500, guided=0.0, rounds=12, method="bri",
             eval_seeds=(7, 8), eval_games=10, log=print, **learn_kw):
    """Fresh (policy=None) or experienced (policy carried) agent on one game."""
    t0 = time.perf_counter()
    env = AtariEnv(game, seed=seed)
    rng = np.random.default_rng(seed)
    ag = Agent(policy, vanish_k=2)
    ag.enter(env, rng, seed=seed)
    f_enter = env.frames + ag.frames_adapter
    if explore_iters > 0:
        demo, ev, root = ag.explore(env, rng, explore_iters, guided=guided)
    else:
        demo, ev, root = [], 0, None
    f_explore = env.frames + ag.frames_adapter
    hist = ag.learn(env, rng, demo, root, method=method, rounds=rounds, log=log, **learn_kw) if (demo or method == "rpi") else {}
    f_learn = env.frames + ag.frames_adapter
    live, mix = {}, {}
    for s in eval_seeds:
        live[s], mix[s] = ag.play_live(game, seed=s, games=eval_games)
    allv = [x for s in eval_seeds for x in live[s]]
    res = dict(game=game, seed=seed, method=method, slots=ag.adapter.mapping(), adapter_info=ag.adapter.info,
               roles={k: list(v) for k, v in ag.roles.items()}, demo_len=len(demo), demo_events=ev,
               frames_enter=f_enter, frames_explore=f_explore - f_enter, frames_learn=f_learn - f_explore,
               live=live, button_mix=mix, live_mean=float(np.mean(allv)),
               live_se=float(np.std(allv, ddof=1) / np.sqrt(len(allv))) if len(allv) > 1 else float("nan"),
               learn=hist, secs=round(time.perf_counter() - t0))
    log(f"{game} seed {seed} {method}: live mean {res['live_mean']:.1f} ± {res['live_se']:.1f} "
        f"({len(allv)} games) | slots {res['slots']} | demo {len(demo)} steps {ev} events | "
        f"frames enter {f_enter} explore {res['frames_explore']} learn {res['frames_learn']} | {res['secs']}s")
    return res, ag
