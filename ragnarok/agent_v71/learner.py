"""One game, end to end: perceive -> explore -> self-imitate -> play live. Every emulated frame is
counted (branches, rewinds and loads included) so learning speed can be compared between a fresh
agent and an agent carrying a policy from earlier games.

The carried object is the policy's WEIGHTS only: an identity-free net over foreground views that
acts over canonical EFFECTS. Everything per game (background, counters, effect->button table,
archive, demonstration, buffers) is rebuilt at entry for every arm.
"""
import time

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.perceiver import Perceiver
from ragnarok.agent_v71.me_buttons import effect_table, EFFECTS
from ragnarok.agent_v71.policy import Policy, ego_view, EffectAdapter
from ragnarok.agent_v71.explore import Archive, me_cell
from ragnarok.agent_v71.hud import HudReader


class Agent:
    def __init__(self, policy=None):
        self.policy = policy or Policy(len(EFFECTS))

    # ------------------------------------------------------------------ entry
    def enter(self, env, rng):
        obs = env.reset()
        self.perc = perc = Perceiver(obs.shape[:2])
        perc.warm(env, rng)
        self.roles = perc.fit_counters(env, rng)
        obs = env.reset()
        perc.tracker.pos = None
        for _ in range(12):                                   # a few probes for the effect table
            perc.tracker.pos = None
            perc.locate(env, obs)
            for _ in range(6):
                obs, over = env.step(int(rng.integers(env.n_actions)))
                if over:
                    obs = env.reset()
        self.adapter = EffectAdapter(effect_table(perc.probes), EFFECTS)
        c = perc.counters or []
        self.s_idx = c.index(self.roles["score"]) if "score" in self.roles else None
        self.l_idx = c.index(self.roles["lives"]) if "lives" in self.roles else None

    # ------------------------------------------------------------------ acting
    def act(self, obs, me, prev, rng, temp=1.0, eps=0.02):
        img, pos, prev = ego_view(obs, self.perc.bg, me, prev)
        logits_mask = self.adapter.available
        e = self.policy.act(img, pos, rng, temp=temp, eps=eps, mask=logits_mask)
        return self.adapter.to_button(e, rng), e, img, pos, prev

    # ------------------------------------------------------------------ explore
    def explore(self, env, rng, iters, steps=100, repeat=0.95, guided=0.0):
        """Novelty x score drive with savestates. With probability `guided` a rollout's actions come
        from the current policy (what the agent already knows) instead of random repeats."""
        perc = self.perc
        arch = Archive()
        obs = env.reset()
        perc.tracker.pos = None
        me = perc.locate(env, obs)
        arch.add(me_cell(me, perc.hud_key(obs), obs.shape), env.clone(), 0, 0)
        first = arch.cells[next(iter(arch.cells))]
        first["frame"], first["traj"] = (obs, me), b""
        for _ in range(iters):
            k = arch.choose(rng)
            c = arch.cells[k]
            env.restore(c["state"])
            perc.cf.ring = []
            perc.hud.reset_tracking()
            perc.tracker.pos, perc.tracker.miss = c["frame"][1], 0
            depth, events = c["depth"], c["events"]
            traj = bytearray(c["traj"])
            use_pol = rng.random() < guided
            obs, prev = c["frame"][0], None
            a = int(rng.integers(env.n_actions))
            for _ in range(steps):
                me = perc.tracker.pos
                if use_pol:
                    a, _, _, _, prev = self.act(obs, me, prev, rng)
                elif rng.random() > repeat:
                    a = int(rng.integers(env.n_actions))
                obs, over = env.step(a)
                traj.append(a)
                depth += 1
                if over:
                    break
                me = perc.locate(env, obs)
                ev = perc.hud.step(obs)
                if self.s_idx is not None and self.s_idx in ev:
                    events += 1
                if self.l_idx is not None and self.l_idx in ev:
                    break
                key = me_cell(me, perc.hud_key(obs), obs.shape)
                cc = arch.cells.get(key)
                if cc is None or events > cc["events"] or (events == cc["events"] and depth < cc["depth"]):
                    if cc is not None:
                        del arch.cells[key]
                    arch.add(key, env.clone(), depth, events)
                    arch.cells[key]["frame"], arch.cells[key]["traj"] = (obs, me), bytes(traj)
        top = max(arch.cells.values(), key=lambda c: (c["events"], -c["depth"]))
        return list(top["traj"]), top["events"]

    # ------------------------------------------------------------------ self-imitation
    def learn(self, env, rng, demo, rounds=30, rollouts=16, horizon=400, bc_steps=1500):
        perc = self.perc
        obs = env.reset()
        perc.tracker.pos = None
        D, states, prev = [], [], None
        for i, a in enumerate(demo):
            me = perc.locate(env, obs)
            img, pos, prev = ego_view(obs, perc.bg, me, prev)
            D.append((img, pos, self.adapter.to_effect(a)))
            if i % 25 == 0:
                states.append((env.clone(), perc.tracker.pos, prev, obs))
            obs, over = env.step(a)
            if over:
                break
        S = list(D)
        if D:
            self.policy.fit(*zip(*S), steps=bc_steps)
        for _ in range(rounds):
            segs = []
            for _ in range(rollouts):
                st, mpos, prev, obs = states[int(rng.integers(len(states)))]
                env.restore(st)
                perc.tracker.pos, perc.tracker.miss = mpos, 0
                hud = HudReader(perc.counters or [])
                hud.step(obs)
                buf, last = [], 0
                for _ in range(horizon):
                    me = perc.tracker.update(obs) or perc.tracker.reacquire(obs)
                    a, e, img, pos, prev = self.act(obs, me, prev, rng)
                    buf.append((img, pos, e))
                    obs, over = env.step(a)
                    ev = hud.step(obs)
                    if self.s_idx is not None and self.s_idx in ev:
                        segs.append(buf[last:]); last = len(buf)
                    if over or (self.l_idx is not None and self.l_idx in ev):
                        break
            if segs:
                med = np.median([len(s) for s in segs])
                for s in segs:
                    if len(s) <= med:
                        S.extend(s)
            if S:
                self.policy.fit(*zip(*S[-60000:]), steps=300)

    # ------------------------------------------------------------------ live
    def play_live(self, game, seed, games=5, max_steps=7000):
        env = AtariEnv(game, seed=seed, sticky=0.25)
        rng = np.random.default_rng(seed)
        out = []
        start = None
        for _ in range(games):
            obs = env.reset()
            tr = self.perc.tracker
            tr.pos, tr.miss = start, 0
            prev = None
            for _ in range(max_steps):
                me = tr.update(obs) if tr.pos is not None else None
                if me is None:
                    me = tr.reacquire(obs)
                if start is None and me is not None:
                    start = me
                a, _, _, _, prev = self.act(obs, me, prev, rng)
                obs, over = env.step(a)
                if over:
                    break
            out.append(env.true_score())
        return out


def run_game(game, policy=None, seed=0, explore_iters=1500, guided=0.0, rounds=30, eval_games=5, log=print):
    """Fresh (policy=None) or experienced (policy carried) agent on one game. Returns a result dict
    with the frames spent per phase and the live scores."""
    t0 = time.perf_counter()
    env = AtariEnv(game, seed=seed)
    rng = np.random.default_rng(seed)
    ag = Agent(policy)
    ag.enter(env, rng)
    f_enter = env.frames
    demo, ev = ag.explore(env, rng, explore_iters, guided=guided)
    f_explore = env.frames
    ag.learn(env, rng, demo, rounds=rounds)
    f_learn = env.frames
    live = ag.play_live(game, seed=7, games=eval_games)
    res = dict(game=game, roles={k: list(v) for k, v in ag.roles.items()},
               effects=list(map(bool, ag.adapter.available)), demo_len=len(demo), demo_events=ev,
               frames_enter=f_enter, frames_explore=f_explore - f_enter, frames_learn=f_learn - f_explore,
               live=live, live_mean=float(np.mean(live)), secs=round(time.perf_counter() - t0))
    log(f"{game}: live {live} mean {res['live_mean']:.1f} | demo {len(demo)} steps {ev} events | "
        f"frames enter {f_enter} explore {res['frames_explore']} learn {res['frames_learn']} | {res['secs']}s")
    return res, ag
