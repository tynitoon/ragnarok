"""Milestone A, step 4 (exploratory): from the explorer's demonstration to a policy that plays LIVE.

1. Replay the explorer's best trajectory (seed 0, deterministic) and record ego views + actions.
2. Behaviour cloning on the demonstration.
3. Backward self-imitation (Salimans & Chen 2018 style, our signal): start the policy from demo
   savestates, latest first; its own rollouts that reach a score event (the agent's reader) are
   added to the training set; move the start earlier once it succeeds from there.
4. LIVE evaluation: other seed, sticky actions 0.25, no savestates, full games; experimenter-side
   true score vs random play with action repeats under the same conditions.

Usage: C:/Python314/python.exe -m scripts.slice_play_v71 --game freeway
"""
import argparse
import json
import time

import numpy as np
import torch

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.perceiver import Perceiver
from ragnarok.agent_v71.policy import Policy, ego_view
from ragnarok.agent_v71.hud import HudReader


def live_episode(env, perc, pol, rng, max_steps=7000, me_size=None, start_pos=None):
    obs = env.reset()
    tr = perc.tracker
    tr.pos, tr.miss = start_pos, 0
    prev = None
    for _ in range(max_steps):
        me = tr.update(obs) if tr.pos is not None else None
        if me is None:
            me = tr.reacquire(obs)
        img, pos, prev = ego_view(obs, perc.bg, me, prev)
        a = pol.act(img, pos, rng) if pol is not None else None
        obs, over = env.step(a)
        if over:
            break
    return env.true_score()


def random_repeat_episode(env, rng, max_steps=7000, repeat=0.95):
    env.reset()
    a = 0
    for _ in range(max_steps):
        if rng.random() > repeat:
            a = int(rng.integers(env.n_actions))
        _, over = env.step(a)
        if over:
            break
    return env.true_score()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="freeway")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--eval-games", type=int, default=5)
    a = ap.parse_args()
    t0 = time.perf_counter()
    demo = json.load(open(f"craft_v6_out/v71_demo_{a.game}.json"))
    env = AtariEnv(a.game, seed=demo["seed"])
    rng = np.random.default_rng(1)
    obs = env.reset()
    perc = Perceiver(obs.shape[:2])
    perc.warm(env, np.random.default_rng(0))
    roles = perc.fit_counters(env, np.random.default_rng(0))
    s_idx = perc.counters.index(roles["score"]) if "score" in roles else None
    l_idx = perc.counters.index(roles["lives"]) if "lives" in roles else None
    print(f"{a.game}: counters {roles} | demo {len(demo['traj'])} steps, true score {demo['true_score']}", flush=True)

    # 1. replay the demonstration
    obs = env.reset()
    perc.tracker.pos = None
    D_img, D_pos, D_act, states, prev = [], [], [], [], None
    start_pos = None
    for i, act in enumerate(demo["traj"]):
        me = perc.locate(env, obs)
        if i == 0:
            start_pos = me
        img, pos, prev = ego_view(obs, perc.bg, me, prev)
        D_img.append(img); D_pos.append(pos); D_act.append(act)
        if i % 25 == 0:
            states.append((i, env.clone(), perc.tracker.pos, prev))
        obs, over = env.step(act)
        if over:
            break
    print(f"   demo replay true score {env.true_score()} (explorer recorded {demo['true_score']})", flush=True)

    # 2. behaviour cloning
    pol = Policy(env.n_actions)
    pol.fit(D_img, D_pos, D_act, steps=1500)

    # 3. backward self-imitation from demo savestates
    S_img, S_pos, S_act = list(D_img), list(D_pos), list(D_act)
    k = len(states) - 1
    for rnd in range(a.rounds):
        i0, st, mpos, prev0 = states[k]
        succ = 0
        wins = []
        for _ in range(a.rollouts):
            env.restore(st)
            perc.tracker.pos, perc.tracker.miss = mpos, 0
            hud = HudReader(perc.counters)
            obs = env.obs()
            hud.step(obs)
            prev = prev0
            buf, got = [], False
            for t in range(400):
                me = perc.tracker.update(obs) or perc.tracker.reacquire(obs)
                img, pos, prev = ego_view(obs, perc.bg, me, prev)
                act = pol.act(img, pos, rng)
                buf.append((img, pos, act))
                obs, over = env.step(act)
                ev = hud.step(obs)
                if l_idx is not None and l_idx in ev or over:
                    break
                if s_idx is not None and s_idx in ev:
                    got = True
                    break
            if got:
                succ += 1
                wins.append(buf)
        # self-imitation of the BETTER half: only successes at least as fast as the median one
        if wins:
            med = np.median([len(b) for b in wins])
            for buf in wins:
                if len(buf) <= med:
                    for img, pos, act in buf:
                        S_img.append(img); S_pos.append(pos); S_act.append(act)
        loss = pol.fit(S_img[-60000:], S_pos[-60000:], S_act[-60000:], steps=300)
        if succ >= a.rollouts // 2 and k > 0:
            k = max(0, k - 2)
        print(f"   round {rnd}: start step {i0} success {succ}/{a.rollouts} data {len(S_act)} loss {loss:.3f} | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    # 4. live evaluation, other seed, sticky actions, no savestates
    ev_env = AtariEnv(a.game, seed=7, sticky=0.25)
    ev_rng = np.random.default_rng(7)
    pol_scores = [live_episode(ev_env, perc, pol, ev_rng, start_pos=start_pos) for _ in range(a.eval_games)]
    rnd_scores = [random_repeat_episode(ev_env, ev_rng) for _ in range(a.eval_games)]
    res = dict(game=a.game, policy=pol_scores, random_repeat=rnd_scores, demo_true=demo["true_score"],
               counters={k: list(v) for k, v in roles.items()}, secs=round(time.perf_counter() - t0))
    print(f"   LIVE (seed 7, sticky 0.25): policy {pol_scores} mean {np.mean(pol_scores):.1f} | "
          f"random-repeat {rnd_scores} mean {np.mean(rnd_scores):.1f}", flush=True)
    json.dump(res, open(f"craft_v6_out/v71_play_{a.game}.json", "w"), indent=1)
    torch.save(pol.net.state_dict(), f"craft_v6_out/v71_policy_{a.game}.pt")


if __name__ == "__main__":
    main()
