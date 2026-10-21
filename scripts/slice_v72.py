"""ARC 4 v72 measurement runner (docs/ARC4_V72_PROTOCOL.md). One process, sequential, RAM-safe.

  prep      GAME SEED            entry (perception, adapter) + exploration -> cache pickle
  learn     GAME SEED METHOD     fresh policy (torch seed) learns from the cached demo, live eval
  baselines GAME                 every constant button, uniform random, random with action repeat,
                                 untrained policy through the agent's own interface; same live
                                 conditions (sticky 0.25, eval seeds 7 and 8, 10 games each)

Usage: C:/Python314/python.exe -m scripts.slice_v72 prep assault 0
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

OUT = "craft_v6_out/v72"
EVAL_SEEDS, EVAL_GAMES = (7, 8), 10
METHODS = {
    "bri": dict(method="bri", hold=32, cont=0),          # the diagnosis' tested rule
    "bri_cont": dict(method="bri", hold=8, cont=24),     # exploratory: short hold then the policy
    "segments": dict(method="segments"),                 # previous rule, for comparison
}


def free_gb():
    import ctypes
    import ctypes.wintypes as wt

    class M(ctypes.Structure):
        _fields_ = [("l", wt.DWORD), ("m", wt.DWORD), ("t", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong),
                    ("x", ctypes.c_ulonglong * 5)]
    m = M(); m.l = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.a / 2 ** 30


def gate(min_gb=14.0, wait_s=900):
    """Do not start while less than `min_gb` is free (Tavern keeps 12 GB)."""
    t0 = time.time()
    while free_gb() < min_gb and time.time() - t0 < wait_s:
        time.sleep(60)
    if free_gb() < min_gb:
        sys.exit(f"RAM gate: only {free_gb():.1f} GB free after {wait_s}s, not starting")


def cache_path(game, seed):
    return f"{OUT}/cache_{game}_s{seed}.pkl"


def prep(game, seed, iters=1500):
    from ragnarok.agent_v71.env import AtariEnv
    from ragnarok.agent_v71.learner import Agent
    t0 = time.perf_counter()
    env = AtariEnv(game, seed=seed)
    rng = np.random.default_rng(seed)
    ag = Agent()
    ag.enter(env, rng, seed=seed)
    f_enter = env.frames + ag.frames_adapter
    demo, ev, root = ag.explore(env, rng, iters)
    rec = dict(game=game, seed=seed, demo_len=len(demo), demo_events=ev, demo_true_score=None,
               frames_enter=f_enter, frames_explore=env.frames + ag.frames_adapter - f_enter,
               slots=ag.adapter.mapping(), adapter_info=ag.adapter.info, roles={k: list(v) for k, v in ag.roles.items()})
    env.restore(root)                                   # experimenter-side: true score of the demo
    for a in demo:
        _, over = env.step(a)
        if over:
            break
    rec["demo_true_score"] = env.true_score()
    ag.policy = None                                    # policies are per arm
    pickle.dump(dict(agent=ag, demo=demo, root=root, rec=rec, rng_state=rng.bit_generator.state),
                open(cache_path(game, seed), "wb"))
    rec["secs"] = round(time.perf_counter() - t0)
    json.dump(rec, open(f"{OUT}/prep_{game}_s{seed}.json", "w"), indent=1)
    print(f"prep {game} s{seed}: slots {rec['slots']} | demo {len(demo)} steps, {ev} events, true {rec['demo_true_score']} | {rec['secs']}s")


def learn(game, seed, method):
    from ragnarok.agent_v71.env import AtariEnv
    from ragnarok.agent_v71.policy import Policy
    from ragnarok.agent_v71.adapter import SLOTS
    C = pickle.load(open(cache_path(game, seed), "rb"))
    ag, demo, root = C["agent"], C["demo"], C["root"]
    t0 = time.perf_counter()
    torch.manual_seed(1000 + seed)
    ag.policy = Policy(len(SLOTS))
    env = AtariEnv(game, seed=seed)
    env.reset()
    rng = np.random.default_rng(10_000 + seed)
    f0 = env.frames
    hist = ag.learn(env, rng, demo, root, rounds=12, log=print, **METHODS[method])
    frames_learn = env.frames - f0
    live, mix = {}, {}
    for s in EVAL_SEEDS:
        live[s], mix[s] = ag.play_live(game, seed=s, games=EVAL_GAMES)
    allv = [x for s in EVAL_SEEDS for x in live[s]]
    res = dict(game=game, seed=seed, method=method, cfg=METHODS[method], live=live, button_mix=mix,
               mean=float(np.mean(allv)), se=float(np.std(allv, ddof=1) / np.sqrt(len(allv))),
               frames_learn=frames_learn, prep=C["rec"], learn=hist, secs=round(time.perf_counter() - t0))
    json.dump(res, open(f"{OUT}/learn_{game}_s{seed}_{method}.json", "w"), indent=1)
    print(f"learn {game} s{seed} {method}: live {res['mean']:.1f} ± {res['se']:.1f} | mix {mix} | {res['secs']}s")


def _play(game, policy_fn, seeds=EVAL_SEEDS, games=EVAL_GAMES, max_steps=7000):
    from ragnarok.agent_v71.env import AtariEnv
    out = []
    for s in seeds:
        env = AtariEnv(game, seed=s, sticky=0.25)
        rng = np.random.default_rng(s + 77)
        for _ in range(games):
            obs = env.reset()
            st = {"a": 0}
            for _ in range(max_steps):
                obs, over = env.step(policy_fn(obs, rng, st, env))
                if over:
                    break
            out.append(env.true_score())
    return out


def baselines(game):
    from ragnarok.agent_v71.env import AtariEnv
    from ragnarok.agent_v71.policy import Policy
    from ragnarok.agent_v71.adapter import SLOTS
    t0 = time.perf_counter()
    n = AtariEnv(game, seed=0).n_actions
    names = [a.name for a in AtariEnv(game, seed=0).actions]
    res = dict(game=game, constant={})
    for b in range(n):
        res["constant"][names[b]] = _play(game, lambda o, r, st, e, b=b: b)
    res["uniform"] = _play(game, lambda o, r, st, e: int(r.integers(e.n_actions)))

    def rep(o, r, st, e):
        if r.random() > 0.95:
            st["a"] = int(r.integers(e.n_actions))
        return st["a"]
    res["random_repeat"] = _play(game, rep)
    if os.path.exists(cache_path(game, 0)):           # untrained policy through the agent's interface
        ag = pickle.load(open(cache_path(game, 0), "rb"))["agent"]
        torch.manual_seed(999)
        ag.policy = Policy(len(SLOTS))
        sc = []
        for s in EVAL_SEEDS:
            sc += ag.play_live(game, seed=s, games=EVAL_GAMES)[0]
        res["untrained"] = sc

    def stat(v):
        return dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1) / np.sqrt(len(v))), n=len(v))
    res["summary"] = {k: stat(v) for k, v in [("uniform", res["uniform"]), ("random_repeat", res["random_repeat"])]
                      + ([("untrained", res["untrained"])] if "untrained" in res else [])}
    res["summary"]["constant"] = {k: stat(v) for k, v in res["constant"].items()}
    best = max(res["summary"]["constant"], key=lambda k: res["summary"]["constant"][k]["mean"])
    res["best_constant"] = dict(button=best, **res["summary"]["constant"][best])
    res["secs"] = round(time.perf_counter() - t0)
    json.dump(res, open(f"{OUT}/baselines_{game}.json", "w"), indent=1)
    print(f"baselines {game}: best constant {best} {res['best_constant']['mean']:.1f} ± {res['best_constant']['se']:.1f} | "
          + " ".join(f"{k} {v['mean']:.1f}" for k, v in res["summary"].items() if k != "constant") + f" | {res['secs']}s")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    gate()
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(sys.argv[2], int(sys.argv[3]))
    elif cmd == "learn":
        learn(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif cmd == "baselines":
        baselines(sys.argv[2])
