"""ARC 4 v74 measurement runner (docs/ARC4_V74_PROTOCOL.md). Sequential, RAM-safe.

v74 = v73 (death signal, tracker seeding) + the explorer uses the validated perception (v73 defect
fixed) + ROLLOUT POLICY IMPROVEMENT: the agent finds its own best constant slot, measures from the
states its own policy visits the value of holding each slot 8 steps then the constant base 96 steps
(its HUD score events - 2 x its death signal), imitates the best slot (base on ties), and acts with
each decision held 8 steps, greedily. FRESH seeds: training 4 and 5, evaluation 13 and 14.

  prep      GAME SEED            entry (perception, adapter, death check) + exploration -> cache
  learn     GAME SEED METHOD     fresh policy learns from the cached demo, live eval
  baselines GAME                 constant buttons, uniform, random-repeat, untrained policy
  all                            the protocol's queue, resumable
  summary                        the frozen criterion (+ the declared secondary reading)

Usage: C:/Python314/python.exe -m scripts.slice_v74 all [shard k/n]
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

OUT = "craft_v6_out/v74"
TRAIN_SEEDS = (4, 5)
EVAL_SEEDS, EVAL_GAMES = (13, 14), 10
VANISH_K = 2
RPI = dict(method="rpi", hold=8, horizon=96, n_label=4, death_penalty=2.0, fit_mode="label", commit=True,
           live_temp=0.05)
METHODS = {
    "rpi": dict(RPI, n_roots=96),        # 12 rounds x 96 roots
    "rpi_big": dict(RPI, n_roots=192),   # the data lever (offline learning curve, Assault)
}
GAMES = ["assault", "freeway", "demon_attack", "name_this_game", "carnival", "breakout"]


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
    ag = Agent(vanish_k=VANISH_K)
    ag.enter(env, rng, seed=seed)
    f_enter = env.frames + ag.frames_adapter
    demo, ev, root = ag.explore(env, rng, iters)
    rec = dict(game=game, seed=seed, demo_len=len(demo), demo_events=ev, demo_true_score=None,
               frames_enter=f_enter, frames_explore=env.frames + ag.frames_adapter - f_enter,
               slots=ag.adapter.mapping(), adapter_info=ag.adapter.info, roles={k: list(v) for k, v in ag.roles.items()},
               death_signal=dict(lives_counter=ag.l_idx is not None, vanish_k=ag.vanish_k,
                                 lost_frac=round(ag.lost_frac, 3), my_sizes=ag.my_sizes))
    env.restore(root)                                   # experimenter-side: true score of the demo
    for a in demo:
        _, over = env.step(a)
        if over:
            break
    rec["demo_true_score"] = env.true_score()
    ag.policy = None                                    # policies are per arm
    pickle.dump(dict(agent=ag, demo=demo, root=root, rec=rec), open(cache_path(game, seed), "wb"))
    # diagnostic (review B1): are single deterministic branch choices worth anything under sticky
    # actions on this game? Recorded, never used by the agent; its frames are not charged.
    try:
        from scripts.check_branch_noise_v74 import main as noise
        rec["branch_noise"] = noise(cache_path(game, seed), n_roots=20, K=4)
    except Exception as e:                              # a diagnostic must not stop the protocol
        rec["branch_noise"] = f"failed: {e!r}"
    rec["secs"] = round(time.perf_counter() - t0)
    json.dump(rec, open(f"{OUT}/prep_{game}_s{seed}.json", "w"), indent=1)
    print(f"prep {game} s{seed}: slots {rec['slots']} | death {rec['death_signal']} | demo {len(demo)} steps, "
          f"{ev} events, true {rec['demo_true_score']} | {rec['secs']}s")


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
    p0 = cache_path(game, TRAIN_SEEDS[0])
    if os.path.exists(p0):                            # untrained policy through the agent's interface
        ag = pickle.load(open(p0, "rb"))["agent"]
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


def queue():
    """The protocol's runs in order; each a separate process (RAM returned between runs), skipped
    when its output exists, so the queue can be restarted after an interruption."""
    jobs = []
    for g in GAMES:
        for s in TRAIN_SEEDS:
            jobs.append((f"{OUT}/prep_{g}_s{s}.json", ["prep", g, str(s)]))
        jobs.append((f"{OUT}/baselines_{g}.json", ["baselines", g]))
        for s in TRAIN_SEEDS:
            for m in METHODS:
                jobs.append((f"{OUT}/learn_{g}_s{s}_{m}.json", ["learn", g, str(s), m]))
    return jobs


def run_all(shard=None):
    """shard 'k/n': only the games with index % n == k (a game's jobs stay in order in one shard)."""
    import subprocess
    k, n = (int(x) for x in shard.split("/")) if shard else (0, 1)
    mine = {g for i, g in enumerate(GAMES) if i % n == k}
    for out, args in queue():
        if args[1] not in mine or os.path.exists(out):
            continue
        name = "_".join(args)
        print(time.strftime("%H:%M:%S"), "start", name, flush=True)
        with open(f"{OUT}/{name}.log", "w") as f:
            r = subprocess.run([sys.executable, "-m", "scripts.slice_v74"] + args, stdout=f, stderr=subprocess.STDOUT)
        print(time.strftime("%H:%M:%S"), "end", name, "exit", r.returncode, flush=True)


def summary():
    """Frozen criterion: pooled method mean (2 training seeds x 20 games) beats the best constant
    button (20 games) when delta > 2 * sqrt(se_m^2 + se_c^2); a game counts only with complete data.
    Secondary reading declared in the protocol before launch: 'robust' if also each training
    seed's own mean is above the best constant's mean."""
    rows, wins, robust = [], {}, {}
    for g in GAMES:
        p = f"{OUT}/baselines_{g}.json"
        if not os.path.exists(p):
            continue
        B = json.load(open(p))
        bc = B["best_constant"]
        line = dict(game=g, best_constant=bc["button"], const=round(bc["mean"], 1), const_se=round(bc["se"], 1),
                    **{k: round(v["mean"], 1) for k, v in B["summary"].items() if k != "constant"})
        for m in METHODS:
            per = {}
            for s in TRAIN_SEEDS:
                q = f"{OUT}/learn_{g}_s{s}_{m}.json"
                if os.path.exists(q):
                    per[s] = [x for v in json.load(open(q))["live"].values() for x in v]
            if not per:
                continue
            vals = [x for v in per.values() for x in v]
            complete = len(per) == len(TRAIN_SEEDS) and all(len(v) == len(EVAL_SEEDS) * EVAL_GAMES for v in per.values()) \
                and bc["n"] == len(EVAL_SEEDS) * EVAL_GAMES
            mu, se = float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            # v74 primary (stricter than v72/v73, review C3): EACH training seed's 20 games beat the
            # best constant by > 2 combined SE; the pooled test of v72/v73 is kept as secondary
            beat = complete and all((np.mean(v) - bc["mean"]) > 2 * np.sqrt(np.var(v, ddof=1) / len(v) + bc["se"] ** 2)
                                    for v in per.values())
            pooled = complete and (mu - bc["mean"]) > 2 * np.sqrt(se ** 2 + bc["se"] ** 2)
            sm = {s: round(float(np.mean(v)), 1) for s, v in per.items()}
            rob = pooled
            line[m] = f"{mu:.1f}±{se:.1f} (n={len(vals)}){' BEATS' if beat else ''}{' pooled' if rob else ''}" \
                      f"{'' if complete else ' INCOMPLETE'}"
            line[m + "_seed_means"] = sm
            if beat:
                wins.setdefault(m, []).append(g)
            if rob:
                robust.setdefault(m, []).append(g)
        rows.append(line)
    for r in rows:
        print(r)
    print("beats the best constant button, each training seed (frozen primary):", wins)
    print("pooled test of v72/v73 (secondary):", robust)
    json.dump(dict(rows=rows, wins=wins, robust=robust), open(f"{OUT}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "all":
        run_all(sys.argv[2] if len(sys.argv) > 2 else None)
        sys.exit(0)
    if cmd == "summary":
        summary()
        sys.exit(0)
    gate()
    if cmd == "prep":
        prep(sys.argv[2], int(sys.argv[3]))
    elif cmd == "learn":
        learn(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif cmd == "baselines":
        baselines(sys.argv[2])
