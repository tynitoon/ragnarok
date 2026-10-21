"""v80 milestone-1 development queue (docs/ARC4_V80_DESIGN.md §4), development seed 98 only.

Steps, in order, each a separate process (memory returned between steps), skipped when its output
exists, each waiting (never skipping) for 14 GB of RAM AND of commit (machine shared with Tavern):
  tests -> prep caches (entry + exploration, seed 98) -> G0 -> train traces -> held-out traces ->
  offline G1/G2 (H = 32), per game (freeway first: it has a RAM judge for G0 and G1).
Usage: C:/Python314/python.exe -m scripts.run_v80_dev [games...]
"""
import os
import pickle
import subprocess
import sys
import time

SEED = int(os.environ.get("V80_SEED", "98"))             # 98: development; 99: confirmation
OUT = "craft_v6_out/v80" + ("" if SEED == 98 else f"/s{SEED}")


def free_gb():
    import ctypes
    import ctypes.wintypes as wt

    class M(ctypes.Structure):
        _fields_ = [("l", wt.DWORD), ("m", wt.DWORD), ("t", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong),
                    ("x", ctypes.c_ulonglong * 5)]
    m = M(); m.l = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return min(m.a, m.x[1]) / 2 ** 30


def wait_memory(min_gb=14.0):
    while free_gb() < min_gb:
        time.sleep(60)


def prep(game):
    """Entry (perception, adapter, death check) + exploration, cached (agent side only)."""
    import numpy as np
    from ragnarok.agent_v71.env import AtariEnv
    from ragnarok.agent_v71.learner import Agent
    env = AtariEnv(game, seed=SEED)
    rng = np.random.default_rng(SEED)
    ag = Agent(vanish_k=2)
    ag.enter(env, rng, seed=SEED)
    demo, ev, root = ag.explore(env, rng, 1500)
    ag.policy = None
    rec = dict(game=game, seed=SEED, demo_len=len(demo), demo_events=ev, frames=env.frames + ag.frames_adapter,
               death_signal=dict(lives_counter=ag.l_idx is not None, vanish_k=ag.vanish_k, lost_frac=ag.lost_frac))
    pickle.dump(dict(agent=ag, demo=demo, root=root, rec=rec), open(f"{OUT}/cache_{game}_s{SEED}.pkl", "wb"))
    print("prep", rec, flush=True)


def jobs(game):
    c = f"{OUT}/cache_{game}_s{SEED}.pkl"
    tr, ho = f"{OUT}/trace_{game}_train.pkl", f"{OUT}/trace_{game}_heldout.pkl"
    g0 = [(f"{OUT}/g0_{game}.json", ["-m", "scripts.check_objects_v80", game])] if SEED == 98 else []
    return [
        (c, ["-m", "scripts.run_v80_dev", "prep", game]),
        *g0,
        (tr, ["-m", "scripts.collect_trace_v80", "train", c, tr, "400"]),
        (ho, ["-m", "scripts.collect_trace_v80", "heldout", c, ho, tr, "300"]),
        (ho.replace(".pkl", "_offline_H32.json"), ["-m", "scripts.theory_offline_v80", tr, ho, "32"]),
    ]


def run(games):
    os.makedirs(OUT, exist_ok=True)
    tlog = f"{OUT}/tests.log"
    if not os.path.exists(tlog):
        wait_memory()
        with open(tlog, "w") as f:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_agent_v80.py"], stdout=f, stderr=subprocess.STDOUT)
        print(time.strftime("%H:%M:%S"), "tests exit", r.returncode, flush=True)
        if r.returncode:
            os.replace(tlog, tlog + ".failed")
            sys.exit("tests failed: see " + tlog + ".failed")
    for g in games:
        for out, args in jobs(g):
            if os.path.exists(out):
                continue
            wait_memory()
            name = os.path.basename(out).rsplit(".", 1)[0]
            print(time.strftime("%H:%M:%S"), "start", name, flush=True)
            with open(f"{OUT}/{name}.log", "w") as f:
                r = subprocess.run([sys.executable] + args, stdout=f, stderr=subprocess.STDOUT)
            print(time.strftime("%H:%M:%S"), "end", name, "exit", r.returncode, flush=True)
            if r.returncode:
                break                                  # later steps of this game need this output


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "prep":
        prep(sys.argv[2])
    else:
        run(sys.argv[1:] or ["freeway", "assault"])
