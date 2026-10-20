"""Run the unified v71 learner on several games, fresh (exploratory baseline for transfer).
Usage: C:/Python314/python.exe -m scripts.slice_learn_v71 --games freeway breakout assault ...
"""
import argparse
import json
from multiprocessing import Pool


def one(args):
    game, iters = args
    from ragnarok.agent_v71.learner import run_game
    try:
        res, _ = run_game(game, explore_iters=iters)
        return res
    except Exception as e:  # noqa: BLE001
        import traceback
        return dict(game=game, error=traceback.format_exc()[-600:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["freeway", "breakout", "assault", "carnival", "demon_attack",
                                                   "name_this_game", "seaquest", "chopper_command"])
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="craft_v6_out/v71_learn_fresh.json")
    a = ap.parse_args()
    with Pool(a.workers) as p:
        res = p.map(one, [(g, a.iters) for g in a.games])
    json.dump(res, open(a.out, "w"), indent=1)
    for r in res:
        print(r.get("error") or f"{r['game']:16s} live {r['live']} mean {r['live_mean']:.1f} random {r['random_mean']:.1f} demo_events {r['demo_events']} "
              f"frames explore {r['frames_explore']} learn {r['frames_learn']}")


if __name__ == "__main__":
    main()
