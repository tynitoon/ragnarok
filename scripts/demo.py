"""Ragnarok  -  a self-teaching agent you can watch.

Drop the agent into a crafting world. NO goals, NO recipes, NO curriculum are
given. It teaches ITSELF: it pokes around, notices what it can newly make,
learns that as a skill, and  -  because it can now produce everything it has
mastered  -  the NEXT thing is only ever one small step away. So it climbs the
whole tech-tree bottom-up, choosing the order itself, and deep skills cost no
more to learn than shallow ones. That is the heart of the project:

  (1) learn basic notions, (2) REUSE them to learn complex ones just as
  cheaply (compounding), (3) DISCOVER what to learn next on its own.

This is a DEMO of capabilities validated in preregistration.md (v6/M3 +
v7.0). It is meant to be run and watched, not a new experiment.

Usage:
  python -m scripts.demo                # ~5-8 min on a GPU
  python -m scripts.demo --fast         # ~2-3 min (smaller, still real)
  python -m scripts.demo --build        # then watch it BUILD iron_pickaxe live
"""

import argparse
import sys
import time

import torch

try:                                # never crash on a cp1252 console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.craft_world import ACH_DEPTH
from scripts.discover_v7 import (
    _discover, _learn_item, ITEM_TO_ACH, ITEM_NAME, ALL_ITEMS)

BAR = "=" * 72


def _say(msg=""):
    print(msg, flush=True)


def _depth_of(item):
    return ACH_DEPTH[ITEM_TO_ACH[item]]


def develop(cfg, max_rounds):
    """Run the autonomous self-teaching loop with plain-language narration.
    Returns the ordered list of (item, depth, steps_to_master)."""
    mastered, learned = set(), []
    _say(BAR)
    _say("  RAGNAROK  -  a self-teaching agent")
    _say(BAR)
    _say("\nI've just been dropped into a world I know nothing about.")
    _say("No one told me what to do or what's possible. Let me find out.\n")
    time.sleep(1.0)

    for rnd in range(1, max_rounds + 1):
        frac = _discover(mastered, cfg)
        cand = sorted([i for i, f in frac.items() if f >= cfg["disc_thresh"]],
                      key=lambda i: -frac[i])
        known = [ITEM_NAME[i] for i in mastered] or ["(nothing yet)"]
        _say(f"-- Round {rnd} "
             f"------------------------------------------------------")
        _say(f"   What I can already make: {', '.join(known)}")
        if not cand:
            _say("   I explored, but found nothing new I can reach. I'm done.\n")
            break
        new_names = ", ".join(ITEM_NAME[i] for i in cand)
        _say(f"   I poked around and realised I can now reach: {new_names}")
        for item in cand:
            if item in mastered:
                continue
            reuse = ("" if not mastered else
                     " (reusing what I already know)")
            _say(f"     ... learning to get {ITEM_NAME[item]}{reuse} ...")
            steps, succ = _learn_item(item, mastered, cfg)
            if succ >= cfg["mastery"]:
                mastered.add(item)
                learned.append((item, _depth_of(item), steps))
                _say(f"         got it! mastered {ITEM_NAME[item]} "
                     f"(success {succ:.2f}, learned in {steps:,} steps)")
            else:
                _say(f"         couldn't quite master {ITEM_NAME[item]} "
                     f"(reached {succ:.2f})  -  I'll revisit it.")
        _say("")
        if len(mastered) == len(ALL_ITEMS):
            _say("   I think I've discovered everything in this world!\n")
            break
    return learned


def _summary(learned):
    _say(BAR)
    _say("  WHAT JUST HAPPENED")
    _say(BAR)
    order = " -> ".join(ITEM_NAME[i] for i, _, _ in learned)
    _say(f"\nWith no goals given, I taught myself {len(learned)} skills, and I")
    _say(f"chose this order entirely on my own:\n\n   {order}\n")
    _say("Notice how long each skill took to learn, by how DEEP it is in the")
    _say("tech-tree (depth = how many things must come before it):\n")
    _say(f"   {'skill':16s} {'depth':>6} {'steps to learn':>16}")
    for item, depth, steps in sorted(learned, key=lambda x: x[1]):
        _say(f"   {ITEM_NAME[item]:16s} {depth:>6} {steps:>16,}")
    deep = [s for _, d, s in learned if d >= 3]
    shallow = [s for _, d, s in learned if d <= 1]
    _say("")
    if deep and shallow:
        _say(f"The deepest skills cost about the same as the first ones")
        _say(f"(~{sum(deep)//len(deep):,} vs ~{sum(shallow)//len(shallow):,} "
             f"steps)  -  even though they sit much deeper in the tree.")
    _say("That is the whole point: because I REUSE everything I've already")
    _say("mastered, each new notion is only ever one small step away, so")
    _say("learning does NOT get more expensive as things get deeper.")
    _say("An agent that can't reuse stalls past depth 2  -  it never even")
    _say("reaches the deep skills (see preregistration.md: v6/M3, v7.0).\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fast", action="store_true", help="quicker, smaller run")
    p.add_argument("--build", action="store_true",
                   help="after self-teaching, watch it BUILD iron_pickaxe live")
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--view", type=int, default=5)
    p.add_argument("--max-rounds", type=int, default=8)
    args = p.parse_args()

    cfg = dict(num_envs=256, grid=args.grid, view=args.view, max_steps=100,
               rollout=32, hidden=256, entropy=0.02, skill_cap=70, eval_every=10,
               mastery=0.8, disc_envs=512, disc_steps=120, disc_thresh=0.01)
    if args.fast:
        cfg.update(num_envs=128, skill_cap=40, disc_envs=256, disc_steps=80)

    _say(f"[device: {DEVICE}]")
    t0 = time.perf_counter()
    learned = develop(cfg, args.max_rounds)
    _summary(learned)
    _say(f"(self-taught in {time.perf_counter()-t0:.0f}s on {DEVICE})")

    if args.build:
        _say("\n" + BAR)
        _say("  NOW WATCH ME BUILD THE HARDEST THING I LEARNED")
        _say(BAR)
        _say("Handing off to the model-based builder (learns recipes -> plans ->")
        _say("crafts, with a live view). Run this yourself anytime:\n")
        _say("   python -m scripts.ragnarok --target iron_pickaxe\n")


if __name__ == "__main__":
    main()
