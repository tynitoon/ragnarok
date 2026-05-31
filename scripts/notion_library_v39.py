"""v39 — a LIBRARY of notions + RECOGNITION + reuse (the missing lock).

The developmental loop (v22/v25 'recognise-or-learn') lifted to the NOTION level,
model-based. The agent holds a library of learned DYNAMICS models (different
'worlds'/physics). Dropped into a context, it RECOGNISES which notion applies (the
model that best predicts a few observed transitions), REUSES it by planning to
solve the task, and DETECTS a NOVEL world (no model fits) -> learns a new one and
adds it. This is exactly the reliable, general reuse the user asked for: not blind
transfer, but recognise-the-right-notion-then-reuse.

Worlds share the action->force mapping but differ in physics (drag / gravity /
response). Builds on v38 (model-based reuse via CEM planning).

Usage: python -m scripts.notion_library_v39 [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE
from scripts.concept_dynamics_v38 import FORCE, TOL, VTOL, TASKS, success, plan_step

WORLDS = [
    dict(name="inertia", resp=1.0, gravity=0.0, drag=0.92),
    dict(name="heavy_drag", resp=1.0, gravity=0.0, drag=0.50),
    dict(name="gravity", resp=1.0, gravity=0.035, drag=0.92),
    dict(name="strong", resp=3.0, gravity=0.0, drag=0.92),
    dict(name="anti_gravity", resp=1.0, gravity=-0.035, drag=0.92),   # held out as NOVEL
]


def step_world(pos, vel, action, w):
    force = (action.float() - 1.0) * FORCE
    vel = (vel + force * w["resp"] - w["gravity"]) * w["drag"]
    pos = (pos + vel).clamp(0.0, 1.0)
    vel = torch.where((pos <= 0.0) | (pos >= 1.0), torch.zeros_like(vel), vel)
    return pos, vel


def learn_model(w, steps, n, gen):
    M = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                      nn.Linear(64, 2)).to(DEVICE)
    opt = torch.optim.Adam(M.parameters(), 1e-3)
    pos = torch.rand(n, generator=gen, device=DEVICE)
    vel = torch.zeros(n, device=DEVICE)
    for _ in range(steps):
        a = torch.randint(0, 3, (n,), generator=gen, device=DEVICE)
        force = (a.float() - 1.0) * FORCE
        npos, nvel = step_world(pos, vel, a, w)
        loss = F.mse_loss(M(torch.stack([pos, vel, force], -1)),
                          torch.stack([npos, nvel], -1))
        opt.zero_grad(); loss.backward(); opt.step()
        pos, vel = npos.detach(), nvel.detach()
        resc = torch.rand(n, generator=gen, device=DEVICE) < 0.05
        pos = torch.where(resc, torch.rand(n, generator=gen, device=DEVICE), pos)
        vel = torch.where(resc, torch.zeros_like(vel), vel)
    return M


@torch.no_grad()
def rollout(w, n, gen, H=8):
    """A short observed TRAJECTORY in world w (states + actions)."""
    pos = torch.rand(n, generator=gen, device=DEVICE)
    vel = (torch.rand(n, generator=gen, device=DEVICE) - 0.5) * 0.2
    acts = torch.randint(0, 3, (H, n), generator=gen, device=DEVICE)
    states = [torch.stack([pos, vel], -1)]
    for h in range(H):
        pos, vel = step_world(pos, vel, acts[h], w)
        states.append(torch.stack([pos, vel], -1))
    return torch.stack(states, 0), acts                  # (H+1,n,2), (H,n)


@torch.no_grad()
def recognise(models, states, acts):
    """Multi-step: roll each model forward along the observed actions; the model
    with lowest accumulated trajectory error is the recognised notion. Compounding
    makes worlds (and novelty) clearly separable."""
    H = acts.shape[0]
    errs = []
    for M in models:
        pos, vel = states[0, :, 0].clone(), states[0, :, 1].clone()
        e = torch.zeros((), device=DEVICE)
        for h in range(H):
            force = (acts[h].float() - 1.0) * FORCE
            out = M(torch.stack([pos, vel, force], -1))
            pos, vel = out[:, 0].clamp(0, 1), out[:, 1]
            e = e + F.mse_loss(torch.stack([pos, vel], -1), states[h + 1])
        errs.append(e / H)
    errs = torch.stack(errs)
    return int(errs.argmin()), float(errs.min())


class WorldEnv:
    def __init__(self, n, w, task, max_steps=60, seed=0):
        self.n = self.num_envs = n; self.w = w; self.task = task; self.max_steps = max_steps
        self._g = torch.Generator(device=DEVICE); self._g.manual_seed(seed)
        self.action_dim = self.obs_dim = 3
        self._reset_all(); self.cum_s = torch.zeros(n, device=DEVICE); self.cum_e = torch.zeros(n, device=DEVICE)

    def _tgt(self):
        if TASKS[self.task]["center"]:
            return torch.full((self.n,), 0.5, device=DEVICE)
        return torch.rand(self.n, generator=self._g, device=DEVICE) * 0.8 + 0.1

    def _reset_all(self):
        self.pos = torch.rand(self.n, generator=self._g, device=DEVICE)
        self.vel = torch.zeros(self.n, device=DEVICE); self.target = self._tgt()
        self.steps = torch.zeros(self.n, dtype=torch.long, device=DEVICE)

    def step(self, action):
        self.pos, self.vel = step_world(self.pos, self.vel, action, self.w)
        self.steps += 1
        s = success(self.pos, self.vel, self.target, self.task)
        done = s | (self.steps >= self.max_steps)
        self.cum_s += s.float(); self.cum_e += done.float()
        if bool(done.any()):
            d = done
            self.pos = torch.where(d, torch.rand(self.n, generator=self._g, device=DEVICE), self.pos)
            self.vel = torch.where(d, torch.zeros_like(self.vel), self.vel)
            self.target = torch.where(d, self._tgt(), self.target)
            self.steps = torch.where(d, torch.zeros_like(self.steps), self.steps)
        return s

    def rate(self):
        return float(self.cum_s.sum() / self.cum_e.sum().clamp(min=1))


@torch.no_grad()
def reuse(M, w_true, task, n=256, steps=240, seed=5):
    env = WorldEnv(n, w_true, task, seed=seed)
    for _ in range(steps):
        env.step(plan_step(M, env.pos, env.vel, env.target, task))
    return env.rate()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dyn-steps", type=int, default=800)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--task", default="stop")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.dyn_steps, args.num_envs = 250, 64

    gen = torch.Generator(device=DEVICE); gen.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    known, novel = WORLDS[:4], WORLDS[4]
    print(f"[v39] device={DEVICE} | LIBRARY of {len(known)} dynamics-notions + "
          f"RECOGNISE-or-LEARN + reuse-by-planning | novel='{novel['name']}'", flush=True)
    t0 = time.perf_counter()

    library = [learn_model(w, args.dyn_steps, args.num_envs, gen) for w in known]
    # known min-MSE (correct model) sets the novelty scale
    known_min = []
    recog_ok = 0
    for i, w in enumerate(known):
        states, acts = rollout(w, 1000, gen)
        r, mse = recognise(library, states, acts)
        recog_ok += int(r == i); known_min.append(mse)
    recog_acc = recog_ok / len(known)
    thresh = max(known_min) * 20.0
    print(f"  recognition over known worlds: {recog_acc:.0%} | known min-MSE<= "
          f"{max(known_min):.2e}, novelty thresh {thresh:.2e} | {time.perf_counter()-t0:.0f}s",
          flush=True)

    # reuse: recognised model vs a WRONG model, per known world
    rec_succ, wrong_succ = [], []
    for i, w in enumerate(known):
        states, acts = rollout(w, 1000, gen)
        r, _ = recognise(library, states, acts)
        rec_succ.append(reuse(library[r], w, args.task))
        wrong_succ.append(reuse(library[(i + 1) % len(known)], w, args.task))
    rs, ws = sum(rec_succ) / len(rec_succ), sum(wrong_succ) / len(wrong_succ)
    print(f"  reuse on known worlds (task={args.task}): RECOGNISED-model {rs:.2f} vs "
          f"WRONG-model {ws:.2f} | {time.perf_counter()-t0:.0f}s", flush=True)

    # novelty: the novel world should fit no known model -> detect -> learn + reuse
    statesN, actsN = rollout(novel, 1000, gen)
    rN, mseN = recognise(library, statesN, actsN)
    is_novel = mseN > thresh
    Mnew = learn_model(novel, args.dyn_steps, args.num_envs, gen)
    novel_reuse = reuse(Mnew, novel, args.task)
    mismatch_reuse = reuse(library[rN], novel, args.task)        # best known (wrong) model
    print(f"  novel world '{novel['name']}': best-known-model MSE {mseN:.2e} "
          f"(>{thresh:.2e}? {is_novel}) -> detected={is_novel}; after LEARN+add, reuse "
          f"{novel_reuse:.2f} vs best-known-model {mismatch_reuse:.2f} | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    ok = (recog_acc >= 0.9 and rs >= ws + 0.25 and is_novel and novel_reuse >= rs - 0.15)
    verdict = (
        f"NOTION LIBRARY + RECOGNITION WORKS — the agent recognises which of "
        f"{len(known)} learned dynamics-notions applies ({recog_acc:.0%}) and reusing "
        f"the RECOGNISED model solves the task {rs:.0%} vs only {ws:.0%} with a WRONG "
        f"model; it DETECTS the novel world (MSE {mseN:.1e} > {thresh:.1e}), learns it, "
        f"and then solves it {novel_reuse:.0%} (vs {mismatch_reuse:.0%} forcing a wrong "
        f"known model). Recognise-the-right-notion-then-reuse is the reliable, general "
        f"reuse mechanism — the missing lock, closed, at the notion level."
        if ok else
        f"PARTIAL — recog {recog_acc:.0%}, reuse recognised {rs:.2f} vs wrong {ws:.2f}, "
        f"novelty detected={is_novel} (MSE {mseN:.1e} vs thresh {thresh:.1e}), "
        f"novel reuse {novel_reuse:.2f} vs mismatch {mismatch_reuse:.2f}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v39_notion_library.json"), "w") as f:
        json.dump(dict(recog_acc=recog_acc, reuse_recognised=rs, reuse_wrong=ws,
                       novelty_mse=mseN, novelty_thresh=thresh, novel_detected=is_novel,
                       novel_reuse=novel_reuse, mismatch_reuse=mismatch_reuse,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
