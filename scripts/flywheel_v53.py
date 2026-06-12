"""v53 — THE SELF-IMITATION FLYWHEEL: does the agent's own accumulated experience make NEW tasks
progressively cheaper to master? (The compounding-mechanism demo. Prereg FROZEN before this file.)

v51 located the wall precisely: a composer trained by sparse-reward RL fails (0.003) even though the
correct strategy is expressible in its observable per-item features (greedy proves it). v53 keeps the
EXACT architecture and changes only the LEARNING SIGNAL: the composer is trained by CROSS-ENTROPY on
hindsight-relabeled samples from the agent's OWN trajectories — every item X first unlocked at step t
turns the episode prefix into demonstrations of "how to achieve X". A lifelong buffer accumulates these
across a STREAM of tasks (Arm A); the control (Arm B) is amnesic (fresh composer+buffer per task).
The claim under test is the COMPOUNDING CURVE: cost-to-master(task k) falls with k / zero-shot mastery
rises, in A but not B. Mastery alone proves nothing here (substrate is greedy-solvable — known ceiling
0.98, context only); the curve of a LEARNED-from-own-data composer is the result.

Usage: python -m scripts.flywheel_v53 [--smoke] [--arm A|B|both] [--seed 0]
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import gen_tree
from scripts.depth_scaling_v49 import N_ITEMS_FOR_DEPTH
from scripts.childhood_v50 import train_childhood
from scripts.meta_manager_v51 import RouterEnv, PerItemRouter, MAX_ITEMS, N_FEAT

GOAL_COL = 4                       # per-item feature index of is_goal


class Composer:
    """The learned general composer. NO RL: trained only by CE on hindsight self-imitation samples."""

    def __init__(self, lr=3e-4):
        self.net = PerItemRouter().to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, state, epsilon=0.0, temp=1.0, deterministic=False):
        logits, _ = self.net(state)
        if deterministic:
            return logits.argmax(-1)
        a = torch.multinomial(F.softmax(logits / temp, -1), 1).squeeze(-1)
        if epsilon > 0:
            valid = state.reshape(-1, MAX_ITEMS, N_FEAT)[..., 6] > 0.5
            rnd = torch.multinomial(valid.float(), 1).squeeze(-1)
            take = torch.rand(a.shape[0], device=DEVICE) < epsilon
            a = torch.where(take, rnd, a)
        return a

    def train_steps(self, buf, n_steps, bs=512):
        if buf.n == 0:
            return float("nan")
        losses = []
        for _ in range(n_steps):
            idx = torch.randint(0, buf.n, (min(bs, buf.n),), device=DEVICE)
            logits, _ = self.net(buf.s[idx])
            loss = F.cross_entropy(logits, buf.a[idx])
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            losses.append(float(loss))
        return sum(losses) / len(losses)


class Buffer:
    """Lifelong FIFO buffer of (state-with-goal, action) self-imitation samples, device-resident."""

    def __init__(self, cap=400_000):
        self.s = torch.zeros(cap, MAX_ITEMS * N_FEAT, device=DEVICE)
        self.a = torch.zeros(cap, dtype=torch.long, device=DEVICE)
        self.cap, self.n, self.ptr = cap, 0, 0

    def add(self, s, a):
        k = s.shape[0]
        if k == 0:
            return
        if k >= self.cap:
            s, a, k = s[-self.cap:], a[-self.cap:], self.cap
        end = self.ptr + k
        if end <= self.cap:
            self.s[self.ptr:end], self.a[self.ptr:end] = s, a
        else:
            r = self.cap - self.ptr
            self.s[self.ptr:], self.a[self.ptr:] = s[:r], a[:r]
            self.s[:k - r], self.a[:k - r] = s[r:], a[r:]
        self.ptr = end % self.cap
        self.n = min(self.n + k, self.cap)


class RecordingEnv(RouterEnv):
    """RouterEnv + stash of pre-reset unlocked mask (so last-step unlocks aren't lost on truncation)."""

    def step(self, g):
        g = g.reshape(self.num_envs).clamp(max=self.n_items - 1)
        N, ar = self.num_envs, torch.arange(self.num_envs, device=DEVICE)
        rew = torch.zeros(N, device=DEVICE)
        is_craft = self.item_is_craft[g]
        cell_of = self.item_cell[g]
        craft_act = self.item_craft_act[g]
        start = self.base.inv[ar, g].float()
        done_opt = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        det = not self.cfg.get("skill_stochastic", False)
        for t in range(self.option_timeout):
            ego = self.base.state[:, :self.ego_dim]
            goh = F.one_hot(cell_of, 24).float()               # MAX_CELLS=24 (childhood obs format)
            a_skill = self.skill.act(torch.cat([ego, goh], -1), deterministic=det)
            a = torch.where(is_craft, craft_act, a_skill)
            _, r, _, _, _ = self.base.step(a)
            self._prim += N
            rew += r * (~done_opt).float()
            cur = self.base.inv[ar, g].float()
            got_craft = is_craft & self.base.unlocked[ar, g]
            done_opt = done_opt | got_craft | (~is_craft & (cur >= start + 1))
            if t % 8 == 7 and bool(done_opt.all()):
                break
        self.post_unlocked = self.base.unlocked.clone()        # BEFORE any truncation reset
        self.msteps += 1
        trunc = self.msteps >= self.macro_budget
        if bool(trunc.any()):
            self.base._reset_done(trunc)
            self.msteps = torch.where(trunc, torch.zeros_like(self.msteps), self.msteps)
        self._set_state()
        return self.state, rew, torch.zeros_like(trunc), trunc, trunc


def collect_episode(env, composer, epsilon, temp):
    """One synchronized macro-episode. Returns states (T,N,D) with the GOAL COLUMN ZEROED (rewritten at
    relabel time), actions (T,N), and unlockstep (N, n_items) = first macro-step where item unlocked."""
    N, T = env.num_envs, env.macro_budget
    states = torch.zeros(T, N, MAX_ITEMS * N_FEAT, device=DEVICE)
    actions = torch.zeros(T, N, dtype=torch.long, device=DEVICE)
    unlockstep = torch.full((N, env.n_items), -1, dtype=torch.long, device=DEVICE)
    env.reset()
    prev = env.base.unlocked.clone()
    obs = env.state
    for t in range(T):
        a = composer.act(obs, epsilon=epsilon, temp=temp)
        s = obs.clone().reshape(N, MAX_ITEMS, N_FEAT)
        s[..., GOAL_COL] = 0.0                                 # store goal-free; goal set at relabel
        states[t] = s.reshape(N, -1)
        actions[t] = a
        obs, _, _, _, _ = env.step(a)
        newly = env.post_unlocked & ~prev
        first = (unlockstep == -1) & newly
        unlockstep[first] = t
        prev = env.post_unlocked.clone()
    return states, actions, unlockstep


def relabel(states, actions, unlockstep, max_samples):
    """Hindsight: (s_t, a_t, goal=X) is a demo iff X was first unlocked at some step >= t."""
    T, N, D = states.shape
    t_idx = torch.arange(T, device=DEVICE).view(T, 1, 1)
    valid = (unlockstep.unsqueeze(0) >= t_idx) & (unlockstep.unsqueeze(0) >= 0)   # (T,N,I)
    idx = valid.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return None, None
    if idx.shape[0] > max_samples:
        idx = idx[torch.randperm(idx.shape[0], device=DEVICE)[:max_samples]]
    t, n, X = idx[:, 0], idx[:, 1], idx[:, 2]
    s = states[t, n].clone().reshape(-1, MAX_ITEMS, N_FEAT)
    s[torch.arange(s.shape[0], device=DEVICE), X, GOAL_COL] = 1.0
    return s.reshape(s.shape[0], -1), actions[t, n]


@torch.no_grad()
def eval_master(spec, skill, composer, cfg, seed, n=256, ablate_goal=False):
    env = RouterEnv(n, spec, skill, cfg, seed=seed + 9)
    got = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["macro_budget"]):
        o = obs
        if ablate_goal:
            o = obs.clone().reshape(n, MAX_ITEMS, N_FEAT)
            o[..., GOAL_COL] = 0.0
            o = o.reshape(n, -1)
        obs, _, _, _, _ = env.step(composer.act(o, deterministic=True))
        got |= env.base.unlocked[:, spec["target"]]
    return float(got.float().mean())


def run_task(spec, skill, composer, buf, cfg, seed):
    """Deploy on one task until mastered (>= thresh) or budget. Returns cost + curve points."""
    zs = eval_master(spec, skill, composer, cfg, seed)
    if zs >= cfg["thresh"]:
        return dict(zero_shot=round(zs, 3), cost=0, mastered=True, final=round(zs, 3), rounds=0)
    env = RecordingEnv(cfg["num_envs"], spec, skill, cfg, seed=seed)
    env._prim = 0
    final, rounds = zs, 0
    while env._prim < cfg["task_budget"]:
        for _ in range(cfg["episodes_per_round"]):
            s, a, us = collect_episode(env, composer, cfg["epsilon"], cfg["temp"])
            ss, aa = relabel(s, a, us, cfg["max_samples_per_ep"])
            if ss is not None:
                buf.add(ss, aa)
        loss = composer.train_steps(buf, cfg["train_steps_per_round"])
        final = eval_master(spec, skill, composer, cfg, seed)
        rounds += 1
        if rounds % 2 == 0:
            print(f"      round {rounds:>3} | prim {env._prim/1e6:.2f}M | buf {buf.n} | "
                  f"loss {loss:.3f} | master {final:.2f}", flush=True)
        if final >= cfg["thresh"]:
            return dict(zero_shot=round(zs, 3), cost=env._prim, mastered=True,
                        final=round(final, 3), rounds=rounds)
    return dict(zero_shot=round(zs, 3), cost=env._prim, mastered=False,
                final=round(final, 3), rounds=rounds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--n-stream", type=int, default=10)
    p.add_argument("--n-heldout", type=int, default=4)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--view", type=int, default=13)
    p.add_argument("--n-resource", type=int, default=4)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--nav-max-steps", type=int, default=40)
    p.add_argument("--skill-iters", type=int, default=400)
    p.add_argument("--option-timeout", type=int, default=40)
    p.add_argument("--macro-budget", type=int, default=20)
    p.add_argument("--task-budget", type=float, default=5e6)
    p.add_argument("--episodes-per-round", type=int, default=4)
    p.add_argument("--train-steps-per-round", type=int, default=300)
    p.add_argument("--max-samples-per-ep", type=int, default=8192)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--thresh", type=float, default=0.6)
    p.add_argument("--arm", choices=["A", "B", "both"], default="both")
    p.add_argument("--resume", action="store_true", help="resume from per-task checkpoints if present")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.depth, args.n_stream, args.n_heldout = 5, 3, 2
        args.skill_iters, args.task_budget = 250, 1.5e6
        args.train_steps_per_round, args.arm = 150, "A"

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "n_resource", "rollout", "entropy", "nav_max_steps",
            "skill_iters", "option_timeout", "macro_budget", "episodes_per_round",
            "train_steps_per_round", "max_samples_per_ep", "epsilon", "temp", "thresh")}
    cfg["task_budget"] = args.task_budget
    cfg["skill_stochastic"] = True
    cfg["mgr_entropy"] = 0.03                                  # unused (no RL), RouterEnv cfg compat
    cfg["router_iters"] = 0
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ni = N_ITEMS_FOR_DEPTH[args.depth]
    skill_specs = [gen_tree(1000 + i, n_items=ni) for i in range(8)]
    stream = [gen_tree(5000 + i, n_items=ni) for i in range(args.n_stream)]
    heldout = [gen_tree(9000 + i, n_items=ni) for i in range(args.n_heldout)]
    B_TASKS = [t for t in (0, 3, 6, 9) if t < args.n_stream]

    print(f"[v53 flywheel] device={DEVICE} | depth~{args.depth} | stream {args.n_stream} tasks | "
          f"arm(s) {args.arm} | hindsight self-imitation, NO RL | budget {args.task_budget/1e6:.1f}M/task",
          flush=True)
    t0 = time.perf_counter()
    skill, c_skill = train_childhood(skill_specs, cfg, args.seed)
    print(f"  shared childhood skill ready ({c_skill/1e6:.2f}M, both arms) | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    results = dict(seed=args.seed, depth=args.depth, c_skill=c_skill, A=[], B=[])
    tag = f"_d{args.depth}" + ("_smoke" if args.smoke else "")        # config-specific files: a smoke
    ckpt_path = os.path.join(args.out_dir, f"v53_flywheel{tag}_s{args.seed}.json")  # can never pollute
    ckpt_pt = os.path.join(args.out_dir, f"v53_ckpt{tag}_s{args.seed}.pt")          # a confirmatory resume

    def _ckpt():
        with open(ckpt_path, "w") as f:
            json.dump(results, f, indent=2)

    done_A, done_B = -1, set()
    if args.resume and os.path.exists(ckpt_path) and os.path.exists(ckpt_pt):   # BOTH or fresh
        with open(ckpt_path) as f:
            prev = json.load(f)
        results["A"], results["B"] = prev.get("A", []), prev.get("B", [])
        done_A = max([r["task"] for r in results["A"]], default=-1)
        done_B = {r["task"] for r in results["B"]}
        print(f"  RESUME: arm-A tasks <= {done_A} already done, arm-B {sorted(done_B)} done", flush=True)

    if args.arm in ("A", "both"):
        print("\n  === ARM A (flywheel: lifelong buffer + persistent composer) ===", flush=True)
        composer, buf = Composer(), Buffer()
        if args.resume and done_A >= 0 and os.path.exists(ckpt_pt):
            st = torch.load(ckpt_pt, map_location=DEVICE)
            composer.net.load_state_dict(st["net"])
            composer.opt.load_state_dict(st["opt"])
            n = st["buf_s"].shape[0]
            buf.s[:n], buf.a[:n], buf.n, buf.ptr = st["buf_s"], st["buf_a"], n, n % buf.cap
            print(f"  RESUME: composer + buffer ({n} samples) restored", flush=True)
        for k, spec in enumerate(stream):
            if k <= done_A:
                continue
            r = run_task(spec, skill, composer, buf, cfg, args.seed + 11 * k + 1)
            r.update(task=k, true_depth=int(spec["depth"][spec["target"]]))
            results["A"].append(r)
            print(f"    [A] task {k} (d{r['true_depth']}): zero-shot {r['zero_shot']:.2f} | "
                  f"cost {r['cost']/1e6:.2f}M | mastered {r['mastered']} ({r['final']:.2f}) | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
            _ckpt()
            torch.save(dict(net=composer.net.state_dict(), opt=composer.opt.state_dict(),
                            buf_s=buf.s[:buf.n].clone(), buf_a=buf.a[:buf.n].clone()), ckpt_pt)
        ho = [round(eval_master(s, skill, composer, cfg, args.seed + 777), 3) for s in heldout]
        hoa = [round(eval_master(s, skill, composer, cfg, args.seed + 777, ablate_goal=True), 3)
               for s in heldout]
        results["heldout_zero_shot"] = ho
        results["heldout_goal_ablated"] = hoa
        print(f"    [A] HELD-OUT zero-shot: {ho} | goal-ablated: {hoa}", flush=True)
        _ckpt()

    if args.arm in ("B", "both"):
        print("\n  === ARM B (amnesic control: fresh composer+buffer per task) ===", flush=True)
        for k in B_TASKS:
            if k in done_B:
                continue
            r = run_task(stream[k], skill, Composer(), Buffer(), cfg, args.seed + 11 * k + 1)
            r.update(task=k, true_depth=int(stream[k]["depth"][stream[k]["target"]]))
            results["B"].append(r)
            print(f"    [B] task {k} (d{r['true_depth']}): cost {r['cost']/1e6:.2f}M | "
                  f"mastered {r['mastered']} ({r['final']:.2f}) | {time.perf_counter()-t0:.0f}s",
                  flush=True)
            _ckpt()

    # FROZEN criteria (prereg v53)
    verdict = "incomplete (single arm)"
    if args.arm == "both" and results["A"]:
        A = results["A"]
        mastered = sum(r["mastered"] for r in A)
        early, late = A[0:3], A[7:10] if len(A) >= 10 else A[-3:]
        c_early = sum(r["cost"] for r in early) / max(1, len(early))
        c_late = sum(r["cost"] for r in late) / max(1, len(late))
        zs_early = sum(r["zero_shot"] for r in early) / max(1, len(early))
        zs_late = sum(r["zero_shot"] for r in late) / max(1, len(late))
        compounding = (c_late <= 0.5 * c_early) or (zs_late >= 0.5 and zs_early < 0.2)
        Bmap = {r["task"]: r for r in results["B"]}
        sep_tasks = [t for t in (6, 9) if t in Bmap and t < len(A)]
        a_sep = sum(A[t]["cost"] for t in sep_tasks) / max(1, len(sep_tasks))
        b_sep = sum(Bmap[t]["cost"] for t in sep_tasks) / max(1, len(sep_tasks))
        separation = a_sep < 0.6 * b_sep if sep_tasks else False
        positive = mastered >= 8 and compounding and separation
        results.update(mastered_A=mastered, cost_early=c_early, cost_late=c_late,
                       zs_early=zs_early, zs_late=zs_late, a_sep=a_sep, b_sep=b_sep,
                       compounding=compounding, separation=separation, positive=positive)
        verdict = (
            f"FLYWHEEL COMPOUNDS (v53, seed {args.seed}) — mastered {mastered}/{len(A)}; cost "
            f"early {c_early/1e6:.2f}M -> late {c_late/1e6:.2f}M; zero-shot {zs_early:.2f} -> "
            f"{zs_late:.2f}; A {a_sep/1e6:.2f}M vs amnesic B {b_sep/1e6:.2f}M on late tasks. The "
            f"agent's own accumulated successes make NEW tasks cheaper, composition LEARNED. "
            f"ADVERSARIAL REVIEW + seeds 1-2 before believing."
            if positive else
            f"PARTIAL/NULL — mastered {mastered}/{len(A)}, cost {c_early/1e6:.2f}->{c_late/1e6:.2f}M, "
            f"zs {zs_early:.2f}->{zs_late:.2f}, sep A {a_sep/1e6:.2f} vs B {b_sep/1e6:.2f} "
            f"(compounding={compounding}, separation={separation}). Honest per prereg.")
    results["verdict"] = verdict
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s total", flush=True)
    with open(os.path.join(args.out_dir, f"v53_flywheel_s{args.seed}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
