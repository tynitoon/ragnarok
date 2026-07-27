"""v57 — THE CREDIT FIX: train only on what the agent was actually ASKED to do.

v55's relabel keeps (s_t, a_t, X) for EVERY item X unlocked at or after t. The leak meter
(scripts/leak_meter_v57.py, run on the committed v55 JSONs) measured the consequence: at depth
(pc >= 10) the median share of collected samples that concerned the COMMANDED goal is 0.31%. So 99.69%
of every gradient step at depth was about something we never asked for — which means every
"from-scratch" control this project has ever run was handed a free ascending curriculum by our own loss
function, including v55's arm E.

This module credits ONLY the commanded goal. v55's hidden_recipe_v55.py is left untouched so the frozen
v55 mechanism still reproduces bit-identically.

THE RISK, stated before running: at depth only ~5-23 of 1024 env-episodes reach the goal, so
commanded-only credit yields roughly 20-80 samples per round against v55's 8k-33k — a ~500x data cut.
It may starve every arm. That is exactly what the gate below is for.
"""

import torch

from ragnarok.infrastructure.device import DEVICE
from scripts.meta_manager_v51 import MAX_ITEMS, N_FEAT
from scripts.hidden_recipe_v55 import HiddenEnv, collect_episode, GOAL_COL

INSTR_MASK = True          # v56 instrument fix: never argmax into an already-failed item at eval


def relabel_commanded(states, actions, unlockstep, max_samples, goal, gamma=0.7):
    """Hindsight, but ONLY toward the goal that was commanded.

    Keeps (s_t, a_t, goal) iff this env actually reached `goal`, at some step u >= t, with the same
    geometric recency weight gamma^(u-t) as v55's M6. Every other item unlocked along the way is
    DISCARDED — that incidental credit is the free curriculum we are removing."""
    T, N, _ = states.shape
    u = unlockstep[:, goal]                                        # (N,) first step goal was unlocked
    t_idx = torch.arange(T, device=DEVICE).view(T, 1)
    lag = u.view(1, N) - t_idx                                     # (T,N)
    valid = (lag >= 0) & (u.view(1, N) >= 0)
    if gamma < 1.0:
        p = torch.pow(torch.tensor(gamma, device=DEVICE), lag.clamp(min=0).float())
        valid = valid & (torch.rand(valid.shape, device=DEVICE) < p)
    idx = valid.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return None, None, 0
    if idx.shape[0] > max_samples:
        idx = idx[torch.randperm(idx.shape[0], device=DEVICE)[:max_samples]]
    t, n = idx[:, 0], idx[:, 1]
    s = states[t, n].clone().reshape(-1, MAX_ITEMS, N_FEAT)
    s[:, goal, GOAL_COL] = 1.0
    return s.reshape(s.shape[0], -1), actions[t, n], int(idx.shape[0])


@torch.no_grad()
def eval_goal_fixed(spec, skill, composer, cfg, seed, goal, n=256, command=None):
    """v55 eval + the v56 instrument fix (mask items already attempted-and-failed this episode, which
    otherwise make the observation bit-identical and trap argmax forever). Applied to EVERY arm."""
    env = HiddenEnv(n, spec, skill, cfg, seed=seed + 9,
                    goal=goal if command is None else command, hidden=True)
    got = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["macro_budget"]):
        logits, _ = composer.net(obs)
        if INSTR_MASK:
            f = obs.reshape(n, MAX_ITEMS, N_FEAT)
            logits = logits.masked_fill((f[..., 2] > 0.5) & (f[..., 3] < 0.5), -1e9)
        obs, _, _, _, _ = env.step(logits.argmax(-1))
        got |= env.post_unlocked[:, goal]
    return float(got.float().mean())


def run_goal_commanded(spec, skill, composer, buf, cfg, seed, goal, gamma=0.7, r_max=None):
    """One goal under the FIXED credit rule. Returns the same fields as v55's run_goal plus the sample
    accounting the gate needs (how much data the commanded-only rule actually yields)."""
    env = HiddenEnv(cfg["num_envs"], spec, skill, cfg, seed=seed, goal=goal, hidden=True)
    env._prim = 0
    zs = eval_goal_fixed(spec, skill, composer, cfg, seed, goal)
    master, rounds, n_eval = zs, 0, 1
    demos, samples = [], []
    for r in range(r_max if r_max is not None else cfg["r_max"]):
        d = k = 0
        for _ in range(cfg["episodes_per_round"]):
            s, a, us = collect_episode(env, composer, cfg["epsilon"], cfg["temp"], goal)
            d += int((us[:, goal] >= 0).sum())
            ss, aa, n_kept = relabel_commanded(s, a, us, cfg["max_samples_per_ep"], goal, gamma)
            k += n_kept
            if ss is not None:
                buf.add(ss, aa)
        demos.append(d); samples.append(k)
        composer.train_steps(buf, cfg["train_steps_per_round"])
        rounds = r + 1
        master = eval_goal_fixed(spec, skill, composer, cfg, seed, goal)
        n_eval += 1
        if master >= cfg["thresh"]:
            break
    ev = n_eval * 256 * cfg["macro_budget"] * cfg["option_timeout"]
    return dict(goal=goal, zero_shot=round(zs, 3), rounds=rounds, prim=env._prim + ev,
                collect_prim=env._prim, eval_prim=ev, demos_per_round=demos,
                samples_per_round=samples, att=env._att, fail=env._fail,
                repeat_prev_succ=env._repeat_prev_succ, first_try_ok=env._first_try_ok,
                buf_n=buf.n, master=round(master, 3), mastered=bool(master >= cfg["thresh"]))
