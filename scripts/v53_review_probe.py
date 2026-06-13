"""v53 ADVERSARIAL REVIEW PROBE (reviewer-run, not part of the prereg).

Questions this probe answers, in order of severity:
 P1. Amnesic B on task 1 (d6, the flywheel's failure; greedy=0.984 there in v51):
     does a FRESH composer master it? If yes -> the lifelong buffer plausibly CAUSED a failure
     (negative transfer), hidden by B only being run on {0,3,6,9}.
 P2. Final (post-t6) composer zero-shot on t1 and t5: did "maturation" ever fix the failed d6?
     If zs(t1) ~ 0 with the final composer, the t9-d6-free vs t1-d6-fail contrast is tree identity,
     not maturation.
 P3. ONE-TASK control: train a fresh composer on task 0 only (exactly arm-B t0), then zero-shot
     eval on t2,t4,t7,t8,t9 + held-out. If it already clears 0.6 on the late tasks, the
     "compounding across a stream" reduces to "the first task teaches a general reflex".
 P4. Skill-sensitivity of the published late-stream numbers: re-eval final-composer zs on
     t7,t8,t9 + held-out with THIS process's independently retrained skill (the published run
     crossed a process boundary and silently swapped skills; checkpoint only stored composer+buffer).
 P5. Buffer forensics on v53_ckpt_d7_s0.pt: per-tree provenance of the 400k "lifelong" samples
     (FIFO cap vs ~630k generated -> is task-0 data even still in there?), fraction of direct
     1-step demos (action == goal), goal-achievable-now stats.
 P6. Greedy baseline on t1/t5 with this skill draw (masterability witness).

Writes craft_v6_out/v53_review_probe.json. READ-ONLY w.r.t. all v53 artifacts.
"""

import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import gen_tree
from scripts.depth_scaling_v49 import N_ITEMS_FOR_DEPTH
from scripts.childhood_v50 import train_childhood
from scripts.meta_manager_v51 import greedy_master_rate, MAX_ITEMS, N_FEAT
from scripts.flywheel_v53 import Composer, Buffer, eval_master, run_task, GOAL_COL

OUT = os.path.join("craft_v6_out", "v53_review_probe.json")
CKPT = os.path.join("craft_v6_out", "v53_ckpt_d7_s0.pt")

CFG = dict(num_envs=256, grid=7, view=13, n_resource=4, rollout=32, entropy=0.02,
           nav_max_steps=40, skill_iters=400, option_timeout=40, macro_budget=20,
           episodes_per_round=4, train_steps_per_round=300, max_samples_per_ep=8192,
           epsilon=0.05, temp=1.0, thresh=0.6, task_budget=5e6, skill_stochastic=True,
           mgr_entropy=0.03, router_iters=0)
SEED = 0


def task_seed(k):
    return SEED + 11 * k + 1


def main():
    t0 = time.perf_counter()
    res = {}
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    ni = N_ITEMS_FOR_DEPTH[7]
    skill_specs = [gen_tree(1000 + i, n_items=ni) for i in range(8)]
    stream = [gen_tree(5000 + i, n_items=ni) for i in range(10)]
    heldout = [gen_tree(9000 + i, n_items=ni) for i in range(4)]
    res["stream_depths"] = [int(s["depth"][s["target"]]) for s in stream]

    # ---------- P5 first: buffer forensics (no skill needed) ----------
    st = torch.load(CKPT, map_location=DEVICE)
    bs, ba = st["buf_s"], st["buf_a"]
    n = bs.shape[0]
    f = bs.reshape(n, MAX_ITEMS, N_FEAT)
    sigs = {}
    for name, specs in (("stream", stream), ("skilltree", skill_specs), ("heldout", heldout)):
        for i, s in enumerate(specs):
            key = tuple(1 if s["kind"][j] == "R" else 0 for j in range(ni))
            sigs.setdefault(key, []).append(f"{name}{i}")
    res["signature_collisions"] = {str(k): v for k, v in sigs.items() if len(v) > 1}
    is_res = (f[:, :ni, 5] > 0.5).to(torch.int8)
    prov = {}
    matched = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    for key, names in sigs.items():
        pat = torch.tensor(key, dtype=torch.int8, device=DEVICE)
        m = (is_res == pat).all(-1)
        prov["/".join(names)] = int(m.sum())
        matched |= m
    prov["UNMATCHED"] = int((~matched).sum())
    res["buffer_provenance"] = prov
    goal_idx = f[..., GOAL_COL].argmax(-1)
    has_goal = f[..., GOAL_COL].sum(-1) > 0.5
    direct = (ba == goal_idx) & has_goal
    res["buffer_n"] = n
    res["buffer_direct_demo_frac"] = round(float(direct.float().mean()), 4)
    ar = torch.arange(n, device=DEVICE)
    g_craftable = f[ar, goal_idx, 2] > 0.5
    g_collectable = f[ar, goal_idx, 3] > 0.5
    res["goal_achievable_now_frac"] = round(float((g_craftable | g_collectable).float().mean()), 4)
    res["goal_is_resource_frac"] = round(float((f[ar, goal_idx, 5] > 0.5).float().mean()), 4)
    res["action_is_goal_when_achievable"] = round(
        float((direct & (g_craftable | g_collectable)).float().sum() /
              max(1, int((g_craftable | g_collectable).sum()))), 4)
    print(f"[P5] buffer {n} samples | provenance {prov} | direct-demo {res['buffer_direct_demo_frac']} "
          f"| goal-achievable-now {res['goal_achievable_now_frac']}", flush=True)

    # ---------- shared skill (3rd independent draw; published run used 2 different ones) ----------
    skill, c_skill = train_childhood(skill_specs, CFG, SEED)
    print(f"[skill] retrained ({c_skill/1e6:.2f}M) | {time.perf_counter()-t0:.0f}s", flush=True)

    # ---------- P6: greedy masterability witness on the two failed tasks ----------
    res["greedy_t1"] = round(greedy_master_rate(stream[1], skill, CFG, task_seed(1)), 3)
    res["greedy_t5"] = round(greedy_master_rate(stream[5], skill, CFG, task_seed(5)), 3)
    res["greedy_t9"] = round(greedy_master_rate(stream[9], skill, CFG, task_seed(9)), 3)
    print(f"[P6] greedy t1={res['greedy_t1']} t5={res['greedy_t5']} t9={res['greedy_t9']}", flush=True)

    # ---------- P2 + P4: final composer (exact ckpt) re-evals with this skill ----------
    fin = Composer()
    fin.net.load_state_dict(st["net"])
    for k in (1, 5, 7, 8, 9):
        res[f"final_zs_t{k}"] = round(eval_master(stream[k], skill, fin, CFG, task_seed(k)), 3)
    res["final_heldout"] = [round(eval_master(s, skill, fin, CFG, SEED + 777), 3) for s in heldout]
    print(f"[P2/P4] final-composer zs: t1={res['final_zs_t1']} t5={res['final_zs_t5']} "
          f"t7={res['final_zs_t7']} t8={res['final_zs_t8']} t9={res['final_zs_t9']} "
          f"heldout={res['final_heldout']} | {time.perf_counter()-t0:.0f}s", flush=True)
    with open(OUT, "w") as fp:
        json.dump(res, fp, indent=2)

    # ---------- P3: ONE-TASK control (fresh composer, task 0 only, then broad zero-shot) ----------
    c0, b0 = Composer(), Buffer()
    r0 = run_task(stream[0], skill, c0, b0, CFG, task_seed(0))
    res["onetask_t0"] = r0
    for k in (2, 4, 7, 8, 9):
        res[f"onetask_zs_t{k}"] = round(eval_master(stream[k], skill, c0, CFG, task_seed(k)), 3)
    res["onetask_zs_t1"] = round(eval_master(stream[1], skill, c0, CFG, task_seed(1)), 3)
    res["onetask_heldout"] = [round(eval_master(s, skill, c0, CFG, SEED + 777), 3) for s in heldout]
    print(f"[P3] one-task(t0 only) -> t0 {r0} | zs t2={res['onetask_zs_t2']} t4={res['onetask_zs_t4']} "
          f"t7={res['onetask_zs_t7']} t8={res['onetask_zs_t8']} t9={res['onetask_zs_t9']} "
          f"t1={res['onetask_zs_t1']} heldout={res['onetask_heldout']} | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    with open(OUT, "w") as fp:
        json.dump(res, fp, indent=2)

    # ---------- P1: amnesic B on the failed task 1 ----------
    r1 = run_task(stream[1], skill, Composer(), Buffer(), CFG, task_seed(1))
    res["B_t1"] = r1
    print(f"[P1] amnesic B on t1 (d6): {r1} | {time.perf_counter()-t0:.0f}s", flush=True)

    with open(OUT, "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"[done] {time.perf_counter()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
