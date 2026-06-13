"""v54 — CLEAN MEASUREMENT of compounding (prereg FROZEN before this file).

v53 was NULL but mostly a MEASUREMENT failure: task difficulty (8..27 productions, some infeasible)
swamped a possibly-real small signal. v54 changes NOTHING in the mechanism (same self-imitation
composer/buffer/options as v53) — it only (1) restricts the task stream to UNIFORM, FEASIBLE difficulty
(production-count in a fixed band, all << macro_budget) and (2) runs the amnesic control B on EVERY task,
so the confound-free per-task benefit Delta_k = cost_B(k) - cost_A(k) is measured everywhere, with
per-seed spread. Primary claim: accumulated experience makes NEW tasks cheaper (mean Delta>0, >=3 seeds).

Usage: python -m scripts.clean_compound_v54 [--smoke] [--seed 0] [--resume]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import gen_tree
from scripts.depth_scaling_v49 import N_ITEMS_FOR_DEPTH
from scripts.childhood_v50 import train_childhood
from scripts.flywheel_v53 import Composer, Buffer, eval_master, run_task


def _prod_tree(spec, i, depth=0):
    """Productions to make ONE unit of item i (inputs CONSUMED -> tree-expanded). Acyclic by gen_tree."""
    if depth > 40 or spec["kind"][i] == "R":
        return 1
    return 1 + sum(c * _prod_tree(spec, j, depth + 1) for j, c in spec["inputs"][i].items())


def _tools_in_closure(spec, i, acc, depth=0):
    if depth > 40:
        return
    if spec["kind"][i] == "R" and spec["tool"][i] >= 0:
        acc.add(spec["tool"][i]); _tools_in_closure(spec, spec["tool"][i], acc, depth + 1)
    for t in spec["tools"][i]:
        acc.add(t); _tools_in_closure(spec, t, acc, depth + 1)
    for j in spec["inputs"][i]:
        _tools_in_closure(spec, j, acc, depth + 1)


def production_count(spec):
    """Difficulty proxy: total productions for the target (consumed inputs tree-expanded; tools once)."""
    tgt = spec["target"]
    tools = set(); _tools_in_closure(spec, tgt, tools)
    return _prod_tree(spec, tgt) + sum(_prod_tree(spec, t) for t in tools)


def build_pool(n_items, lo, hi, n_stream, n_heldout, base_seed):
    """Scan candidate trees; keep those with production_count in [lo,hi] (uniform, feasible)."""
    stream, held, counts, s = [], [], [], base_seed
    while len(stream) + len(held) < n_stream + n_heldout and s < base_seed + 400:
        spec = gen_tree(s, n_items=n_items)
        pc = production_count(spec)
        if lo <= pc <= hi:
            (stream if len(stream) < n_stream else held).append((s, spec, pc))
        s += 1
    return stream, held, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--n-stream", type=int, default=16)
    p.add_argument("--n-heldout", type=int, default=4)
    p.add_argument("--prod-lo", type=int, default=10)
    p.add_argument("--prod-hi", type=int, default=16)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--view", type=int, default=13)
    p.add_argument("--n-resource", type=int, default=4)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--nav-max-steps", type=int, default=40)
    p.add_argument("--skill-iters", type=int, default=400)
    p.add_argument("--option-timeout", type=int, default=40)
    p.add_argument("--macro-budget", type=int, default=40)
    p.add_argument("--task-budget", type=float, default=3e6)
    p.add_argument("--episodes-per-round", type=int, default=4)
    p.add_argument("--train-steps-per-round", type=int, default=300)
    p.add_argument("--max-samples-per-ep", type=int, default=8192)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--thresh", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.depth, args.n_stream, args.n_heldout = 5, 4, 2
        args.skill_iters, args.task_budget, args.train_steps_per_round = 250, 1.2e6, 150

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "n_resource", "rollout", "entropy", "nav_max_steps",
            "skill_iters", "option_timeout", "macro_budget", "episodes_per_round",
            "train_steps_per_round", "max_samples_per_ep", "epsilon", "temp", "thresh")}
    cfg["task_budget"] = args.task_budget
    cfg["skill_stochastic"] = True
    cfg["mgr_entropy"] = 0.03; cfg["router_iters"] = 0
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ni = N_ITEMS_FOR_DEPTH[args.depth]
    skill_specs = [gen_tree(1000 + i, n_items=ni) for i in range(8)]
    stream, held, _ = build_pool(ni, args.prod_lo, args.prod_hi, args.n_stream, args.n_heldout, 5000)
    tag = f"_d{args.depth}" + ("_smoke" if args.smoke else "")
    jpath = os.path.join(args.out_dir, f"v54_clean{tag}_s{args.seed}.json")
    ppath = os.path.join(args.out_dir, f"v54_ckpt{tag}_s{args.seed}.pt")

    print(f"[v54 clean] device={DEVICE} | depth~{args.depth} | stream {len(stream)} UNIFORM tasks "
          f"(prod {args.prod_lo}-{args.prod_hi}, macro {args.macro_budget}) | A vs amnesic-B every task | "
          f"3-seed clean measure", flush=True)
    print(f"  stream production-counts: {[pc for _, _, pc in stream]} | held-out: "
          f"{[pc for _, _, pc in held]}", flush=True)
    t0 = time.perf_counter()
    skill, c_skill = train_childhood(skill_specs, cfg, args.seed)
    print(f"  childhood skill ready ({c_skill/1e6:.2f}M) | {time.perf_counter()-t0:.0f}s", flush=True)

    results = dict(seed=args.seed, depth=args.depth, c_skill=c_skill,
                   stream_pc=[pc for _, _, pc in stream], rows=[])
    composer, buf = Composer(), Buffer()
    done = -1
    if args.resume and os.path.exists(jpath) and os.path.exists(ppath):
        prev = json.load(open(jpath))
        results["rows"] = prev.get("rows", [])
        done = max([r["task"] for r in results["rows"]], default=-1)
        st = torch.load(ppath, map_location=DEVICE)
        composer.net.load_state_dict(st["net"]); composer.opt.load_state_dict(st["opt"])
        n = st["buf_s"].shape[0]; buf.s[:n], buf.a[:n], buf.n, buf.ptr = st["buf_s"], st["buf_a"], n, n % buf.cap
        print(f"  RESUME: tasks <= {done} done, buffer {n}", flush=True)

    def _ck():
        json.dump(results, open(jpath, "w"), indent=2)
        torch.save(dict(net=composer.net.state_dict(), opt=composer.opt.state_dict(),
                        buf_s=buf.s[:buf.n].clone(), buf_a=buf.a[:buf.n].clone()), ppath)

    for k, (sd, spec, pc) in enumerate(stream):
        if k <= done:
            continue
        rA = run_task(spec, skill, composer, buf, cfg, args.seed + 11 * k + 1)       # accumulating
        rB = run_task(spec, skill, Composer(), Buffer(), cfg, args.seed + 11 * k + 1)  # amnesic control
        delta = rB["cost"] - rA["cost"]
        row = dict(task=k, pc=pc, zs_A=rA["zero_shot"], cost_A=rA["cost"], master_A=rA["mastered"],
                   cost_B=rB["cost"], master_B=rB["mastered"], delta=delta)
        results["rows"].append(row)
        print(f"    task {k:>2} (pc{pc}): A zs {rA['zero_shot']:.2f} cost {rA['cost']/1e6:.2f}M "
              f"({'M' if rA['mastered'] else 'x'}) | B cost {rB['cost']/1e6:.2f}M "
              f"({'M' if rB['mastered'] else 'x'}) | Delta {delta/1e6:+.2f}M | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
        _ck()

    ho = [round(eval_master(s, skill, composer, cfg, args.seed + 777), 3) for _, s, _ in held]
    results["heldout_zero_shot"] = ho
    rows = results["rows"]
    half = len(rows) // 2
    ca_e = sum(r["cost_A"] for r in rows[:half]) / max(1, half)
    ca_l = sum(r["cost_A"] for r in rows[half:]) / max(1, len(rows) - half)
    zs_e = sum(r["zs_A"] for r in rows[:half]) / max(1, half)
    zs_l = sum(r["zs_A"] for r in rows[half:]) / max(1, len(rows) - half)
    mean_delta = sum(r["delta"] for r in rows) / max(1, len(rows))
    pos_delta = sum(r["delta"] > 0 for r in rows)
    primary = mean_delta > 0 and pos_delta >= 0.6 * len(rows)
    secondary = (ca_l < 0.6 * ca_e) or (zs_l >= 0.5 and zs_e < 0.2)
    results.update(heldout_zero_shot=ho, cost_A_early=ca_e, cost_A_late=ca_l, zs_early=zs_e,
                   zs_late=zs_l, mean_delta=mean_delta, pos_delta_frac=pos_delta / max(1, len(rows)),
                   primary=primary, secondary=secondary)
    verdict = (
        f"v54 seed {args.seed}: mean Delta {mean_delta/1e6:+.2f}M ({pos_delta}/{len(rows)} tasks B>A) | "
        f"cost_A early {ca_e/1e6:.2f}->late {ca_l/1e6:.2f}M | zs {zs_e:.2f}->{zs_l:.2f} | held-out {ho} "
        f"| primary {primary}, secondary {secondary}")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    _ck()


if __name__ == "__main__":
    main()
