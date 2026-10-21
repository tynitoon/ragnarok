"""v80 gates G1 and G2 offline, plus the label check (docs/ARC4_V80_DESIGN.md §4; thresholds there).

The theory is fitted on the agent's TRAIN traces only, then judged on the evaluator's HELD-OUT
roots (fresh starts, distinct screens), with the agent's own labels. Every interval is a 95 %
bootstrap over ROLLOUTS (roots of one rollout are near duplicates). Branches censored by ALE's
game over (without the agent's own death signal) before the window ends are excluded and counted.
  label check : declared before running: a death label at step t is right if the RAM shows a life
                lost or game over in [t-8, t+60] (RAM may lag the screen); a score label is right if
                the true score rises in [t-4, t+4]. Precision >= 0.9 with >= 20 labels, on held-out
                AND on the sampled train branches; below 0.9: STOP; fewer than 20: no verdict.
  G1 (Assault): within-root death ranking >= 0.70, lower bound > 0.5, >= 30 pairs; Brier skill of
                P(death within 32) >= 0.10 against the HELD-OUT per-slot climatology (lower bound
                > 0); same for the score count within 32 (1 - MSE / MSE_climatology).
  G1 (Freeway): imagined position of the objects at k = 16 within 4 px (L-infinity, one-to-one by
                type) for >= 80 %, and median error <= 0.8 x constant-velocity extrapolation.
  G2          : real 64-step value of the theory's choice minus that of the best constant
                deviation (chosen on train) >= +0.10, lower bound > 0, AND theory minus the conv_q
                reflex trained on the same train roots >= +0.05, paired lower bound > 0.
Usage: C:/Python314/python.exe -m scripts.theory_offline_v80 TRAIN.pkl HELDOUT.pkl [H]
"""
import json
import pickle
import sys
import time

import numpy as np

from ragnarok.agent_v80.imagine import Model, Scene, decide, imagine
from ragnarok.agent_v80.lab import branch_value
from ragnarok.agent_v80.theory import I, Theory, me_row, predict_motion

K_EVAL = 32
D_BEFORE, D_AFTER, S_WIN, MIN_LABELS = 8, 60, 4, 20


def cboot(stat, G, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    m = []
    for _ in range(n):
        idx = np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))])
        x = stat(idx)
        if np.isfinite(x):
            m.append(x)
    if not m:
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def label_check(D, truth):
    dp = dn = sp = sn = 0
    for i, tr_t in zip(truth["idx"], truth["truth"]):
        for s, tr in D["sibsets"][i].items():
            T = tr_t[s]
            for t in np.nonzero(tr["death"])[0]:
                dn += 1
                lo, hi = max(0, t - D_BEFORE), t + D_AFTER + 1
                dp += int(T["lives_lost"][lo:hi].any() or T["game_over"][lo:hi].any())
            for t in np.nonzero(tr["score"])[0]:
                sn += 1
                sp += int((T["score_gain"][max(0, t - S_WIN):t + S_WIN + 1] > 0).any())
    f = lambda a, b: round(a / b, 3) if b else None
    verdict = lambda p, n: None if n < MIN_LABELS else bool(p >= 0.9)
    return dict(death_labels=dn, death_precision=f(dp, dn), death_ok=verdict(dp / max(1, dn), dn),
                score_labels=sn, score_precision=f(sp, sn), score_ok=verdict(sp / max(1, sn), sn))


def first(tr, n):
    """(died within n, score events within n), or None when censored by `over` before n steps."""
    if tr["over"] and tr["end"] < n:
        return None
    return bool(tr["death"][:n].any()), float(tr["score"][:n].sum())


def conv_q_gain(T, Hd, const_dev, seeds=(0, 1, 2)):
    """The reflex baseline on the SAME train roots: PolicyNet regressed on the measured 64-step
    values of every slot, choosing by argmax. Input rebuilt exactly as offline_v75 (previous view
    from the previous screen, centred on my previous position)."""
    from ragnarok.agent_v71.policy import ego_view
    from ragnarok.agent_v71.adapter import SLOTS
    from scripts.offline_v75 import train, choose

    def inputs(D):
        Im, Ps = [], []
        for r, sib in zip(D["roots"], D["sibsets"]):
            m = next(iter(sib.values()))["me"][0]
            me = (float(m[0]), float(m[1])) if np.isfinite(m[0]) else None
            pm = None if r["pme"] is None else (float(r["pme"][0]), float(r["pme"][1]))
            pf = None if r["prev_obs"] is None else ego_view(r["prev_obs"], D["bg"], pm, None)[2]
            img, pos, _ = ego_view(r["obs"], D["bg"], me, pf)
            Im.append(img.astype(np.uint8)); Ps.append(pos)
        return np.stack(Im), np.stack(Ps)

    def qmat(D):
        Q = np.full((len(D["sibsets"]), len(SLOTS)), np.nan, np.float32)
        for i, sib in enumerate(D["sibsets"]):
            for s, tr in sib.items():
                Q[i, s] = branch_value(tr)
        return Q
    It, Pt = inputs(T); Ih, Ph = inputs(Hd)
    Qt, Qh = qmat(T), qmat(Hd)
    M = np.broadcast_to(np.asarray(T["mask"], bool), Qt.shape).copy()
    Mh = np.broadcast_to(np.asarray(T["mask"], bool), Qh.shape).copy()
    vals = []
    for sd in seeds:
        net = train(It, Pt, Qt, M, T["base"], "q", sd)
        ch = choose(net, Ih, Ph, Mh)
        vals.append([Qh[i, c] - Qh[i, const_dev] for i, c in enumerate(ch)])
    return np.mean(vals, 0)


def freeway_g1(th, Hd, k=16, tol=4.0):
    from scipy.optimize import linear_sum_assignment
    eth, ecv = [], []
    for sib in Hd["sibsets"]:
        tr = sib[Hd["base"]]
        if len(tr["objs"]) <= k:
            continue
        A, B = tr["objs"][0], tr["objs"][k]
        km = me_row(A, tr["me"][0])
        for key, ty in th.types.items():
            a = [r for j, r in enumerate(A) if j != km and r[I["missed"]] == 0 and th.key_of(r) == key and not th._is_gauge_row(r)]
            Y = np.array([(r[I["cx"]], r[I["cy"]]) for r in B if r[I["missed"]] == 0 and th.key_of(r) == key])
            if not a or not len(Y):
                continue
            Pth = np.array([predict_motion(ty["rule"], r[I["cx"]], r[I["cy"]], r[I["vx"]], r[I["vy"]], k, ty["bounds"], speed=ty["speed"]) for r in a])
            Pcv = np.array([(r[I["cx"]] + k * r[I["vx"]], r[I["cy"]] + k * r[I["vy"]]) for r in a])
            for P, out in ((Pth, eth), (Pcv, ecv)):
                C = np.abs(P[:, None, :] - Y[None]).max(-1)
                ri, ci = linear_sum_assignment(C)
                out += C[ri, ci].tolist()
    if not eth:
        return dict(n=0)
    return dict(n=len(eth), within=round(float(np.mean(np.array(eth) <= tol)), 3),
                ratio=round(float(np.median(eth) / max(1e-9, np.median(ecv))), 3),
                passed=bool(np.mean(np.array(eth) <= tol) >= 0.8 and np.median(eth) <= 0.8 * max(1e-9, np.median(ecv))))


def main(train_p, held_p, H32=32):
    t0 = time.perf_counter()
    if "_truth" in train_p or "_truth" in held_p:
        raise ValueError("pass the agent files; the truth files are found by name")
    T = pickle.load(open(train_p, "rb"))
    Hd = pickle.load(open(held_p, "rb"))
    hold = T["hold"]
    slots = sorted(T["sibsets"][0])
    th = Theory(slots, T["base"]).fit(T["sibsets"], hold=hold)
    open(train_p.replace(".pkl", "_theory.json"), "w").write(th.export())
    res = dict(game=T["game"], train_roots=len(T["sibsets"]), heldout_roots=len(Hd["sibsets"]),
               train_frames=T.get("frames"), base=T["base"], H=H32,
               labels_heldout=label_check(Hd, pickle.load(open(held_p.replace(".pkl", "_truth.pkl"), "rb"))),
               labels_train=label_check(T, pickle.load(open(train_p.replace(".pkl", "_truth.pkl"), "rb"))))
    rid = np.array([r["rid"] for r in Hd["roots"]])
    G = [np.nonzero(rid == g)[0] for g in np.unique(rid)]
    nr, k = len(Hd["sibsets"]), len(slots)
    # held-out climatology per slot (uncensored branches)
    clim = {}
    for s in slots:
        F = [first(sib[s], H32) for sib in Hd["sibsets"]]
        F = [f for f in F if f is not None]
        clim[s] = (np.mean([f[0] for f in F]) if F else 0.0, np.mean([f[1] for f in F]) if F else 0.0)
    rng = np.random.default_rng(0)
    B = np.full((nr, k), np.nan); Bc = np.full((nr, k), np.nan); Mx = np.full((nr, k), np.nan); Mc = np.full((nr, k), np.nan)
    ps, pc = np.zeros(nr), np.zeros(nr)
    censored, no_me = 0, 0
    const_dev = max(slots, key=lambda c: np.mean([branch_value(sib[c]) - branch_value(sib[T["base"]]) for sib in T["sibsets"]]))
    cg = np.zeros(nr)
    choice = []
    for ri, sib in enumerate(Hd["sibsets"]):
        tr0 = sib[slots[0]]
        sc = Scene(th, tr0["objs"][0], tr0["me"][0])
        no_me += int(sc.no_me)
        seeds = rng.integers(2 ** 31, size=K_EVAL)
        md = Model(th, sc.keys)
        pred, ys = {}, {}
        for j, s in enumerate(slots):
            P = imagine(th, sc, s, seeds, hold=hold, horizon=H32 - hold, leaf=False, md=md)
            pred[s] = (float(P[:, 2].mean()), float(P[:, 1].mean()))
            f = first(sib[s], H32)
            ys[s] = f
            if f is None:
                censored += 1
                continue
            B[ri, j] = (pred[s][0] - f[0]) ** 2; Bc[ri, j] = (clim[s][0] - f[0]) ** 2
            Mx[ri, j] = (pred[s][1] - f[1]) ** 2; Mc[ri, j] = (clim[s][1] - f[1]) ** 2
        for a in slots:
            for b in slots:
                if a < b and ys[a] is not None and ys[b] is not None and ys[a][0] != ys[b][0]:
                    pa, pb = pred[a][0], pred[b][0]
                    right = (pa > pb) if ys[a][0] else (pb > pa)
                    ps[ri] += 1.0 if right else (0.5 if pa == pb else 0.0)
                    pc[ri] += 1
        c, _ = decide(th, sc, rng, K=16, hold=hold, horizon=H32 - hold)
        choice.append(c)
        cg[ri] = branch_value(sib[c]) - branch_value(sib[const_dev])
    skill = lambda X, Xc: (lambda s: 1 - np.nansum(X[s]) / max(1e-9, np.nansum(Xc[s])))
    wrd = lambda s: ps[s].sum() / pc[s].sum() if pc[s].sum() else np.nan
    allr = np.arange(nr)
    deaths_seen = any(clim[s][0] > 0 for s in slots)        # no death in the held-out: death metrics do not apply
    res["G1"] = dict(wrd=round(float(wrd(allr)), 3) if pc.sum() else None, wrd_pairs=int(pc.sum()), wrd_ci=cboot(wrd, G),
                     death_brier_skill=round(float(skill(B, Bc)(allr)), 3) if deaths_seen else None,
                     death_ci=cboot(skill(B, Bc), G) if deaths_seen else None,
                     death_note=None if deaths_seen else "no death in the held-out branches: not applicable",
                     score_skill=round(float(skill(Mx, Mc)(allr)), 3), score_ci=cboot(skill(Mx, Mc), G),
                     censored_branches=censored, roots_without_me=no_me, rollouts=len(G))
    if T["game"] == "freeway":
        res["G1_freeway"] = freeway_g1(th, Hd)
    cq = conv_q_gain(T, Hd, const_dev)
    d = cg - cq
    res["G2"] = dict(const_dev=int(const_dev), gain_over_const_dev=round(float(cg.mean()), 3), ci=cboot(lambda s: cg[s].mean(), G),
                     deviate=round(float(np.mean([c != T["base"] for c in choice])), 3),
                     conv_q_gain_over_const_dev=round(float(np.mean(cq)), 3), conv_q_ci=cboot(lambda s: cq[s].mean(), G),
                     theory_minus_conv_q=round(float(np.mean(d)), 3), paired_ci=cboot(lambda s: d[s].mean(), G),
                     note="conv_q uses the v74 input whose whole-screen view misses Assault's heat gauge (ARC4_V75_OFFLINE.md)")
    res["secs"] = round(time.perf_counter() - t0)
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(held_p.replace(".pkl", f"_offline_H{H32}.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:3]))
