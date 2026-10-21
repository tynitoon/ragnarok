"""v80 gates G1 and G2 offline, plus the label check (docs/ARC4_V80_DESIGN.md §4; thresholds there).

The theory is fitted on the agent's TRAIN traces only, then judged on the evaluator's HELD-OUT
roots (distinct screens), with the agent's own labels; the evaluator's truth file is read only for
the label check (precision of the agent's death and score labels against RAM).
  label check : precision >= 0.9 for death labels (a RAM life loss or game over within +-8 steps)
                and score labels (a true-score increase within +-4 steps); else STOP.
  G1          : within-root death ranking (pairs of sibling slots, one dying within 32 steps, the
                other not) >= 0.70 with lower bound > 0.5 and >= 30 pairs; Brier skill of P(death
                within 32) against per-slot climatology >= 0.10 (lower bound > 0); same for the
                score count within 32 (1 - MSE / MSE_climatology).
  G2          : the real 64-step value of the theory's choice minus that of the best constant
                deviation (chosen on train) >= +0.10 events, lower bound > 0, AND theory minus the
                conv_q reflex trained on the same train roots >= +0.05, paired lower bound > 0.
Usage: C:/Python314/python.exe -m scripts.theory_offline_v80 TRAIN.pkl HELDOUT.pkl [H]
"""
import json
import pickle
import sys
import time

import numpy as np

from ragnarok.agent_v80.imagine import Scene, decide, imagine
from ragnarok.agent_v80.lab import branch_value
from ragnarok.agent_v80.theory import Theory

K_EVAL = 32


def boot(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    m = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def label_check(H, truth):
    dp, dn, sp, sn = 0, 0, 0, 0
    for sib, tr_t in zip(H["sibsets"], truth["truth"]):
        for s, tr in sib.items():
            T = tr_t[s]
            L = len(tr["death"])
            for t in np.nonzero(tr["death"])[0]:
                dn += 1
                lo, hi = max(0, t - 8), min(L, t + 9)
                dp += int(T["lives_lost"][lo:hi].any() or (tr["over"] and t >= L - 9))
            for t in np.nonzero(tr["score"])[0]:
                sn += 1
                lo, hi = max(0, t - 4), min(L, t + 5)
                sp += int((T["score_gain"][lo:hi] > 0).any())
    return dict(death_labels=dn, death_precision=round(dp / max(1, dn), 3),
                score_labels=sn, score_precision=round(sp / max(1, sn), 3))


def conv_q_gain(T, Hd, slots, const_dev, seeds=(0, 1, 2)):
    """The reflex baseline of v74/v75 on the SAME train roots: PolicyNet regressed on the measured
    64-step values of every slot, choosing by argmax; scored by the real held-out values."""
    from ragnarok.agent_v71.policy import ego_view
    from ragnarok.agent_v71.adapter import SLOTS
    from scripts.offline_v75 import train, choose

    def inputs(D):
        I, P = [], []
        for r in D["roots"]:
            me = None if r["me"] is None else (float(r["me"][0]), float(r["me"][1]))
            img, pos, _ = ego_view(r["obs"], D["bg"], me, r["prev"])
            I.append(img.astype(np.uint8)); P.append(pos)
        return np.stack(I), np.stack(P)

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


def main(train_p, held_p, H32=32):
    t0 = time.perf_counter()
    if "_truth" in train_p or "_truth" in held_p:
        raise ValueError("pass the agent files; the truth file is found by name")
    T = pickle.load(open(train_p, "rb"))
    Hd = pickle.load(open(held_p, "rb"))
    truth = pickle.load(open(held_p.replace(".pkl", "_truth.pkl"), "rb"))
    hold = T["hold"]
    slots = sorted(T["sibsets"][0])
    th = Theory(slots, T["base"]).fit(T["sibsets"], hold=hold)
    res = dict(game=T["game"], train_roots=len(T["sibsets"]), heldout_roots=len(Hd["sibsets"]),
               train_frames=T.get("frames"), base=T["base"], H=H32)
    res["labels"] = label_check(Hd, truth)
    open(train_p.replace(".pkl", "_theory.json"), "w").write(th.export())

    # climatology per slot on TRAIN: death within H32, score count within H32
    def first(tr, n):
        return bool(tr["death"][:n].any()), float(tr["score"][:n].sum())
    clim = {s: np.mean([[*first(sib[s], H32)] for sib in T["sibsets"]], 0) for s in slots}
    rng = np.random.default_rng(0)
    pairs, bs, bs_c, ms, ms_c = [], [], [], [], []
    choice_gain, choice = [], []
    q64 = lambda tr: branch_value(tr)
    const_dev = max(slots, key=lambda c: np.mean([q64(sib[c]) - q64(sib[T["base"]]) for sib in T["sibsets"]]))
    for sib in Hd["sibsets"]:
        tr0 = sib[slots[0]]
        sc = Scene(th, tr0["objs"][0], tr0["me"][0])
        seeds = rng.integers(2 ** 31, size=K_EVAL)
        pred = {}
        for s in slots:
            P = imagine(th, sc, s, seeds, hold=hold, horizon=H32 - hold, leaf=False)
            pred[s] = (float(P[:, 2].mean()), float(P[:, 1].mean()))
            y_d, y_s = first(sib[s], H32)
            bs.append((pred[s][0] - y_d) ** 2); bs_c.append((clim[s][0] - y_d) ** 2)
            ms.append((pred[s][1] - y_s) ** 2); ms_c.append((clim[s][1] - y_s) ** 2)
        for a in slots:
            for b in slots:
                if a < b:
                    da, db = first(sib[a], H32)[0], first(sib[b], H32)[0]
                    if da != db:
                        pa, pb = pred[a][0], pred[b][0]
                        right = (pa > pb) if da else (pb > pa)
                        pairs.append(1.0 if right else (0.5 if pa == pb else 0.0))
        c, _ = decide(th, sc, rng, K=16, hold=hold, horizon=H32 - hold)
        choice.append(c)
        choice_gain.append(q64(sib[c]) - q64(sib[const_dev]))
    bsk = 1 - np.sum(bs) / max(1e-9, np.sum(bs_c))
    msk = 1 - np.sum(ms) / max(1e-9, np.sum(ms_c))
    # paired bootstrap per root for the skills
    nr = len(Hd["sibsets"])
    k = len(slots)
    B = np.array(bs).reshape(nr, k); Bc = np.array(bs_c).reshape(nr, k)
    Mx = np.array(ms).reshape(nr, k); Mc = np.array(ms_c).reshape(nr, k)
    rb = np.random.default_rng(1)
    sk_b, sk_m = [], []
    for _ in range(2000):
        i = rb.integers(0, nr, nr)
        sk_b.append(1 - B[i].sum() / max(1e-9, Bc[i].sum())); sk_m.append(1 - Mx[i].sum() / max(1e-9, Mc[i].sum()))
    res["G1"] = dict(wrd=round(float(np.mean(pairs)), 3) if pairs else None, wrd_pairs=len(pairs), wrd_ci=boot(pairs),
                     death_brier_skill=round(float(bsk), 3), death_ci=[round(float(np.percentile(sk_b, 2.5)), 3), round(float(np.percentile(sk_b, 97.5)), 3)],
                     score_skill=round(float(msk), 3), score_ci=[round(float(np.percentile(sk_m, 2.5)), 3), round(float(np.percentile(sk_m, 97.5)), 3)])
    res["G2"] = dict(const_dev=int(const_dev), gain_over_const_dev=round(float(np.mean(choice_gain)), 3),
                     ci=boot(choice_gain), deviate=round(float(np.mean([c != T["base"] for c in choice])), 3))
    cq = conv_q_gain(T, Hd, slots, const_dev)
    d = np.array(choice_gain) - cq
    res["G2"].update(conv_q_gain_over_const_dev=round(float(np.mean(cq)), 3), conv_q_ci=boot(cq),
                     theory_minus_conv_q=round(float(np.mean(d)), 3), paired_ci=boot(d))
    res["secs"] = round(time.perf_counter() - t0)
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(held_p.replace(".pkl", f"_offline_H{H32}.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:3]))
