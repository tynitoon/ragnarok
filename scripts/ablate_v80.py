"""v80 diagnostic (development): which part of the theory drives its death predictions on Assault?

Same G1 death metrics as theory_offline_v80 (held-out roots, H = 32, held-out climatology, agent
labels), for ablated theories: full, without omens, without gauges, without both, without contact
deaths. Also prints the mean imagined vs actual death rate.
Usage: C:/Python314/python.exe -m scripts.ablate_v80 TRAIN.pkl HELDOUT.pkl"""
import copy
import pickle
import sys

import numpy as np

from ragnarok.agent_v80.imagine import Model, Scene, imagine
from ragnarok.agent_v80.theory import OUTCOMES, Theory
from scripts.theory_offline_v80 import first

H32, K = 32, 32


def run(th, Hd, slots, hold, label):
    rng = np.random.default_rng(0)
    clim = {s: np.mean([f[0] for f in (first(sib[s], H32) for sib in Hd["sibsets"]) if f is not None]) for s in slots}
    bs, bc, pairs, pred_all, act_all = [], [], [], [], []
    for sib in Hd["sibsets"]:
        tr0 = sib[slots[0]]
        sc = Scene(th, tr0["objs"][0], tr0["me"][0])
        seeds = rng.integers(2 ** 31, size=K)
        md = Model(th, sc.keys)
        pred, ys = {}, {}
        for s in slots:
            P = imagine(th, sc, s, seeds, hold=hold, horizon=H32 - hold, leaf=False, md=md)
            pred[s] = float(P[:, 2].mean())
            f = first(sib[s], H32)
            ys[s] = f
            if f is None:
                continue
            bs.append((pred[s] - f[0]) ** 2); bc.append((clim[s] - f[0]) ** 2)
            pred_all.append(pred[s]); act_all.append(float(f[0]))
        for a in slots:
            for b in slots:
                if a < b and ys[a] is not None and ys[b] is not None and ys[a][0] != ys[b][0]:
                    pa, pb = pred[a], pred[b]
                    pairs.append(1.0 if ((pa > pb) if ys[a][0] else (pb > pa)) else (0.5 if pa == pb else 0.0))
    print(f"{label:22s} brier_skill {1 - sum(bs) / max(1e-9, sum(bc)):+.3f}  wrd {np.mean(pairs):.3f} ({len(pairs)})  "
          f"imagined death {np.mean(pred_all):.3f} vs actual {np.mean(act_all):.3f}", flush=True)


def main(tp, hp):
    T, Hd = pickle.load(open(tp, "rb")), pickle.load(open(hp, "rb"))
    slots, hold = sorted(T["sibsets"][0]), T["hold"]
    th = Theory(slots, T["base"]).fit(T["sibsets"], hold=hold)
    variants = {"full": th}
    t = copy.deepcopy(th); t.omens = {}; variants["no omens"] = t
    t = copy.deepcopy(th); t.gauges = []; variants["no gauge"] = t
    t = copy.deepcopy(th); t.omens = {}; t.gauges = []; variants["no omens, no gauge"] = t
    t = copy.deepcopy(th); t.omens = {}; t.gauges = []
    d = OUTCOMES.index("death")
    for k in t.contacts:                                   # no deaths from contacts either
        for kind in ("me", "mine"):
            for part in ("int", "obs"):
                t.contacts[k][kind][part][d] = 0
    variants["nothing kills"] = t
    for name, v in variants.items():
        run(v, Hd, slots, hold, name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
