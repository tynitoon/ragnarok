"""Motor interface (port of agent_v71.adapter.SignatureAdapter to the v83 senses).

Slots are canonical across games: a movement class (sign of dx, sign of dy, diagonals kept) times an
'extra' bit (the button changes the screen differently from the plainest button of its class), plus
private slots for further distinct behaviours. Buttons that behave identically share a slot. Probes
are spread over ordinary play (every `every` steps). The adapter plays on its own emulator, built by
`make_env()` (the senses' sibling factory: no game name), and its frames are billed to the agent.
Changes from v71: factory instead of game name, 4-tuple steps, handles freed, no button names.
"""
import numpy as np

from ragnarok.agent_v83.probe import probe, NOOP

N_PRIV = 8
CANON = [f"{sx},{sy},{ex}" for sx in (0, 1, -1) for sy in (0, 1, -1) for ex in (0, 1)]
SLOTS = CANON + [f"priv{k}" for k in range(N_PRIV)]


class SignatureAdapter:
    def __init__(self, make_env, bg, seed, n_states=24, every=100, horizons=(2, 4), dist_thr=0.15, repeat=0.95):
        env = make_env()
        b0 = env.billed
        rng = np.random.default_rng(seed + 991)
        n = env.n_actions
        env.reset()
        a = 0
        moves, frames = [], []
        for t in range(n_states * every):
            if rng.random() > repeat:
                a = int(rng.integers(n))
            over = env.step(a)[3]
            if over:
                env.reset()
            if (t + 1) % every:
                continue
            s = env.clone()
            fr = {}
            for h in horizons:
                fr[h] = []
                for b in range(n):
                    env.restore(s)
                    for _ in range(h):
                        o = env.step(b)[0]
                    fr[h].append(o)
            env.restore(s)
            env.free(s)
            if not any((fr[h][b] != fr[h][NOOP]).any() for h in horizons for b in range(n)):
                continue                              # nothing responds here (menus, intros)
            out, _ = probe(env, bg.copy(), horizon=horizons[0])
            moves.append(out)
            frames.append(fr)
        self.frames_used = env.billed - b0
        self.n_resp = len(frames)
        R = max(1, len(frames))

        def differs(b, c):
            return float(np.mean([any((f[h][b] != f[h][c]).any() for h in horizons) for f in frames])) if frames else 0.0

        cls, self.info = [], []
        for b in range(n):
            mv = [m[b] for m in moves if m[b]["moved"]]
            frac = len(mv) / R
            sx = sy = 0
            dx = dy = 0.0
            if b != NOOP and frac >= 0.5:
                dx = float(np.median([m["dx"] for m in mv])); dy = float(np.median([m["dy"] for m in mv]))
                thr = max(0.5, 0.4 * max(abs(dx), abs(dy)))
                sx = int(np.sign(dx)) if abs(dx) >= thr else 0
                sy = int(np.sign(dy)) if abs(dy) >= thr else 0
            cls.append((sx, sy))
            self.info.append(dict(button=b, moved_frac=round(frac, 2), dx=round(dx, 2), dy=round(dy, 2),
                                  diff_noop=round(differs(b, NOOP), 2)))
        slot_of, priv, groups = {}, 0, {}
        for b in range(n):
            groups.setdefault(cls[b], []).append(b)
        for (sx, sy), members in groups.items():
            base = NOOP if (sx, sy) == (0, 0) else min(members, key=lambda b: (differs(b, NOOP), b))
            extras = []
            for b in members:
                if b == base or differs(b, base) < dist_thr:
                    slot_of[b] = f"{sx},{sy},0"
                    continue
                rep = next((r for r in extras if differs(b, r) < dist_thr), None)
                if rep is not None:
                    slot_of[b] = slot_of[rep]
                elif not extras:
                    extras.append(b); slot_of[b] = f"{sx},{sy},1"
                elif priv < N_PRIV:
                    extras.append(b); slot_of[b] = f"priv{priv}"; priv += 1
                else:
                    slot_of[b] = slot_of[extras[-1]]
        self.slot_of = {b: SLOTS.index(s) for b, s in slot_of.items()}
        self.buttons = {}
        for b, s in sorted(self.slot_of.items()):
            self.buttons.setdefault(s, []).append(b)
        self.available = np.array([s in self.buttons for s in range(len(SLOTS))])
        for b in range(n):
            self.info[b]["slot"] = SLOTS[self.slot_of[b]]

    def to_button(self, e, rng=None):
        """Deterministic: the first button of the slot (buttons sharing a slot behave identically)."""
        opts = self.buttons.get(int(e)) or self.buttons[self.slot_of[NOOP]]
        return int(opts[0])

    def to_effect(self, a):
        return self.slot_of[int(a)]

    def mapping(self):
        return {SLOTS[s]: list(bs) for s, bs in self.buttons.items()}
