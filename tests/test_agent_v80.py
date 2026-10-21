"""v80 honesty and mechanics tests (docs/ARC4_V80_DESIGN.md §3-4). No emulator needed."""
import ast
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENT = ROOT / "ragnarok" / "agent_v80"


def _calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            out.append(n.attr)
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Call):
            f = n.value.func
            if isinstance(f, ast.Attribute) and f.attr == "clone":
                out.append("clone()[]")
    return out


def test_agent_modules_never_read_oracles():
    """RAM, lives(), the true score and the true-score slot of a savestate are evaluator-only."""
    banned = {"ram", "lives", "true_score", "_true_score", "clone()[]", "getRAM"}
    for f in AGENT.glob("*.py"):
        if f.name == "evaluate.py":
            continue
        hits = banned & set(_calls(f))
        assert not hits, f"{f.name} reads {hits}"


def test_reserved_games_never_named():
    for f in list(AGENT.glob("*.py")) + list((ROOT / "scripts").glob("*_v80.py")):
        txt = f.read_text(encoding="utf-8").lower()
        for g in ("boxing", "frostbite", "riverraid", "kangaroo"):
            assert g not in txt, f"{f.name} names reserved game {g}"


def test_truth_files_refused():
    from scripts.collect_trace_v80 import load_agent_file
    with pytest.raises(ValueError):
        load_agent_file("x_heldout_truth.pkl")


def _frame(objs, H=210, W=160):
    """A synthetic screen: black background, coloured rectangles."""
    img = np.zeros((H, W, 3), np.uint8)
    for x, y, w, h, c in objs:
        img[y:y + h, x:x + w] = c
    return img


def test_parser_and_tracker_follow_a_moving_object():
    from ragnarok.agent_v80.objects import Parser, Tracker
    bg = np.zeros((210, 160, 3), np.uint8)
    p, t = Parser(bg), Tracker()
    ids = set()
    for k in range(10):
        objs = p.parse(_frame([(10 + 3 * k, 50, 4, 4, (200, 50, 50)), (100, 100, 6, 6, (50, 200, 50))]))
        assert len(objs) == 2
        t.step(objs)
        ids |= set(t.tracks)
    assert len(ids) == 2, "each object keeps one track"
    mover = [tr for tr in t.tracks.values() if tr["colour"] == (200 << 16) + (50 << 8) + 50][0]
    assert abs(mover["vx"] - 3.0) < 1e-6 and abs(mover["vy"]) < 1e-6


def test_counter_boxes_are_masked():
    from ragnarok.agent_v80.objects import Parser
    bg = np.zeros((210, 160, 3), np.uint8)
    p = Parser(bg, boxes=[(0, 20, 0, 160)])
    objs = p.parse(_frame([(10, 5, 8, 8, (255, 255, 255)), (50, 100, 4, 4, (255, 0, 0))]))
    assert len(objs) == 1 and objs[0]["cy"] > 90


def _synthetic_traces(n_roots=40, hold=8, horizon=56):          # as long as real lab branches
    """A toy world: me at the bottom moving with my slot; a 'bomb' type falling at 2 px/step that
    kills me on contact; slot 1 moves me left, slot 2 right, slot 0 stays. Labels follow the rule."""
    from ragnarok.agent_v80.lab import COLS
    rng = np.random.default_rng(0)
    sibsets = []
    for r in range(n_roots):
        bx = float(rng.integers(20, 140))
        sib = {}
        for s, dxs in ((0, 0.0), (1, -3.0), (2, 3.0)):
            mx, my = 80.0, 180.0
            by = 120.0
            objs, me, sl, sc, de = [], [], [], [], []
            for t in range(hold + horizon):
                rows = np.array([[1, mx, my, 0, 0, 36, 111, mx - 3, my - 3, mx + 3, my + 3, 0],
                                 [2, bx, by, 0, 2, 16, 222, bx - 2, by - 2, bx + 2, by + 2, 0]], float)
                objs.append(rows); me.append((mx, my))
                a = s if t < hold else 0
                sl.append(a)
                died = abs(bx - mx) <= 5 and abs(by - my) <= 5      # touching on this frame
                sc.append(False); de.append(died)
                if died:
                    break
                mx = float(np.clip(mx + (dxs if t < hold else 0.0), 0, 159)); by += 2.0
            objs.append(objs[-1]); me.append((mx, my))
            sib[s] = dict(objs=objs, me=np.array(me, np.float32), slot=np.array(sl, np.int16), score=np.array(sc),
                          death=np.array(de), over=False, end=len(sl))
        sibsets.append(sib)
    return sibsets


def test_theory_learns_falling_bomb_and_my_card():
    from ragnarok.agent_v80.theory import Theory
    th = Theory([0, 1, 2], 0).fit(_synthetic_traces())
    bomb = [k for k in th.types if k[0] == 222]
    assert bomb, "the bomb type is found"
    assert th.types[bomb[0]]["rule"] in ("cv", "cv_wrap", "cv_bounce")
    assert th.me_card[1][0] < -2 and th.me_card[2][0] > 2 and abs(th.me_card[0][0]) < 0.5
    p = th.contact_probs(bomb[0], "me")
    assert p[0] > 0.5, f"touching the bomb kills me: {p}"


def test_imagination_dodges_the_bomb():
    from ragnarok.agent_v80.imagine import Scene, decide
    from ragnarok.agent_v80.theory import Theory
    th = Theory([0, 1, 2], 0).fit(_synthetic_traces())
    rows = np.array([[1, 80, 180, 0, 0, 36, 111, 77, 177, 83, 183, 0],
                     [2, 80, 140, 0, 2, 16, 222, 78, 138, 82, 142, 0]], float)
    c, V = decide(th, Scene(th, rows, np.array([80.0, 180.0])), np.random.default_rng(0), K=16, hold=8, horizon=24)
    assert c in (1, 2), f"a bomb straight above: move away, got {c} {[(s, V[s].mean()) for s in V]}"


class _Env:
    """A deterministic stub emulator: one red square moving right 3 px per step."""

    def __init__(self):
        self.t, self.frames = 0, 0

    def _img(self):
        return _frame([(10 + 3 * self.t, 60, 4, 4, (200, 50, 50))])

    def clone(self):
        return self.t

    def restore(self, s):
        self.t = s
        self.frames += 60

    def obs(self):
        return self._img()

    def step(self, a):
        self.t += 1
        self.frames += 4
        return self._img(), False


class _Agent:
    class _P:
        class _T:
            pos, miss = None, 0
        tracker = _T()
        bg = np.zeros((210, 160, 3), np.uint8)
        counters = []

    def __init__(self):
        self.perc = self._P()
        self.gone, self.s_idx = 0, None
        self.mask = np.array([True, True])

        class _A:
            @staticmethod
            def to_button(s):
                return int(s)
        self.adapter = _A()

    def _track(self, obs):
        return (100.0, 180.0)

    def _dead(self, ev):
        return False


def test_branch_first_step_keeps_real_velocity():
    """Review v80 item 1: the root frame must enter the object tracker exactly once."""
    from ragnarok.agent_v80 import lab
    from ragnarok.agent_v71.hud import HudReader
    ag, env = _Agent(), _Env()
    parser = lab.make_parser(ag)
    obs = env.obs()
    states = [(env.clone(), None, None, obs, HudReader([]), 0)]
    roots = lab.roots_from(ag, env, states, 0, np.random.default_rng(0), parser, n_roll=1, length=20, mix=0.0)
    assert roots
    tr = lab.trace(ag, env, roots[0], 1, 0, parser, hold=2, horizon=2)
    row = tr["objs"][0][tr["objs"][0][:, 0] >= 0][0]
    vx = row[lab.COLS.index("vx")]
    assert abs(vx - 3.0) < 1e-6, f"velocity at the root frame should be 3, got {vx}"


def test_touching_objects_of_different_colours_stay_apart():
    """v80 Freeway: a car over the chicken must remain two objects, or no contact is ever seen."""
    from ragnarok.agent_v80.objects import Parser
    bg = np.zeros((210, 160, 3), np.uint8)
    objs = Parser(bg).parse(_frame([(40, 100, 8, 8, (252, 252, 84)), (46, 102, 10, 6, (167, 26, 26))]))
    assert len(objs) == 2


def test_slow_sprite_gets_a_fractional_velocity():
    """A sprite moving 1 px every other step: velocity over several steps, not 0/1 alternation."""
    from ragnarok.agent_v80.objects import Parser, Tracker
    bg = np.zeros((210, 160, 3), np.uint8)
    p, t = Parser(bg), Tracker()
    for k in range(12):
        t.step(p.parse(_frame([(20 + k // 2, 80, 4, 4, (200, 50, 50))])))
    v = list(t.tracks.values())[0]["vx"]
    assert 0.4 <= v <= 0.6, v
