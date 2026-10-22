"""v83 isolation (AST): agent code reads no RAM, no true score, no game name, no emulator, no
evaluator module, no torch and no earlier learner; it touches no private attribute of objects it is
handed (the senses), and importing it loads none of those modules."""
import ast
import pathlib
import re
import subprocess
import sys

from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV, EXCLUDED, RESERVED, SEALED, SECOND

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENT = ROOT / "ragnarok" / "agent_v83"
FILES = sorted(AGENT.glob("*.py"))
BAD_ATTR = {"ram", "getRAM", "true_score", "game", "ale", "lives", "actions", "cloneState", "restoreState",
            "getScreenRGB", "getMinimalActionSet"}
BAD_NAMES = {"getattr", "setattr", "vars", "eval", "exec", "globals", "__import__", "compile", "open"}
ALLOWED_IMPORTS = {"numpy", "scipy", "scipy.ndimage", "scipy.optimize", "zlib", "math", "collections", "heapq", "json", "time",
                   "ragnarok.agent_v71.tracker", "scripts.ref_extract_v70"}


def _trees():
    return [(f, ast.parse(f.read_text(encoding="utf-8"))) for f in FILES]


def test_agent_package_exists():
    assert len(FILES) >= 4


def test_imports_whitelisted():
    for f, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            else:
                continue
            for m in mods:
                assert m in ALLOWED_IMPORTS or m.startswith("ragnarok.agent_v83"), (f.name, m)


def test_no_oracle_attributes_or_private_access():
    for f, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                own = isinstance(node.value, ast.Name) and node.value.id in ("self", "cls")
                assert own or node.attr not in BAD_ATTR, (f.name, node.lineno, node.attr)
                if node.attr.startswith("_") and not node.attr.startswith("__"):
                    assert own, (f.name, node.lineno, node.attr)
            if isinstance(node, ast.Name):
                assert node.id not in BAD_NAMES, (f.name, node.lineno, node.id)


def test_no_game_names():
    names = DEV + CALIBRATION + SEALED + SECOND + RESERVED + EXCLUDED
    pat = re.compile(r"\b(" + "|".join(names) + r")\b", re.I)
    for f in FILES:
        assert not pat.search(f.read_text(encoding="utf-8")), f.name


def test_import_loads_nothing_forbidden():
    mods = ", ".join(f"ragnarok.agent_v83.{f.stem}" for f in FILES if f.stem != "__init__")
    code = (f"import sys, {mods}\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'ale_py') or "
            "m.startswith('ragnarok.evaluation') or m in ('ragnarok.agent_v71.env', 'ragnarok.agent_v71.learner', "
            "'ragnarok.agent_v71.policy', 'ragnarok.agent_v71.hud', 'ragnarok.agent_v71.scorer')]\n"
            "print(bad); assert not bad\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
