"""v83 pools: disjoint, reserved/excluded games absent, ROMs present, sealed names unused in code."""
import os
import pathlib
import re

from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV, EXCLUDED, RESERVED, SEALED, SECOND

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_pools_disjoint_and_sized():
    pools = [set(DEV), set(CALIBRATION), set(SEALED), set(SECOND)]
    assert [len(p) for p in pools] == [35, 6, 24, 9]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not pools[i] & pools[j]
    every = set().union(*pools)
    assert not every & set(RESERVED) and not every & set(EXCLUDED)


def test_roms_present():
    from ale_py import roms
    d = os.path.dirname(roms.get_rom_path("pong"))
    have = {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".bin")}
    assert set(DEV + CALIBRATION + SEALED + SECOND) <= have


def test_sealed_names_unused_in_code():
    """Sealed and second-round names may appear only in this module and in its test."""
    allowed = {"pools_v83.py", "test_pools_v83.py"}
    pat = re.compile(r"\b(" + "|".join(SEALED + SECOND) + r")\b")
    hits = []
    for sub in ("ragnarok", "scripts", "tests"):
        for f in (ROOT / sub).rglob("*.py"):
            if f.name in allowed:
                continue
            if pat.search(f.read_text(encoding="utf-8", errors="ignore")):
                hits.append(str(f))
    assert not hits, hits
