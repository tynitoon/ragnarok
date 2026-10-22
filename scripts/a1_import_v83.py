"""Import the finished A1 outputs from the frozen run worktree into this worktree (evaluator side).

Copies craft_v6_out/v83/{bars,room} from SRC, checks every copy byte for byte,
and writes craft_v6_out/v83/a1_manifest.json (SHA-256 of every file). The JSON files and the manifest
are committed; the .npz row logs stay on disk (about 5 MB each), pinned by their hashes.

  python scripts/a1_import_v83.py F:/dev/ragnarok_v83run
"""
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.evaluation.provenance_v81 import full_hash, stamp   # noqa: E402


def main(src):
    src = pathlib.Path(src) / "craft_v6_out" / "v83"
    dst = ROOT / "craft_v6_out" / "v83"
    files = sorted([p for d in ("bars", "room") for p in (src / d).glob("*") if p.suffix in (".json", ".npz")])
    # the verdict is computed here, on the imported outputs (build notes 20)
    man = {}
    for p in files:
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        h = full_hash(p)
        assert full_hash(q) == h, q
        man[str(p.relative_to(src)).replace("\\", "/")] = h
    out = dict(source=str(src), files=man, provenance=stamp())
    with open(dst / "a1_manifest.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(len(man), "files imported")


if __name__ == "__main__":
    main(sys.argv[1])
