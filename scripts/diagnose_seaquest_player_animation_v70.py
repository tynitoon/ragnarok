"""Offline evaluator-development diagnostic; no ROM or agent endpoint."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames


def diagnose():
    base = Path("craft_v6_out/v70_seaquest_calibration")
    patterns = Counter()
    by_byte = [defaultdict(Counter) for _ in range(128)]
    manifests = {}
    for name in ("development", "coverage_development", "coverage2_development"):
        directory = base/name
        manifests[name] = hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest()
        for rgb, ram, _, _ in cold_fixture_frames(directory):
            x, y, missile = int(ram[70]), int(ram[97])+33, int(ram[103])
            if ram[105] or y < 57 or x < 8 or x+16 > 160 or (missile and missile < x+16 and missile+8 > x):
                continue
            part = rgb[y:y+11, x:x+16]
            if part.shape != (11, 16, 3):
                continue
            mask = np.all(part == (187, 187, 53), axis=2) | np.all(part == (236, 236, 236), axis=2)
            if ram[86] == 0:
                mask = mask[:, ::-1]
            pattern = tuple("".join("#" if v else "." for v in row) for row in mask)
            patterns[pattern] += 1
            for byte, value in enumerate(ram):
                by_byte[byte][int(value)][pattern] += 1
    n = sum(patterns.values())
    selectors = sorted((dict(byte=i, values=len(groups), modal_purity=sum(max(c.values()) for c in groups.values())/n)
                        for i, groups in enumerate(by_byte)), key=lambda d: (-d["modal_purity"], d["values"]))
    result = dict(kind="seaquest_player_animation_development_diagnostic", frames=n,
                  patterns=[dict(rows=list(p), count=c) for p, c in patterns.most_common(10)],
                  byte_selectors=selectors[:16], fixture_manifests=manifests,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  agent_training_allowed=False, calibration_pass=None)
    path = base/"player_animation_diagnostic.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    diagnose()
