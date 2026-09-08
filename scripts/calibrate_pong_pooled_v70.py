"""Independent calibration of a pooled RAM adapter; never a gate or learner."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ragnarok.evaluation.pong_v70 import KINDS, edges, screen_calibration_boxes
from ragnarok.evaluation.pooled_pong_v70 import POOLED_REVISION, pooled_ram_objects, pooled_screen_objects
from ragnarok.evaluation.pooled_fixtures_v70 import pooled_fixture_frames


def calibrate(directory):
    totals = {k: dict(pairs=0, presence_agree=0, jointly_visible=0, within_one_pixel=0,
                     exact=0, max_edge_error=0, naive_palette_exact=0, examples=[]) for k in KINDS}
    for frame in pooled_fixture_frames(directory):
        decoded = {o.kind: o.box for o in pooled_ram_objects(frame["ram_previous"], frame["ram_current"])}
        visible = {o.kind: o.box for o in pooled_screen_objects(frame["raw_previous"], frame["raw_current"])}
        naive = {o.kind: o.box for o in screen_calibration_boxes(frame["pooled"])}
        for kind in KINDS:
            stat = totals[kind]
            stat["pairs"] += 1
            agree = (kind in decoded) == (kind in visible)
            stat["presence_agree"] += int(agree)
            stat["naive_palette_exact"] += int(naive.get(kind) == visible.get(kind))
            error = None
            if kind in decoded and kind in visible:
                stat["jointly_visible"] += 1
                error = int(np.abs(edges(decoded[kind]) - edges(visible[kind])).max())
                stat["within_one_pixel"] += int(error <= 1)
                stat["exact"] += int(error == 0)
                stat["max_edge_error"] = max(stat["max_edge_error"], error)
            if (not agree or error is not None and error > 1) and len(stat["examples"]) < 12:
                stat["examples"].append(dict(decision=frame["index"], decoded=decoded.get(kind),
                                              visible=visible.get(kind), edge_error=error))
    for stat in totals.values():
        stat["presence_agreement"] = stat["presence_agree"] / stat["pairs"] if stat["pairs"] else None
        stat["box_agreement"] = stat["within_one_pixel"] / stat["jointly_visible"] if stat["jointly_visible"] else None
        stat["pass"] = (stat["jointly_visible"] >= 50 and stat["presence_agreement"] >= .99 and stat["box_agreement"] >= .99)
    root = Path(__file__).resolve().parents[1]
    sources = ["ragnarok/evaluation/pong_v70.py", "ragnarok/evaluation/pooled_pong_v70.py",
               "ragnarok/evaluation/pooled_fixtures_v70.py", "scripts/calibrate_pong_pooled_v70.py",
               "docs/ARC4_PONG_POOLED_PROTOCOL.json"]
    return dict(kind="development_pong_pooled_calibration", pooled_revision=POOLED_REVISION,
                objects=totals, calibration_pass=all(s["pass"] for s in totals.values()),
                fixture_manifest_sha256=hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
                source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
                all_wrapper_pools_verified=True, gate_result=None, no_agent_training=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Keep the old calibration; choose a new output")
    result = calibrate(args.fixtures)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
