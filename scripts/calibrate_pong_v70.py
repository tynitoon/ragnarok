"""Calibrate RAM geometry against the independent Pong pixel verifier; no reference imports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ragnarok.evaluation.fixtures_v70 import fixture_frames
from ragnarok.evaluation.pong_v70 import KINDS, DECODER_REVISION, decode_pong_ram, screen_calibration_boxes, edges


def calibrate(directory):
    totals = {k: dict(frames=0, presence_agree=0, jointly_visible=0, within_one_pixel=0,
                     exact=0, max_edge_error=0, examples=[]) for k in KINDS}
    for frame in fixture_frames(directory):
        decoded = {o.kind: o.box for o in decode_pong_ram(frame["ram"])}
        visible = {o.kind: o.box for o in screen_calibration_boxes(frame["rgb"])}
        for kind in KINDS:
            stat = totals[kind]
            stat["frames"] += 1
            agree = (kind in decoded) == (kind in visible)
            stat["presence_agree"] += int(agree)
            error = None
            if kind in decoded and kind in visible:
                stat["jointly_visible"] += 1
                error = int(np.abs(edges(decoded[kind]) - edges(visible[kind])).max())
                stat["within_one_pixel"] += int(error <= 1)
                stat["exact"] += int(error == 0)
                stat["max_edge_error"] = max(stat["max_edge_error"], error)
            if (not agree or error is not None and error > 1) and len(stat["examples"]) < 12:
                stat["examples"].append(dict(frame=frame["index"], decoded=decoded.get(kind),
                                              visible=visible.get(kind), edge_error=error,
                                              ram={str(i): int(frame["ram"][i]) for i in (49,50,51,54)}))
    for stat in totals.values():
        stat["presence_agreement"] = stat["presence_agree"] / stat["frames"] if stat["frames"] else None
        stat["box_agreement"] = stat["within_one_pixel"] / stat["jointly_visible"] if stat["jointly_visible"] else None
        stat["pass"] = (stat["jointly_visible"] >= 50 and stat["presence_agreement"] >= .99 and stat["box_agreement"] >= .99)
    root = Path(__file__).resolve().parents[1]
    sources = ["ragnarok/evaluation/pong_v70.py", "ragnarok/evaluation/fixtures_v70.py", "scripts/calibrate_pong_v70.py",
               "docs/ARC4_PONG_CALIBRATION_PROTOCOL.json"]
    return dict(kind="development_pong_ram_calibration", decoder_revision=DECODER_REVISION,
                fixture=str(directory), fixture_manifest_sha256=hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
                source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
                objects=totals, calibration_pass=all(s["pass"] for s in totals.values()), gate_result=None,
                score_calibrated=False, no_agent_training=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Keep the old result; choose a new path")
    result = calibrate(args.fixtures)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
