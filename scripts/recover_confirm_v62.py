"""One reviewed recovery after the recorded Windows Update interruption.

No experimental source or old controller state is changed. The first attempt
has no complete arm checkpoint; frozen --resume commands replay that arm.
The original experiment log is append-only and its saved prefix is verified.
"""
import argparse
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import time

from scripts import confirm_resume_v62 as base

EVIDENCE="v62_confirm_interruption_20260909"
STATE="v62_confirm_recovery_state.json"
REVIEW="v62_confirm_recovery_review.json"


def recovery_budget(evidence,correction,old):
    start=datetime.fromisoformat(evidence["original_started_utc"])
    end=datetime.fromisoformat(evidence["last_boot_utc"])
    base.require(start.utcoffset() is not None and end.utcoffset() is not None,
                 "Timezone-aware interruption times required")
    charged=math.ceil((end-start).total_seconds())
    base.require(0<charged==correction["charged_upper_bound_seconds"] and
                 evidence["original_started_utc"]==old["started_utc"],"Interruption interval differs")
    remaining=old["provenance"]["design"]["remaining_seconds"]-charged
    base.require(remaining==correction["remaining_seconds"],"Recovered budget differs")
    return charged,base.design(remaining)


def verify_recovery_inputs(root,proof):
    base.verify_inputs(root,proof)
    out=root/"craft_v6_out"
    recovery=proof["recovery"]
    for relative,expected in recovery["immutable_files"].items():
        base.require(base.digest(root/relative)==expected,f"Recovery evidence/source changed: {relative}")
    for name in ("v62_confirm_state.json","v62_confirm_stdout.log","v62_confirm_stderr.log"):
        base.require((out/name).read_bytes()==(out/EVIDENCE/name).read_bytes(),"Historical controller record changed")
    prefix=(out/EVIDENCE/"v61_confirm_s0.log").read_bytes()
    base.require((out/"v61_confirm_s0.log").read_bytes().startswith(prefix),"Historical experiment log prefix changed")


def preflight(root):
    out=root/"craft_v6_out"
    directory=out/EVIDENCE
    old=base.read_json(out/"v62_confirm_state.json")
    evidence=base.read_json(directory/"interruption.json")
    correction=base.read_json(directory/"budget_correction.json")
    base.require(old["status"]=="running" and old["stage"]=="confirm_s0" and old["completed"]==[],
                 "This recovery applies only to the reviewed interrupted first lineage")
    base.require(evidence["kind"]=="v62_confirm_windows_update_interruption" and
                 correction["evidence_sha256"]==base.digest(directory/"interruption.json"),"Mismatched interruption evidence")
    base.require(any(e["id"]==1074 and e["planned_update_restart"] for e in evidence["events"]),
                 "Recorded operating-system update restart required")
    for snapshot in evidence["snapshots"]:
        path=directory/snapshot["path"]
        base.require(path.parent==directory and path.stat().st_size==snapshot["bytes"] and
                     base.digest(path)==snapshot["sha256"],"Interruption snapshot changed")
    base.require(len(evidence["snapshots"])==4 and evidence["observed_checkpoints"]==[],"Unexpected completed checkpoint")
    base.require((out/"v61_confirm_ckpt").is_dir() and not any((out/"v61_confirm_ckpt").iterdir()),
                 "Preserve and independently review any existing arm checkpoint")
    existing=[p.name for p in out.glob("v61_confirm*.json") if "_smoke" not in p.name]
    base.require(not existing,"Unexpected confirmatory aggregate; do not replay reviewed outputs")
    base.require((out/"v61_confirm_s0.log").read_bytes()==(directory/"v61_confirm_s0.log").read_bytes(),
                 "Experiment progressed after interruption was recorded")
    base.require(not any(out.glob("v62_confirm_recovery_state*")),"Preserve existing recovery attempt")
    charged,selected=recovery_budget(evidence,correction,old)
    proof=copy.deepcopy(old["provenance"])
    base.verify_inputs(root,proof)
    base.require(proof["design"]["b_max"]==6 and proof["design"]["g_units"]==2,
                 "Unexpected original confirmatory design")
    review_path=out/REVIEW
    review=base.read_json(review_path)
    launcher=root/"scripts/recover_confirm_v62.py"
    base.require(review.get("kind")=="v62_confirm_recovery_independent_review" and review.get("passed") is True and
                 review.get("interruption_sha256")==base.digest(directory/"interruption.json") and
                 review.get("budget_correction_sha256")==base.digest(directory/"budget_correction.json") and
                 review.get("original_state_sha256")==base.digest(out/"v62_confirm_state.json") and
                 review.get("launcher_sha256")==base.digest(launcher),"Matching independent recovery review required")
    immutable=[*directory.iterdir(),review_path,launcher]
    for path in immutable:
        base.require(path.is_file(),"Only plain evidence files permitted")
        base.committed(root,path)
    proof["design"]=selected
    for command in proof["commands"]:
        argv=command["argv"]
        argv[argv.index("--b-max")+1]=str(selected["b_max"])
        argv[argv.index("--g-units")+1]=str(selected["g_units"])
    proof["recovery"]=dict(charged_interruption_seconds=charged,
                          immutable_files={p.relative_to(root).as_posix():base.digest(p) for p in immutable})
    verify_recovery_inputs(root,proof)
    return proof


def run_pipeline(root,proof):
    out=root/"craft_v6_out"
    path=out/STATE
    state=dict(status="starting",controller_pid=os.getpid(),started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
               provenance=proof,completed=[],elapsed_seconds=0.0)
    base.save_state(path,state,create=True)
    started=time.monotonic()
    code=1
    def remaining():
        value=proof["design"]["remaining_seconds"]-(time.monotonic()-started)
        if value<=0:
            raise TimeoutError("ARC3 hard budget exhausted during recovery")
        return value
    try:
        for step in proof["commands"]:
            verify_recovery_inputs(root,proof)
            base.verify_completed(out,state["completed"])
            remaining()
            state.update(status="running",stage=step["stage"],elapsed_seconds=time.monotonic()-started)
            base.save_state(path,state)
            print(f"Starting recovery {step['stage']} ({remaining()/3600:.2f} h remaining)",flush=True)
            base.run_child(step["argv"],root,remaining())
            verify_recovery_inputs(root,proof)
            base.verify_completed(out,state["completed"])
            artifacts=base.validate_lineage(out,step["lineage"],proof["design"]["b_max"],proof["design"]["g_units"])
            remaining()
            state["completed"].append(dict(stage=step["stage"],artifacts=artifacts))
            state["elapsed_seconds"]=time.monotonic()-started
            base.save_state(path,state)
        merged=base.read_json(out/"v61_confirm.json")
        expected=[u for s in range(3) for u in base.read_json(out/f"v61_confirm_s{s}.json")["units"]]
        base.require(merged==dict(b_max=proof["design"]["b_max"],store_persists=True,units=expected),"Incomplete merged confirmation")
        base.verify_completed(out,state["completed"])
        remaining()
        state.update(status="confirm_complete_review_required",merged_sha256=base.digest(out/"v61_confirm.json"))
        code=0
    except (TimeoutError,subprocess.TimeoutExpired) as error:
        state.update(status="budget_exhausted",error=str(error))
    except KeyboardInterrupt:
        state.update(status="interrupted",error="Interrupted; child stopped and outputs preserved")
        code=130
    except Exception as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}")
    finally:
        state["elapsed_seconds"]=time.monotonic()-started
        base.save_state(path,state)
        print(f"Stopped: {state['status']}",flush=True)
    return code


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    proof=preflight(root)
    if args.dry_run:
        print(json.dumps(proof,indent=2,allow_nan=False))
    else:
        raise SystemExit(run_pipeline(root,proof))
