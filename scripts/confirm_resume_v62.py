"""Run three frozen confirmatory lineages after a committed independent probe review.

Separate orchestration only. No test world is constructed in preflight, no
recovery over previous outputs, no changes to the frozen experimental scripts.
"""
import argparse
import math
import os
from pathlib import Path
import subprocess
import time

from scripts.resume_v62 import (
    CRITICAL,FREEZE,REMAINING_SECONDS,digest,read_json,require,finite,verify_provenance,
    validate_pretraining,validate_probe,run_child,save_state,
)

T6_SECONDS=11832/18  # ARC3_PLAN13.1: actual gate K1 goal time, excluding K0.


def design(remaining):
    for b,g in ((6,2),(6,1),(5,1)):
        per_goal=T6_SECONDS*(126*b+8)/764
        projected=(81+3*g)*per_goal
        if projected<=remaining:
            return dict(b_max=b,g_units=g,n_units_per_lineage=3,num_envs=64,
                        projected_seconds=projected,remaining_seconds=remaining)
    raise ValueError("Insufficient budget even after the frozen G then B cuts")


def committed(root,path):
    relative=path.relative_to(root).as_posix()
    def git(*args):
        return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()
    require(git("hash-object",f"--path={relative}",str(path))==git("rev-parse",f"HEAD:{relative}"),
            f"Commit the review and launcher before confirmation: {relative}")


def validate_lineage(out,lineage,b,g):
    path=out/f"v61_confirm_s{lineage}.json"
    data=read_json(path)
    require(data["lineage"]==lineage and data["b_max"]==b and data["store_persists"] is True,
            "Confirmatory lineage design differs from the frozen command")
    units=data["units"]
    require(len(units)==3,"Incomplete confirmatory lineage: exactly three units required")
    worlds=[u["world"] for u in units]
    artifacts={path.name:digest(path)}
    require(worlds==sorted(set(worlds)) and all(8300+10*lineage<=w<8310+10*lineage for w in worlds),
            "Wrong, duplicated or reordered confirmatory worlds")
    for index,u in enumerate(units):
        require(u["id"]==f"s{lineage}_{u['world']}" and u["lineage"]==lineage and
                u["b_max"]==b and u["store_persists"] is True and u["tiers"]==[2,3,4] and
                len(set(u["goals"]))==3 and len(u["goals"])==3,"Invalid confirmatory unit")
        require(set(u["arms"])=={"M4","Fa","Fb","L","G"},"Missing or extra confirmatory arm")
        for arm,rows in u["arms"].items():
            if arm=="G" and not (lineage==0 and index<g):
                require(rows is None,"G must use only the declared lineage0 units")
                continue
            require(isinstance(rows,list) and len(rows)==3,"Incomplete arm goal stream")
            for goal,tier,row in zip(u["goals"],u["tiers"],rows):
                require(row["goal"]==goal and row["tier"]==tier and len(row["curve"])==b+1 and
                        all(finite(v) for v in row["curve"]),"Incomplete or invalid learning curve")
                if arm in ("M4","Fa","Fb"):
                    require(all(isinstance(row.get(k),list) and len(row[k])==b and
                                all(type(v) is int and v>=0 for v in row[k]) for k in ("rows","demos")),
                            "Incomplete training round counters")
            checkpoint=out/"v61_confirm_ckpt"/f"s{lineage}_{u['world']}_{arm}.json"
            require(checkpoint.is_file() and read_json(checkpoint)==rows,
                    "Per-arm checkpoint differs from the lineage aggregate")
            artifacts[checkpoint.relative_to(out).as_posix()]=digest(checkpoint)
    return artifacts


def verify_completed(out,completed):
    for stage in completed:
        for name,expected in stage["artifacts"].items():
            require(digest(out/name)==expected,f"Completed confirmatory artifact changed: {name}")


def preflight(root):
    out=root/"craft_v6_out"
    state_path=out/"v62_resume_state.json"
    prior=read_json(state_path)
    require(prior["status"]=="probe_passed_review_required","Completed passed probe review state required")
    require([v["stage"] for v in prior["completed"]]==["pretrain_s0","pretrain_s1","pretrain_s2","probe"],
            "All pretraining and probe stages must be complete")
    provenance=prior["provenance"]
    require(set(provenance["files"])==set(CRITICAL),"Incomplete frozen source registry")
    commit=subprocess.check_output(["git","-C",str(root),"rev-parse",f"{FREEZE}^{{commit}}"],text=True).strip()
    require(commit==provenance["freeze_commit"]=="a189ea5bb0f9c8b7e9016a50aefc1cbca603426f","Freeze changed")
    subprocess.run(["git","-C",str(root),"diff","--exit-code",FREEZE,"--",*CRITICAL],check=True,capture_output=True)
    verify_provenance(root,provenance)
    require(digest(root/"scripts/resume_v62.py")==provenance["launcher_sha256"],"Prior launcher changed")
    for lineage in range(3):
        require(validate_pretraining(out,lineage)==prior["completed"][lineage]["artifacts"],"Pretraining artifacts changed")
    require(validate_probe(out),"Frozen probe says STOP")
    require(prior["completed"][3]["artifacts"]=={"v61_probe.json":digest(out/"v61_probe.json")},"Probe artifact changed")
    review_path=out/"v62_probe_review.json"
    review=read_json(review_path)
    require(review.get("kind")=="v62_probe_independent_review" and review.get("passed") is True and
            review.get("probe_sha256")==digest(out/"v61_probe.json") and
            review.get("controller_state_sha256")==digest(state_path),"A matching independent probe review is required")
    for path in (review_path,out/"v61_probe.json",root/"scripts/confirm_resume_v62.py"):
        committed(root,path)
    existing=[p.name for pattern in ("v61_confirm*","v62_confirm_state*") for p in out.glob(pattern) if "_smoke" not in p.name]
    require(not existing,f"Preserve previous confirmatory outputs or attempt: {existing}")
    require(math.isfinite(prior["elapsed_seconds"]) and prior["elapsed_seconds"]>0,"Invalid previous elapsed time")
    require(provenance["remaining_seconds"]==REMAINING_SECONDS,"Original hard budget differs")
    remaining=REMAINING_SECONDS-prior["elapsed_seconds"]
    selected=design(remaining)
    python=provenance["python"]
    require(Path(python).is_file(),"Original GPU Python environment missing")
    commands=[dict(stage=f"confirm_s{s}",lineage=s,argv=[python,"-u","-m","scripts.confirm_v61",
              "--lineage",str(s),"--n-units","3","--b-max",str(selected["b_max"]),"--num-envs","64",
              "--store-persists","--g-units",str(selected["g_units"]),"--resume"]) for s in range(3)]
    return dict(prior_provenance=provenance,prior_completed=prior["completed"],
                prior_state_sha256=digest(state_path),probe_review_sha256=digest(review_path),
                launcher_sha256=digest(root/"scripts/confirm_resume_v62.py"),
                prior_launcher_sha256=digest(root/"scripts/resume_v62.py"),design=selected,commands=commands)


def verify_inputs(root,proof):
    verify_provenance(root,proof["prior_provenance"])
    out=root/"craft_v6_out"
    require(digest(out/"v62_resume_state.json")==proof["prior_state_sha256"] and
            digest(out/"v62_probe_review.json")==proof["probe_review_sha256"],"Prior state or review changed")
    require(digest(root/"scripts/confirm_resume_v62.py")==proof["launcher_sha256"] and
            digest(root/"scripts/resume_v62.py")==proof["prior_launcher_sha256"],"Orchestration source changed")
    for stage in proof["prior_completed"]:
        for name,value in stage["artifacts"].items():
            require(digest(out/name)==value,f"Prior checkpoint or probe changed: {name}")


def run_pipeline(root,proof):
    out=root/"craft_v6_out"
    path=out/"v62_confirm_state.json"
    state=dict(status="starting",controller_pid=os.getpid(),started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
               provenance=proof,completed=[],elapsed_seconds=0.0)
    save_state(path,state,create=True)
    started=time.monotonic()
    code=1
    try:
        for step in proof["commands"]:
            verify_inputs(root,proof)
            verify_completed(out,state["completed"])
            remaining=proof["design"]["remaining_seconds"]-(time.monotonic()-started)
            if remaining<=0:
                raise TimeoutError("ARC3 hard budget exhausted")
            state.update(status="running",stage=step["stage"],elapsed_seconds=time.monotonic()-started)
            save_state(path,state)
            print(f"Starting {step['stage']} ({remaining/3600:.2f} h remaining)",flush=True)
            remaining=proof["design"]["remaining_seconds"]-(time.monotonic()-started)
            if remaining<=0:
                raise TimeoutError("ARC3 hard budget exhausted before dispatch")
            run_child(step["argv"],root,remaining)
            verify_inputs(root,proof)
            verify_completed(out,state["completed"])
            artifacts=validate_lineage(out,step["lineage"],proof["design"]["b_max"],proof["design"]["g_units"])
            if time.monotonic()-started>=proof["design"]["remaining_seconds"]:
                raise TimeoutError("ARC3 hard budget exhausted after child validation")
            state["completed"].append(dict(stage=step["stage"],artifacts=artifacts))
            state["elapsed_seconds"]=time.monotonic()-started
            save_state(path,state)
        merged=read_json(out/"v61_confirm.json")
        expected=[u for s in range(3) for u in read_json(out/f"v61_confirm_s{s}.json")["units"]]
        require(merged==dict(b_max=proof["design"]["b_max"],store_persists=True,units=expected),"Merged confirmatory differs from its complete lineages")
        verify_completed(out,state["completed"])
        if time.monotonic()-started>=proof["design"]["remaining_seconds"]:
            raise TimeoutError("ARC3 hard budget exhausted before completion")
        state.update(status="confirm_complete_review_required",merged_sha256=digest(out/"v61_confirm.json"))
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
        save_state(path,state)
        print(f"Stopped: {state['status']}",flush=True)
    return code


if __name__=="__main__":
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    proof=preflight(root)
    if args.dry_run:
        print(json.dumps(proof,indent=2,allow_nan=False))
    else:
        raise SystemExit(run_pipeline(root,proof))
