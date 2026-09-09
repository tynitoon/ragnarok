import json
import subprocess

import pytest

from scripts import recover_confirm_v62 as runner


def proof(remaining=10000):
    return dict(design=dict(b_max=6,g_units=2,remaining_seconds=remaining),
                commands=[dict(stage=f"confirm_s{s}",lineage=s,argv=[str(s)]) for s in range(3)])


def test_utc_restart_interval_does_not_depend_on_windows_local_timezone():
    evidence=dict(original_started_utc="2026-09-09T04:03:09Z",last_boot_utc="2026-09-09T04:39:14.5020020Z")
    old=dict(started_utc=evidence["original_started_utc"],provenance=dict(design=dict(remaining_seconds=86908.0902837)))
    correction=dict(charged_upper_bound_seconds=2166,remaining_seconds=84742.0902837)
    charged,selected=runner.recovery_budget(evidence,correction,old)
    assert charged==2166 and selected["b_max"]==6 and selected["g_units"]==2
    correction["charged_upper_bound_seconds"]=9366
    with pytest.raises(ValueError,match="interval"):
        runner.recovery_budget(evidence,correction,old)


@pytest.mark.parametrize("error,status",[(subprocess.CalledProcessError(2,["fake"]),"failed"),
                                         (TimeoutError("limit"),"budget_exhausted"),
                                         (KeyboardInterrupt(),"interrupted")])
def test_recovery_failure_preserves_original_state_and_excludes_duplicate(tmp_path,monkeypatch,error,status):
    out=tmp_path/"craft_v6_out";out.mkdir()
    original=out/"v62_confirm_state.json";original.write_text("original interrupted record",encoding="utf-8")
    monkeypatch.setattr(runner,"verify_recovery_inputs",lambda *a:None)
    calls=[]
    def fail(*args):
        calls.append(args);raise error
    monkeypatch.setattr(runner.base,"run_child",fail)
    assert runner.run_pipeline(tmp_path,proof())!=0
    assert original.read_text(encoding="utf-8")=="original interrupted record"
    path=out/runner.STATE;before=path.read_bytes()
    assert json.loads(before)["status"]==status and len(calls)==1
    with pytest.raises(ValueError):runner.run_pipeline(tmp_path,proof())
    assert path.read_bytes()==before and len(calls)==1


def test_elapsed_during_validation_forbids_next_dispatch(tmp_path,monkeypatch):
    (tmp_path/"craft_v6_out").mkdir();clock=[0.0];calls=[]
    monkeypatch.setattr(runner.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(runner,"verify_recovery_inputs",lambda *a:None)
    monkeypatch.setattr(runner.base,"run_child",lambda *a:calls.append(a))
    def validate(*args):
        clock[0]=101;return {}
    monkeypatch.setattr(runner.base,"validate_lineage",validate)
    assert runner.run_pipeline(tmp_path,proof(remaining=100))==1
    state=json.loads((tmp_path/"craft_v6_out"/runner.STATE).read_text(encoding="utf-8"))
    assert state["status"]=="budget_exhausted" and len(calls)==1 and state["completed"]==[]


def test_saved_history_prefix_is_checked_without_forbidding_append(tmp_path,monkeypatch):
    out=tmp_path/"craft_v6_out";directory=out/runner.EVIDENCE;directory.mkdir(parents=True)
    monkeypatch.setattr(runner.base,"verify_inputs",lambda *a:None)
    for name in ("v62_confirm_state.json","v62_confirm_stdout.log","v62_confirm_stderr.log"):
        (out/name).write_bytes(b"old immutable");(directory/name).write_bytes(b"old immutable")
    (directory/"v61_confirm_s0.log").write_bytes(b"old result\n")
    log=out/"v61_confirm_s0.log";log.write_bytes(b"old result\nnew appended result\n")
    p=dict(recovery=dict(immutable_files={}))
    runner.verify_recovery_inputs(tmp_path,p)
    log.write_bytes(b"new replaced result\n")
    with pytest.raises(ValueError,match="prefix"):
        runner.verify_recovery_inputs(tmp_path,p)


@pytest.mark.parametrize("mutate_completed",[False,True])
def test_complete_recovery_or_changed_prior_lineage(tmp_path,monkeypatch,mutate_completed):
    out=tmp_path/"craft_v6_out";out.mkdir();(out/"v61_confirm_ckpt").mkdir()
    monkeypatch.setattr(runner,"verify_recovery_inputs",lambda *a:None)
    def write(path,value):
        path.write_text(json.dumps(value),encoding="utf-8")
    calls=[];lineages=[]
    def child(argv,*args):
        s=int(argv[0]);calls.append(s);units=[]
        for index in range(3):
            world=8300+10*s+index;arms={}
            for name in ("M4","Fa","Fb","L","G"):
                rows=None if name=="G" and not(s==0 and index<2) else [
                    dict(goal=g,tier=t,curve=[.5]*7,rows=[1]*6,demos=[0]*6) for g,t in zip((1,2,3),(2,3,4))]
                arms[name]=rows
                if rows is not None:write(out/"v61_confirm_ckpt"/f"s{s}_{world}_{name}.json",rows)
            units.append(dict(id=f"s{s}_{world}",lineage=s,world=world,b_max=6,store_persists=True,
                              goals=[1,2,3],tiers=[2,3,4],arms=arms))
        lineages.append(units)
        write(out/f"v61_confirm_s{s}.json",dict(lineage=s,b_max=6,store_persists=True,units=units))
        write(out/"v61_confirm.json",dict(b_max=6,store_persists=True,units=[u for group in lineages for u in group]))
        if mutate_completed and s==1:
            path=out/"v61_confirm_ckpt/s0_8300_M4.json"
            value=json.loads(path.read_text(encoding="utf-8"));value[0]["curve"][0]=.9;write(path,value)
    monkeypatch.setattr(runner.base,"run_child",child)
    result=runner.run_pipeline(tmp_path,proof())
    state=json.loads((out/runner.STATE).read_text(encoding="utf-8"))
    if mutate_completed:
        assert result==1 and state["status"]=="failed" and calls==[0,1]
        assert "artifact changed" in state["error"]
    else:
        assert result==0 and state["status"]=="confirm_complete_review_required" and calls==[0,1,2]
        assert len(state["completed"])==3 and state["merged_sha256"]==runner.base.digest(out/"v61_confirm.json")
