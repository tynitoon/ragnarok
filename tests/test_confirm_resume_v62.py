import copy
import json
import subprocess
import pytest
from scripts import confirm_resume_v62 as runner


def data(lineage=0,b=6,g=2):
    units=[]
    for index in range(3):
        arms={}
        for name in ("M4","Fa","Fb","L","G"):
            if name=="G" and not (lineage==0 and index<g):
                arms[name]=None
            else:
                arms[name]=[dict(goal=goal,tier=tier,curve=[.5]*(b+1),rows=[2]*b,demos=[1]*b)
                            for goal,tier in zip((1,2,3),(2,3,4))]
        world=8300+10*lineage+index
        units.append(dict(id=f"s{lineage}_{world}",lineage=lineage,world=world,b_max=b,
                          store_persists=True,goals=[1,2,3],tiers=[2,3,4],arms=arms))
    return dict(lineage=lineage,b_max=b,store_persists=True,units=units)


def write(path,value):
    path.write_text(json.dumps(value),encoding="utf-8")


def write_lineage(out,value):
    write(out/f"v61_confirm_s{value['lineage']}.json",value)
    directory=out/"v61_confirm_ckpt"
    directory.mkdir(exist_ok=True)
    for unit in value["units"]:
        for arm,rows in unit["arms"].items():
            if rows is not None:
                write(directory/f"{unit['id']}_{arm}.json",rows)


def test_budget_cuts_g_before_b_and_never_a_lineage():
    t=runner.T6_SECONDS
    assert (runner.design(88*t)["b_max"],runner.design(88*t)["g_units"])==(6,2)
    assert (runner.design(85*t)["b_max"],runner.design(85*t)["g_units"])==(6,1)
    assert (runner.design(83*t)["b_max"],runner.design(83*t)["g_units"])==(5,1)
    assert runner.design(83*t)["n_units_per_lineage"]==3
    with pytest.raises(ValueError,match="Insufficient"):
        runner.design(1)


@pytest.mark.parametrize("fault",["short_unit","duplicate_world","store_reset","missing_Fb","short_round","nan_curve","extra_G"])
def test_reject_invalid_or_incomplete_confirmatory(tmp_path,fault):
    value=data()
    if fault=="short_unit":value["units"].pop()
    if fault=="duplicate_world":value["units"][1]["world"]=8300
    if fault=="store_reset":value["units"][0]["store_persists"]=False
    if fault=="missing_Fb":del value["units"][0]["arms"]["Fb"]
    if fault=="short_round":value["units"][0]["arms"]["Fb"][2]["rows"].pop()
    if fault=="nan_curve":value["units"][0]["arms"]["M4"][0]["curve"][3]=float("nan")
    if fault=="extra_G":value["units"][2]["arms"]["G"]=copy.deepcopy(value["units"][0]["arms"]["G"])
    write_lineage(tmp_path,value)
    with pytest.raises(ValueError):runner.validate_lineage(tmp_path,0,6,2)


def proof(remaining=10000):
    return dict(design=dict(b_max=6,g_units=2,remaining_seconds=remaining),
                commands=[dict(stage=f"confirm_s{s}",lineage=s,argv=[str(s)]) for s in range(3)])


@pytest.mark.parametrize("error,status",[(subprocess.CalledProcessError(2,["fake"]),"failed"),
                                         (TimeoutError("ceiling"),"budget_exhausted"),
                                         (KeyboardInterrupt(),"interrupted")])
def test_child_failure_stops_pipeline_and_preserves_attempt(tmp_path,monkeypatch,error,status):
    (tmp_path/"craft_v6_out").mkdir()
    monkeypatch.setattr(runner,"verify_inputs",lambda *args:None)
    calls=[]
    def fail(*args):
        calls.append(args)
        raise error
    monkeypatch.setattr(runner,"run_child",fail)
    assert runner.run_pipeline(tmp_path,proof())!=0
    path=tmp_path/"craft_v6_out/v62_confirm_state.json"
    state=json.loads(path.read_text(encoding="utf-8"))
    assert state["status"]==status and len(calls)==1 and not state["completed"]
    before=path.read_bytes()
    with pytest.raises(ValueError):runner.run_pipeline(tmp_path,proof())
    assert path.read_bytes()==before and len(calls)==1


def test_complete_three_lineages_stops_before_scoring_or_figure(tmp_path,monkeypatch):
    out=tmp_path/"craft_v6_out"
    out.mkdir()
    monkeypatch.setattr(runner,"verify_inputs",lambda *args:None)
    calls=[]
    def complete(argv,*args):
        s=int(argv[0]);calls.append(s)
        write_lineage(out,data(s))
        write(out/"v61_confirm.json",dict(b_max=6,store_persists=True,units=[u for i in range(s+1) for u in data(i)["units"]]))
    monkeypatch.setattr(runner,"run_child",complete)
    assert runner.run_pipeline(tmp_path,proof())==0 and calls==[0,1,2]
    state=json.loads((out/"v62_confirm_state.json").read_text(encoding="utf-8"))
    assert state["status"]=="confirm_complete_review_required" and len(state["completed"])==3


def test_previous_valid_lineage_mutation_during_next_child_stops_pipeline(tmp_path,monkeypatch):
    out=tmp_path/"craft_v6_out";out.mkdir()
    monkeypatch.setattr(runner,"verify_inputs",lambda *args:None)
    calls=[]
    def complete(argv,*args):
        s=int(argv[0]);calls.append(s)
        write_lineage(out,data(s))
        if s==1:
            changed=data(0)
            changed["units"][0]["arms"]["M4"][0]["curve"][0]=.7
            write_lineage(out,changed)  # both aggregate and arm file changed consistently
    monkeypatch.setattr(runner,"run_child",complete)
    assert runner.run_pipeline(tmp_path,proof())==1 and calls==[0,1]
    state=json.loads((out/"v62_confirm_state.json").read_text(encoding="utf-8"))
    assert state["status"]=="failed" and "artifact changed" in state["error"]


def test_arm_checkpoint_mismatch_rejected(tmp_path):
    write_lineage(tmp_path,data())
    write(tmp_path/"v61_confirm_ckpt/s0_8300_M4.json",[])
    with pytest.raises(ValueError,match="checkpoint differs"):
        runner.validate_lineage(tmp_path,0,6,2)


def test_elapsed_budget_after_child_prevents_next_dispatch(tmp_path,monkeypatch):
    out=tmp_path/"craft_v6_out";out.mkdir()
    monkeypatch.setattr(runner,"verify_inputs",lambda *args:None)
    tick=[0.0]
    monkeypatch.setattr(runner.time,"monotonic",lambda:tick[0])
    calls=[]
    def complete(argv,*args):
        s=int(argv[0]);calls.append(s)
        write_lineage(out,data(s))
        tick[0]=2.0
    monkeypatch.setattr(runner,"run_child",complete)
    assert runner.run_pipeline(tmp_path,proof(remaining=1))==1 and calls==[0]
    state=json.loads((out/"v62_confirm_state.json").read_text(encoding="utf-8"))
    assert state["status"]=="budget_exhausted"
