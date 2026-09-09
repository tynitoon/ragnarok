"""Prospective archive runner safeguards; only generated temporary zero RGB."""
from dataclasses import fields
import json

import numpy as np
import pytest
import torch

from scripts import check_archived_pixel_diagnostics_v70 as check
from ragnarok.environments.ordinary_rgb_trace_v70 import OrdinaryRGBTrace


def archive_case(name,has_branch):
    arrays={};selected=[]
    for index in check.SPEC['ordinary_indices']:
        for field in fields(OrdinaryRGBTrace):
            leading={'start_pair':(2,),'raw':(4,),'pooled':(4,),'endpoint':()}[field.name]
            arrays[f's{index:03d}_{field.name}']=np.zeros((*leading,210,160,3),np.uint8)
        selected.append(dict(index=index,action=(index//4)%18,operation_id=['owned-ordinary-v0/'+name,index+1],
            epoch=0,start_frame=index*4,end_frame=index*4+4,visual_discontinuity=False,
            actual_frames=4,effect_assumption=[0.]*12))
    branch=None
    if has_branch:
        arrays['branch_start_pair']=np.zeros((2,210,160,3),np.uint8)
        for side in ('intervention','control'):
            for field in ('raw','pooled','endpoint'):
                leading=() if field=='endpoint' else (8,)
                arrays[f'branch_{side}_{field}']=np.zeros((*leading,210,160,3),np.uint8)
        branch=dict(operation_id=['owned-ordinary-v0/'+name,145],requested_action=5,control_action=0,
            requested_frames=8,effect_assumption=[0.]*12)
    return dict(name=name,selected_steps=selected,branch=branch,selected_trace_archive=name+'.npz'),arrays


class TinyParser(torch.nn.Module):
    def __init__(self):super().__init__();self.scalar=torch.nn.Parameter(torch.tensor(1.))


class SyntheticDiagnostics:
    def __init__(self,parser,*,stream_id):self.parser=parser
    def step(self,index,record):return dict(index=index,prediction_eligible=True,identity_anchors=0)
    def branch(self,*args,**kwargs):return dict(supervised_pixels=0)


def setup(monkeypatch,tmp_path):
    monkeypatch.setattr(check,'ROOT',tmp_path);monkeypatch.setattr(check,'FOLDER',tmp_path)
    for field in ('OUTPUT','ATTEMPT','PREFLIGHT'):monkeypatch.setattr(check,field,tmp_path/(field+'.json'))
    monkeypatch.setattr(check,'NATIVE_RESULT','native.json');monkeypatch.setattr(check,'PROTOCOL','protocol.json')
    monkeypatch.setattr(check,'preflight',lambda:{'synthetic':True})
    monkeypatch.setattr(check,'CounterfactualParser',TinyParser)
    monkeypatch.setattr(check,'ArchivedPixelDiagnostics',SyntheticDiagnostics)
    cases=[]
    for i,name in enumerate(check.SPEC['cases']):
        case,arrays=archive_case(name,i<6)
        np.savez_compressed(tmp_path/case['selected_trace_archive'],**arrays);cases.append(case)
    (tmp_path/'native.json').write_text(json.dumps({'cases':cases}))
    (tmp_path/'protocol.json').write_text(json.dumps({'limitations':['synthetic only']}))


def test_missing_preflight_stops_before_weights_or_archives(monkeypatch,tmp_path):
    monkeypatch.setattr(check,'OUTPUT',tmp_path/'out.json');monkeypatch.setattr(check,'ATTEMPT',tmp_path/'attempt.json')
    def reject():raise ValueError('independent review absent')
    monkeypatch.setattr(check,'preflight',reject)
    monkeypatch.setattr(check,'CounterfactualParser',lambda:pytest.fail('model before proof'))
    with pytest.raises(ValueError,match='review absent'):check.run()
    assert not check.ATTEMPT.exists()


def test_complete_population_and_identical_initial_weights_are_required(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    result=check.run()
    assert result['passed'] and result['ordinary_records']==72 and result['branch_pairs']==6
    assert len({r['initial_weight_sha256'] for r in result['cases']})==1
    assert result['optimizer_updates']==result['new_emulated_frames']==0
    with pytest.raises(ValueError,match='no retry'):check.run()


def test_late_objective_failure_retains_completed_cases_and_current_rows(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    calls=[]
    def fail(self,index,record):
        calls.append(index)
        if len(calls)==11:raise ValueError('synthetic nonfinite graph')
        return dict(index=index,prediction_eligible=True,identity_anchors=0)
    monkeypatch.setattr(SyntheticDiagnostics,'step',fail)
    with pytest.raises(ValueError,match='nonfinite graph'):check.run()
    attempt=json.loads(check.ATTEMPT.read_text())
    assert attempt['status']=='failed' and attempt['completed_ordinary_records']==10
    assert len(attempt['completed_cases'])==1 and len(attempt['current_case_rows'])==2
    assert attempt['current_case']=='breakout' and attempt['current_index']==2
    assert attempt['completed_branch_pairs']==1 and attempt['optimizer_updates']==0
    assert not check.OUTPUT.exists()


def test_forged_button_or_extra_archive_field_is_rejected_before_inference():
    case,arrays=archive_case('pong',True)
    case['selected_steps'][0]['action']=17
    with pytest.raises(ValueError,match='source command'):check.read_steps(case,arrays)
    case['selected_steps'][0]['action']=0
    arrays['ram']=np.zeros(128,np.uint8)
    with pytest.raises(ValueError,match='extra fields'):check.read_steps(case,arrays)


def test_unexpected_parameter_mutation_fails_instead_of_claiming_no_training(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    def mutate(self,index,record):
        with torch.no_grad():self.parser.scalar.add_(1.)
        return dict(index=index,prediction_eligible=True,identity_anchors=0)
    monkeypatch.setattr(SyntheticDiagnostics,'step',mutate)
    with pytest.raises(ValueError,match='modified model weights'):check.run()
    attempt=json.loads(check.ATTEMPT.read_text())
    assert attempt['status']=='failed' and not check.OUTPUT.exists()


def test_next_archive_load_failure_cannot_misattribute_previous_case_rows(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    load=np.load
    def fail(path,*args,**kwargs):
        if path.name=='breakout.npz':raise OSError('synthetic unavailable next archive')
        return load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',fail)
    with pytest.raises(OSError,match='next archive'):check.run()
    attempt=json.loads(check.ATTEMPT.read_text())
    assert attempt['status']=='failed' and attempt['current_case']=='breakout'
    assert attempt['current_phase']=='load_and_initialize' and attempt['current_case_rows']==[]
    assert attempt['current_branch'] is None and attempt['current_index'] is None
    assert attempt['completed_ordinary_records']==8 and attempt['completed_branch_pairs']==1
    assert len(attempt['completed_cases'])==1 and attempt['completed_cases'][0]['case']=='pong'
    assert len(attempt['completed_cases'][0]['ordinary'])==8 and not check.OUTPUT.exists()
