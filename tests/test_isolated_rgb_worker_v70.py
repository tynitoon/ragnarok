"""Single-worker dispatch on synthetic engines; subprocess smoke only closes."""
from dataclasses import asdict
import io
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import serve_ordinary_rgb_v70 as worker
from ragnarok.environments import rgb_wire_v70 as wire
from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.ordinary_rgb_trace_v70 import validate_trace
from tests.test_ordinary_rgb_trace_v70 import subject as base_subject


def subject():
    engine=base_subject()
    # This fixture deliberately has no native bootstrap/runtime fields.
    engine.accounting=engine._budget.report
    return engine


def conversation(commands,factory=None):
    calls=[]
    if factory is None:
        def factory(config):
            calls.append(config);return subject()
    source=io.BytesIO(b''.join(wire.encode((i+1,name,args)) for i,(name,args) in enumerate(commands)))
    output=io.BytesIO()
    worker.serve(source,output,factory=factory,runtime_metadata={'fixture':'synthetic'})
    output.seek(0);responses=[]
    while output.tell()<len(output.getvalue()):responses.append(wire.read(output))
    return responses,calls


INIT=('init',(asdict(EmulatorConfig()),))


def test_owned_pixel_prefix_branch_restore_reports_and_closed_commands():
    replies,calls=conversation([INIT,('step',(1,)),('branch_pair_trace',(1,0,8)),
        ('observe_pair',()),('getRAM',()),('report',()),('close',())])
    assert len(calls)==1 and replies[0][2]['kind']=='ordinary_rgb_worker_v0'
    assert validate_trace(replies[2][2])==4
    pair=replies[3][2]
    assert len(pair)==2 and all(len(t.raw)==8 for t in pair)
    assert not replies[5][1] and replies[5][2]['error_type']=='ValueError'
    assert replies[6][2]['accounting']['charged_frames']==740
    np.testing.assert_array_equal(np.maximum(*replies[4][2]),replies[2][2].endpoint)
    assert replies[-1]==(7,True,None)


def test_constructor_failure_has_unknown_cost_and_cannot_repeat():
    calls=[]
    def fail(config):calls.append(config);raise RuntimeError('synthetic constructor failed')
    replies,_=conversation([INIT,INIT,('report',()),('close',())],fail)
    assert len(calls)==1 and not replies[1][1] and not replies[2][1]
    assert replies[1][2]['worker_stopped']
    assert replies[3][2]['accounting'] is None and replies[3][2]['accounting_complete'] is False


def test_paid_failure_preserves_known_ledger_and_refuses_more_steps():
    engine=subject();calls=[]
    def failed(action):
        calls.append(action);engine._budget.record_play(1)
        raise MemoryError('synthetic postpaid step')
    engine.step_trace=failed
    replies,_=conversation([INIT,('step',(1,)),('step',(2,)),('report',()),('close',())],lambda config:engine)
    assert calls==[1] and not replies[2][1] and not replies[3][1]
    assert replies[4][2]['accounting']['charged_frames']==601
    assert replies[4][2]['failure']['accounting_complete'] is False


def test_heldout_configuration_is_rejected_before_any_constructor():
    for game in ('boxing','frostbite','riverraid','kangaroo'):
        config=asdict(EmulatorConfig(game=game))
        replies,calls=conversation([('init',(config,)),('close',())])
        assert not calls and not replies[1][1]


def test_wire_sequence_violation_stops_without_executing_duplicate_action():
    source=io.BytesIO(wire.encode((1,*INIT))+wire.encode((1,'step',(1,))))
    engine=subject();output=io.BytesIO()
    with pytest.raises(ValueError,match='sequential'):
        worker.serve(source,output,factory=lambda config:engine,runtime_metadata={})
    assert engine.accounting()['charged_frames']==600


def test_partial_response_write_does_not_append_an_error_or_continue():
    engine=subject();source=io.BytesIO(wire.encode((1,*INIT))+wire.encode((2,'step',(1,))))
    class Broken(io.BytesIO):
        calls=0
        def write(self,data):
            self.calls+=1
            if self.calls==2:return super().write(data[:10])
            if self.calls==3:raise MemoryError('synthetic mid-response allocation')
            return super().write(data)
    output=Broken()
    with pytest.raises(MemoryError):worker.serve(source,output,factory=lambda config:engine,runtime_metadata={})
    assert output.calls==3 and engine.accounting()['charged_frames']==600


def test_separate_interpreter_hello_and_close_construct_no_emulator():
    # No init command: ALE version/file inspection only, no ROM or ALEInterface.
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    process=subprocess.run([sys.executable,'-u','-m','scripts.serve_ordinary_rgb_v70'],
        input=wire.encode((1,'close',())),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        cwd=Path(__file__).resolve().parents[1],env=env,timeout=30,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert process.returncode==0,process.stderr.decode(errors='replace')
    stream=io.BytesIO(process.stdout)
    hello=wire.read(stream);closed=wire.read(stream)
    assert hello[0:2]==(0,True) and hello[2]['runtime']['ale_py']=='0.12.1'
    assert hello[2]['runtime']['cuda_visible_devices']==''
    assert closed==(1,True,None) and not stream.read()
