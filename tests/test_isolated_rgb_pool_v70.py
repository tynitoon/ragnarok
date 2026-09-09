"""Cross-runtime pool accounting/stop mechanics with synthetic direct peers."""
from pathlib import Path
import sys
import time

import numpy as np
import pytest

from ragnarok.environments import isolated_rgb_pool_v70 as isolated
from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.owned_rgb_collector_v70 import OwnedRGBCollector
from tests.test_isolated_rgb_worker_v70 import subject


RUNTIME={'fixture':'synthetic'}


def install(monkeypatch,*,init_failure=None,step_failure=None,transport_failure=None,runtime=None):
    peers=[];events=[]
    class Peer:
        def __init__(self,executable):
            self.index=len(peers);peers.append(self)
            self.engine=None;self._dead=False;self.closed=False;self.sequence=0
            self.reply=(True,{'kind':'ordinary_rgb_worker_v0','runtime':RUNTIME if runtime is None else runtime})
        def ledger(self):
            return ({'accounting':None,'failure':{'accounting_complete':False},'accounting_complete':False}
                if self.engine is None else {'accounting':self.engine.accounting(),'failure':self.engine._trace_failure})
        def send(self,command,args):
            assert not self._dead and not self.closed
            self.sequence+=1;events.append(('send',self.index,command))
            self.command=command
            if command=='init':
                if self.index==init_failure:
                    self.reply=(False,{'error_type':'RuntimeError','worker_report':self.ledger()})
                    return self.sequence
                self.engine=subject();value={'pair':self.engine.observe_pair(),'report':self.ledger()}
            elif command=='report':value=self.ledger()
            elif command=='observe_pair':value=self.engine.observe_pair()
            elif command=='step':
                if self.index==step_failure:
                    self.engine._budget.record_play(1)
                    self.engine._trace_failure={'error_type':'SyntheticPaidFailure','accounting_complete':True}
                    self.reply=(False,{'error_type':'ValueError','worker_report':self.ledger()})
                    return self.sequence
                value=self.engine.step_trace(args[0])
            elif command=='branch_pair_trace':value=self.engine.branch_pair_trace(*args)
            else:raise AssertionError(command)
            self.reply=(True,value)
            return self.sequence
        def receive(self,sequence,deadline):
            events.append(('receive',self.index,sequence))
            if sequence and self.command=='step' and self.index==transport_failure:
                self._dead=True;raise EOFError('synthetic lost paid response')
            return self.reply
        def close(self,deadline):self.closed=True;self._dead=True
    monkeypatch.setattr(isolated,'_Peer',Peer)
    return peers,events


def pool(workers=2):
    return isolated.IsolatedRGBPool([EmulatorConfig(seed=i) for i in range(workers)],
        python_executable=sys.executable,expected_runtime=RUNTIME,timeout=1.)


def test_collector_uses_exact_own_steps_and_branches_across_closed_transport(monkeypatch):
    peers,_=install(monkeypatch);transport=pool(2)
    own=OwnedRGBCollector(transport,2,session_id='synthetic-isolated')
    try:
        records=own.take(own.step([1,0]))
        assert len(records)==2 and all(len(r.trace.raw)==4 for r in records)
        anchor=own.current_anchor(1)
        branch,=own.take(own.branch(1,worker=1,expected_anchor=anchor.token,frames=8))
        assert len(branch.intervention.raw)==8 and len(branch.control.raw)==8
        assert own.report(0)['worker']['accounting']['charged_frames']==604
        assert own.report(1)['worker']['accounting']['charged_frames']==740
        np.testing.assert_array_equal(own.current_anchor(1).pair,anchor.pair)
    finally:own.close()
    assert all(p.closed for p in peers)


def test_runtime_mismatch_prevents_every_init_and_closes_all_children(monkeypatch):
    peers,events=install(monkeypatch,runtime={'different':True})
    with pytest.raises(isolated.IsolatedWorkerError,match='handshake'):pool(3)
    assert not any(e[0]=='send' and e[2]=='init' for e in events)
    assert len(peers)==3 and all(p.closed for p in peers)


def test_partial_startup_retains_successful_peer_ledgers_and_unknown_failed_one(monkeypatch):
    peers,_=install(monkeypatch,init_failure=1)
    with pytest.raises(isolated.IsolatedWorkerError) as caught:pool(3)
    reports=caught.value.known_worker_reports
    assert set(reports)=={0,1,2} and reports[0]['accounting']['charged_frames']==600
    assert reports[2]['accounting']['charged_frames']==600 and reports[1]['accounting'] is None
    assert reports[1]['accounting_complete'] is False and all(p.closed for p in peers)


def test_paid_error_drains_all_peers_and_error_carries_refreshed_costs(monkeypatch):
    peers,events=install(monkeypatch,step_failure=1);transport=pool(3)
    try:
        with pytest.raises(isolated.IsolatedWorkerError) as caught:transport.step([1,1,1])
        reports=caught.value.known_worker_reports
        assert [reports[i]['accounting']['charged_frames'] for i in range(3)]==[604,601,604]
        assert sum(e[0]=='send' and e[2]=='step' for e in events)==3
        before=list(events)
        with pytest.raises(ValueError,match='stopped'):transport.step([0,0,0])
        assert events==before and transport.report(1)['failure']['error_type']=='SyntheticPaidFailure'
    finally:transport.close()
    assert all(p.closed for p in peers)


def test_lost_paid_reply_kills_that_peer_and_keeps_its_last_ledger_as_lower_bound(monkeypatch):
    peers,_=install(monkeypatch,transport_failure=1);transport=pool(3)
    try:
        with pytest.raises(isolated.IsolatedWorkerError) as caught:transport.step([1,1,1])
        reports=caught.value.known_worker_reports
        assert reports[0]['accounting']['charged_frames']==reports[2]['accounting']['charged_frames']==604
        assert reports[1]['accounting']['charged_frames']==600  # Four more may be paid but were not acknowledged.
        assert reports[1]['accounting_complete'] is False and peers[1].closed
        assert transport.report(1)['failure']['accounting_complete'] is False
    finally:transport.close()


@pytest.mark.parametrize('later_accounting',[None,{'charged_frames':0}])
def test_failed_report_recovery_never_erases_a_previously_received_paid_ledger(monkeypatch,later_accounting):
    peers,_=install(monkeypatch,step_failure=0);transport=pool(1)
    original_send=peers[0].send
    def failed_report(command,args):
        sequence=original_send(command,args)
        if command=='report':
            peers[0].reply=(False,{'error_type':'SyntheticReportFailure','worker_report':{
                'accounting':later_accounting,'accounting_complete':False,
                'failure':{'error_type':'SyntheticReportFailure','accounting_complete':False}}})
        return sequence
    peers[0].send=failed_report
    try:
        with pytest.raises(isolated.IsolatedWorkerError) as caught:transport.step([1])
        report=caught.value.known_worker_reports[0]
        assert report['accounting']['charged_frames']==601
        assert report['accounting_complete'] is False and report['failure']['error_type']=='SyntheticReportFailure'
        assert transport.report()['accounting']['charged_frames']==601
    finally:transport.close()


def test_external_mutation_of_report_cannot_rewrite_known_ledger(monkeypatch):
    install(monkeypatch);transport=pool(1)
    try:
        value=transport.report();value['accounting']['charged_frames']=0
        assert transport.report()['accounting']['charged_frames']==600
        transport._lock.acquire()
        try:
            with pytest.raises(ValueError,match='active'):transport.step([1])
            assert not transport._stopped
        finally:transport._lock.release()
        assert len(transport.step([1]))==1
    finally:transport.close()


def test_heldouts_and_missing_runtime_are_refused_without_child(monkeypatch):
    peers,_=install(monkeypatch)
    for game in ('boxing','frostbite','riverraid','kangaroo'):
        with pytest.raises(ValueError):
            isolated.IsolatedRGBPool([EmulatorConfig(game=game)],python_executable=sys.executable,expected_runtime=RUNTIME)
    with pytest.raises(ValueError):
        isolated.IsolatedRGBPool([EmulatorConfig()],python_executable=sys.executable,expected_runtime={})
    assert peers==[]


def test_actual_peer_uses_explicit_ale_python_for_hello_close_only():
    executable=Path(__file__).resolve().parents[1]/'.venv-arc4/Scripts/python.exe'
    if not executable.exists():pytest.skip('Workstation ARC4 runtime is not installed here')
    peer=isolated._Peer(executable)
    try:
        ok,hello=peer.receive(0,time.monotonic()+30)
        assert ok and hello['runtime']['ale_py']=='0.12.1'
        assert Path(hello['runtime']['executable'])==executable
        sequence=peer.send('close',())
        assert peer.receive(sequence,time.monotonic()+5)==(True,None)
    finally:peer.close(time.monotonic()+2)
    assert peer.process.poll()==0 and peer._dead
    assert not any(thread.is_alive() for thread in peer._threads)
