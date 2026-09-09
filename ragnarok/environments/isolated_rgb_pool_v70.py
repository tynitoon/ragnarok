"""Prospective RGB pool with explicit child interpreters and a closed binary wire.

Only the child needs ALE. No environment mutation, model or native-state channel.
One process per engine; pipes have a shared batch deadline and are never retried.
"""
import copy
from dataclasses import asdict
import os
from pathlib import Path
from queue import Queue,Empty,Full
import subprocess
from threading import Lock,Thread,Event
import time
from collections import deque

import numpy as np

from ragnarok.environments import rgb_wire_v70 as wire
from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import ROM_SPECS
from ragnarok.environments.ordinary_rgb_trace_v70 import validate_trace
from ragnarok.environments.owned_rgb_collector_v70 import _pair,_branch_trace


ROOT=Path(__file__).resolve().parents[2]


class IsolatedWorkerError(ValueError):
    def __init__(self,message,*,reports):
        super().__init__(message)
        self.known_worker_reports=copy.deepcopy(reports)


class _Peer:
    """One direct child; persistent pipe threads keep both reads and writes bounded."""
    def __init__(self,executable):
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        self._incoming=Queue(maxsize=1);self._outgoing=Queue(maxsize=1)
        self._stderr=deque(maxlen=16);self._dead=False;self._sequence=0
        self._shutdown=Event();self._threads=[]
        self.process=subprocess.Popen([str(executable),'-u','-m','scripts.serve_ordinary_rgb_v70'],
            cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            bufsize=0,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        def offer(value):
            while not self._shutdown.is_set():
                try:self._incoming.put(value,timeout=.1);return True
                except Full:pass
            return False
        def reading():
            try:
                while not self._shutdown.is_set():
                    if not offer((True,wire.read(self.process.stdout))):return
            except BaseException as error:offer((False,error))
        def writing():
            try:
                while not self._shutdown.is_set():
                    try:packet=self._outgoing.get(timeout=.1)
                    except Empty:continue
                    offset=0
                    while offset<len(packet):
                        count=self.process.stdin.write(packet[offset:])
                        if type(count) is not int or count<=0:raise BrokenPipeError('Incomplete isolated RGB request')
                        offset+=count
                    self.process.stdin.flush()
            except BaseException as error:offer((False,error))
        def logging():
            try:
                while True:
                    block=self.process.stderr.read(4096)
                    if not block:return
                    self._stderr.append(block)
            except (OSError,ValueError):pass
        try:
            for target in (reading,writing,logging):
                thread=Thread(target=target,daemon=True)
                self._threads.append(thread);thread.start()
        except BaseException:
            self._dead=True;self.close(time.monotonic()+2.)
            raise

    def send(self,command,args):
        if self._dead:raise ValueError('Isolated RGB connection already failed')
        try:
            sequence=self._sequence+1
            packet=wire.encode((sequence,command,args))
            if len(packet)>4096:raise ValueError('Isolated RGB requests exceed the closed command size limit')
            self._sequence=sequence
            self._outgoing.put_nowait(packet)
            return sequence
        except BaseException:
            self._dead=True
            raise

    def receive(self,sequence,deadline):
        try:
            valid,value=self._incoming.get(timeout=max(0.,deadline-time.monotonic()))
            if not valid:raise value
            if (type(value) is not tuple or len(value)!=3 or type(value[0]) is not int
                    or value[0]!=sequence or type(value[1]) is not bool):
                raise ValueError('Mismatched or malformed isolated RGB response')
            return value[1],value[2]
        except BaseException:
            self._dead=True
            raise

    def close(self,deadline):
        # EOF is the worker's close signal even when its last response was lost.
        self._shutdown.set()
        if self._dead and self.process.poll() is None:self.process.terminate()
        try:self.process.stdin.close()
        except (OSError,ValueError):pass
        try:self.process.wait(timeout=max(.01,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:self.process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                self.process.kill();self.process.wait(timeout=2.)
        for stream in (self.process.stdout,self.process.stderr):
            try:stream.close()
            except (OSError,ValueError):pass
        for thread in self._threads:
            if thread.ident is not None:thread.join(timeout=max(0.,deadline-time.monotonic()))
        self._dead=True


class IsolatedRGBPool:
    """Exclusive synchronous ownership, with explicit expected child runtime.

Known reports are external accounting only. A lost transport marks them as lower
bounds. They cannot restore a connection or authorize another paid command.
"""
    def __init__(self,configs,*,python_executable,expected_runtime,timeout=30.):
        configs=list(configs);executable=Path(python_executable).resolve()
        if (not 1<=len(configs)<=16 or any(not isinstance(c,EmulatorConfig) or c.game not in ROM_SPECS
                or c.frame_skip!=4 or c.mode!='lab' for c in configs) or not executable.is_file()
                or type(expected_runtime) is not dict or not expected_runtime
                or type(timeout) not in (int,float) or not np.isfinite(timeout) or not 0<timeout<=60):
            raise ValueError('Explicit child Python, expected runtime,1..16 development configs and deadline required')
        self._lock=Lock();self._peers=[];self._reports={};self._stopped=False;self._closed=False
        self._timeout=float(timeout);self._runtime=copy.deepcopy(expected_runtime)
        try:
            for _ in configs:self._peers.append(_Peer(executable))
            deadline=time.monotonic()+self._timeout
            errors=[]
            for index,peer in enumerate(self._peers):
                try:
                    ok,hello=peer.receive(0,deadline)
                    if (not ok or type(hello) is not dict or hello.get('kind')!='ordinary_rgb_worker_v0'
                            or hello.get('runtime')!=self._runtime):
                        raise ValueError('Child runtime differs from its prospectively declared identity')
                except BaseException as error:errors.append((index,error))
            if errors:raise ValueError('At least one child runtime handshake failed')
            replies=self._calls([(i,'init',(asdict(config),)) for i,config in enumerate(configs)])
            # Keep each returned ledger before allocating/validating any image.
            for i,value in enumerate(replies):self._remember(i,value['report'])
            for value in replies:_pair(value['pair'])
        except BaseException as error:
            self._stopped=True
            try:self._refresh_reports()
            finally:self.close()
            raise IsolatedWorkerError(f'Isolated RGB startup failed: {type(error).__name__}: {error}',reports=self._reports) from error

    def _remember(self,index,report):
        if type(report) is not dict or 'accounting' not in report or 'failure' not in report:
            raise ValueError('An explicit worker ledger or unknown-accounting report is required')
        # Deserialization gave this pool its own detached report. Retain it
        # immediately; an extra copy could fail after the ledger was received.
        self._reports[index]=report

    def _calls(self,operations):
        deadline=time.monotonic()+self._timeout;sent=[];errors=[];values={}
        for index,command,args in operations:
            try:sent.append((index,command,self._peers[index].send(command,args)))
            except BaseException as error:errors.append((index,error))
        for index,command,sequence in sent:
            try:
                ok,value=self._peers[index].receive(sequence,deadline)
                if not ok:
                    if type(value) is dict and 'worker_report' in value:self._remember(index,value['worker_report'])
                    raise ValueError(f'Worker {index} {command} failed: {value}')
                if command=='init':self._remember(index,value['report'])
                elif command=='report':self._remember(index,value)
                values[index]=value
            except BaseException as error:errors.append((index,error))
        if errors:
            self._stopped=True
            # Results from every sent command have been drained or declared lost.
            raise IsolatedWorkerError(str(errors[0][1]),reports=self._reports)
        return [values[index] for index,_,_ in operations]

    def _refresh_reports(self):
        available=[(i,'report',()) for i,p in enumerate(self._peers) if not p._dead]
        try:
            if available:self._calls(available)
        except BaseException:pass
        for i,peer in enumerate(self._peers):
            if peer._dead and i in self._reports:
                report=copy.deepcopy(self._reports[i])
                report.update(accounting_complete=False,failure=dict(error_type='LostIsolatedTransport',
                    error='Last available ledger is a known lower bound only',accounting_complete=False))
                self._reports[i]=report
        deadline=time.monotonic()+2.
        for peer in self._peers:
            if peer._dead:
                try:peer.close(deadline)
                except BaseException:pass

    def _enter(self,*,report=False):
        if not self._lock.acquire(blocking=False):raise ValueError('Another isolated pool operation is active')
        if self._closed or self._stopped and not report:
            self._lock.release();raise ValueError('Isolated RGB pool stopped after failure or closure')

    def _index(self,index):
        if type(index) is not int or not 0<=index<len(self._peers):raise ValueError('Unknown isolated worker')

    def step(self,actions):
        actions=list(actions)
        if len(actions)!=len(self._peers) or any(type(a) is not int or not 0<=a<18 for a in actions):
            raise ValueError('One own button per worker required')
        self._enter()
        try:
            traces=self._calls([(i,'step',(a,)) for i,a in enumerate(actions)])
            for trace in traces:validate_trace(trace)
            return traces
        except BaseException as error:
            self._stopped=True;self._refresh_reports()
            raise IsolatedWorkerError(str(error),reports=self._reports) from error
        finally:self._lock.release()

    def observe_pair(self,index=0):
        self._index(index);self._enter()
        try:return _pair(self._calls([(index,'observe_pair',())])[0])
        except BaseException as error:
            self._stopped=True;self._refresh_reports()
            raise IsolatedWorkerError(str(error),reports=self._reports) from error
        finally:self._lock.release()

    def branch_pair_trace(self,action,*,control=0,frames=4,index=0):
        self._index(index)
        if type(action) is not int or not 0<=action<18 or type(control) is not int or control!=0 or type(frames) is not int or not 1<=frames<=64:
            raise ValueError('Own action, NOOP control and explicit horizon required')
        self._enter()
        try:
            start=_pair(self._calls([(index,'observe_pair',())])[0])
            result=self._calls([(index,'branch_pair_trace',(action,control,frames))])[0]
            if type(result) is not tuple or len(result)!=2:raise ValueError('One paid pair must return both own branches')
            for trace in result:_branch_trace(trace,start,frames)
            after=_pair(self._calls([(index,'observe_pair',())])[0])
            if not np.array_equal(start,after):raise ValueError('Isolated pair did not restore its RGB anchor')
            return result
        except BaseException as error:
            self._stopped=True;self._refresh_reports()
            raise IsolatedWorkerError(str(error),reports=self._reports) from error
        finally:self._lock.release()

    def report(self,index=0):
        self._index(index);self._enter(report=True)
        try:
            if not self._peers[index]._dead:
                try:self._calls([(index,'report',())])
                except BaseException:self._stopped=True;self._refresh_reports()
            if index not in self._reports:raise ValueError('No known worker ledger is available')
            return copy.deepcopy(self._reports[index])
        finally:self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):raise ValueError('Cannot close during an isolated pool operation')
        try:
            self._stopped=True;self._closed=True;deadline=time.monotonic()+2.
            errors=[]
            for peer in self._peers:
                try:peer.close(deadline)
                except BaseException as error:errors.append(error)
            if errors:raise errors[0]
        finally:self._lock.release()
