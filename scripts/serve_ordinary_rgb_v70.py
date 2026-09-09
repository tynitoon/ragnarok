"""Prospective single-engine RGB worker for an explicitly selected interpreter.

Its stdout is a binary protocol. Human/native stdout is redirected to stderr
before runtime inspection or engine construction. No multiprocessing descendants
are created here: the parent can stop this one subprocess without orphan workers.
"""
from dataclasses import fields
import hashlib
import os
from pathlib import Path
import sys

from ragnarok.environments import rgb_wire_v70 as wire
from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import ROM_SPECS
from ragnarok.environments.ordinary_rgb_trace_v70 import _OrdinaryRGBTraceEngine


COMMANDS=frozenset(('init','observe_pair','step','branch_pair_trace','report','close'))


def runtime():
    import ale_py
    import numpy
    return dict(executable=str(Path(sys.executable).resolve()),python=sys.version,numpy=numpy.__version__,
        ale_py=ale_py.__version__,ale_binary_sha256=hashlib.sha256(Path(ale_py._ale_py.__file__).read_bytes()).hexdigest(),
        numerical_threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def serve(reader,writer,*,factory=_OrdinaryRGBTraceEngine,runtime_metadata=None):
    """Closed dispatch; runtime_metadata/factory injection is for synthetic tests."""
    engine=None;initialized=False;stopped=False;last_sequence=0
    def send(value):
        frame=wire.encode(value)
        offset=0
        while offset<len(frame):
            count=writer.write(frame[offset:])
            if type(count) is not int or count<=0:raise BrokenPipeError('Incomplete worker response write')
            offset+=count
        writer.flush()
    def report():
        if engine is None:
            return dict(accounting=None,accounting_complete=False,
                failure=dict(error_type='NoConstructedEngine',accounting_complete=False,
                    error='No complete engine ledger available after construction failure or before initialization'))
        failure=engine._trace_failure
        if stopped and failure is None:
            failure=dict(error_type='StoppedIsolatedWorker',accounting_complete=False,
                error='Worker stopped without a complete native failure description')
        return dict(accounting=engine.accounting(),failure=failure)
    def safe_report():
        try:return report()
        except BaseException as error:
            return dict(accounting=None,accounting_complete=False,
                failure=dict(error_type=type(error).__name__,accounting_complete=False,error=str(error)))
    send((0,True,dict(kind='ordinary_rgb_worker_v0',runtime=runtime() if runtime_metadata is None else runtime_metadata)))
    while True:
        try:request=wire.read(reader)
        except EOFError:return  # Parent closed the pipe; no further command can be issued.
        if (type(request) is not tuple or len(request)!=3 or type(request[0]) is not int
                or request[0]!=last_sequence+1 or type(request[1]) is not str or type(request[2]) is not tuple):
            raise ValueError('Strict sequential closed RGB requests required')
        sequence,command,args=request;last_sequence=sequence;closing=False
        try:
            if command not in COMMANDS:raise ValueError('Unknown isolated RGB command')
            if command=='close':
                if args:raise ValueError('Close takes no arguments')
                value=None;closing=True
            elif command=='init':
                if initialized:raise ValueError('An isolated worker can initialize only once')
                if len(args)!=1 or type(args[0]) is not dict or set(args[0])!={f.name for f in fields(EmulatorConfig)}:
                    raise ValueError('One complete explicit emulator configuration required')
                config=EmulatorConfig(**args[0])
                if config.game not in ROM_SPECS or config.frame_skip!=4 or config.mode!='lab':
                    raise ValueError('Only reviewed development games with repeat4/lab are admitted')
                initialized=True  # A failed native constructor must never be retried.
                try:engine=factory(config)
                except BaseException:
                    stopped=True
                    raise
                value=dict(pair=engine.observe_pair(),report=report())
            elif command=='report':
                if args:raise ValueError('Report takes no arguments')
                value=report()
            else:
                if not initialized or engine is None or stopped:
                    raise ValueError('Isolated RGB worker is not active after failure or before initialization')
                if command=='observe_pair':
                    if args:raise ValueError('Observe pair takes no arguments')
                    value=engine.observe_pair()
                elif command=='step':
                    if len(args)!=1 or type(args[0]) is not int or not 0<=args[0]<18:
                        raise ValueError('One own button in0..17 required')
                    try:value=engine.step_trace(args[0])
                    except BaseException:
                        stopped=True
                        raise
                elif command=='branch_pair_trace':
                    if (len(args)!=3 or type(args[0]) is not int or not 0<=args[0]<18
                            or type(args[1]) is not int or args[1]!=0 or type(args[2]) is not int
                            or not 1<=args[2]<=64):
                        raise ValueError('Own action, NOOP control and explicit horizon1..64 required')
                    try:value=engine.branch_pair_trace(*args)
                    except BaseException:
                        stopped=True
                        raise
        except BaseException as error:
            if command in ('step','branch_pair_trace') or command=='init' and initialized:stopped=True
            response=(sequence,False,dict(error_type=type(error).__name__,error=str(error),
                worker_stopped=stopped,worker_report=safe_report()))
        else:response=(sequence,True,value)
        # Encoding/writing is deliberately outside the dispatch error handler.
        # Never append an error after a partly written successful response.
        send(response)
        if closing:return


def main():
    # Keep an independent handle to the original stdout pipe for wire messages.
    output=os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    if os.name=='nt':
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(),os.O_BINARY)
        msvcrt.setmode(output.fileno(),os.O_BINARY)
    try:serve(sys.stdin.buffer,output)
    finally:output.close()


if __name__=='__main__':main()
