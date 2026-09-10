"""One hidden WMI-launched continuation, with exclusive logs and launch receipt.

Run this module with the original Python through Win32_Process.Create,
Win32_ProcessStartup.ShowWindow=0. No service or recurring task is installed.
The controller itself independently refuses membership in any Windows Job.
"""
import os
from pathlib import Path
import subprocess
import sys

from scripts import recover_confirm_after_interrupt_v62 as controller

LAUNCH = controller.LAUNCH
LAUNCH_ERROR = 'v62_confirm_second_recovery_launch_error.json'
STDOUT = 'v62_confirm_second_recovery_stdout.log'
STDERR = 'v62_confirm_second_recovery_stderr.log'


def launch(root):
    controller.require_detached()
    proof = controller.preflight(root)
    base = controller.base
    python = proof['command']['argv'][0]
    base.require(Path(sys.executable).resolve() == Path(python).resolve(),
                 'Use the original reviewed Python environment')
    # Do not inherit the CPU-only diagnostic shell's CUDA mask.
    base.require('CUDA_VISIBLE_DEVICES' not in os.environ,
                 'The original GPU environment must not have a diagnostic CUDA mask')
    out = root / 'craft_v6_out'
    receipt = out / LAUNCH
    base.require(not any((out / p).exists() for p in (LAUNCH, LAUNCH + '.tmp', LAUNCH_ERROR, STDOUT, STDERR)),
                 'Preserve the existing detached launch evidence')
    argv = [python, '-u', '-m', 'scripts.recover_confirm_after_interrupt_v62']
    state = dict(kind='v62_detached_second_recovery_launch', status='starting',
                 launcher_pid=os.getpid(), started_utc=controller.utc_now().isoformat(),
                 argv=argv, cwd=str(root), detached_from_job=True,
                 source_sha256=proof['source_sha256'], review_sha256=proof['review_sha256'],
                 absolute_deadline_utc=proof['absolute_deadline_utc'])
    base.save_state(receipt, state, create=True)
    child = None
    try:
        with (out / STDOUT).open('xb') as stdout, (out / STDERR).open('xb') as stderr:
            child = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, shell=False,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        # Close parent-side handles before publishing the release handshake.
        state.update(status='dispatched', controller_pid=child.pid)
        base.save_state(receipt, state)
        return child.pid
    except BaseException as error:
        failure = dict(kind='v62_detached_launch_error', error=f'{type(error).__name__}: {error}',
                       stopped_utc=controller.utc_now().isoformat())
        if child is None:
            state.update(status='failed', **failure)
            base.save_state(receipt, state)
        else:
            # Never kill a supervisor after releasing it: its child could then
            # lose deadline supervision. A missing release times out before GPU.
            # Preserve any successfully published receipt even if this launcher
            # is interrupted immediately after atomic publication.
            failure['controller_pid'] = child.pid
            base.save_state(out / LAUNCH_ERROR, failure, create=True)
        raise


if __name__ == '__main__':
    launch(Path(__file__).resolve().parents[1])
