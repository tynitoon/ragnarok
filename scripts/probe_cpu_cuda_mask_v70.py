"""One bounded environment diagnostic after the aborted trial; no model or images.

Two fresh Python processes inspect the empty and -1 CUDA masks. They construct
no tensor, network or optimizer and never import a Ragnarok learning module.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'craft_v6_out/v70_cpu_cuda_mask_diagnostic'
CODE = '''import json, os, torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
before = torch.cuda.is_initialized()
available = torch.cuda.is_available()
print(json.dumps(dict(received_mask=os.environ.get('CUDA_VISIBLE_DEVICES'),
    torch_version=torch.__version__, cuda_available=available,
    torch_cuda_initialized_before=before, torch_cuda_initialized_after=torch.cuda.is_initialized(),
    model_constructions=0, tensor_constructions=0, optimizer_updates=0)))
'''


def write(name, value):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    OUT.mkdir(exist_ok=True)
    started = time.perf_counter()
    source = Path(__file__).read_bytes()
    intent = dict(started_utc=datetime.now(timezone.utc).isoformat(), max_wall_seconds=30,
                  source_sha256=hashlib.sha256(source).hexdigest(), child_code=CODE,
                  masks=['', '-1'], new_rgb_frames=0, model_constructions=0, optimizer_updates=0)
    write('probe_v0_attempt.json', intent)
    rows, failure = [], None
    try:
        for index, mask in enumerate(intent['masks']):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=mask, OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run([sys.executable, '-c', CODE], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=max(.001, 30-(time.perf_counter()-started)))
            row = dict(requested_mask=mask, returncode=result.returncode,
                       stdout=result.stdout, stderr=result.stderr)
            write(f'probe_v0_child_{index}.json', row)
            rows.append(row)
            if result.returncode:
                raise RuntimeError('Environment probe failed')
        parsed = [json.loads(row['stdout']) for row in rows]
        if parsed[1]['received_mask'] != '-1' or parsed[1]['cuda_available'] is not False:
            raise ValueError('Explicit -1 mask did not disable CUDA availability')
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != intent['source_sha256']:
            raise ValueError('Diagnostic source changed')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
    seconds = time.perf_counter()-started
    if seconds >= 30:
        failure = (failure+'; ' if failure else '')+'budget exceeded'
    write('probe_v0.json', dict(intent, rows=rows, failure=failure,
                               passed=failure is None, elapsed_seconds=seconds))
    print(json.dumps(dict(passed=failure is None, elapsed_seconds=seconds, rows=rows)))
    if failure:
        raise RuntimeError(failure)


if __name__ == '__main__':
    main()
