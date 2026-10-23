"""Provenance for every v81 output (docs/ARC4_V81_ATLAS.md §6): code SHA, dirty flag, inputs' hashes,
launch time. A reader can refuse an output that is older than its inputs or than the code it claims."""
import hashlib
import os
import subprocess
import time


def _git(*args, cwd=None):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=20, cwd=cwd).stdout.strip()
    except Exception:
        return ""


def file_hash(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n))
    h.update(str(os.path.getsize(path)).encode())
    return h.hexdigest()[:16]


def full_hash(path):
    """Full SHA-256 of a file (file_hash reads only the first MiB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stamp(inputs=(), cwd=None, paths=("ragnarok", "scripts"), full=False):
    """cwd: repository root (None = process cwd, the v81-v83 behaviour); paths: code trees checked;
    full: full SHA-256 of the inputs instead of the first-MiB hash."""
    h = full_hash if full else file_hash
    return dict(code_sha=_git("rev-parse", "HEAD", cwd=cwd),
                dirty=bool(_git("status", "--porcelain", "--", *paths, cwd=cwd)),
                code_tree=":".join(_git("rev-parse", f"HEAD:{p}", cwd=cwd) for p in paths),
                launched=time.strftime("%Y-%m-%d %H:%M:%S"), launched_ts=time.time(),
                inputs={str(p): h(p) for p in inputs if os.path.exists(p)})


def fresh(out_path, launched_ts):
    """True if `out_path` was written by the run that started at `launched_ts`."""
    return os.path.exists(out_path) and os.path.getmtime(out_path) >= launched_ts
