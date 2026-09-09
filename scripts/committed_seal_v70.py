"""Verify a committed experiment seal before any reserved emulator is opened."""
import hashlib
import json
from pathlib import Path
import subprocess


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True).stdout.decode("utf-8").strip()


def _relative(root, name):
    path=(root/name).resolve()
    if not path.is_relative_to(root) or path==root:
        raise ValueError("Seal paths must stay inside the repository")
    return path, path.relative_to(root).as_posix()


def verify_committed_seal(root, seal_path, expected_sources):
    """Check exact working bytes and tracked content, including the seal itself.

    Git's clean conversion handles CRLF; source SHA256 separately fixes the
    exact executable bytes. Call again after capture and preserve any failed run.
    The caller supplies the complete source set; a seal cannot omit a dependency.
    """
    root=Path(root).resolve()
    seal_file, seal_name=_relative(root, seal_path)
    seal=json.loads(seal_file.read_text(encoding="utf-8"))
    hashes=seal["source_sha256"]
    if set(hashes)!=set(expected_sources):
        raise ValueError("Seal source set differs from the required dependencies")
    source_commit=seal["source_commit"]
    if len(source_commit)!=40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("Seal requires a complete source commit ID")
    _git(root, "merge-base", "--is-ancestor", source_commit, "HEAD")
    # rev-parse fails for an untracked seal, even when an earlier git add failed.
    committed_seal=_git(root, "rev-parse", f"HEAD:{seal_name}")
    if _git(root, "hash-object", f"--path={seal_name}", str(seal_file))!=committed_seal:
        raise ValueError("Seal metadata must be committed before reserved capture")
    for name, expected_hash in hashes.items():
        path, relative=_relative(root, name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hash:
            raise ValueError(f"Sealed source bytes changed: {relative}")
        blob=_git(root, "hash-object", f"--path={relative}", str(path))
        if blob!=_git(root, "rev-parse", f"{source_commit}:{relative}"):
            raise ValueError(f"Source was not committed at the declared revision: {relative}")
    return seal
