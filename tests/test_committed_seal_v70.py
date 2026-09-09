import hashlib
import json
import subprocess

import pytest

from scripts.committed_seal_v70 import verify_committed_seal


@pytest.fixture
def sealed_repository(tmp_path):
    def git(*args):
        return subprocess.run(["git","-C",str(tmp_path),*args],check=True,
                              capture_output=True).stdout.decode("utf-8").strip()
    git("init","--quiet")
    git("config","user.name","Seal regression test")
    git("config","user.email","seal-test@example.invalid")
    git("config","core.autocrlf","true")
    source=tmp_path/"candidate.py"
    source.write_bytes(b"candidate = 1\r\n")
    git("add","candidate.py")
    git("commit","--quiet","-m","candidate")
    seal=dict(source_commit=git("rev-parse","HEAD"),
              source_sha256={"candidate.py":hashlib.sha256(source.read_bytes()).hexdigest()})
    (tmp_path/"seal.json").write_text(json.dumps(seal),encoding="utf-8")
    return tmp_path,git,seal


def test_rejects_uncommitted_seal_even_when_sources_are_committed(sealed_repository):
    root,git,_=sealed_repository
    with pytest.raises(subprocess.CalledProcessError):
        verify_committed_seal(root,"seal.json",("candidate.py",))
    git("add","seal.json")
    with pytest.raises(subprocess.CalledProcessError):
        verify_committed_seal(root,"seal.json",("candidate.py",))


def test_committed_metadata_exact_bytes_and_full_dependency_set(sealed_repository):
    root,git,seal=sealed_repository
    git("add","seal.json")
    git("commit","--quiet","-m","seal")
    assert verify_committed_seal(root,"seal.json",("candidate.py",))==seal
    with pytest.raises(ValueError,match="source set"):
        verify_committed_seal(root,"seal.json",("candidate.py","missing.py"))
    (root/"candidate.py").write_bytes(b"candidate = 2\r\n")
    with pytest.raises(ValueError,match="bytes changed"):
        verify_committed_seal(root,"seal.json",("candidate.py",))


def test_rejects_metadata_changed_after_commit(sealed_repository):
    root,git,seal=sealed_repository
    git("add","seal.json")
    git("commit","--quiet","-m","seal")
    seal["extra"]="uncommitted change"
    (root/"seal.json").write_text(json.dumps(seal),encoding="utf-8")
    with pytest.raises(ValueError,match="metadata must be committed"):
        verify_committed_seal(root,"seal.json",("candidate.py",))


def test_rejects_path_outside_repository(sealed_repository):
    root,_,_=sealed_repository
    with pytest.raises(ValueError,match="inside"):
        verify_committed_seal(root,"../outside.json",("candidate.py",))


def test_new_hash_in_committed_seal_cannot_hide_uncommitted_candidate(sealed_repository):
    root,git,seal=sealed_repository
    source=root/"candidate.py"
    source.write_bytes(b"candidate = 2\r\n")
    seal["source_sha256"]["candidate.py"]=hashlib.sha256(source.read_bytes()).hexdigest()
    (root/"seal.json").write_text(json.dumps(seal),encoding="utf-8")
    git("add","seal.json")
    git("commit","--quiet","-m","metadata only")
    with pytest.raises(ValueError,match="not committed"):
        verify_committed_seal(root,"seal.json",("candidate.py",))
