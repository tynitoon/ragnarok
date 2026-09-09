"""Build the local Windows x64 reference arithmetic, with explicit provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=root/"ragnarok/native/reference_update_v70.c"
    compiler=args.compiler.resolve()
    if os.name!="nt" or not compiler.is_file() or not (compiler.parent/"link.exe").is_file():
        parser.error("Supply the installed Windows x64 MSVC compiler")
    destination=args.out.resolve()
    destination.mkdir(parents=True,exist_ok=False)
    dll=destination/"reference_update_v70.dll"
    command=[str(compiler),"/nologo","/O2","/LD","/GS-","/Zl","/TC",
             f"/Fo{destination/'reference_update_v70.obj'}",f"/Fe{dll}",str(source),
             "/link","/NOENTRY","/NODEFAULTLIB","/MACHINE:X64"]
    env=os.environ.copy()
    env["PATH"]=str(compiler.parent)+os.pathsep+env.get("PATH","")
    result=subprocess.run(command,env=env,cwd=destination,text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    manifest=dict(kind="local_freestanding_reference_arithmetic_build",created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                  source=str(source.relative_to(root)).replace("\\","/"),source_sha256=sha(source),
                  dll=dll.name,dll_sha256=sha(dll),compiler=str(compiler),compiler_sha256=sha(compiler),
                  linker_sha256=sha(compiler.parent/"link.exe"),command=command,build_output=result.stdout+result.stderr)
    with (destination/"build.json").open("x",encoding="utf-8") as stream:
        json.dump(manifest,stream,indent=2)
    print(json.dumps(manifest,indent=2))
