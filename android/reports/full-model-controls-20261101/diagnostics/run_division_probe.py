"""Isolated GPU arithmetic diagnostic; never model qualification."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
adb="/Users/sergios/Library/Android/sdk/platform-tools/adb"
serial="RFCR81Y1TBJ"
remote="/data/local/tmp/fold3_divprobe"
source="/data/local/tmp/fold3_divprobe.cl"
def call(*args):
  return subprocess.check_output([adb,"-s",serial,*args],text=True,timeout=60)
try:
  call("push",str(root/"opencl-division-probe"),remote)
  call("push",str(root/"opencl-division-probe.cl"),source)
  call("shell","chmod","700",remote)
  result=json.loads(call("shell","timeout","45",remote,source))
  result.update(device=serial,fingerprint=call("shell","getprop","ro.build.fingerprint").strip(),
    source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
      "opencl-division-probe.cpp","opencl-division-probe.cl","opencl-division-probe")})
  (root/"fold45r-division-probe.json").write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps(result,indent=2))
finally:
  call("shell","rm","-f",remote,source)
