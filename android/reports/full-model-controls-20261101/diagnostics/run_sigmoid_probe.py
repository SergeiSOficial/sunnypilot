"""Isolated GPU arithmetic diagnostic; never model qualification."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
adb="/Users/sergios/Library/Android/sdk/platform-tools/adb"
serial="RFCR81Y1TBJ"
remote="/data/local/tmp/fold3_sigprobe"
source="/data/local/tmp/fold3_sigprobe.cl"
def call(*args):
  return subprocess.check_output([adb,"-s",serial,*args],text=True,timeout=60)
try:
  call("push",str(root/"opencl-sigmoid-probe"),remote)
  call("push",str(root/"opencl-sigmoid-probe.cl"),source)
  call("shell","chmod","700",remote)
  inputs="/data/local/tmp/fold3_sigprobe-input.bin"
  expected="/data/local/tmp/fold3_sigprobe-expected.bin"
  call("push",str(root/"sigmoid-actual-input.bin"),inputs)
  call("push",str(root/"sigmoid-host-expected.bin"),expected)
  result=json.loads(call("shell","timeout","45",remote,source,inputs,expected))
  result.update(device=serial,fingerprint=call("shell","getprop","ro.build.fingerprint").strip(),
    source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
      "opencl-sigmoid-probe.cpp","opencl-sigmoid-probe.cl","opencl-sigmoid-probe","sigmoid-actual-input.bin","sigmoid-host-expected.bin")})
  (root/"fold45t-sigmoid-probe.json").write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps(result,indent=2))
finally:
  call("shell","rm","-f",remote,source,"/data/local/tmp/fold3_sigprobe-input.bin","/data/local/tmp/fold3_sigprobe-expected.bin")
