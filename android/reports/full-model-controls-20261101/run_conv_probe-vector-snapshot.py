#!/usr/bin/env python3
"""Isolated shell-GPU operator diagnosis; never model qualification evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "android/scripts"))
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from experiment import Device
from fold3_accel import accuracy_passed


def main():
  candidate = sys.argv[1]
  variant = sys.argv[2] if len(sys.argv) > 2 else "register"
  assert variant in ("register", "vector")
  suffix = candidate.removeprefix("candidate")
  destination = REPORT / f"fold{suffix}-conv-{variant}-probe.json"
  assert not destination.exists()
  device = Device("RFCR81Y1TBJ", "/Users/sergios/Library/Android/sdk/platform-tools/adb", "com.samsung.sunnypilot.fold3")
  artifact = json.loads((ROOT / "android/build/candidates" / candidate / "artifacts.json").read_text())
  apk = artifact["files"]["app-debug.apk"]["sha256"]
  assert device.artifact() == apk
  for model in ("road", "driver"):
    proof = json.loads((REPORT / f"fold{suffix}-mnn-{model}-accuracy-complete.json").read_text())
    entry = next(iter(proof["models"].values()))
    assert entry["identity"]["sha256"]["apk"] == apk and accuracy_passed(entry, entry["identity"])
  binary = ROOT / f"android/build/conv_{variant}_probe"
  kernel = ROOT / "android/build/native/accelerators/mnn-patched/MNN-7e3c6fb5225d3ccf22807ee641332669df34650d/source/backend/opencl/execution/cl/fold3_host_conv_buf.cl"
  directory = "/data/local/tmp/fold3-conv-" + variant + "-probe-" + suffix
  remote_binary = directory + "/f3convprobe"
  report = {"diagnostic_only": True, "qualifies_model": False, "apk_sha256": apk,
            "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),
                         REPORT / f"conv_{variant}_probe.cpp", binary, kernel)}, "status": "running"}
  process = None
  created = False
  try:
    device.command("shell", "mkdir", directory)
    created = True
    device.command("push", str(binary), remote_binary)
    device.command("push", str(kernel), directory + "/kernel.cl")
    device.command("shell", "chmod", "700", remote_binary)
    process = subprocess.Popen([device.adb, "-s", device.serial, "shell", "timeout", "120", remote_binary,
                                directory + "/kernel.cl"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
      device.assert_thermal_safe()
      try:
        stdout, stderr = process.communicate(timeout=1)
        break
      except subprocess.TimeoutExpired:
        pass
    report.update(returncode=process.returncode, stderr=stderr)
    if process.returncode != 0:
      report.update(status="blocked", error=stderr.strip(), stdout=stdout)
    else:
      report["probe"] = json.loads(stdout)
      report["status"] = "passed" if report["probe"]["all_exact"] else "failed"
  except Exception as exc:
    report.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
  finally:
    if created:
      try:
        pids = device.command("shell", "pidof", "f3convprobe").split()
      except subprocess.CalledProcessError:
        pids = []
      for pid in pids:
        command = device.command("exec-out", "cat", f"/proc/{int(pid)}/cmdline")
        if command.split("\0")[0] == remote_binary:
          device.command("shell", "kill", "-9", str(int(pid)))
      if process is not None and process.poll() is None:
        process.kill()
        process.communicate(timeout=10)
      device.command("shell", "rm", "-f", remote_binary, directory + "/kernel.cl")
      device.command("shell", "rmdir", directory)
    report["completed_utc"] = time.time()
    destination.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps({key: report[key] for key in ("status", "diagnostic_only", "qualifies_model")}))
  return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
  raise SystemExit(main())
