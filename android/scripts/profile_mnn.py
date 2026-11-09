#!/usr/bin/env python3
"""Profile pinned MNN kernels in a separate shell process; never qualification."""
import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from experiment import Device

ROOT = Path(__file__).resolve().parents[2]


def parse_events(log):
  runs = []
  current = None
  for line in log.splitlines():
    if line.startswith("FOLD3_PROFILE_BEGIN "):
      if current is not None:
        raise ValueError("Unfinished profile run")
      current = {"run": int(line.split()[-1]), "kernels": []}
    elif line.startswith("FOLD3_PROFILE_END "):
      if current is None or current["run"] != int(line.split()[-1]) or not current["kernels"]:
        raise ValueError("Incomplete or empty GPU event profile")
      current["kernel_total_ms"] = sum(k["gpu_ms"] for k in current["kernels"])
      if abs(current["kernel_total_ms"] - current.get("reported_kernel_total_ms", -1)) > .000001:
        raise ValueError("GPU events do not match native profiler total")
      runs.append(current)
      current = None
    elif current is not None and (match := re.match(r"kernel time = (\d+)\s+us (.+)", line)):
      current["kernels"].append({"name": match[2], "gpu_ms": int(match[1]) / 1000})
    elif current is not None and (match := re.match(r"total kernel time = (\d+)\s+us,", line)):
      if "reported_kernel_total_ms" in current:
        raise ValueError("Duplicate native profiler total")
      current["reported_kernel_total_ms"] = int(match[1]) / 1000
  if current is not None or [run["run"] for run in runs] != [0, 1, 2]:
    raise ValueError("Expected three complete GPU event profiles")
  if any([k["name"] for k in run["kernels"]] != [k["name"] for k in runs[0]["kernels"]] for run in runs[1:]):
    raise ValueError("GPU event coverage differs between runs; reject incomplete or changed execution")
  return runs


def sha(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--model", choices=("dmonitoring_model", "driving_supercombo"), required=True)
  parser.add_argument("--report", type=Path, required=True)
  args = parser.parse_args()
  if args.report.exists():
    raise FileExistsError(args.report)
  native = ROOT / "android/build/native/accelerators/mnn-patched"
  stamp = json.loads((native / "build-profile.json").read_text())
  if (not stamp.get("gpu_profiling") or stamp.get("profile_output") != "stdout" or
      stamp["patch_sha256"] != sha(ROOT / "android/patches/mnn-half-cast.patch") or
      stamp["recipe_sha256"] != sha(Path(__file__).with_name("build_mnn.py"))):
    raise ValueError("Missing or stale diagnostic build; run build_mnn.py --profile-gpu")
  source = native / ("MNN-" + stamp["source"])
  libs = native / "android-profile/libs/arm64-v8a"
  sdk = Path.home() / "Library/Android/sdk"
  toolchain = sdk / ("ndk/" + stamp["ndk"] + "/toolchains/llvm/prebuilt/darwin-x86_64")
  binary = ROOT / "android/build/f3mnnprofile"
  cpp = Path(__file__).with_name("mnn_profile.cpp")
  subprocess.run([str(toolchain / "bin/aarch64-linux-android35-clang++"), "-std=c++17", "-O2",
                  "-I" + str(source / "include"), "-I" + str(source / "source"), str(cpp),
                  "-L" + str(libs), "-lMNN", "-Wl,--no-as-needed", "-lMNN_CL", "-Wl,--as-needed",
                  "-o", str(binary)], check=True)
  model = ROOT / "android/app/build/model-reference/accelerators/mnn" / (args.model + ".mnn")
  files = [binary, model, libs / "libMNN.so", libs / "libMNN_CL.so",
           toolchain / "sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"]
  device = Device(args.serial, args.adb, "com.samsung.sunnypilot.fold3")
  directory = f"/data/local/tmp/fold3-profile-{time.time_ns()}"
  report = {"diagnostic_only": True, "qualifies_model": False, "vehicle_tx_enabled": False,
            "inputs": "all-zero; no recurrent updates", "model": args.model,
            "apk_sha256": device.artifact(), "build": stamp,
            "sha256": {p.name: sha(p) for p in [*files, cpp, Path(__file__)]}, "status": "running"}
  created = False
  process = None
  try:
    device.command("shell", "mkdir", directory)
    created = True
    for path in files:
      device.command("push", str(path), directory + "/" + path.name)
    device.command("shell", "chmod", "700", directory + "/f3mnnprofile")
    process = subprocess.Popen([device.adb, "-s", device.serial, "shell", "env", "LD_LIBRARY_PATH=" + directory,
                                "timeout", "180", directory + "/f3mnnprofile", directory + "/" + model.name],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
      device.assert_thermal_safe()
      try:
        stdout, stderr = process.communicate(timeout=1)
        break
      except subprocess.TimeoutExpired:
        pass
    args.report.with_suffix(".stdout.log").write_text(stdout)
    args.report.with_suffix(".stderr.log").write_text(stderr)
    if process.returncode:
      raise RuntimeError(f"Native profile failed: {process.returncode}: {stderr[-500:]}")
    report["runs"] = parse_events(stdout)
    report["status"] = "passed"
  except Exception as error:
    report.update(status="blocked", error=f"{type(error).__name__}: {error}")
  finally:
    if created:
      try:
        pids = device.command("shell", "pidof", "f3mnnprofile").split()
      except subprocess.CalledProcessError:
        pids = []
      for pid in pids:
        command = device.command("exec-out", "cat", f"/proc/{int(pid)}/cmdline")
        if command.split("\0")[0] == directory + "/f3mnnprofile":
          device.command("shell", "kill", "-9", str(int(pid)))
      if process is not None and process.poll() is None:
        process.kill()
        process.communicate(timeout=10)
      for path in files:
        device.command("shell", "rm", "-f", directory + "/" + path.name)
      device.command("shell", "rmdir", directory)
    report["completed_utc"] = datetime.now(UTC).isoformat()
    args.report.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps({key: report[key] for key in ("status", "diagnostic_only", "qualifies_model")}))
  return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
  raise SystemExit(main())
