#!/usr/bin/env python3
"""Run one ADB experiment with verified exits and a measured cool baseline."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

PACKAGE = "ai.sunnypilot.fold3"
CAMERA_COMPAT_PACKAGE = "com.samsung.sunnypilot.fold3"


def parse_state(processes, cameras, thermal, package=PACKAGE):
  status = re.search(r"^Thermal Status: (\d+)", thermal, re.MULTILINE)
  skin = re.findall(r"Temperature\{mValue=([\d.]+), mType=3,", thermal)
  active = cameras.split("Active Camera Clients:", 1)[-1].split("Allowed user IDs:", 1)[0]
  packages = (PACKAGE, CAMERA_COMPAT_PACKAGE, package)
  return {"processes": [line.strip() for line in processes.splitlines() if any(p in line for p in packages)],
          "camera_provider_processes": [line.strip() for line in processes.splitlines()
                                        if "camera.provider" in line or "cameraserver" in line],
          "camera_owned": any(p in active for p in packages), "thermal_status": int(status[1]) if status else None,
          "skin_c": max(map(float, skin)) if skin else None,
          "thermal_override": "IsStatusOverride: true" in thermal}


class Device:
  def __init__(self, serial, adb="adb", package=PACKAGE):
    self.serial, self.adb, self.package = serial, adb, package

  def command(self, *args):
    return subprocess.check_output([self.adb, "-s", self.serial, *args], text=True, timeout=30)

  def state(self):
    return parse_state(self.command("shell", "ps", "-A", "-o", "PID,NAME"),
                       self.command("shell", "dumpsys", "media.camera"),
                       self.command("shell", "dumpsys", "thermalservice"), self.package)

  def assert_thermal_safe(self):
    thermal = parse_state("", "", self.command("shell", "dumpsys", "thermalservice"), self.package)
    if thermal["thermal_override"] or thermal["thermal_status"] is None or thermal["thermal_status"] >= 2:
      raise RuntimeError(f"Unsafe or unavailable Android thermal state: {thermal['thermal_status']}")

  def artifact(self):
    path = self.command("shell", "pm", "path", self.package).strip().removeprefix("package:")
    return self.command("shell", "sha256sum", path).split()[0]

  def forwards(self):
    return [fields[1:] for line in self.command("forward", "--list").splitlines()
            if len(fields := line.split()) == 3 and fields[0] == self.serial]

  def clean_forwards(self, baseline):
    removed = []
    for local, remote in self.forwards():
      if [local, remote] not in baseline:
        self.command("forward", "--remove", local)
        removed.append([local, remote])
    return removed

  def stop(self):
    for name in dict.fromkeys((PACKAGE, CAMERA_COMPAT_PACKAGE, self.package)):
      for package in (name, name + ".test"):
        self.command("shell", "am", "force-stop", package)
    deadline = time.monotonic() + 20
    while True:
      state = self.state()
      if not state["processes"] and not state["camera_owned"]:
        return state
      if time.monotonic() >= deadline:
        raise RuntimeError(f"Experiment did not exit cleanly: {state}")
      time.sleep(1)

  def cool(self, idle_seconds=30, timeout=900):
    start = time.monotonic()
    normal_since = None
    while True:
      state = self.state()
      now = time.monotonic()
      if state["processes"] or state["camera_owned"] or state["thermal_override"]:
        raise RuntimeError(f"Unclean or overridden baseline: {state}")
      normal = (state["thermal_status"] == 0 and
                (self.serial.startswith("emulator-") or state["skin_c"] is not None and state["skin_c"] <= 32))
      normal_since = (now if normal_since is None else normal_since) if normal else None
      if normal_since is not None and now - normal_since >= idle_seconds:
        return {**state, "idle_seconds": now - start, "normal_thermal_seconds": now - normal_since}
      if now - start >= timeout:
        raise TimeoutError(f"Device did not cool: {state}")
      time.sleep(5)

  def start_runtime(self, simulation, full_chain=False):
    def read(name):
      try:
        return json.loads(self.command("shell", "run-as", self.package, "cat", f"files/{name}.json"))
      except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {}
    previous_ui = read("ui-report").get("pid")
    launched = float(self.command("shell", "cat", "/proc/uptime").split()[0]) * 1000
    self.command("shell", "am", "start", "-n", self.package + "/ai.sunnypilot.fold3.MainActivity", "--ez", "run_runtime", "true",
                 "--ez", "simulation", str(simulation).lower(), "--ez", "full_chain", str(full_chain).lower(), "--ez", "raylib", "true")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
      manager, ui = read("manager"), read("ui-report")
      if manager.get("updated_elapsed_ms", -1) >= launched:
        if manager.get("vehicle_tx_enabled") is not False or manager.get("engagement_allowed") is not False:
          raise RuntimeError("Runtime transmission/engagement guard failed")
        if (manager.get("simulation") == simulation and manager.get("full_chain", False) == full_chain and ui.get("pid") != previous_ui
            and ui.get("status") == "running" and ui.get("full_upstream_ui") and ui.get("frames", 0) >= 30):
          return {"manager_pid": manager["pid"], "ui_pid": ui["pid"], "simulation": simulation}
      time.sleep(1)
    raise TimeoutError(f"Fresh runtime and Raylib UI did not become ready: manager={manager}, ui={ui}")


def run(device, command, output, idle_seconds=30, runtime=None):
  report = {"serial": device.serial, "command": command, "vehicle_tx_enabled": False, "status": "starting",
            "validation_status": "not_run", "isolation_status": "starting"}
  try:
    report["forwards_before"] = device.forwards()
    report["before_cleanup"] = device.stop()
    report["apk_sha256"] = device.artifact()
    report["baseline"] = device.cool(idle_seconds)
    report["apk_sha256_at_start"] = device.artifact()
    if report["apk_sha256_at_start"] != report["apk_sha256"]:
      raise RuntimeError("Installed APK changed during cooldown; experiment was not started")
    if runtime is not None:
      report["runtime"] = device.start_runtime(runtime in ("simulation", "full-chain"), full_chain=runtime == "full-chain")
    report["returncode"] = subprocess.run(command, check=False).returncode
    report["validation_status"] = "passed" if report["returncode"] == 0 else "failed"
    report["status"] = report["validation_status"]
  except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
  finally:
    try:
      report["after_cleanup"] = device.stop()
      report["forwards_removed"] = device.clean_forwards(report.get("forwards_before", []))
      before = report.get("baseline", {}).get("camera_provider_processes")
      after = report["after_cleanup"].get("camera_provider_processes")
      if before is not None and after != before:
        report.update(status="failed", camera_provider_error="Camera service/provider restarted during experiment")
      report["apk_sha256_after"] = device.artifact()
      report["artifact_stable"] = ("apk_sha256_at_start" in report and
                                    report["apk_sha256"] == report["apk_sha256_at_start"] == report["apk_sha256_after"])
      if not report["artifact_stable"]:
        report.update(status="failed", artifact_error="Installed APK changed or could not be verified throughout the experiment")
    except Exception as exc:
      report.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
    report["isolation_status"] = ("passed" if "baseline" in report and report.get("artifact_stable") and
      not any(key in report for key in ("cleanup_error", "camera_provider_error", "artifact_error")) else "failed")
    report["wrapper_status"] = report["status"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--report", required=True)
  parser.add_argument("--package", default=PACKAGE)
  parser.add_argument("--idle-seconds", type=int, default=30, choices=range(30, 901))
  parser.add_argument("--runtime", choices=("physical", "simulation", "full-chain"), help="Start a fresh manager and Raylib after cooldown")
  parser.add_argument("command", nargs=argparse.REMAINDER)
  args = parser.parse_args()
  command = args.command[1:] if args.command[:1] == ["--"] else args.command
  if not command:
    parser.error("A command after -- is required")
  result = run(Device(args.serial, args.adb, args.package), command, args.report, args.idle_seconds, args.runtime)
  print(json.dumps(result, indent=2))
  raise SystemExit(0 if result["status"] == "passed" else 1)
