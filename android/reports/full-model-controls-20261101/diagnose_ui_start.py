#!/usr/bin/env python3
"""Collect native evidence for a renderer which stops after its first frame."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "android/scripts"))
from experiment import Device

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--serial", required=True)
parser.add_argument("--adb", required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
device = Device(args.serial, args.adb, "com.samsung.sunnypilot.fold3")
device.command("shell", "run-as", device.package, "rm", "-f", "files/ui-report.json", "files/ui-startup-stack.txt")
launch = device.command("shell", "am", "start", "-n", device.package + "/ai.sunnypilot.fold3.MainActivity",
               "--ez", "run_runtime", "true", "--ez", "simulation", "true", "--ez", "full_chain", "true", "--ez", "raylib", "true")
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
  try:
    ui = json.loads(device.command("exec-out", "run-as", device.package, "cat", "files/ui-report.json"))
    if "pid" in ui:
      break
  except (subprocess.CalledProcessError, ValueError):
    pass
  time.sleep(1)
time.sleep(35)
report = {"vehicle_tx_enabled": False, "status": "diagnostic_only", "launch": launch,
          "processes": device.command("shell", "ps", "-A")}
for label, command in (("window_state", ("shell", "dumpsys", "window", "windows")),
                       ("activity_state", ("shell", "dumpsys", "activity", "activities"))):
  report[label] = device.command(*command)
for name in ("ui-report", "ui-startup-stack"):
  try:
    suffix = ".json" if name == "ui-report" else ".txt"
    text = device.command("exec-out", "run-as", device.package, "cat", "files/" + name + suffix)
    report[name] = json.loads(text) if name == "ui-report" else text
  except Exception as exc:
    report[name] = str(exc)
pid = report.get("ui-report", {}).get("pid")
if pid:
  for label, command in (("native_stack", ("shell", "debuggerd", "-b", str(pid))),
                         ("renderer_log", ("logcat", "-d", "--pid=" + str(pid)))):
    try:
      report[label] = device.command(*command)
    except subprocess.CalledProcessError as exc:
      report[label] = {"error": str(exc), "output": exc.output}
args.report.write_text(json.dumps(report, indent=2) + "\n")
