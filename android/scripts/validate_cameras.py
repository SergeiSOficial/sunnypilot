#!/usr/bin/env python3
"""One real Camera2 trial; use experiment.py for stop/cooldown/cleanup isolation."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from experiment import Device, PACKAGE


def validate(device, ids, logical, seconds, width, height, output, samsung_multiview=False):
  previous = read(device)
  launched = time.monotonic()
  args = ["shell", "am", "start", "-n", device.package + "/ai.sunnypilot.fold3.MainActivity",
          "--ez", "run_probe", "true", "--es", "camera_ids", ",".join(ids),
          "--ei", "duration_seconds", str(seconds), "--ei", "width", str(width), "--ei", "height", str(height)]
  if logical:
    args += ["--es", "logical_camera", logical]
  if samsung_multiview:
    args += ["--ez", "samsung_multiview", "true", "--ez", "front_camera", str("1" in ids).lower()]
  device.command(*args)
  deadline = time.monotonic() + seconds + 90
  report = {}
  while time.monotonic() < deadline:
    candidate = read(device)
    if (candidate.get("started_utc") != previous.get("started_utc") and
        (candidate.get("completed_utc") or candidate.get("fatal_error"))):
      report = candidate
      break
    time.sleep(1)
  if not report:
    report = {"status": "failed", "error": "Fresh camera trial did not finish", "host_launch_monotonic_s": launched}
  else:
    attempts = report.get("attempts", [])
    report["status"] = attempts[0].get("status", "failed") if len(attempts) == 1 else "failed"
    if report["status"] not in ("passed", "failed", "blocked"):
      report["status"] = "failed"
    if report.get("vehicle_tx_enabled") is not False or report.get("driving_runtime_ready") is not False:
      raise RuntimeError("Camera diagnostic transmission/engagement guard failed")
  Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(json.dumps({"status": report["status"], "logical_access": report.get("logical_access"),
                    "attempts": report.get("attempts", []), "error": report.get("error")}, allow_nan=False))
  return report["status"] == "passed"


def read(device):
  try:
    return json.loads(device.command("shell", "run-as", device.package, "cat", "files/camera-report.json"))
  except (subprocess.CalledProcessError, json.JSONDecodeError):
    return {}


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--package", default=PACKAGE)
  parser.add_argument("--ids", required=True, help="Distinct physical IDs, for example 0,2,1")
  parser.add_argument("--logical", help="Logical camera serving the rear physical outputs")
  parser.add_argument("--samsung-multiview", action="store_true", help="Diagnose Samsung's composite transport; never qualifies physical streams")
  parser.add_argument("--seconds", type=int, default=10, choices=range(3, 1801))
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  ids = args.ids.split(",")
  if not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids) or any(i not in ("0", "1", "2", "3") for i in ids):
    parser.error("Use one to three distinct public physical camera IDs 0,1,2,3")
  if args.samsung_multiview and (args.logical != "20" or set(ids) not in ({"0", "2"}, {"0", "2", "1"})):
    parser.error("Samsung multiview requires --logical 20 and --ids 0,2 or 0,2,1")
  raise SystemExit(0 if validate(Device(args.serial, args.adb, args.package), ids, args.logical,
                               args.seconds, args.width, args.height, args.output, args.samsung_multiview) else 1)
