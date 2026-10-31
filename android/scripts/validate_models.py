#!/usr/bin/env python3
"""Collect a fresh model benchmark, with parity and performance as separate gates."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from experiment import Device, PACKAGE


def validate(device, profile, destination):
  def read():
    try:
      return json.loads(device.command("exec-out", "run-as", PACKAGE, "cat", "files/model-report.json"))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
      return {}

  previous = read().get("started_utc")
  device.command("shell", "am", "start", "-n", PACKAGE + "/.MainActivity", "--ez", "run_models", "true",
                 "--es", "model_profile", profile)
  deadline = time.monotonic() + 240
  observed_process = False
  while time.monotonic() < deadline:
    report = read()
    fresh = report.get("started_utc") is not None and report["started_utc"] != previous
    if fresh and "completed_utc" in report:
      models = report.get("models", {})
      if any(v.get("status") == "failed" or v.get("performance_status") == "failed" for v in models.values()):
        report["validation_status"] = "failed"
      elif (set(models) == {"dmonitoring_model", "driving_supercombo"}
            and all(v.get("status") == v.get("performance_status") == "passed" for v in models.values())):
        report["validation_status"] = "passed"
      else:
        report["validation_status"] = "blocked"
      Path(destination).write_text(json.dumps(report, indent=2) + "\n")
      # A completed benchmark is evidence, even when the model gates fail.
      return report
    running = PACKAGE + ":models" in device.command("shell", "ps", "-A", "-o", "NAME").split()
    observed_process |= running
    if (fresh or observed_process) and not running:
      if not fresh:
        report = {"vehicle_tx_enabled": False}
      report.update(validation_status="failed", error="Model service exited without completing the benchmark")
      Path(destination).write_text(json.dumps(report, indent=2) + "\n")
      raise RuntimeError(report["error"])
    time.sleep(1)
  Path(destination).write_text(json.dumps(report, indent=2) + "\n")
  raise TimeoutError("No fresh completed model report")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--profile", required=True)
  parser.add_argument("--report", required=True)
  args = parser.parse_args()
  report = validate(Device(args.serial, args.adb), args.profile, args.report)
  print(json.dumps({name: {k: v for k, v in result.items() if k != "traceback"} for name, result in report.get("models", {}).items()}, indent=2))
  raise SystemExit(0 if report["validation_status"] == "passed" else 1)
