#!/usr/bin/env python3
"""Collect a fresh model benchmark, with parity and performance as separate gates."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from experiment import Device, PACKAGE, CAMERA_COMPAT_PACKAGE


def validate(device, profile, destination, timeout=600):
  def read():
    try:
      return json.loads(device.command("exec-out", "run-as", device.package, "cat", "files/model-report.json"))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
      return {}

  previous = read().get("started_utc")
  device.command("shell", "am", "start", "-n", device.package + "/ai.sunnypilot.fold3.MainActivity", "--ez", "run_models", "true",
                 "--es", "model_profile", profile)
  deadline = time.monotonic() + timeout
  observed_process = False
  report = {}
  while time.monotonic() < deadline:
    try:
      device.assert_thermal_safe()
    except Exception as exc:
      device.stop()
      if report.get("started_utc") == previous:
        report = {}
      report.update(validation_status="blocked", error=str(exc))
      Path(destination).write_text(json.dumps(report, indent=2) + "\n")
      raise
    report = read()
    fresh = report.get("started_utc") is not None and report["started_utc"] != previous
    if fresh and "completed_utc" in report:
      assert report.get("vehicle_tx_enabled") is False, "Unsafe or missing TX guard"
      models = report.get("models", {})
      if any(v.get("status") == "failed" or v.get("performance_status") == "failed" for v in models.values()):
        report["validation_status"] = "failed"
      elif profile.startswith("mnn-buffer-accuracy-"):
        # This result establishes numerical evidence only. Timing and live
        # qualification remain separate and require both model records.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app/src/main/python"))
        from fold3_accel import accuracy_passed, accuracy_part_passed
        from fold3_models import accel_profile
        staged = accel_profile(profile)
        name, part = staged[1:] if staged is not None else (None, None)
        entry = models.get(name, {})
        numerical = (accuracy_passed(entry, entry.get("identity")) if part is None else
                     accuracy_part_passed(entry, entry.get("identity"), part))
        report["validation_status"] = "passed" if (set(models) == {name} and report.get("stage") == "accuracy" and
          report.get("accuracy_part") == part and numerical and
          entry.get("status") == "passed" and entry.get("performance_status") == "not_run" and
          entry.get("diagnostic_only") is True and entry.get("qualified") is False) else "blocked"
      elif profile in ("mnn-buffer-timing-driver", "mnn-buffer-timing-road"):
        name = "dmonitoring_model" if profile.endswith("driver") else "driving_supercombo"
        entry = models.get(name, {})
        gate = report.get("timing_accuracy_gate", {})
        report["validation_status"] = "passed" if (set(models) == {name} and report.get("stage") == "timing" and
          gate.get("passed") is True and gate.get("comparisons") == 795 and
          entry.get("status") == entry.get("performance_status") == "passed" and entry.get("qualified") is True) else "blocked"
      elif profile in ("profile-ncnn-vulkan", *(f"profile-ncnn-dispatch-{limit}" for limit in (131072, 262144, 524288))):
        report["validation_status"] = "passed" if (report.get("accuracy_comparisons") == 795 and
          report.get("diagnostic_only") is True and set(models) == {"dmonitoring_model", "driving_supercombo"} and
          all(v.get("status") == "passed" and v.get("qualified") is False and
              v.get("assignment", {}).get("profiling_enabled") is True and v.get("ranked_layers") for v in models.values())) else "blocked"
      elif profile in ("accel-probes", "qnn-htp-probes"):
        from prepare_htp_probes import MODES
        engines = ("mnn-opencl", "ncnn-vulkan") if profile == "accel-probes" else MODES
        expected = {engine + "-" + op for engine in engines for op in ("conv", "gemm", "activation", "layout")}
        report["validation_status"] = "passed" if set(models) == expected and report.get("diagnostic_only") is True and all(
          v.get("status") == "passed" for v in models.values()) else "blocked"
      elif (set(models) == {"dmonitoring_model", "driving_supercombo"}
            and all(v.get("status") == v.get("performance_status") == "passed" for v in models.values())):
        report["validation_status"] = "passed"
      else:
        report["validation_status"] = "blocked"
      Path(destination).write_text(json.dumps(report, indent=2) + "\n")
      # A completed benchmark is evidence, even when the model gates fail.
      return report
    running = device.package + ":models" in device.command("shell", "ps", "-A", "-o", "NAME").split()
    observed_process |= running
    if (fresh or observed_process) and not running:
      if not fresh:
        report = {}
      report.update(validation_status="failed", error="Model service exited without completing the benchmark")
      Path(destination).write_text(json.dumps(report, indent=2) + "\n")
      raise RuntimeError(report["error"])
    time.sleep(1)
  if report.get("started_utc") is None or report.get("started_utc") == previous:
    report = {}
  report.update(validation_status="blocked", error="No fresh completed model report before timeout")
  Path(destination).write_text(json.dumps(report, indent=2) + "\n")
  device.stop()
  raise TimeoutError("No fresh completed model report")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--profile", required=True)
  parser.add_argument("--package", choices=(PACKAGE, CAMERA_COMPAT_PACKAGE), default=PACKAGE)
  parser.add_argument("--report", required=True)
  parser.add_argument("--timeout", type=int, default=600, help="Maximum benchmark seconds")
  args = parser.parse_args()
  if args.timeout <= 0:
    parser.error("--timeout must be positive")
  report = validate(Device(args.serial, args.adb, args.package), args.profile, args.report, args.timeout)
  print(json.dumps({name: {k: v for k, v in result.items() if k != "traceback"} for name, result in report.get("models", {}).items()}, indent=2))
  raise SystemExit(0 if report["validation_status"] == "passed" else 1)
