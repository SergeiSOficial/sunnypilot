#!/usr/bin/env python3
"""Verify camera thermal aborts using Android's dedicated emulator override."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def validate(serial, adb, output):
  if not serial.startswith("emulator-"):
    raise ValueError("Thermal overrides are restricted to the dedicated emulator")

  def command(*args):
    return subprocess.check_output([adb, "-s", serial, *args], text=True, timeout=30)

  def read(name):
    try:
      return json.loads(command("shell", "run-as", "ai.sunnypilot.fold3", "cat", f"files/{name}.json"))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
      return {}

  report = {"serial": serial, "vehicle_tx_enabled": False, "status": "starting", "cases": []}
  try:
    for during_capture in (False, True):
      command("shell", "am", "force-stop", "ai.sunnypilot.fold3")
      command("shell", "cmd", "thermalservice", "reset")
      if not during_capture:
        command("shell", "cmd", "thermalservice", "override-status", "4")
      previous = read("vision-report").get("started_utc")
      command("shell", "am", "start", "-n", "ai.sunnypilot.fold3/.MainActivity", "--ez", "run_vision", "true",
              "--ei", "vision_seconds", "30")
      if during_capture:
        time.sleep(6)
        command("shell", "cmd", "thermalservice", "override-status", "4")
      start = time.monotonic()
      deadline = start + 30
      while time.monotonic() < deadline:
        result = read("vision-report")
        if result.get("started_utc") != previous and result.get("status") in ("passed", "failed", "stopped"):
          break
        time.sleep(.25)
      else:
        raise TimeoutError("Camera diagnostic did not terminate after critical thermal status")
      case = {"during_capture": during_capture, "abort_wait_seconds": time.monotonic() - start, "result": result}
      report["cases"].append(case)
      assert result["status"] == "failed", result
      assert result["vehicle_tx_enabled"] is False and result["valid"] is False
      assert result["recoveries"] == 0 and len(result["sessions"]) == 1, "Critical heat must not trigger a retry"
      session = result["sessions"][0]
      assert "critical_thermal_status:4" in session.get("error", ""), session
      assert not session["completed_requested_duration"]
      frames = sum(stream["frames"] for stream in session["streams"])
      assert (frames > 0) if during_capture else (frames == 0), "Unexpected camera startup/capture evidence"
      assert case["abort_wait_seconds"] < 10, "Thermal abort exceeded the shutdown budget"
    validation_path = Path(output).with_name("emulator-thermal-live-validator.json")
    validation = subprocess.run([sys.executable, str(Path(__file__).with_name("validate_vision.py")),
                                 "--serial", serial, "--adb", adb, "--seconds", "15", "--report", str(validation_path)],
                                capture_output=True, text=True, timeout=45)
    result = json.loads(validation_path.read_text())
    report["live_validator"] = {key: result.get(key) for key in ("status", "error", "cleanup", "elapsed_s")}
    assert validation.returncode == 1 and result["status"] == "failed", "Live validator accepted critical heat"
    assert "critical" in result.get("error", "").lower() and result.get("cleanup"), result
    assert result["elapsed_s"] < 15, "Live validator did not abort promptly"
    command("shell", "cmd", "thermalservice", "reset")
    previous = read("vision-report").get("started_utc")
    command("shell", "am", "start", "-n", "ai.sunnypilot.fold3/.MainActivity", "--ez", "run_vision", "true",
            "--ei", "vision_seconds", "30")
    time.sleep(6)
    # Force onCreate rather than onNewIntent: the driving UI may be on top.
    command("shell", "am", "start", "-f", "0x18000000", "-n", "ai.sunnypilot.fold3/.MainActivity",
            "--ez", "stop_vision", "true")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
      result = read("vision-report")
      if result.get("started_utc") != previous and result.get("status") == "stopped":
        break
      time.sleep(.25)
    else:
      raise TimeoutError("A new activity instance did not stop live capture")
    report["new_activity_stop"] = result
    assert result["vehicle_tx_enabled"] is False and result["valid"] is False
    assert result["recoveries"] == 0 and len(result["sessions"]) == 1
    assert sum(stream["frames"] for stream in result["sessions"][0]["streams"]) > 0
    report["status"] = "passed"
  except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
  finally:
    try:
      command("shell", "am", "force-stop", "ai.sunnypilot.fold3")
    finally:
      command("shell", "cmd", "thermalservice", "reset")
      report["override_reset"] = True
      Path(output).write_text(json.dumps(report, indent=2) + "\n")
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--report", default="android/reports/emulator-thermal-abort.json")
  args = parser.parse_args()
  result = validate(args.serial, args.adb, args.report)
  print(json.dumps({key: result[key] for key in ("serial", "status", "override_reset")}, indent=2))
  raise SystemExit(0 if result["status"] == "passed" else 1)
