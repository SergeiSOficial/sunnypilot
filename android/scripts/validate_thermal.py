#!/usr/bin/env python3
"""Verify camera thermal aborts using Android's dedicated emulator override."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from experiment import Device, PACKAGE, CAMERA_COMPAT_PACKAGE


def validate(serial, adb, output, package=PACKAGE):
  if not serial.startswith("emulator-"):
    raise ValueError("Thermal overrides are restricted to the dedicated emulator")

  def command(*args):
    return subprocess.check_output([adb, "-s", serial, *args], text=True, timeout=30)

  def read(name):
    try:
      return json.loads(command("shell", "run-as", package, "cat", f"files/{name}.json"))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
      return {}

  def wait_camera(previous_ui):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
      ui = read("ui-report")
      previews = ui.get("camera_preview", [])
      if (ui.get("pid") != previous_ui and len(previews) == 2
          and all(stream.get("frame_id", 0) > 0 for stream in previews)):
        return
      time.sleep(.25)
    raise TimeoutError("No current camera frames before fault injection")

  device = Device(serial, adb, package)
  activity = package + "/ai.sunnypilot.fold3.MainActivity"
  report = {"serial": serial, "package": package, "vehicle_tx_enabled": False, "status": "starting", "cases": []}
  try:
    device.stop()
    command("shell", "cmd", "thermalservice", "reset")
    device.start_runtime(False)
    # Importing Python/native modules can take much longer than six seconds on
    # an emulator. Require current workers before injecting a thermal event.
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
      manager = read("manager")
      current = [read(name) for name in ("modeld", "dmonitoringmodeld")]
      if all(item.get("pid") == manager.get("processes", {}).get(worker, {}).get("pid") and "thermal_status" in item
             for item, worker in zip(current, ("RoadModel", "DriverModel"), strict=True)):
        break
      time.sleep(.5)
    else:
      raise TimeoutError("Current model workers did not initialize")
    manager_pid = manager["pid"]
    command("shell", "cmd", "thermalservice", "override-status", "2")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
      manager = read("manager")
      paused = {name: read(name) for name in ("modeld", "dmonitoringmodeld")}
      workers = manager.get("processes", {})
      if (manager.get("pid") == manager_pid and all(
          item.get("pid") == workers.get(worker, {}).get("pid") and item.get("phase") == "thermal_paused"
          for (name, item), worker in zip(paused.items(), ("RoadModel", "DriverModel"), strict=True))):
        break
      time.sleep(.25)
    else:
      raise TimeoutError("Both current model workers did not pause at MODERATE")
    assert all(item.get("target_suspended") is True and item.get("valid") is False for item in paused.values())
    report["moderate_pause"] = paused
    command("shell", "cmd", "thermalservice", "reset")
    time.sleep(2)
    assert all(read(name).get("phase") == "thermal_paused" for name in paused), "Resumed before the cool interval"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
      resumed = {name: read(name) for name in paused}
      if all(item.get("pid") == paused[name]["pid"] and item.get("target_suspended") is False
             and item.get("phase") != "thermal_paused" for name, item in resumed.items()):
        break
      time.sleep(.25)
    else:
      raise TimeoutError("Models did not resume after ten cool seconds")
    report["moderate_resumed"] = resumed
    for during_capture in (False, True):
      device.stop()
      command("shell", "cmd", "thermalservice", "reset")
      if not during_capture:
        command("shell", "cmd", "thermalservice", "override-status", "4")
      previous = read("vision-report").get("started_utc")
      previous_ui = read("ui-report").get("pid")
      command("shell", "am", "start", "-n", activity, "--ez", "run_vision", "true",
              "--ei", "vision_seconds", "90" if during_capture else "30")
      if during_capture:
        wait_camera(previous_ui)
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
      # With heat already critical, activity/Python cold startup precedes the
      # camera worker. Measure its rejection inside that worker; startup is
      # separately bounded above. An active camera must stop within ten seconds
      # of injecting critical heat, including report delivery.
      if during_capture:
        case["shutdown_seconds"] = case["abort_wait_seconds"]
        case["shutdown_clock"] = "critical override to completed report"
      else:
        assert not session["opened_camera_ids"], "Critical heat must prevent camera opening"
        case["shutdown_seconds"] = session["elapsed_ms"] / 1000
        case["shutdown_clock"] = "camera worker startup to rejection"
      assert 0 <= case["shutdown_seconds"] < 10, "Thermal abort exceeded the shutdown budget"
    validation_path = Path(output).with_name("emulator-thermal-live-validator.json")
    validation = subprocess.run([sys.executable, str(Path(__file__).with_name("validate_vision.py")),
                                 "--serial", serial, "--adb", adb, "--package", package,
                                 "--seconds", "15", "--report", str(validation_path)],
                                capture_output=True, text=True, timeout=45)
    result = json.loads(validation_path.read_text())
    report["live_validator"] = {key: result.get(key) for key in ("status", "error", "cleanup", "elapsed_s")}
    assert validation.returncode == 1 and result["status"] == "failed", "Live validator accepted critical heat"
    assert "critical" in result.get("error", "").lower() and result.get("cleanup"), result
    assert result["elapsed_s"] < 15, "Live validator did not abort promptly"
    command("shell", "cmd", "thermalservice", "reset")
    previous = read("vision-report").get("started_utc")
    previous_ui = read("ui-report").get("pid")
    command("shell", "am", "start", "-n", activity, "--ez", "run_vision", "true",
            "--ei", "vision_seconds", "90")
    wait_camera(previous_ui)
    # Force onCreate rather than onNewIntent: the driving UI may be on top.
    command("shell", "am", "start", "-f", "0x18000000", "-n", activity,
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
      device.stop()
    finally:
      command("shell", "cmd", "thermalservice", "reset")
      report["override_reset"] = True
      Path(output).write_text(json.dumps(report, indent=2) + "\n")
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--package", choices=(PACKAGE, CAMERA_COMPAT_PACKAGE), default=PACKAGE)
  parser.add_argument("--report", default="android/reports/emulator-thermal-abort.json")
  args = parser.parse_args()
  result = validate(args.serial, args.adb, args.report, args.package)
  print(json.dumps({key: result[key] for key in ("serial", "status", "override_reset")}, indent=2))
  raise SystemExit(0 if result["status"] == "passed" else 1)
