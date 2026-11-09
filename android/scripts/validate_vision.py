#!/usr/bin/env python3
"""Exercise physical-namespace live preview and record diagnostic model blockers."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from experiment import Device, PACKAGE, CAMERA_COMPAT_PACKAGE


def validate(serial, adb, seconds, output, package=PACKAGE, samsung=False):
  if samsung and package != CAMERA_COMPAT_PACKAGE:
    raise ValueError("Samsung multiview needs the compatibility APK")
  device = Device(serial, adb, package)
  def command(*args):
    return subprocess.check_output([adb, "-s", serial, *args], text=True, timeout=30)

  def read(name):
    try:
      return json.loads(command("shell", "run-as", package, "cat", f"files/{name}.json"))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
      return {}

  report = {"serial": serial, "status": "starting", "requested_seconds": seconds, "vehicle_tx_enabled": False,
            "road_readiness": "blocked", "calibration": "unqualified", "model_parity": "failed",
            "physical_vehicle_verified": False, "package": package, "samsung_multiview": samsung, "resource_sampling_seconds": 30}
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  samples = output.with_suffix(".jsonl")
  fresh = dict.fromkeys(("road", "driver"), 0)
  frame_ids = {name: set() for name in fresh}
  next_resources = 0.
  previous_ui_pid = read("ui-report").get("pid")
  previous_manager_pid = read("manager").get("pid")
  previous_vision_start = read("vision-report").get("started_utc")
  launch_elapsed_ms = int(float(command("shell", "cat", "/proc/uptime").split()[0]) * 1000)
  device.stop()
  start = time.monotonic()
  try:
    device.assert_thermal_safe()
    command("shell", "am", "start", "-n", package + "/ai.sunnypilot.fold3.MainActivity", "--ez", "run_vision", "true",
            "--ei", "vision_seconds", str(seconds), "--ez", "samsung_multiview", str(samsung).lower())
    with samples.open("w") as log:
      # Native/Python imports and renderer startup are slower on the emulator.
      # Stop immediately on completion, but do not truncate a fresh session.
      while time.monotonic() - start < seconds + 60:
        device.assert_thermal_safe()
        sample = {"elapsed_s": time.monotonic() - start}
        for name in ("ui-report", "manager", "dmonitoringmodeld", "modeld", "sensord", "vision-report"):
          sample[name] = read(name)
        if sample["elapsed_s"] >= next_resources:
          memory = command("shell", "dumpsys", "meminfo", "--package", package)
          pss = [int(value) for value in re.findall(r"TOTAL PSS:\s+(\d+)", memory)]
          sample["resources"] = {"process_pss_kib": pss, "total_pss_kib": sum(pss)}
          thermal = command("shell", "dumpsys", "thermalservice")
          skin = re.findall(r"Temperature\{mValue=([\d.]+), mType=3,", thermal)
          sample["resources"].update(skin_c=max(map(float, skin)) if skin else None,
                                     thermal_override="IsStatusOverride: true" in thermal)
          next_resources = sample["elapsed_s"] + 30
        log.write(json.dumps(sample) + "\n")
        log.flush()
        manager, ui = sample["manager"], sample["ui-report"]
        report["last_sample"] = sample
        if manager.get("pid") != previous_manager_pid and manager.get("updated_elapsed_ms", -1) >= launch_elapsed_ms:
          assert not manager["vehicle_tx_enabled"] and not manager["engagement_allowed"], "Unsafe manager state"
          assert not manager["simulation"] and "Simulator" not in manager["processes"], "Simulation manager in physical diagnostic"
          assert all(not worker.get("error") and not worker.get("restarts")
                     for worker in manager["processes"].values()), "Worker failure/restart"
          sensor_pid = manager["processes"].get("Sensors", {}).get("pid", 0)
          if sensor_pid and sample["sensord"].get("pid") == sensor_pid:
            assert sample["sensord"].get("thermal_status", 0) < 4, "Critical Android thermal status; aborting workload"
        vision = sample["vision-report"]
        if vision.get("started_utc") != previous_vision_start:
          errors = [session["error"] for session in vision.get("sessions", []) if session.get("error")]
          assert vision.get("status") not in ("failed", "stopped"), \
            f"Camera capture {vision.get('status')}; recoveries={vision.get('recoveries')}; errors={errors}"
        # Ignore old persisted startup reports; advancing IDs prove new delivery.
        if sample["elapsed_s"] > 5 and ui.get("pid") != previous_ui_pid:
          for stream in ui.get("camera_preview", []):
            name = stream["stream"]
            frame_ids[name].add(stream["frame_id"])
            fresh[name] += int(stream["fresh"])
        if (vision.get("started_utc") != previous_vision_start and vision.get("completed_utc")
            and sample["elapsed_s"] > 5):
          break
        time.sleep(2)
    report.update(fresh_samples=fresh, distinct_frame_ids={name: len(ids) for name, ids in frame_ids.items()})
    session = report["last_sample"]["vision-report"]
    assert session.get("transport_status", session.get("status")) == "passed", f"Camera transport failed: {session.get('status')}"
    assert all(count > 1 for count in fresh.values()), f"Insufficient fresh preview samples: {fresh}"
    assert all(len(ids) > 1 for ids in frame_ids.values()), "Preview frame IDs did not advance"
    workers = report["last_sample"]["manager"]["processes"]
    assert report["last_sample"]["manager"]["updated_elapsed_ms"] >= launch_elapsed_ms, "Stale manager report"
    assert report["last_sample"]["manager"]["pid"] != previous_manager_pid, "Previous manager process report"
    assert set(workers) == {"Controls", "Monitoring", "Sensors", "RoadModel", "DriverModel", "Vehicle",
                            "Localization", "Calibration", "Parameters", "Torque", "Delay", "Radar", "Planning", "Selfdrive"}
    assert all(not worker.get("error") and not worker.get("restarts") for worker in workers.values()), "Worker failure/restart"
    road = report["last_sample"]["modeld"]
    if samsung:
      assert road.get("pid") == workers["RoadModel"]["pid"], "Previous road model process report"
      assert road.get("computed_frames", 0) > 0, "Separate candidate rear inputs did not reach the model"
      assert road.get("valid") is False and road.get("calibrated") is False, "Unqualified model validity"
      assert "unverified" in road.get("stream_origins", {}).get("2", ""), "Missing candidate sensor disclosure"
    else:
      assert "2" in road.get("missing_physical_sensors", []), "Missing ultrawide requirement was bypassed"
    report.update(transport_status="passed", status="blocked" if samsung else "passed")
  except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
  finally:
    report["elapsed_s"] = time.monotonic() - start
    report["panda_receive_only"] = read("panda-rx")
    try:
      report["cleanup"] = device.stop()
    except Exception as exc:
      report.update(status="failed", cleanup_error=str(exc))
    output.write_text(json.dumps(report, indent=2) + "\n")
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--seconds", type=int, default=60, choices=range(5, 1801), metavar="5..1800")
  parser.add_argument("--package", choices=(PACKAGE, CAMERA_COMPAT_PACKAGE), default=PACKAGE)
  parser.add_argument("--samsung-multiview", action="store_true")
  parser.add_argument("--report", default="android-vision.json")
  args = parser.parse_args()
  result = validate(args.serial, args.adb, args.seconds, args.report, args.package, args.samsung_multiview)
  print(json.dumps({key: value for key, value in result.items() if key != "last_sample"}, indent=2))
  raise SystemExit(0 if result["status"] == "passed" else 1)
