#!/usr/bin/env python3
"""Exercise physical-namespace live preview and record diagnostic model blockers."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from experiment import Device


def validate(serial, adb, seconds, output):
  def command(*args):
    return subprocess.check_output([adb, "-s", serial, *args], text=True, timeout=30)

  def read(name):
    try:
      return json.loads(command("shell", "run-as", "ai.sunnypilot.fold3", "cat", f"files/{name}.json"))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
      return {}

  report = {"serial": serial, "status": "starting", "requested_seconds": seconds, "vehicle_tx_enabled": False,
            "road_readiness": "blocked", "calibration": "unqualified", "model_parity": "failed",
            "physical_vehicle_verified": False, "resource_sampling_seconds": 30}
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
  Device(serial, adb).stop()
  command("shell", "am", "start", "-n", "ai.sunnypilot.fold3/.MainActivity", "--ez", "run_vision", "true",
          "--ei", "vision_seconds", str(seconds))
  start = time.monotonic()
  try:
    with samples.open("w") as log:
      while time.monotonic() - start < seconds + 15:
        sample = {"elapsed_s": time.monotonic() - start}
        for name in ("ui-report", "manager", "dmonitoringmodeld", "modeld", "sensord", "vision-report"):
          sample[name] = read(name)
        if sample["elapsed_s"] >= next_resources:
          memory = command("shell", "dumpsys", "meminfo", "--package", "ai.sunnypilot.fold3")
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
        time.sleep(2)
    report.update(fresh_samples=fresh, distinct_frame_ids={name: len(ids) for name, ids in frame_ids.items()})
    session = report["last_sample"]["vision-report"]
    assert session.get("status") == "passed", f"Camera capture failed or recovered: {session.get('status')}"
    assert all(count > 1 for count in fresh.values()), f"Insufficient fresh preview samples: {fresh}"
    assert all(len(ids) > 1 for ids in frame_ids.values()), "Preview frame IDs did not advance"
    workers = report["last_sample"]["manager"]["processes"]
    assert report["last_sample"]["manager"]["updated_elapsed_ms"] >= launch_elapsed_ms, "Stale manager report"
    assert report["last_sample"]["manager"]["pid"] != previous_manager_pid, "Previous manager process report"
    assert set(workers) == {"Controls", "Monitoring", "Sensors", "RoadModel", "DriverModel"}
    assert all(not worker.get("error") and not worker.get("restarts") for worker in workers.values()), "Worker failure/restart"
    # The absent ultrawide must block modeld rather than duplicating road frames.
    road = report["last_sample"]["modeld"]
    assert "2" in road.get("missing_physical_sensors", []), "Missing ultrawide requirement was bypassed"
    report["status"] = "passed"
  except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
  finally:
    report["elapsed_s"] = time.monotonic() - start
    report["panda_receive_only"] = read("panda-rx")
    try:
      report["cleanup"] = Device(serial, adb).stop()
    except Exception as exc:
      report.update(status="failed", cleanup_error=str(exc))
    output.write_text(json.dumps(report, indent=2) + "\n")
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--seconds", type=int, default=60, choices=range(5, 1801), metavar="5..1800")
  parser.add_argument("--report", default="android-vision.json")
  args = parser.parse_args()
  result = validate(args.serial, args.adb, args.seconds, args.report)
  print(json.dumps({key: value for key, value in result.items() if key != "last_sample"}, indent=2))
  raise SystemExit(0 if result["status"] == "passed" else 1)
