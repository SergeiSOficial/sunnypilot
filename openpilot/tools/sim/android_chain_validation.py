#!/usr/bin/env python3
"""Debug source fixtures for the complete receive-only Android process chain.

Derived calibration, plans, vehicle state and selfdrive state must be produced
on Android. Synthetic CAN is confined to the app's simulation IPC namespace.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from openpilot.tools.sim.android_bridge import connect
from openpilot.tools.sim.lib.android_sources import Sources, observe
from openpilot.tools.sim.lib.android_protocol import CHAIN_OUTBOUND, receive, send


def validate(serial, adb, seconds, report_path, package, steady_inputs=False, device_replay=False):
  report = {"status": "starting", "vehicle_tx_enabled": False, "physical_engagement_allowed": False,
            "synthetic_sources": True, "physical_vehicle_verified": False, "camera_model_parity_verified": False,
            "derived_inputs_injected": False, "received_services": [], "packets": 0, "unsafe_outputs": 0,
            "input_pacing": "absolute_deadline_100_hz", "steady_inputs": steady_inputs,
            "fixture_sha256": hashlib.sha256((Path(__file__).parent / "lib/android_sources.py").read_bytes()).hexdigest(), "source_deadline_misses": 0}
  samples, seen, port = {}, set(), None
  try:
    port = int(subprocess.check_output([adb, "-s", serial, "forward", "--no-rebind", "tcp:0",
                                       "localabstract:sunnypilot_fold3_sim"], text=True).strip())
    source = Sources()
    from openpilot.cereal import messaging
    source.events(0)  # Resolve lazy imports before the relay's 100 ms transport deadline.
    connection, profile = connect(port, timeout=90, full_chain=True)
    report["profile"] = profile
    start = time.monotonic()
    next_frame, work_ms = start, []
    with connection:
      if device_replay:
        send(connection, {"sequence": 0, "op": "replay", "seconds": seconds, "steady_inputs": steady_inputs})
        connection.settimeout(seconds + 90)
        ready = receive(connection)
        if ready.get("op") != "replay_ready" or ready.get("vehicle_tx_enabled") is not False:
          raise ValueError("Invalid device replay readiness")
        result = receive(connection)
        if (result.get("op") != "replay_complete" or result.get("vehicle_tx_enabled") is not False or
            result.get("derived_inputs_injected") is not False or not set(result.get("output_samples", {})) <= set(CHAIN_OUTBOUND)):
          raise ValueError("Invalid device replay response")
        report.update({key: value for key, value in result.items() if key != "op"})
        seen = set(report["output_samples"])
      else:
        while (elapsed := time.monotonic() - start) < seconds:
          iteration = time.monotonic()
          sequence = report["packets"]
          # Cancellation is represented by factory ACC disabling in passive CAN.
          phase = "cruise" if steady_inputs or elapsed < seconds * .6 else "cancel" if elapsed < seconds * .8 else "stale"
          events = source.events(sequence, cruise=phase == "cruise")
          send(connection, {"sequence": sequence, "messages": {name: base64.b64encode(event.to_bytes()).decode()
                            for name, event in events.items()}, "fault": "stale" if phase == "stale" else None})
          reply = receive(connection)
          if reply.get("sequence") != sequence or reply.get("vehicle_tx_enabled") is not False:
            raise ValueError("Invalid complete-chain response")
          for name, payload in reply["messages"].items():
            if name not in CHAIN_OUTBOUND:
              raise ValueError("Unexpected Android output")
            event = messaging.log_from_bytes(base64.b64decode(payload, validate=True))
            if event.which() != name:
              raise ValueError("Cereal output service mismatch")
            observe(report, phase, name, getattr(event, name))
          seen = set(report.get("output_samples", {}))
          samples = report.get("output_samples", {})
          report["packets"] += 1
          finished = time.monotonic()
          work_ms.append((finished - iteration) * 1000)
          next_frame += .01
          report["source_deadline_misses"] += int(finished > next_frame)
          # Absolute deadlines avoid adding host sleep overshoot to every period.
          time.sleep(max(0, next_frame - time.monotonic()))
        report["input_elapsed_s"] = time.monotonic() - start
        report["source_mean_hz"] = report["packets"] / report["input_elapsed_s"]
        report["source_exchange_work_p99_ms"] = sorted(work_ms)[math.ceil(len(work_ms) * .99) - 1]
        report["received_services"], report["output_samples"] = sorted(seen), samples
    report["processes"] = json.loads(subprocess.check_output([adb, "-s", serial, "shell", "run-as", package,
                                                             "cat", "files/manager.json"], text=True))
    report["runtime_reports"] = {}
    for name in ("controlsd", "selfdrived", "plannerd", "ui-report"):
      try:
        report["runtime_reports"][name] = json.loads(subprocess.check_output(
          [adb, "-s", serial, "shell", "run-as", package, "cat", f"files/{name}.json"], text=True))
      except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        report["runtime_reports"][name] = {"error": str(exc)}
    required = {"carState", "deviceMotion", "extrinsicsCalibration", "vehicleParameters", "lateralTorqueParameters",
                "lateralDelay", "radarState", "longitudinalPlan", "selfdriveState", "carControl", "driverMonitoringState"}
    report["missing_outputs"] = sorted(required - seen)
    report["factory_acc_cancellation_passed"] = (None if steady_inputs else
      report.get("cruise_acc_enabled_samples", 0) > 0 and report.get("cancel_acc_disabled_samples", 0) > 0)
    report["fault_checks"] = "not_run_steady_inputs" if steady_inputs else "factory_acc_cancel_and_stale_can"
    source_checks = (report.get("cruise_can_valid_samples", 0) > 0 and report.get("cruise_acc_enabled_samples", 0) > 0
                     if steady_inputs else report["factory_acc_cancellation_passed"] and report.get("stale_can_invalid_samples", 0) > 0)
    workers = report["processes"]["processes"]
    report["worker_failures"] = {name: worker for name, worker in workers.items() if worker.get("error") or worker.get("restarts")}
    report["status"] = ("passed" if not report["missing_outputs"] and not report["unsafe_outputs"] and not report["worker_failures"]
                        and source_checks else "failed")
    report["elapsed_s"] = time.monotonic() - start
  except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    report["failure_reports"] = {}
    time.sleep(1.2)  # Worker reports are flushed once per second.
    for name in ("manager", "sim_bridge", "controlsd", "selfdrived", "ui-report"):
      try:
        report["failure_reports"][name] = json.loads(subprocess.check_output(
          [adb, "-s", serial, "shell", "run-as", package, "cat", f"files/{name}.json"], text=True, timeout=15))
      except (subprocess.SubprocessError, json.JSONDecodeError) as read_error:
        report["failure_reports"][name] = {"error": str(read_error)}
  finally:
    if port:
      subprocess.run([adb, "-s", serial, "forward", "--remove", f"tcp:{port}"], capture_output=True, check=False)
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--package", default="com.samsung.sunnypilot.fold3")
  parser.add_argument("--seconds", type=int, choices=range(10, 121), default=30)
  parser.add_argument("--steady-inputs", action="store_true", help="Keep factory ACC and source CAN present for timing; skip fault phases")
  parser.add_argument("--device-replay", action="store_true", help="Run synthetic 100 Hz sources on Android; ADB carries setup/results only")
  parser.add_argument("--report", required=True)
  args = parser.parse_args()
  result = validate(args.serial, args.adb, args.seconds, args.report, args.package, args.steady_inputs, args.device_replay)
  print(json.dumps(result, indent=2))
  raise SystemExit(0 if result["status"] == "passed" else 1)
