#!/usr/bin/env python3
"""Reject malformed debug ADB sessions and verify disconnect disengagement.

Requires an explicitly started simulation manager. No CAN transport is opened.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time

from openpilot.tools.sim.android_bridge import connect
from openpilot.tools.sim.lib.android_protocol import MAX_PACKET, receive, send


def validate(serial, adb, output):
  def command(*args):
    return subprocess.check_output([adb, "-s", serial, *args], text=True, timeout=30)
  def state(name):
    return json.loads(command("shell", "run-as", "ai.sunnypilot.fold3", "cat", f"files/{name}.json"))
  report = {"serial": serial, "status": "starting", "simulation": True,
            "vehicle_tx_enabled": False, "physical_vehicle_verified": False, "rejected_cases": []}
  port = int(command("forward", "--no-rebind", "tcp:0", "localabstract:sunnypilot_fold3_sim").strip())
  try:
    before = state("manager")
    assert before["simulation"] is True and before["vehicle_tx_enabled"] is False
    import openpilot.cereal.messaging as messaging
    from openpilot.tools.sim.android_validation import inputs
    car_state = base64.b64encode(messaging.new_message("carState").to_bytes()).decode()
    cases = {
      "can_service": {"sequence": 0, "messages": {"sendcan": ""}},
      "output_service": {"sequence": 0, "messages": {"carControl": ""}},
      "sequence_gap": {"sequence": 1, "messages": {}},
      "negative_age": {"sequence": 0, "messages": {"carState": car_state}, "ages_ms": {"carState": -1}},
      "service_payload_mismatch": {"sequence": 0, "messages": {"driverStateV2": car_state}},
      "malformed_cereal": {"sequence": 0, "messages": {"carState": "eA=="}},
      "invalid_base64": {"sequence": 0, "messages": {"carState": "!"}},
    }
    for name, packet in cases.items():
      connection, _ = connect(port)
      with connection:
        send(connection, packet)
        assert connection.recv(1) == b"", f"{name}: invalid session was accepted"
      report["rejected_cases"].append(name)
    for name, wire in (("oversized_packet", struct.pack("!I", MAX_PACKET + 1)),
                       ("malformed_json", struct.pack("!I", 1) + b"x")):
      connection, _ = connect(port)
      with connection:
        connection.sendall(wire)
        assert connection.recv(1) == b"", f"{name}: invalid framing was accepted"
      report["rejected_cases"].append(name)
    with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
      send(connection, {"op": "hello", "simulation": False})
      assert connection.recv(1) == b"", "non-simulation handshake accepted"
    report["rejected_cases"].append("non_simulation_handshake")

    connection, profile = connect(port)
    counts = {"active": 0, "missing_model": 0, "recovered": 0}
    with connection:
      sequence = 0
      for phase, duration in (("active", 3), ("missing_model", 1), ("recovered", 1)):
        started = time.monotonic()
        while (elapsed := time.monotonic() - started) < duration:
          packet = inputs(elapsed, 20., 0., profile["min_steer_speed_m_s"])
          if phase == "missing_model":
            del packet["modelV2"]
          send(connection, {"sequence": sequence, "messages": packet})
          reply = receive(connection)
          assert reply["vehicle_tx_enabled"] is False
          if payload := reply["messages"].get("carControl"):
            event = messaging.log_from_bytes(base64.b64decode(payload))
            assert not event.carControl.longActive
            if phase == "missing_model" and elapsed > .25:
              assert not event.carControl.latActive, "live ADB transport kept stale model input active"
              counts[phase] += 1
            elif phase != "missing_model":
              counts[phase] += int(event.carControl.latActive)
          sequence += 1
          time.sleep(.01)
    assert all(count > 10 for count in counts.values()), f"insufficient active/stale/recovery outputs: {counts}"
    # Abrupt host socket closure must withdraw both active-state services.
    time.sleep(.3)
    connection, _ = connect(port)
    inactive_samples = 0
    with connection:
      started, sequence = time.monotonic(), 0
      while time.monotonic() - started < 1:
        send(connection, {"sequence": sequence, "messages": {}})
        reply = receive(connection)
        assert reply["vehicle_tx_enabled"] is False
        if payload := reply["messages"].get("carControl"):
          event = messaging.log_from_bytes(base64.b64decode(payload))
          assert not event.carControl.latActive and not event.carControl.longActive, "control remained active after disconnect"
          inactive_samples += 1
        sequence += 1
        time.sleep(.01)
    assert inactive_samples > 10, "no control output observed after disconnect"
    time.sleep(1.1)
    after = state("manager")
    assert before["pid"] == after["pid"] and after["vehicle_tx_enabled"] is False
    for name, child in before["processes"].items():
      assert child["pid"] == after["processes"][name]["pid"], f"{name} restarted after malformed session"
    report.update(status="passed", active_samples_before_disconnect=counts["recovered"],
                  input_freshness_samples=counts,
                  inactive_samples_after_disconnect=inactive_samples, manager_before=before,
                  manager_after=after, relay=state("sim_bridge"))
  except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
  finally:
    command("forward", "--remove", f"tcp:{port}")
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--report", default="adb-faults.json")
  args = parser.parse_args()
  with tempfile.TemporaryDirectory(prefix="fold3-adb-faults-") as directory:
    os.environ["PARAMS_ROOT"] = directory + "/params"
    os.environ["OPENPILOT_PREFIX"] = f"fold3_sim_faults_{os.getpid()}"
    result = validate(args.serial, args.adb, args.report)
  print(json.dumps({key: result[key] for key in ("status", "rejected_cases", "vehicle_tx_enabled")}, indent=2))
  raise SystemExit(0 if result["status"] == "passed" else 1)
