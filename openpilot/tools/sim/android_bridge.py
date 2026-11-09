#!/usr/bin/env python3
"""Relay an existing host simulator's Cereal services to the Android workers.

Run alongside MetaDrive and the host perception/planning stack, excluding host
controlsd and dmonitoringd. This transport never forwards CAN or Panda sockets.
"""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

from openpilot.tools.sim.lib.android_protocol import INBOUND, OUTBOUND, receive, send


def connect(port, timeout=30, full_chain=False):
  """ADB forwarding can open before the Android worker is listening."""
  deadline = time.monotonic() + timeout
  while True:
    connection = None
    try:
      connection = socket.create_connection(("127.0.0.1", port), timeout=5)
      connection.settimeout(5)
      send(connection, {"op": "hello", "simulation": True, "full_chain": full_chain})
      hello = receive(connection)
      if hello.get("simulation") is not True or hello.get("vehicle_tx_enabled") is not False or hello.get("factory_acc") is not True:
        raise ValueError("Android did not confirm isolated, stock-ACC simulation")
      if bool(hello.get("full_chain", False)) != full_chain:
        raise ValueError("Android process-chain mode differs from requested test")
      return connection, hello
    except (OSError, EOFError):
      if connection is not None:
        connection.close()
      if time.monotonic() >= deadline:
        raise
      time.sleep(.25)
    except Exception:
      if connection is not None:
        connection.close()
      raise


def relay(serial, port, duration, report_path, probe=False, fault=None, fault_after=5, adb="adb"):
  if not probe and not os.environ.get("OPENPILOT_PREFIX", "").startswith("fold3_sim_"):
    raise ValueError("Android relay requires an isolated OPENPILOT_PREFIX starting with fold3_sim_")
  command = [adb, "-s", serial, "forward", f"tcp:{port}", "localabstract:sunnypilot_fold3_sim"]
  subprocess.run(command, check=True, capture_output=True)
  report = {"status": "starting", "vehicle_tx_enabled": False, "simulation": True,
            "closed_loop_verified": False, "physical_vehicle_verified": False,
            "packets": 0, "received_services": [], "fault": fault, "transport_only": probe}
  timings, seen, inputs = [], set(), set()
  try:
    connection, hello = connect(port)
    with connection:
      report["android_profile"] = hello
      if not probe:
        import openpilot.cereal.messaging as messaging
        sm = messaging.SubMaster(sorted(INBOUND))
        pm = messaging.PubMaster(list(OUTBOUND))
      started, sequence = time.monotonic(), 0
      while time.monotonic() - started < duration:
        iteration = time.monotonic_ns()
        messages, ages = {}, {}
        if not probe:
          sm.update(0)
          for service in INBOUND:
            if sm.updated[service]:
              message = messaging.new_message(service, valid=sm.valid[service])
              setattr(message, service, sm[service])
              messages[service] = base64.b64encode(message.to_bytes()).decode()
              ages[service] = min(10000, max(0, (iteration - sm.logMonoTime[service]) / 1e6))
          inputs.update(messages)
        active_fault = fault if time.monotonic() - started >= fault_after else None
        send(connection, {"sequence": sequence, "messages": messages, "ages_ms": ages, "fault": active_fault})
        if active_fault == "disconnect":
          report["disconnect_injected"] = True
          break
        reply = receive(connection)
        if reply.get("sequence") != sequence or reply.get("vehicle_tx_enabled") is not False:
          raise ValueError("invalid Android simulator response")
        outputs = reply.get("messages", {})
        if not isinstance(outputs, dict) or not outputs.keys() <= set(OUTBOUND):
          raise ValueError("Android returned disallowed simulator services")
        if not probe:
          for service, payload in outputs.items():
            message = messaging.log_from_bytes(base64.b64decode(payload, validate=True)).as_builder()
            if message.which() != service:
              raise ValueError("Android output Cereal service mismatch")
            message.logMonoTime = time.monotonic_ns()
            pm.send(service, message)
        seen.update(outputs)
        timings.append((time.monotonic_ns() - iteration) / 1e6)
        sequence += 1
        report["packets"] = sequence
        time.sleep(max(0, .01 - (time.monotonic_ns() - iteration) / 1e9))
      report["status"] = "passed" if probe and sequence > 0 else "measured"
      report["received_services"] = sorted(seen)
      report["sent_services"] = sorted(inputs)
      required = {"carState", "modelV2", "selfdriveState", "extrinsicsCalibration", "driverStateV2"}
      report["missing_inputs"] = sorted(required - inputs) if not probe else []
      if not probe and (report["missing_inputs"] or "carControl" not in seen):
        report["status"] = "blocked"
      if timings:
        report["p99_round_trip_ms"] = sorted(timings)[math.ceil(len(timings) * .99) - 1]
      report["qualification_blocker"] = "Transport results alone do not qualify closed-loop perception, steering or vehicle integration"
  except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    raise
  finally:
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    subprocess.run([adb, "-s", serial, "forward", "--remove", f"tcp:{port}"], check=False, capture_output=True)
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--port", type=int, default=8766)
  parser.add_argument("--duration", type=float, default=30)
  parser.add_argument("--report", default="android-relay-report.json")
  parser.add_argument("--probe", action="store_true", help="Test framing and handshake only; do not publish host IPC")
  parser.add_argument("--fault", choices=("cancel", "stale", "disconnect"))
  parser.add_argument("--fault-after", type=float, default=5)
  args = parser.parse_args()
  if not 0 < args.duration <= 3600 or not 1024 <= args.port <= 65535 or not 0 <= args.fault_after <= args.duration:
    parser.error("duration, port or fault timing outside bounds")
  print(json.dumps(relay(args.serial, args.port, args.duration, args.report, args.probe,
                         args.fault, args.fault_after, args.adb), indent=2))


if __name__ == "__main__":
  main()
