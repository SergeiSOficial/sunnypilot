"""Debug-only Cereal relay over `adb forward ... localabstract:sunnypilot_fold3_sim`."""
import base64
import math
import os
import select
import socket
import struct
import time

from android_protocol import CHAIN_INBOUND, CHAIN_OUTBOUND, INBOUND, OUTBOUND, receive, send, stale_control_inputs, validate_exchange, validate_replay


def disengage(pm, full_chain=False):
  import openpilot.cereal.messaging as messaging
  for service in (("can", "modelV2", "driverStateV2", "cameraOdometry") if full_chain else ("selfdriveState", "selfdriveStateSP")):
    inactive = messaging.new_message(service, 0, valid=False) if service == "can" else messaging.new_message(service, valid=False)
    pm.send(service, inactive)


def receive_input(connection, pm, timestamps, full_chain=False):
  # An idle debug client must withdraw inputs after 100 ms, but need not lose
  # its framed connection. A timeout halfway through a packet still disconnects.
  while not select.select([connection], [], [], .1)[0]:
    timestamps.clear()
    disengage(pm, full_chain)
  return receive(connection)


def prepare_params(full_chain=False):
  from opendbc.car.car_helpers import interfaces
  from opendbc.car.hyundai.values import CAR
  from openpilot.common.params import Params
  from openpilot.selfdrive.car.helpers import convert_to_capnp
  if not os.environ["PARAMS_ROOT"].endswith("/sim-params") or not os.environ["MSGQ_ANDROID_PATH"].endswith("/sim-ipc"):
    raise PermissionError("simulator must have an isolated Params and IPC namespace")
  candidate = CAR.HYUNDAI_IONIQ_PHEV_2019
  interface = interfaces[candidate]
  cp = interface.get_non_essential_params(candidate)
  cp_sp = interface.get_non_essential_params_sp(cp, candidate)
  if cp.openpilotLongitudinalControl or not cp.pcmCruise:
    raise ValueError("simulator Hyundai profile must preserve factory ACC")
  params = Params()
  params.put_bool("AlphaLongitudinalEnabled", False)
  params.put_bool("IsRhdDetected", True)
  if not full_chain:
    params.put("CarParamsSP", convert_to_capnp(cp_sp).to_bytes(), block=True)
    params.put("CarParams", cp.to_bytes(), block=True)
  return {"car_fingerprint": str(candidate), "factory_acc": True, "min_steer_speed_m_s": cp.minSteerSpeed,
          "physical_fingerprint_verified": False}


def encode_outputs(sm, services):
  from openpilot.cereal import messaging
  outputs = {}
  for service in services:
    if not sm.updated[service]:
      continue
    # radarTracks is a RadarData struct; only onroadEvents is an Event list.
    message = (messaging.new_message(service, len(sm[service]), valid=sm.valid[service])
               if service == "onroadEvents" else messaging.new_message(service, valid=sm.valid[service]))
    setattr(message, service, sm[service])
    outputs[service] = base64.b64encode(message.to_bytes()).decode()
  return outputs


def replay(connection, pm, sm, report, seconds, steady_inputs):
  from android_sources import Sources, observe
  import openpilot.cereal.messaging as messaging
  source = Sources()
  source.events(0)  # Resolve imports before the 100 Hz measurement.
  result = {"op": "replay_complete", "vehicle_tx_enabled": False, "synthetic_sources": True,
            "derived_inputs_injected": False, "packets": 0, "unsafe_outputs": 0, "output_samples": {},
            "source_deadline_misses": 0, "input_pacing": "android_absolute_deadline_100_hz"}
  send(connection, {"op": "replay_ready", "vehicle_tx_enabled": False})
  start, work_ms = time.monotonic(), []
  deadline = start
  while (elapsed := time.monotonic() - start) < seconds:
    # A disconnected ADB peer must stop autonomous fixtures immediately.
    if select.select([connection], [], [], 0)[0]:
      raise EOFError("Debug replay client disconnected or sent an unexpected command")
    entered = time.monotonic()
    phase = "cruise" if steady_inputs or elapsed < seconds * .6 else "cancel" if elapsed < seconds * .8 else "stale"
    events = source.events(result["packets"], cruise=phase == "cruise")
    if phase == "stale":
      for name in ("can", "modelV2", "driverStateV2"):
        events[name] = messaging.new_message(name, 0, valid=False) if name == "can" else messaging.new_message(name, valid=False)
    for name, event in events.items():
      pm.send(name, event)
    sm.update(0)
    for name in CHAIN_OUTBOUND:
      if sm.updated[name]:
        observe(result, phase, name, sm[name])
    result["packets"] += 1
    finished = time.monotonic()
    work_ms.append((finished - entered) * 1000)
    deadline += .01
    result["source_deadline_misses"] += int(finished > deadline)
    report.update("replaying_debug_fixtures", replay_packets=result["packets"], vehicle_tx_enabled=False)
    time.sleep(max(0, deadline - time.monotonic()))
  result["input_elapsed_s"] = time.monotonic() - start
  result["source_mean_hz"] = result["packets"] / result["input_elapsed_s"]
  result["source_exchange_work_p99_ms"] = sorted(work_ms)[math.ceil(len(work_ms) * .99) - 1]
  result["received_services"] = sorted(result["output_samples"])
  disengage(pm, True)
  send(connection, result)


def serve(report, full_chain=False):
  import capnp
  from java import jclass
  if not jclass("ai.sunnypilot.fold3.BuildConfig").DEBUG:
    raise PermissionError("simulator transport is unavailable in release builds")
  import openpilot.cereal.messaging as messaging
  inbound, outbound = (CHAIN_INBOUND, CHAIN_OUTBOUND) if full_chain else (INBOUND, OUTBOUND)
  pm = messaging.PubMaster(sorted(inbound))
  sm = messaging.SubMaster(list(outbound))
  accepted, rejected = 0, 0
  with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
    server.bind("\0sunnypilot_fold3_sim")
    server.listen(1)
    server.settimeout(1)
    while True:
      report.update("waiting_for_adb", accepted_packets=accepted, rejected_sessions=rejected)
      try:
        connection, _ = server.accept()
      except TimeoutError:
        continue
      with connection:
        connection.settimeout(1)
        try:
          # Abstract sockets have no filesystem permissions. Restrict clients
          # to this application UID and the ADB shell/root forwarding daemon.
          _, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
          if uid not in (os.getuid(), 2000, 0):
            raise PermissionError("simulator peer is not the application or ADB")
          hello = receive(connection)
          if hello.get("op") != "hello" or hello.get("simulation") is not True:
            raise ValueError("explicit simulator handshake required")
          if bool(hello.get("full_chain", False)) != full_chain:
            raise ValueError("simulator process-chain mode mismatch")
          profile = prepare_params(full_chain)
          send(connection, {"op": "hello", "simulation": True, "full_chain": full_chain, "vehicle_tx_enabled": False, **profile})
          connection.settimeout(.1)
          sequence = -1
          timestamps = {}
          while True:
            packet = receive_input(connection, pm, timestamps, full_chain)
            sequence = validate_exchange(packet, sequence, allowed=inbound)
            if packet.get("op") == "replay":
              seconds, steady = validate_replay(packet, full_chain)
              replay(connection, pm, sm, report, seconds, steady)
              break
            if packet.get("fault") == "disconnect":
              break
            now = time.monotonic_ns()
            decoded = {}
            for service, payload in packet.get("messages", {}).items():
              data = base64.b64decode(payload, validate=True)
              # Validate every message before publishing any part of a packet.
              message = messaging.log_from_bytes(data).as_builder()
              if message.which() != service:
                raise ValueError("Cereal service/payload mismatch")
              age_ms = packet.get("ages_ms", {}).get(service, 0)
              if packet.get("fault") == "stale":
                age_ms = max(age_ms, 2000)
              original_time = message.logMonoTime
              message.logMonoTime = now - int(age_ms * 1e6)
              # Rebase debug source clocks by the same offset as their Event.
              # Physical Camera2 capture timestamps never use this transport.
              source = getattr(message, service)
              fields = (("timestamp",) if service in ("accelerometer", "gyroscope") else
                        ("timestampEof",) if service in ("cameraOdometry", "modelV2") else ())
              for field in fields:
                stamp = getattr(source, field)
                if stamp:
                  rebased = stamp + message.logMonoTime - original_time
                  if rebased <= 0:
                    raise ValueError("invalid rebased simulator capture time")
                  setattr(source, field, rebased)
              if age_ms > 100:
                message.valid = False
              if service == "selfdriveState" and packet.get("fault") == "cancel":
                message.selfdriveState.active = False
                message.selfdriveState.enabled = False
              decoded[service] = message
            for service, message in decoded.items():
              if message.valid:
                timestamps[service] = message.logMonoTime
              else:
                timestamps.pop(service, None)
            # controlsd relies on selfdrived's active state. The debug relay is
            # responsible for withdrawing it on transport and injected faults,
            # including sunnypilot's independent MADS state.
            stale = (sorted(s for s in ("can", "modelV2", "driverStateV2")
                            if s not in timestamps or not 0 <= now - timestamps[s] <= 100_000_000)
                     if full_chain else stale_control_inputs(timestamps, now))
            if stale or packet.get("fault") in ("cancel", "stale") or any(age > 100 for age in packet.get("ages_ms", {}).values()):
              # Replace active states before publishing. Sending active then
              # inactive allowed controlsd to observe the intermediate state.
              for service in (("can", "modelV2", "driverStateV2") if full_chain else ("selfdriveState", "selfdriveStateSP")):
                decoded[service] = (messaging.new_message(service, 0, valid=False) if service == "can"
                                    else messaging.new_message(service, valid=False))
            for service, message in decoded.items():
              pm.send(service, message)
            sm.update(0)
            outputs = encode_outputs(sm, outbound)
            send(connection, {"sequence": sequence, "messages": outputs, "vehicle_tx_enabled": False})
            accepted += 1
            report.update("running", accepted_packets=accepted, rejected_sessions=rejected,
                          latest_input_services=sorted(decoded), stale_control_inputs=stale, **profile)
        except (EOFError, TimeoutError, ValueError, PermissionError, OSError, capnp.KjException) as exc:
          rejected += 1
          report.update("waiting_for_adb", last_disconnect=str(exc), accepted_packets=accepted, rejected_sessions=rejected)
        finally:
          # A disconnected simulator must not leave a cached active state.
          disengage(pm, full_chain)
