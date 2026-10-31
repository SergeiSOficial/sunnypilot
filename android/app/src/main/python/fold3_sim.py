"""Debug-only Cereal relay over `adb forward ... localabstract:sunnypilot_fold3_sim`."""
import base64
import os
import socket
import struct
import time

from android_protocol import INBOUND, OUTBOUND, receive, send, stale_control_inputs, validate_exchange


def disengage(pm):
  import openpilot.cereal.messaging as messaging
  for service in ("selfdriveState", "selfdriveStateSP"):
    inactive = messaging.new_message(service, valid=False)
    pm.send(service, inactive)


def prepare_params():
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
  params.put("CarParamsSP", convert_to_capnp(cp_sp).to_bytes(), block=True)
  params.put("CarParams", cp.to_bytes(), block=True)
  return {"car_fingerprint": str(candidate), "factory_acc": True, "min_steer_speed_m_s": cp.minSteerSpeed,
          "physical_fingerprint_verified": False}


def serve(report):
  import capnp
  from java import jclass
  if not jclass("ai.sunnypilot.fold3.BuildConfig").DEBUG:
    raise PermissionError("simulator transport is unavailable in release builds")
  import openpilot.cereal.messaging as messaging
  pm = messaging.PubMaster(sorted(INBOUND))
  sm = messaging.SubMaster(list(OUTBOUND))
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
          profile = prepare_params()
          send(connection, {"op": "hello", "simulation": True, "vehicle_tx_enabled": False, **profile})
          connection.settimeout(.1)
          sequence = -1
          timestamps = {}
          while True:
            packet = receive(connection)
            sequence = validate_exchange(packet, sequence)
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
              message.logMonoTime = now - int(age_ms * 1e6)
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
            stale = stale_control_inputs(timestamps, now)
            if stale or packet.get("fault") in ("cancel", "stale") or any(age > 100 for age in packet.get("ages_ms", {}).values()):
              # Replace active states before publishing. Sending active then
              # inactive allowed controlsd to observe the intermediate state.
              for service in ("selfdriveState", "selfdriveStateSP"):
                decoded[service] = messaging.new_message(service, valid=False)
            for service, message in decoded.items():
              pm.send(service, message)
            sm.update(0)
            outputs = {}
            # SubMaster stores inner structs; construct full Cereal Events.
            for service in OUTBOUND:
              if not sm.updated[service]:
                continue
              message = messaging.new_message(service, valid=sm.valid[service])
              setattr(message, service, sm[service])
              outputs[service] = base64.b64encode(message.to_bytes()).decode()
            send(connection, {"sequence": sequence, "messages": outputs, "vehicle_tx_enabled": False})
            accepted += 1
            report.update("running", accepted_packets=accepted, rejected_sessions=rejected,
                          latest_input_services=sorted(decoded), stale_control_inputs=stale, **profile)
        except (EOFError, TimeoutError, ValueError, PermissionError, OSError, capnp.KjException) as exc:
          rejected += 1
          report.update("waiting_for_adb", last_disconnect=str(exc), accepted_packets=accepted, rejected_sessions=rejected)
        finally:
          # A disconnected simulator must not leave a cached active state.
          disengage(pm)
