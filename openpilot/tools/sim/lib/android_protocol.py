"""Bounded debug ADB framing. Shared by Android and the host MetaDrive relay."""
import json
import struct

VERSION = 1
MAX_PACKET = 2 * 1024 * 1024
INBOUND = frozenset(("carState", "modelV2", "selfdriveState", "extrinsicsCalibration", "deviceMotion",
                     "longitudinalPlan", "lateralManeuverPlan", "vehicleParameters", "lateralTorqueParameters",
                     "lateralDelay", "driverStateV2", "onroadEvents", "driverAssistance", "radarState",
                     "selfdriveStateSP", "carOutput", "accelerometer", "gyroscope", "gpsLocationExternal"))
OUTBOUND = ("carControl", "carControlSP", "controlsState", "driverMonitoringState")
CONTROL_INPUTS = frozenset(("carState", "modelV2", "selfdriveState", "selfdriveStateSP", "driverStateV2"))
# Complete-chain tests inject sources only. Derived state must come from the
# Android processes; accepting host plans/calibration here would mask failures.
CHAIN_INBOUND = frozenset(("can", "modelV2", "driverStateV2", "cameraOdometry", "accelerometer", "gyroscope",
                           "gpsLocationExternal", "deviceState", "pandaStates", "peripheralState",
                           "narrowRoadCameraState", "wideRoadCameraState", "cabinCameraState"))
CHAIN_OUTBOUND = OUTBOUND + ("carState", "carOutput", "radarTracks", "radarState", "deviceMotion",
                            "extrinsicsCalibration", "vehicleParameters", "lateralTorqueParameters", "lateralDelay",
                            "longitudinalPlan", "driverAssistance", "selfdriveState", "selfdriveStateSP", "onroadEvents")


def stale_control_inputs(timestamps, now_ns):
  """A live transport must not keep stale or missing driving inputs active."""
  return sorted(service for service in CONTROL_INPUTS
                if service not in timestamps or not 0 <= now_ns - timestamps[service] <= 100_000_000)


def read_exact(connection, length):
  result = bytearray()
  while len(result) < length:
    data = connection.recv(length - len(result))
    if not data:
      raise EOFError("ADB stream disconnected")
    result.extend(data)
  return bytes(result)


def receive(connection):
  size, = struct.unpack("!I", read_exact(connection, 4))
  if not 0 < size <= MAX_PACKET:
    raise ValueError("invalid ADB packet length")
  packet = json.loads(read_exact(connection, size))
  if not isinstance(packet, dict) or packet.get("version") != VERSION:
    raise ValueError("unsupported ADB protocol")
  return packet


def send(connection, packet):
  data = json.dumps({"version": VERSION, **packet}, allow_nan=False, separators=(",", ":")).encode()
  if not 0 < len(data) <= MAX_PACKET:
    raise ValueError("ADB packet exceeds limit")
  connection.sendall(struct.pack("!I", len(data)) + data)


def validate_exchange(packet, previous_sequence, allowed=INBOUND):
  sequence = packet.get("sequence")
  if type(sequence) is not int or sequence != previous_sequence + 1:
    raise ValueError("out-of-order ADB packet")
  messages = packet.get("messages", {})
  ages = packet.get("ages_ms", {})
  if not isinstance(messages, dict) or not messages.keys() <= allowed or not isinstance(ages, dict) or not ages.keys() <= messages.keys():
    raise ValueError("disallowed simulator service")
  if any(not isinstance(data, str) for data in messages.values()):
    raise ValueError("Cereal payload must be base64")
  if any(type(age) not in (int, float) or not 0 <= age <= 10000 for age in ages.values()):
    raise ValueError("invalid simulated input age")
  if packet.get("fault") not in (None, "cancel", "stale", "disconnect"):
    raise ValueError("unsupported simulator fault")
  return sequence


def validate_replay(packet, full_chain):
  if (not full_chain or packet.get("op") != "replay" or packet.get("messages") or packet.get("fault") is not None or
      packet.get("ages_ms") or type(packet.get("seconds")) is not int or not 10 <= packet["seconds"] <= 120 or
      type(packet.get("steady_inputs")) is not bool):
    raise ValueError("Invalid complete-chain debug replay")
  return packet["seconds"], packet["steady_inputs"]
