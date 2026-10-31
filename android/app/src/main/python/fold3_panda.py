"""Panda receive-only diagnostics. There is intentionally no USB write binding."""
import json
from pathlib import Path
import struct
import time

from fold3_panda_protocol import PACKET_VERSIONS, Panda, unpack_can_buffer


def require_silent(health):
  if health["safety_mode"] != 0 or any(health[key] for key in
      ("controls_allowed", "controls_allowed_lateral", "controls_allowed_longitudinal", "faults", "fault_status")):
    raise PermissionError("Panda must already be in silent safety mode with controls disallowed and no faults; no mode changes are sent")


class Decoder(Panda):
  def __init__(self, reader):
    self._handle = self
    self.reader = reader

  def controlRead(self, request_type, request, value, index, length):
    if request_type != self.REQUEST_IN or index != 0:
      raise PermissionError("IN-only Panda decoder")
    return self.reader(request, value, length)


def receive(directory, service, descriptor):
  from java import jclass
  from fold3_runtime import configure
  configure(directory)  # Always the physical namespace; never the simulator socket.
  import openpilot.cereal.messaging as messaging
  native = jclass("ai.sunnypilot.fold3.NativeBridge")
  path = Path(directory) / "panda-rx.json"
  report = {"status": "starting", "vehicle_tx_enabled": False, "firmware_qualified": False,
            "vehicle_integration_verified": False, "frames": 0, "bus_counts": {}, "fingerprint": {},
            "timestamp_source": "host USB arrival; Panda CAN packets have no sensor timestamp"}
  handle = 0

  def save():
    report["updated_monotonic_ns"] = time.monotonic_ns()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)

  try:
    save()
    handle = int(native.openPandaReceiver(descriptor))

    def read(request, value, length):
      data = bytes(native.readPandaControl(handle, request, value, length))
      if len(data) != length:
        raise ValueError(f"Short Panda reply for {request:#x}: {len(data)}/{length}")
      return data

    if read(0xc1, 0, 1) != b"\x07":
      raise ValueError("Expected red Panda hardware type 7")
    versions = struct.unpack("<II", read(0xdd, 0, 8))
    report["packet_versions"] = versions
    if versions != PACKET_VERSIONS:
      raise ValueError(f"Incompatible pinned CAN/health protocols: {versions} != {PACKET_VERSIONS}")
    report["protocol_compatible"] = True
    report["firmware_signature_hex"] = (read(0xd3, 0, 64) + read(0xd4, 0, 64)).hex()
    report["firmware_qualification_blocker"] = "No approved signed firmware reference; signature is evidence only"
    decoder = Decoder(read)
    pm = messaging.PubMaster(["can"])
    pending, last_health, last_frame = b"", 0., time.monotonic()
    while not service.isStopping():
      now = time.monotonic()
      if now - last_health >= .25:
        report["health"] = decoder.health()
        require_silent(report["health"])
        report["status"] = "receiving_unqualified"
        report["can_stale"] = now - last_frame > 1.
        save()
        last_health = time.monotonic()
      # No CAN read occurs until a compatible, silent health packet was checked.
      pending += bytes(native.readPandaCan(handle))
      frames, pending = unpack_can_buffer(pending)
      if len(pending) > 69:
        raise ValueError("Oversized incomplete CAN packet")
      if frames:
        message = messaging.new_message("can", len(frames), valid=False)
        for target, (address, data, bus) in zip(message.can, frames, strict=True):
          if bus not in (0, 1, 2) or address > 0x1fffffff:
            raise ValueError("Unexpected returned/rejected frame or invalid bus/address in silent mode")
          target.address, target.dat, target.src = address, data, bus
          key = str(bus)
          report["bus_counts"][key] = report["bus_counts"].get(key, 0) + 1
          fingerprint = report["fingerprint"].setdefault(key, {})
          if len(fingerprint) >= 2048 and str(address) not in fingerprint:
            raise ValueError("Fingerprint table limit exceeded")
          fingerprint[str(address)] = len(data)
        pm.send("can", message)
        report["frames"] += len(frames)
        last_frame = time.monotonic()
    report["status"] = "stopped"
  except Exception as exc:
    report.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
  finally:
    if handle:
      native.closePandaReceiver(handle)
    save()
