import json
import base64
import math
import os
import socket
import struct
import unittest
from unittest import mock

from openpilot.tools.sim.lib.android_protocol import CONTROL_INPUTS, MAX_PACKET, receive, send, stale_control_inputs, validate_exchange


class TestAndroidProtocol(unittest.TestCase):
  def test_relay_requires_real_msgq_namespace_variable(self):
    from openpilot.tools.sim.android_bridge import relay
    with mock.patch("openpilot.tools.sim.android_bridge.subprocess.run") as transport:
      for environment in ({}, {"OPENPILOTPREFIX": "fold3_sim_wrong_variable"}):
        with self.subTest(environment=environment), mock.patch.dict(os.environ, environment, clear=True):
          with self.assertRaisesRegex(ValueError, "OPENPILOT_PREFIX"):
            relay("unused", 0, 1, "unused")
      transport.assert_not_called()
      with mock.patch.dict(os.environ, {"OPENPILOT_PREFIX": "fold3_sim_test"}, clear=True):
        transport.side_effect = RuntimeError("transport reached")
        with self.assertRaisesRegex(RuntimeError, "transport reached"):
          relay("unused", 0, 1, "unused")

  def test_round_trip(self):
    left, right = socket.socketpair()
    with left, right:
      send(left, {"sequence": 0, "messages": {"carState": "AA=="}})
      packet = receive(right)
      self.assertEqual(validate_exchange(packet, -1), 0)
      self.assertEqual(packet["messages"], {"carState": "AA=="})

  def test_rejects_can_and_output_injection(self):
    for service in ("can", "sendcan", "pandaStates", "carControl", "driverMonitoringState"):
      with self.subTest(service=service), self.assertRaises(ValueError):
        validate_exchange({"sequence": 0, "messages": {service: "AA=="}}, -1)

  def test_sequence_and_age_bounds(self):
    for sequence in (False, -1, 1, 0.0, None):
      with self.subTest(sequence=sequence), self.assertRaises(ValueError):
        validate_exchange({"sequence": sequence}, -1)
    for age in (-1, 10001, math.inf, math.nan, True, "0"):
      with self.subTest(age=age), self.assertRaises(ValueError):
        validate_exchange({"sequence": 0, "messages": {"carState": "AA=="}, "ages_ms": {"carState": age}}, -1)
    for fault in ("cancel", "stale", "disconnect"):
      self.assertEqual(validate_exchange({"sequence": 0, "fault": fault}, -1), 0)

  def test_invalid_framing_and_versions(self):
    for size in (0, MAX_PACKET + 1):
      left, right = socket.socketpair()
      with left, right:
        left.sendall(struct.pack("!I", size))
        with self.assertRaises(ValueError):
          receive(right)
    for value in ({"version": 99}, []):
      left, right = socket.socketpair()
      with left, right:
        data = json.dumps(value).encode()
        left.sendall(struct.pack("!I", len(data)) + data)
        with self.assertRaises(ValueError):
          receive(right)

  def test_truncated_packet(self):
    left, right = socket.socketpair()
    with left, right:
      left.sendall(struct.pack("!I", 10) + b"{")
      left.shutdown(socket.SHUT_WR)
      with self.assertRaises(EOFError):
        receive(right)

  def test_missing_stale_and_future_control_inputs(self):
    now = 1_000_000_000
    fresh = dict.fromkeys(CONTROL_INPUTS, now - 100_000_000)
    self.assertEqual(stale_control_inputs(fresh, now), [])
    self.assertEqual(stale_control_inputs({}, now), sorted(CONTROL_INPUTS))
    for timestamp in (now - 100_000_001, now + 1):
      with self.subTest(timestamp=timestamp):
        self.assertEqual(stale_control_inputs({**fresh, "modelV2": timestamp}, now), ["modelV2"])

  def test_fixture_preserves_steering_minimum(self):
    from openpilot.cereal import messaging
    from openpilot.tools.sim.android_validation import inputs
    minimum = 51.5 / 3.6
    for speed, active in ((0., False), (minimum - .01, False), (minimum, True), (20., True)):
      with self.subTest(speed=speed):
        packet = inputs(0., speed, 0., minimum)
        state = messaging.log_from_bytes(base64.b64decode(packet["selfdriveState"]))
        self.assertEqual(state.selfdriveState.active, active)
        self.assertNotIn("sendcan", packet)
        self.assertNotIn("carControl", packet)


if __name__ == "__main__":
  unittest.main()
