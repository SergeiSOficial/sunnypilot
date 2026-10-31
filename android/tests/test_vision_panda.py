"""Host checks of Android input gates and the pinned Panda decoder."""
import importlib.util
import json
import struct
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from fold3_vision import DiagnosticWarp, FrameGate, RoadWarp, ThermalGate, camera_available, carry_model_state, inference_delay_ns, load_road_calibration

spec = importlib.util.spec_from_file_location("stage_panda", ROOT / "android/scripts/stage_panda.py")
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)
protocol = types.ModuleType("fold3_panda_protocol")
exec(compile(stage.generate(), "pinned-panda-decoder", "exec"), protocol.__dict__)
sys.modules["fold3_panda_protocol"] = protocol
from fold3_panda import Decoder, receive, require_silent


class VisionTest(unittest.TestCase):
  def test_measured_rear_warps_match_upstream_geometry_and_chroma_scaling(self):
    from openpilot.common.transformations.model import get_warp_matrix, medmodel_intrinsics, sbigmodel_intrinsics
    data = np.arange(512 * 256 * 3 // 2, dtype=np.uint8)
    for wide, intrinsic in ((False, medmodel_intrinsics), (True, sbigmodel_intrinsics)):
      calibration = {"image_size": [512, 256], "intrinsics": intrinsic.tolist(),
                     "device_from_calib_euler": [0, 0, 0], "opencv_distortion": [0]*5}
      warp = RoadWarp(512, 256, 512, 512*256, calibration, wide)
      np.testing.assert_array_equal(warp(data), DiagnosticWarp(512, 256, 512, 512*256, 512, 256)(data))
      calibration["device_from_calib_euler"] = [.012, -.03, .045]
      warp = RoadWarp(512, 256, 512, 512*256, calibration, wide)
      matrix = get_warp_matrix(np.array(calibration["device_from_calib_euler"]), intrinsic, bigmodel_frame=wide)
      for x, y in ((30, 20), (100, 100), (200, 110)):
        xy = matrix @ [2*x, 2*y, 1]
        sx, sy = np.rint(xy[:2] / xy[2] / 2).astype(int)
        expected = 512*256 + np.clip(sy, 0, 127)*512 + np.clip(sx, 0, 255)*2
        self.assertEqual(warp.uv_indices[y*256+x], expected)
      calibration["opencv_distortion"][0] = .01
      distorted = RoadWarp(512, 256, 512, 512*256, calibration, wide)
      self.assertFalse(np.array_equal(distorted.y_indices, warp.y_indices))
      calibration["intrinsics"][0][0] = float("nan")
      with self.assertRaises(ValueError):
        RoadWarp(512, 256, 512, 512*256, calibration, wide)

  def test_calibration_rejects_unmounted_changed_and_duplicate_sensor_streams(self):
    origins = {"0": "test:wide", "2": "test:ultrawide"}
    data = {"schema_version": 1, "mount_status": "measured", "cameras": {
      sensor: {"physical_sensor_id": sensor, "stream_origin": source, "measurement_evidence": "test-only"}
      for sensor, source in origins.items()}}
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "calibration.json"
      path.write_text(json.dumps(data))
      self.assertEqual(set(load_road_calibration(path, origins)), {"0", "2"})
      for change in ("mount", "source", "sensor"):
        broken = json.loads(json.dumps(data))
        if change == "mount":
          broken["mount_status"] = "bench"
        else:
          broken["cameras"]["2"]["stream_origin" if change == "source" else "physical_sensor_id"] = "changed" if change == "source" else "0"
        path.write_text(json.dumps(broken))
        with self.assertRaises(ValueError):
          load_road_calibration(path, origins)

  def test_thermal_pause_requires_continuous_cooldown_before_resuming(self):
    gate = ThermalGate()
    self.assertFalse(gate.update(0, 0))
    self.assertTrue(gate.update(2, 1))
    self.assertTrue(gate.update(1, 2))
    self.assertTrue(gate.update(0, 9_000_000_002))
    self.assertTrue(gate.update(3, 9_000_000_003))
    self.assertTrue(gate.update(0, 10_000_000_003))
    self.assertTrue(gate.update(1, 19_000_000_003))
    self.assertFalse(gate.update(0, 20_000_000_003))

  def test_model_cadence_is_20hz_without_catchup_after_stalls(self):
    self.assertEqual(inference_delay_ns(100_000_000, 125_000_000), 25_000_000)
    self.assertEqual(inference_delay_ns(100_000_000, 150_000_000), 0)
    self.assertEqual(inference_delay_ns(100_000_000, 500_000_000), 0)
    self.assertEqual(inference_delay_ns(500_000_000, 500_000_001), 49_999_999)

  def test_nonfinite_prediction_resets_recurrence_and_is_rejected(self):
    inputs = {"state_feat": np.ones((1, 2)), "traffic_convention": np.array([[0., 1.]])}
    for bad in (np.nan, np.inf, -np.inf):
      self.assertFalse(carry_model_state(inputs, {"outputs": np.array([bad]), "next_state_feat": np.ones((1, 2))}))
      np.testing.assert_array_equal(inputs["state_feat"], [[0., 0.]])
      np.testing.assert_array_equal(inputs["traffic_convention"], [[0., 1.]])
    self.assertTrue(carry_model_state(inputs, {"outputs": np.array([1.]), "next_state_feat": np.array([[2., 3.]])}))
    np.testing.assert_array_equal(inputs["state_feat"], [[2., 3.]])

  def test_empty_vipc_handshake_does_not_satisfy_required_camera(self):
    for connected, count, expected in ((True, 0, False), (False, 4, False), (True, 4, True)):
      self.assertEqual(camera_available(types.SimpleNamespace(is_connected=lambda connected=connected: connected, num_buffers=count)), expected)

  def test_distinct_sensor_requirement(self):
    with self.assertRaises(ValueError):
      FrameGate(("0", "0"))

  def test_one_to_one_pairing_rejects_stale_duplicate_future_and_skew(self):
    gate = FrameGate(("0", "2"), max_age_ns=100, max_skew_ns=10)
    self.assertTrue(gate.add("0", 1000, "road", 1000))
    self.assertFalse(gate.add("0", 1000, "duplicate", 1001))
    self.assertFalse(gate.add("2", 1005, "future", 1001))
    self.assertTrue(gate.add("2", 1015, "skew", 1015))
    self.assertIsNone(gate.pair(1015))
    gate.add("0", 1016, "fresh", 1016)
    self.assertEqual(gate.pair(1016), {"0": (1016, "fresh"), "2": (1015, "skew")})
    self.assertIsNone(gate.pair(1016))
    gate.add("0", 1020, "stale", 1020)
    self.assertIsNone(gate.pair(1200))
    self.assertGreaterEqual(gate.rejected, 4)

  def test_queues_are_bounded_and_wait_for_required_wide(self):
    gate = FrameGate(("0", "2"))
    for stamp in range(1, 100):
      gate.add("0", stamp, b"frame", stamp)
    self.assertEqual(len(gate.queues["0"]), 2)
    self.assertIsNone(gate.pair(100))
    self.assertEqual(gate.rejected, 97)

  def test_nv12_packing_matches_upstream_channel_order_with_stride_padding(self):
    # 4x2 luma, 6-byte stride, UV starts after a padded luma row.
    data = bytes([1,2,3,4,99,99,5,6,7,8,99,99,9,10,11,12,99,99])
    warp = DiagnosticWarp(4, 2, 6, 12, 4, 2)
    expected = np.array([[[1,3]], [[5,7]], [[2,4]], [[6,8]], [[9,11]], [[10,12]]], dtype=np.uint8)
    np.testing.assert_array_equal(warp(data), expected)
    identity = id(warp(data))
    self.assertEqual(identity, id(warp(bytes(reversed(data)))))
    with self.assertRaises(ValueError):
      warp(data[:10])

  def test_luma_does_not_mix_uv_and_reuses_output(self):
    warp = DiagnosticWarp(4, 2, 4, 8, 4, 2, luma=True)
    np.testing.assert_array_equal(warp(bytes(range(12))), np.arange(8, dtype=np.uint8).reshape(1, 8))
    for args in ((3, 2, 4, 8), (4, 2, 3, 8), (4, 2, 4, 7)):
      with self.assertRaises(ValueError):
        DiagnosticWarp(*args, 4, 2)

  def test_fixture_model_outputs_use_real_upstream_cereal_parsers(self):
    from fold3_checks import output_messages
    result = json.loads(output_messages(ROOT / "android/app/build/model-reference"))
    self.assertEqual(result["invalid_messages"], ["driverStateV2", "modelV2", "cameraOdometry"])
    self.assertFalse(result["vehicle_tx_enabled"])


class PandaTest(unittest.TestCase):
  def run_receiver(self, *, incompatible=False, unsafe=False, disconnect=False, corrupt=False):
    native = Mock()
    native.openPandaReceiver.return_value = 17
    native.readPandaControl.side_effect = lambda handle, request, value, length: (
      b"\x07" if request == 0xc1 else struct.pack("<II", *( (0, 0) if incompatible else protocol.PACKET_VERSIONS ))
      if request == 0xdd else bytes(length))
    packet = bytes.fromhex("201809000020aabb")
    if corrupt:
      packet = packet[:-1] + b"\x00"
    native.readPandaCan.side_effect = OSError("Panda disconnected") if disconnect else [packet[:5], packet[5:]]
    service = types.SimpleNamespace(isStopping=lambda: native.readPandaCan.call_count >= 2)
    publisher = Mock()
    with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"java": types.SimpleNamespace(jclass=lambda _: native)}), \
        patch("fold3_runtime.configure") as configure, patch("openpilot.cereal.messaging.PubMaster", return_value=publisher), \
        patch.object(Decoder, "health", return_value={"safety_mode": 1 if unsafe else 0, "controls_allowed": 0,
          "controls_allowed_lateral": 0, "controls_allowed_longitudinal": 0, "faults": 0, "fault_status": 0}):
      receive(directory, service, 123)
      configure.assert_called_once_with(directory)
      report = json.loads((Path(directory) / "panda-rx.json").read_text())
    native.closePandaReceiver.assert_called_once_with(17)
    self.assertFalse(report["vehicle_tx_enabled"])
    self.assertFalse(report["firmware_qualified"])
    for args in native.readPandaControl.call_args_list:
      self.assertIn(args.args[1], (0xc1, 0xdd, 0xd3, 0xd4))
    return report, native, publisher

  def test_receive_only_integration_reassembles_and_publishes_invalid_can(self):
    report, _, publisher = self.run_receiver()
    self.assertEqual(report["status"], "stopped")
    self.assertEqual(report["frames"], 1)
    self.assertEqual(report["fingerprint"], {"0": {"291": 2}})
    message = publisher.send.call_args.args[1]
    self.assertFalse(message.valid)
    self.assertEqual(message.can[0].dat, b"\xaa\xbb")

  def test_receive_only_incompatible_and_unsafe_boards_never_read_can(self):
    for kwargs in ({"incompatible": True}, {"unsafe": True}):
      with self.subTest(kwargs=kwargs):
        report, native, publisher = self.run_receiver(**kwargs)
        self.assertEqual(report["status"], "blocked")
        native.readPandaCan.assert_not_called()
        publisher.send.assert_not_called()

  def test_disconnect_and_corrupt_packets_block_and_close_receiver(self):
    for kwargs in ({"disconnect": True}, {"corrupt": True}):
      with self.subTest(kwargs=kwargs):
        report, _, publisher = self.run_receiver(**kwargs)
        self.assertEqual(report["status"], "blocked")
        publisher.send.assert_not_called()

  def test_fragmented_usb_packets_match_pinned_decoder(self):
    data = bytes.fromhex("201809000020aabb")
    for split in range(len(data)):
      frames, pending = protocol.unpack_can_buffer(data[:split])
      self.assertEqual(frames, [])
      frames, pending = protocol.unpack_can_buffer(pending + data[split:])
      self.assertEqual(frames, [(0x123, b"\xaa\xbb", 0)])
      self.assertEqual(pending, b"")
    corrupted = bytearray(data)
    corrupted[-1] ^= 1
    with self.assertRaises(AssertionError):
      protocol.unpack_can_buffer(corrupted)
    optimized = {}
    exec(compile(stage.generate(), "optimized-pinned-panda-decoder", "exec", optimize=2), optimized)
    with self.assertRaises(AssertionError):
      optimized["unpack_can_buffer"](corrupted)

  def test_pinned_health_and_all_engagement_flags_are_checked(self):
    calls = []
    def reader(request, value, length):
      calls.append((request, value, length))
      return bytes(protocol.Panda.HEALTH_STRUCT.size)
    health = Decoder(reader).health()
    self.assertEqual(calls, [(0xd2, 0, protocol.Panda.HEALTH_STRUCT.size)])
    require_silent(health)
    for key in ("safety_mode", "controls_allowed", "controls_allowed_lateral", "controls_allowed_longitudinal", "faults", "fault_status"):
      with self.subTest(key=key), self.assertRaises(PermissionError):
        require_silent({**health, key: 1})
    with self.assertRaises(struct.error):
      Decoder(lambda *args: b"\x00").health()

  def test_packet_versions_come_from_exact_pinned_sources(self):
    import hashlib
    files = (ROOT / "opendbc_repo/opendbc/safety/can.h", ROOT / "panda/board/health.h")
    expected = tuple(int.from_bytes(hashlib.sha256(p.read_bytes().replace(b"\r", b"")).digest()[:4], "little") for p in files)
    self.assertEqual(protocol.PACKET_VERSIONS, expected)
    self.assertFalse(any("write" in name.lower() or "send" in name.lower() for name in vars(protocol.Panda)))


if __name__ == "__main__":
  unittest.main()
