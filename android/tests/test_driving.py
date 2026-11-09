"""Receive-only entry points reject actuation even if directly invoked."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app/src/main/python"))
from fold3_model_trace import compare
from fold3_daemons import Report
from openpilot.selfdrive.car.card import Car, can_comm_callbacks


class DrivingTest(unittest.TestCase):
  def test_idle_relay_withdraws_inputs_before_accepting_another_packet(self):
    from openpilot.tools.sim.lib import android_protocol
    with patch.dict(sys.modules, {"android_protocol": android_protocol}):
      import fold3_sim
    connection, publisher = Mock(), Mock()
    for full_chain, services in ((False, {"selfdriveState", "selfdriveStateSP"}),
                                 (True, {"can", "modelV2", "driverStateV2", "cameraOdometry"})):
      publisher.reset_mock()
      timestamps = {"can": 1}
      with patch.object(fold3_sim.select, "select", side_effect=[([], [], []), ([connection], [], [])]), \
           patch.object(fold3_sim, "receive", return_value={"sequence": 3}) as receive:
        self.assertEqual(fold3_sim.receive_input(connection, publisher, timestamps, full_chain), {"sequence": 3})
      self.assertFalse(timestamps)
      self.assertEqual({call.args[0] for call in publisher.send.call_args_list}, services)
      self.assertTrue(all(not call.args[1].valid for call in publisher.send.call_args_list))
      receive.assert_called_once_with(connection)

  def test_android_memory_params_remap_only_shared_memory_path(self):
    from openpilot.common.params import Params
    with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"ANDROID_PARAMS_MEMORY_ROOT": directory + "/memory"}):
      memory, normal = Params("/dev/shm/params"), Params(directory + "/normal")
      memory.put_bool("IsMetric", True, block=True)
      self.assertEqual(memory.d, directory + "/memory")
      self.assertTrue(Params("/dev/shm/params").get_bool("IsMetric"))
      self.assertFalse(normal.get_bool("IsMetric"))
      self.assertEqual(normal.d, directory + "/normal")

  def test_packaged_metadata_preserves_dirty_source_identity(self):
    from dataclasses import asdict
    from openpilot.common.version import build_metadata_from_dict
    data = {"channel": "codex/fold3-android-port", "openpilot": {"git_commit": "source", "is_dirty": True,
             "build_style": "android-debug-rx-only"}}
    metadata = build_metadata_from_dict(data)
    self.assertTrue(metadata.openpilot.is_dirty)
    self.assertEqual(asdict(metadata)["openpilot"]["git_commit"], "source")
    self.assertEqual(build_metadata_from_dict(asdict(metadata)), metadata)

  def test_android_metadata_uses_extracted_package_data(self):
    import json
    from openpilot.common import version
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / "openpilot/common").mkdir(parents=True)
      (root / "openpilot/build.json").write_text(json.dumps({"openpilot": {"git_commit": "packaged", "is_dirty": True}}))
      with patch.dict("os.environ", {"ANDROID_DATA_ROOT": directory}), patch.object(version, "BASEDIR", directory), \
           patch.object(version, "__file__", str(root / "openpilot/common/version.py")):
        metadata = version.get_build_metadata(directory)
      self.assertEqual(metadata.openpilot.git_commit, "packaged")
      self.assertTrue(metadata.openpilot.is_dirty)

  def test_calibration_fault_withdraws_validity_without_consuming_empty_odometry(self):
    from opendbc.car.structs import car
    from openpilot.cereal import messaging
    from openpilot.selfdrive.locationd import calibrationd
    class Fixture(dict):
      frame = -5
      updated = {"cameraOdometry": True}
      valid = {"cameraOdometry": False}
      def update(self, timeout):
        self.frame += 5
        if self.frame > 9:
          raise StopIteration
        self.valid["cameraOdometry"] = self.frame == 5
      def all_checks(self):
        return self.valid["cameraOdometry"]
    fixture = Fixture(cameraOdometry=messaging.new_message("cameraOdometry").cameraOdometry,
                      carState=car.CarState.new_message())
    with patch.object(calibrationd, "config_realtime_process"), patch.object(calibrationd.messaging, "PubMaster"), \
         patch.object(calibrationd.messaging, "SubMaster", return_value=fixture), \
         patch.object(calibrationd, "Params") as params, patch.object(calibrationd, "Calibrator") as calibrator:
      params.return_value.get.return_value = car.CarParams.new_message().to_bytes()
      calibrator.return_value.handle_cam_odom.return_value = None
      with self.assertRaises(StopIteration):
        calibrationd.main()
      calibrator.return_value.handle_cam_odom.assert_called_once()
      # Faults never update geometry, and publication still marks their inputs invalid.
      self.assertEqual([call.args[1] for call in calibrator.return_value.send_data.call_args_list], [False, True])

  def test_complete_chain_serializes_struct_and_list_outputs_as_cereal_events(self):
    import base64
    from openpilot.cereal import messaging
    from openpilot.tools.sim.lib.android_protocol import CHAIN_OUTBOUND
    with patch.dict(sys.modules, {"android_protocol": sys.modules["openpilot.tools.sim.lib.android_protocol"]}):
      from fold3_sim import encode_outputs
    messages = {name: messaging.new_message(name, 1, valid=True) if name == "onroadEvents"
                else messaging.new_message(name, valid=True) for name in CHAIN_OUTBOUND}
    class Fixture(dict):
      updated = valid = dict.fromkeys(messages, True)
    fixture = Fixture({name: getattr(message.as_reader(), name) for name, message in messages.items()})
    encoded = encode_outputs(fixture, CHAIN_OUTBOUND)
    self.assertEqual(set(encoded), set(CHAIN_OUTBOUND))
    for name, data in encoded.items():
      decoded = messaging.log_from_bytes(base64.b64decode(data))
      self.assertEqual(decoded.which(), name)
      self.assertTrue(decoded.valid)
      actual, expected = getattr(decoded, name), getattr(messages[name], name)
      self.assertEqual([item.to_dict() for item in actual] if name == "onroadEvents" else actual.to_dict(),
                       [item.to_dict() for item in expected] if name == "onroadEvents" else expected.to_dict())

  def test_complete_chain_allows_sources_and_rejects_derived_inputs(self):
    from openpilot.tools.sim.lib.android_protocol import CHAIN_INBOUND, validate_exchange
    for service in ("carState", "selfdriveState", "extrinsicsCalibration", "longitudinalPlan"):
      with self.assertRaises(ValueError):
        validate_exchange({"sequence": 0, "messages": {service: "fixture"}}, -1, allowed=CHAIN_INBOUND)
    self.assertEqual(validate_exchange({"sequence": 0, "messages": {"can": "fixture"}}, -1, allowed=CHAIN_INBOUND), 0)

  def test_device_replay_requires_bounded_debug_source_only_request(self):
    from openpilot.tools.sim.lib.android_protocol import validate_replay
    packet = {"op": "replay", "seconds": 10, "steady_inputs": False}
    self.assertEqual(validate_replay(packet, True), (10, False))
    self.assertEqual(validate_replay({**packet, "seconds": 120, "steady_inputs": True}, True), (120, True))
    with self.assertRaises(ValueError):
      validate_replay(packet, False)
    for key, value in (("seconds", 9), ("seconds", 121), ("seconds", True), ("seconds", 10.),
                       ("steady_inputs", 1), ("fault", "stale"), ("messages", {"carState": "x"}),
                       ("ages_ms", {"can": 1}), ("op", "exchange")):
      with self.subTest(key=key, value=value), self.assertRaises(ValueError):
        validate_replay({**packet, key: value}, True)

  def test_device_replay_disconnect_stops_sources(self):
    from openpilot.tools.sim.lib import android_protocol, android_sources
    with patch.dict(sys.modules, {"android_protocol": android_protocol, "android_sources": android_sources}):
      import fold3_sim
      publisher, connection = Mock(), Mock()
      with patch.object(fold3_sim.select, "select", return_value=([connection], [], [])), patch.object(fold3_sim, "send"):
        with self.assertRaises(EOFError):
          fold3_sim.replay(connection, publisher, Mock(), Mock(), 10, True)
      publisher.send.assert_not_called()

  def test_passive_hyundai_fixture_preserves_acc_cancellation_and_aeb(self):
    from openpilot.tools.sim.android_chain_validation import Sources
    from openpilot.cereal import messaging
    sources = Sources()
    for frame in range(100):
      state, _ = sources.ci.update([(frame * 10_000_000, sources.can())])
    self.assertTrue(state.canValid)
    self.assertAlmostEqual(state.vEgo, 20.)
    self.assertTrue(state.cruiseState.enabled)
    state, _ = sources.ci.update([(1_000_000_000, sources.can(cruise=False))])
    self.assertFalse(state.cruiseState.enabled)
    state, _ = sources.ci.update([(1_010_000_000, sources.can(aeb=True))])
    self.assertTrue(state.stockAeb)
    self.assertFalse(state.stockFcw)
    self.assertFalse(sources.ci.CP.openpilotLongitudinalControl)
    self.assertTrue(sources.ci.CP.pcmCruise)
    events = sources.events(0)
    self.assertEqual(messaging.log_from_bytes(events["can"].to_bytes()).which(), "can")
    self.assertEqual(len(events["modelV2"].modelV2.meta.disengagePredictions.brakeDisengageProbs), 5)
    from openpilot.selfdrive.monitoring.policy import DriverMonitoring
    monitoring = DriverMonitoring(rhd_saved=True)
    for side in ("leftDriverData", "rightDriverData"):
      driver = getattr(events["driverStateV2"].driverStateV2, side)
      self.assertEqual(len(driver.facePosition), 2)
      self.assertEqual(len(driver.faceOrientationStd), 3)
    monitoring._update_states(events["driverStateV2"].driverStateV2, [0.] * 3, 20., False, False)
    self.assertTrue(monitoring.face_detected)

  def test_legacy_sim_fixture_runs_upstream_driver_monitoring(self):
    import base64
    from openpilot.cereal import messaging
    from openpilot.tools.sim.android_validation import inputs
    from openpilot.selfdrive.monitoring.policy import DriverMonitoring
    encoded = inputs(0., 20., 0., 14.3)
    messages = {name: messaging.log_from_bytes(base64.b64decode(data)) for name, data in encoded.items()}
    fixture = {name: getattr(message, name) for name, message in messages.items()}
    fixture["carControl"] = messaging.new_message("carControl").carControl
    with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"PARAMS_ROOT": directory}):
      monitoring = DriverMonitoring(rhd_saved=True)
      monitoring.run_step(fixture)
      self.assertTrue(monitoring.face_detected)
      self.assertTrue(monitoring.get_state_packet(valid=True).valid)

  def test_accumulation_diagnostic_keeps_bias_before_output_rounding(self):
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.onnx import OnnxRunner, onnx_ops
    from fold3_precision import accumulation_diagnostic
    runner = OnnxRunner.__new__(OnnxRunner)
    runner.onnx_ops = onnx_ops
    accumulation_diagnostic(runner)
    for kind, x, w, bias in (
        ("Conv", [[[[1.]], [[1.]]]], [[[[4096.]], [[.5]]]], [-4096.]),
        ("Gemm", [[1., 1.]], [[4096.], [.5]], [-4096.])):
      tensors = [Tensor(value, dtype=dtypes.float16, device="CPU") for value in (x, w, bias)]
      result = runner.onnx_ops[kind](*tensors)
      self.assertEqual(result.dtype, dtypes.float16)
      np.testing.assert_array_equal(result.numpy().reshape(-1), [.5])

  def test_receive_only_requires_identification_before_opening_ipc(self):
    with patch("openpilot.selfdrive.car.card.messaging.sub_sock") as socket:
      with self.assertRaises(ValueError):
        Car(receive_only=True)
      socket.assert_not_called()

  def test_receive_only_callback_and_controls_reject_transmission(self):
    _, send = can_comm_callbacks(Mock(), None)
    with self.assertRaises(PermissionError):
      send([])
    process = Car.__new__(Car)
    process.receive_only = True
    process.CI = Mock()
    with self.assertRaises(PermissionError):
      process.controls_update(None, None, None)
    process.CI.init.assert_not_called()
    process.CI.apply.assert_not_called()

  def test_trace_comparison_cannot_broadcast_or_accept_nonfinite_values(self):
    self.assertFalse(compare(np.zeros((1, 2)), np.zeros(2))["passed"])
    self.assertFalse(compare(np.array([np.inf]), np.array([np.inf]))["passed"])
    self.assertTrue(compare(np.ones(3), np.ones(3))["passed"])

  def test_controls_reports_cycle_misses_including_input_handling(self):
    with tempfile.TemporaryDirectory() as directory:
      report = Report("controlsd", directory, Mock())
      report.controls_progress(True, 2_000_000, {"input_handling_ms": 13., "cycle_before_reporting_ms": 15.,
                                              "release_to_completion_before_reporting_ms": 15., "skipped_slots": 0})
      self.assertEqual(report.data["deadline_misses"], 0)
      self.assertEqual(report.data["cycle_deadline_misses"], 1)
      self.assertGreaterEqual(report.data["complete_cycle_ms"], 15.)
      report.controls_progress(False, 11_000_000, {"input_handling_ms": 0., "cycle_before_reporting_ms": 11.,
                                                "release_to_completion_before_reporting_ms": 11., "skipped_slots": 1})
      self.assertEqual(report.data["work_samples"], 2)
      self.assertEqual(report.data["updated_input_samples"], 1)
      self.assertEqual(report.data["deadline_misses"], 1)
      self.assertEqual(report.data["cycle_deadline_misses"], 2)
      self.assertFalse(report.data["vehicle_tx_enabled"])
      report.close()

  def test_controls_counts_reporting_overhead_in_deadlines(self):
    with tempfile.TemporaryDirectory() as directory:
      report = Report("controlsd", directory, Mock())
      with patch.object(report, "update"), patch("fold3_daemons.time.monotonic_ns", side_effect=[0, 3_000_000]):
        report.controls_progress(True, 8_000_000, {"cycle_before_reporting_ms": 8.,
                                                "release_to_completion_before_reporting_ms": 8., "skipped_slots": 0})
      self.assertEqual(report.data["cycle_deadline_misses"], 1)
      self.assertEqual(report.data["complete_cycle_ms"], 11.)
      self.assertEqual(report.control_buffer[1], 11.)
      self.assertEqual(report.data["scheduled_deadline_misses"], 1)
      report.close()

  def test_non_controls_heartbeat_is_bounded_and_phase_changes_are_immediate(self):
    with tempfile.TemporaryDirectory() as directory:
      worker = Mock()
      report = Report("plannerd", directory, worker)
      with patch("fold3_daemons.time.monotonic", return_value=10.):
        for frame in range(100):
          report.update("running", frame=frame)
      self.assertEqual(worker.progress.call_count, 1)
      self.assertEqual(report.data["frame"], 99)
      with patch("fold3_daemons.time.monotonic", return_value=10.25):
        report.update("running")
        report.update("waiting_for_selfdrive_state")
      self.assertEqual(worker.progress.call_count, 3)
      self.assertEqual(worker.progress.call_args.args, ("waiting_for_selfdrive_state",))
      report.close()

  def test_report_disk_io_cannot_block_controls_and_failure_is_flushed(self):
    import json
    import threading
    entered, release = threading.Event(), threading.Event()
    original = Path.write_text

    def slow_write(path, data, *args, **kwargs):
      entered.set()
      if not release.wait(3):
        raise TimeoutError("test writer was not released")
      return original(path, data, *args, **kwargs)

    with tempfile.TemporaryDirectory() as directory:
      report = Report("controlsd", directory, Mock())
      try:
        with patch("fold3_daemons.Path.write_text", slow_write):
          report.update("running")
          self.assertTrue(entered.wait(2))
          report.controls_progress(True, 2_000_000, {"cycle_before_reporting_ms": 2.,
                                                  "release_to_completion_before_reporting_ms": 2., "skipped_slots": 0})
          self.assertEqual(report.data["work_samples"], 1)
          self.assertFalse(release.is_set())
          release.set()
          report.update("failed", error="injected")
        batches = [json.loads(path.read_text()) for path in sorted((Path(directory) / "controlsd-batches").glob("*.json"))]
        saved = batches[-1]["data"]
        self.assertEqual(saved["work_samples"], 1)
        self.assertEqual(saved["error"], "injected")
        import base64
        import struct
        self.assertEqual(struct.unpack_from("<d", base64.b64decode(batches[-1]["rows_base64"]))[0], 2_000_000)
        self.assertNotIn("p99_work_ms", saved)
      finally:
        release.set()
        report.close()
