"""Experiment isolation must hold even when a workload fails."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("experiment", Path(__file__).resolve().parents[1] / "scripts/experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class ExperimentTest(unittest.TestCase):
  def test_device_lease_excludes_other_packages_but_not_other_devices(self):
    with tempfile.TemporaryDirectory() as directory, patch.object(experiment.tempfile, "gettempdir", return_value=directory):
      with experiment.device_lease("phone"):
        with self.assertRaisesRegex(RuntimeError, "Another experiment owns device"):
          experiment.device_lease("phone").__enter__()
        with experiment.device_lease("emulator"):
          pass
      # Exceptions and normal completion release ownership.
      with self.assertRaises(ValueError):
        with experiment.device_lease("phone"):
          raise ValueError("failed workload")
      with experiment.device_lease("phone"):
        pass

  def test_busy_device_does_not_stop_an_existing_experiment(self):
    with tempfile.TemporaryDirectory() as directory, patch.object(experiment.tempfile, "gettempdir", return_value=directory):
      device = Mock(serial="phone")
      with experiment.device_lease("phone"):
        with self.assertRaisesRegex(RuntimeError, "Another experiment owns device"):
          experiment.run(device, ["test"], Path(directory) / "result.json")
      self.assertEqual(device.mock_calls, [])

  def test_cold_emulator_ui_budget_does_not_change_phone_deadline_or_tx_guard(self):
    for serial, unsafe in (("phone", False), ("emulator-5560", False), ("emulator-5560", True)):
      with self.subTest(serial=serial, unsafe=unsafe):
        clock = [0.]
        def command(*args, clock=clock, unsafe=unsafe):
          if args[-1] == "/proc/uptime":
            return "100 0"
          if args[-1] == "files/manager.json":
            return json.dumps({"pid": 2, "updated_elapsed_ms": 100001,
              "vehicle_tx_enabled": unsafe, "engagement_allowed": False,
              "simulation": True, "full_chain": True})
          if args[-1] == "files/ui-report.json":
            return json.dumps({"pid": 1 if clock[0] == 0 else 3,
              "status": "running" if clock[0] >= 70 else "starting",
              "full_upstream_ui": True, "frames": 30})
          return ""
        device = experiment.Device(serial)
        device.command = command
        def sleep(seconds, clock=clock):
          clock[0] += seconds
        with patch.object(experiment.time, "monotonic", side_effect=lambda clock=clock: clock[0]), \
             patch.object(experiment.time, "sleep", side_effect=sleep):
          if unsafe:
            with self.assertRaisesRegex(RuntimeError, "transmission/engagement"):
              device.start_runtime(True, full_chain=True)
          elif serial == "phone":
            with self.assertRaises(TimeoutError):
              device.start_runtime(True, full_chain=True)
            self.assertEqual(clock[0], 60)
          else:
            receipt = device.start_runtime(True, full_chain=True)
            self.assertEqual(receipt["startup_elapsed_s"], 70)
            self.assertEqual(receipt["startup_budget_s"], 180)

  def test_model_validator_must_target_the_hashed_package(self):
    command = ["python", "android/scripts/validate_models.py"]
    experiment.check_model_target(command, experiment.PACKAGE)
    with self.assertRaisesRegex(ValueError, "pass matching"):
      experiment.check_model_target(command, experiment.CAMERA_COMPAT_PACKAGE)
    experiment.check_model_target(command + ["--package", experiment.CAMERA_COMPAT_PACKAGE], experiment.CAMERA_COMPAT_PACKAGE)
    experiment.check_model_target(command + ["--package=" + experiment.CAMERA_COMPAT_PACKAGE], experiment.CAMERA_COMPAT_PACKAGE)
    with self.assertRaisesRegex(ValueError, "requires a value"):
      experiment.check_model_target(command + ["--package"], experiment.PACKAGE)
    experiment.check_model_target(["unrelated-command"], experiment.CAMERA_COMPAT_PACKAGE)

  def test_failed_validator_does_not_mean_failed_isolation(self):
    with tempfile.TemporaryDirectory() as directory:
      device = Mock(serial="phone")
      device.forwards.return_value = device.clean_forwards.return_value = []
      device.artifact.return_value = "same-apk"
      device.stop.return_value = {"processes": [], "camera_owned": False}
      device.cool.return_value = {"processes": [], "camera_owned": False}
      with patch.object(experiment.subprocess, "run", return_value=Mock(returncode=1)):
        report = experiment.run(device, ["test"], Path(directory) / "result.json")
      self.assertEqual(report["wrapper_status"], "failed")
      self.assertEqual(report["validation_status"], "failed")
      self.assertEqual(report["isolation_status"], "passed")

  def test_workload_thermal_guard_fails_closed(self):
    device = experiment.Device("phone")
    for dump, allowed in (("Thermal Status: 0", True), ("Thermal Status: 1", True),
                          ("Thermal Status: 2", False), ("Thermal Status: 4", False),
                          ("Thermal Status: 0\nIsStatusOverride: true", False), ("", False)):
      device.command = Mock(return_value=dump)
      if allowed:
        device.assert_thermal_safe()
      else:
        with self.assertRaises(RuntimeError):
          device.assert_thermal_safe()

  def test_only_new_forwards_on_the_selected_device_are_removed(self):
    device = experiment.Device("phone")
    device.command = Mock(side_effect=["\nphone tcp:123 localabstract:old\nphone tcp:456 localabstract:new\nemulator-5554 tcp:789 localabstract:other\n", ""])
    self.assertEqual(device.clean_forwards([["tcp:123", "localabstract:old"]]), [["tcp:456", "localabstract:new"]])
    device.command.assert_called_with("forward", "--remove", "tcp:456")
  def test_both_package_variants_are_detected_and_stopped(self):
    state = experiment.parse_state("12 com.samsung.sunnypilot.fold3:camerad\n13 ai.sunnypilot.fold3:models",
      "Active Camera Clients:\ncom.samsung.sunnypilot.fold3\nAllowed user IDs: 0", "Thermal Status: 0")
    self.assertEqual(len(state["processes"]), 2)
    self.assertTrue(state["camera_owned"])
    device = experiment.Device("phone", package=experiment.CAMERA_COMPAT_PACKAGE)
    device.command = Mock()
    device.state = Mock(return_value={"processes": [], "camera_owned": False})
    device.stop()
    self.assertEqual([call.args[-1] for call in device.command.call_args_list],
                     [experiment.PACKAGE, experiment.PACKAGE + ".test",
                      experiment.CAMERA_COMPAT_PACKAGE, experiment.CAMERA_COMPAT_PACKAGE + ".test"])

  def test_active_client_and_test_process_are_detected_without_historic_clients(self):
    state = experiment.parse_state("1 ai.sunnypilot.fold3.test\n2 ai.sunnypilot.fold3:models",
      "Active Camera Clients:\n[]\nAllowed user IDs: 0\nDISCONNECT ai.sunnypilot.fold3",
      "IsStatusOverride: false\nThermal Status: 0\nTemperature{mValue=31.5, mType=3, mName=SKIN}")
    self.assertEqual(len(state["processes"]), 2)
    self.assertFalse(state["camera_owned"])
    self.assertEqual(state["skin_c"], 31.5)
    self.assertFalse(state["thermal_override"])
    self.assertIsNone(experiment.parse_state("", "", "")["thermal_status"])

  def test_stop_reasserts_force_stop_for_late_worker_startup(self):
    device = experiment.Device("phone", package=experiment.CAMERA_COMPAT_PACKAGE)
    device.command = Mock()
    device.state = Mock(side_effect=lambda: {"processes": ["late-ui"] if device.command.call_count < 8 else [],
                                           "camera_owned": False})
    with patch.object(experiment.time, "monotonic", side_effect=[0., 1., 21.]), patch.object(experiment.time, "sleep"):
      result = device.stop()
    self.assertEqual(result["processes"], [])
    self.assertEqual(device.command.call_count, 8)

  def test_final_cleanup_always_runs_and_failed_cleanup_fails_experiment(self):
    for workload_error, cleanup_error in ((True, False), (False, True), (False, False)):
      with self.subTest(workload_error=workload_error, cleanup_error=cleanup_error), tempfile.TemporaryDirectory() as directory:
        device = Mock(serial="phone")
        device.forwards.return_value = []
        device.clean_forwards.return_value = []
        device.artifact.return_value = "test"
        device.stop.side_effect = [{"processes": []}, RuntimeError("still running") if cleanup_error else {"processes": []}]
        device.cool.return_value = {"thermal_status": 0}
        with patch.object(experiment.subprocess, "run", side_effect=RuntimeError("workload failed") if workload_error else None,
                          return_value=Mock(returncode=0)):
          report = experiment.run(device, ["test"], Path(directory) / "result.json")
        self.assertEqual(device.stop.call_count, 2)
        self.assertEqual(report["status"], "failed" if workload_error or cleanup_error else "passed")

  def test_cooldown_refuses_running_processes_and_thermal_overrides(self):
    for change in ({"processes": ["model"]}, {"thermal_override": True}, {"camera_owned": True}):
      device = experiment.Device("phone")
      device.state = Mock(return_value={"processes": [], "camera_owned": False, "thermal_override": False, **change})
      with self.assertRaises(RuntimeError):
        device.cool()

  def test_provider_restart_fails_even_when_workload_returns_success(self):
    with tempfile.TemporaryDirectory() as directory:
      device = Mock(serial="phone")
      device.forwards.return_value = []
      device.clean_forwards.return_value = []
      device.stop.side_effect = [{"processes": []}, {"processes": [], "camera_provider_processes": ["2 provider"]}]
      device.cool.return_value = {"camera_provider_processes": ["1 provider"]}
      device.artifact.return_value = "test"
      with patch.object(experiment.subprocess, "run", return_value=Mock(returncode=0)):
        report = experiment.run(device, ["test"], Path(directory) / "result.json")
      self.assertEqual(report["status"], "failed")
      self.assertIn("restarted", report["camera_provider_error"])

  def test_runtime_starts_after_cooldown_and_startup_failure_still_cleans(self):
    for startup_error in (False, True):
      with self.subTest(startup_error=startup_error), tempfile.TemporaryDirectory() as directory:
        device = Mock(serial="phone")
        device.forwards.return_value = []
        device.clean_forwards.return_value = []
        device.stop.return_value = {"processes": []}
        device.artifact.return_value = "test"
        device.cool.return_value = {"thermal_status": 0}
        device.start_runtime.side_effect = RuntimeError("startup failed") if startup_error else None
        device.start_runtime.return_value = {"simulation": True}
        with patch.object(experiment.subprocess, "run", return_value=Mock(returncode=0)) as workload:
          report = experiment.run(device, ["test"], Path(directory) / "result.json", runtime="simulation")
        self.assertEqual(device.stop.call_count, 2)
        self.assertEqual(report["status"], "failed" if startup_error else "passed")
        self.assertEqual(workload.call_count, 0 if startup_error else 1)
        self.assertEqual([call[0] for call in device.mock_calls][:6], ["forwards", "stop", "artifact", "cool", "artifact", "start_runtime"])
        device.start_runtime.assert_called_once_with(True, full_chain=False)

  def test_installation_races_invalidate_provenance_and_prevent_a_cooldown_race(self):
    for hashes, started in ((["old", "new", "new"], False), (["old", "old", "new"], True), (["same"] * 3, True)):
      with self.subTest(hashes=hashes), tempfile.TemporaryDirectory() as directory:
        device = Mock(serial="phone")
        device.forwards.return_value = []
        device.clean_forwards.return_value = []
        device.stop.return_value = {"processes": []}
        device.cool.return_value = {"thermal_status": 0}
        device.artifact.side_effect = hashes
        with patch.object(experiment.subprocess, "run", return_value=Mock(returncode=0)) as workload:
          report = experiment.run(device, ["test"], Path(directory) / "result.json")
        stable = len(set(hashes)) == 1
        self.assertEqual(workload.called, started)
        self.assertEqual(report["artifact_stable"], stable)
        self.assertEqual(report["status"], "passed" if stable else "failed")
        self.assertEqual(device.stop.call_count, 2)

  def test_cooldown_requires_continuous_normal_temperature(self):
    device = experiment.Device("phone")
    state = {"processes": [], "camera_owned": False, "thermal_override": False, "thermal_status": 0, "skin_c": 31.}
    device.state = Mock(side_effect=[state, {**state, "skin_c": 33.}, state, state, state])
    with patch.object(experiment.time, "monotonic", side_effect=[0., 0., 20., 21., 50., 51.]), patch.object(experiment.time, "sleep"):
      baseline = device.cool()
    self.assertEqual(baseline["normal_thermal_seconds"], 30.)
    self.assertEqual(baseline["idle_seconds"], 51.)
