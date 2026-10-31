"""Experiment isolation must hold even when a workload fails."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("experiment", Path(__file__).resolve().parents[1] / "scripts/experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class ExperimentTest(unittest.TestCase):
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

  def test_final_cleanup_always_runs_and_failed_cleanup_fails_experiment(self):
    for workload_error, cleanup_error in ((True, False), (False, True), (False, False)):
      with self.subTest(workload_error=workload_error, cleanup_error=cleanup_error), tempfile.TemporaryDirectory() as directory:
        device = Mock(serial="phone")
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
        self.assertEqual([call[0] for call in device.mock_calls][:4], ["stop", "artifact", "cool", "start_runtime"])
        device.start_runtime.assert_called_once_with(True)
