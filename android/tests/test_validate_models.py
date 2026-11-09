"""A crashed native model process must leave evidence, never stale acceptance."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment import PACKAGE, CAMERA_COMPAT_PACKAGE
from validate_models import validate


class Device:
  def __init__(self, reports, processes=(), package=PACKAGE):
    self.reports, self.processes = iter(reports), iter(processes)
    self.package, self.calls = package, []

  def command(self, *args):
    self.calls.append(args)
    if args[0] == "exec-out":
      return json.dumps(next(self.reports))
    if args[:2] == ("shell", "ps"):
      return next(self.processes)
    return ""

  def assert_thermal_safe(self):
    pass

  def stop(self):
    self.stopped = True


class ModelsTest(unittest.TestCase):
  def test_thermal_abort_stops_work_and_saves_blocked_evidence(self):
    device = Device([{}])
    device.assert_thermal_safe = lambda: (_ for _ in ()).throw(RuntimeError("moderate thermal state"))
    with tempfile.TemporaryDirectory() as folder:
      destination = Path(folder) / "report.json"
      with self.assertRaisesRegex(RuntimeError, "thermal"):
        validate(device, "qnn-htp", destination)
      self.assertTrue(device.stopped)
      self.assertEqual(json.loads(destination.read_text())["validation_status"], "blocked")

  def test_completed_failed_parity_is_evidence(self):
    report = {"started_utc": 2, "completed_utc": 3, "vehicle_tx_enabled": False, "models": {"driver": {"status": "failed"}}}
    with tempfile.TemporaryDirectory() as folder:
      destination = Path(folder) / "report.json"
      expected = {**report, "validation_status": "failed"}
      self.assertEqual(validate(Device([{"started_utc": 1}, report]), "ort-cpu", destination), expected)
      self.assertEqual(json.loads(destination.read_text()), expected)

  def test_both_models_and_both_gates_are_required(self):
    passed = {"status": "passed", "performance_status": "passed"}
    cases = [({"dmonitoring_model": passed}, "blocked"),
             ({name: passed.copy() for name in ("dmonitoring_model", "driving_supercombo")}, "passed"),
             ({"dmonitoring_model": passed, "driving_supercombo": {**passed, "performance_status": "failed"}}, "failed")]
    with tempfile.TemporaryDirectory() as folder:
      for models, expected in cases:
        with self.subTest(expected=expected):
          report = {"started_utc": 2, "completed_utc": 3, "vehicle_tx_enabled": False, "models": models}
          result = validate(Device([{}, report]), "ort-cpu", Path(folder) / "report.json")
          self.assertEqual(result["validation_status"], expected)

  def test_compatibility_package_uses_real_activity_class_and_tx_guard(self):
    report = {"started_utc": 2, "completed_utc": 3, "vehicle_tx_enabled": False, "models": {}}
    device = Device([{}, report], package=CAMERA_COMPAT_PACKAGE)
    with tempfile.TemporaryDirectory() as folder:
      validate(device, "ort-cpu", Path(folder) / "report.json")
      self.assertIn(CAMERA_COMPAT_PACKAGE + "/ai.sunnypilot.fold3.MainActivity", device.calls[1])
      self.assertEqual(device.calls[0][2], CAMERA_COMPAT_PACKAGE)
      for tx in (True, None):
        with self.assertRaisesRegex(AssertionError, "TX guard"):
          validate(Device([{}, {**report, "vehicle_tx_enabled": tx}]), "ort-cpu", Path(folder) / "report.json")

  def test_diagnostic_probes_are_separate_from_full_model_qualification(self):
    models = {engine + "-" + op: {"status": "passed", "performance_status": "diagnostic_only"}
              for engine in ("mnn-opencl", "ncnn-vulkan") for op in ("conv", "gemm", "activation", "layout")}
    report = {"started_utc": 2, "completed_utc": 3, "vehicle_tx_enabled": False, "diagnostic_only": True, "models": models}
    with tempfile.TemporaryDirectory() as folder:
      destination = Path(folder) / "report.json"
      self.assertEqual(validate(Device([{}, report]), "accel-probes", destination)["validation_status"], "passed")
      self.assertEqual(validate(Device([{}, report]), "ort-cpu", destination)["validation_status"], "blocked")
      models.pop("ncnn-vulkan-gemm")
      self.assertEqual(validate(Device([{}, report]), "accel-probes", destination)["validation_status"], "blocked")

  @patch("validate_models.time.sleep")
  def test_crash_before_new_report_rejects_stale_completion(self, _sleep):
    old = {"started_utc": 1, "completed_utc": 2, "models": {"driver": {"status": "passed"}}}
    with tempfile.TemporaryDirectory() as folder:
      destination = Path(folder) / "report.json"
      with self.assertRaisesRegex(RuntimeError, "exited without completing"):
        validate(Device([old, old, old], [PACKAGE + ":models", ""]), "ort-cpu", destination)
      saved = json.loads(destination.read_text())
      self.assertEqual(saved["validation_status"], "failed")
      self.assertNotIn("completed_utc", saved)
      self.assertNotIn("models", saved)
      self.assertNotIn("vehicle_tx_enabled", saved)

  def test_crash_after_new_report_preserves_progress(self):
    progress = {"started_utc": 2, "models": {"driver": {"status": "running"}}}
    with tempfile.TemporaryDirectory() as folder:
      destination = Path(folder) / "report.json"
      with self.assertRaises(RuntimeError):
        validate(Device([{"started_utc": 1}, progress], [""]), "ort-cpu", destination)
      self.assertEqual(json.loads(destination.read_text())["models"], progress["models"])

  @patch("validate_models.time.monotonic", side_effect=[0, 1])
  def test_timeout_stops_work_and_does_not_save_stale_acceptance(self, _clock):
    old = {"started_utc": 1, "completed_utc": 2, "models": {"driver": {"status": "passed"}}}
    device = Device([old])
    with tempfile.TemporaryDirectory() as folder:
      destination = Path(folder) / "report.json"
      with self.assertRaises(TimeoutError):
        validate(device, "qnn-gpu", destination, timeout=1)
      self.assertTrue(device.stopped)
      saved = json.loads(destination.read_text())
      self.assertEqual(saved["validation_status"], "blocked")
      self.assertNotIn("completed_utc", saved)
      self.assertNotIn("models", saved)


if __name__ == "__main__":
  unittest.main()
