"""A crashed native model process must leave evidence, never stale acceptance."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment import PACKAGE
from validate_models import validate


class Device:
  def __init__(self, reports, processes=()):
    self.reports, self.processes = iter(reports), iter(processes)

  def command(self, *args):
    if args[0] == "exec-out":
      return json.dumps(next(self.reports))
    if args[:2] == ("shell", "ps"):
      return next(self.processes)
    return ""


class ModelsTest(unittest.TestCase):
  def test_completed_failed_parity_is_evidence(self):
    report = {"started_utc": 2, "completed_utc": 3, "models": {"driver": {"status": "failed"}}}
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
          report = {"started_utc": 2, "completed_utc": 3, "models": models}
          result = validate(Device([{}, report]), "ort-cpu", Path(folder) / "report.json")
          self.assertEqual(result["validation_status"], expected)

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

  def test_crash_after_new_report_preserves_progress(self):
    progress = {"started_utc": 2, "models": {"driver": {"status": "running"}}}
    with tempfile.TemporaryDirectory() as folder:
      destination = Path(folder) / "report.json"
      with self.assertRaises(RuntimeError):
        validate(Device([{"started_utc": 1}, progress], [""]), "ort-cpu", destination)
      self.assertEqual(json.loads(destination.read_text())["models"], progress["models"])


if __name__ == "__main__":
  unittest.main()
