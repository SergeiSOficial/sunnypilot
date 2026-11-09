"""Retain partial diagnostics and stop promptly when the model process dies."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_model_trace import collect


class Device:
  package = "com.samsung.sunnypilot.fold3"

  def __init__(self, report, processes=()):
    self.report, self.processes = report, iter(processes)

  def command(self, *args):
    if args[:2] == ("exec-out", "run-as"):
      return json.dumps(self.report)
    if args[:3] == ("shell", "ps", "-A"):
      return next(self.processes)
    return ""


class CollectModelTraceTest(unittest.TestCase):
  @patch("collect_model_trace.time.sleep")
  def test_native_crash_retains_partial_report_and_stops_without_timeout(self, sleep):
    report = {"vehicle_tx_enabled": False, "models": {"driver": {"engines": {}}}}
    device = Device(report, (Device.package + ":models", ""))
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / "trace.json"
      with self.assertRaisesRegex(RuntimeError, "exited before completing"):
        collect(device, destination, "trace-fused")
      self.assertEqual(json.loads(destination.read_text()), report)
      self.assertEqual(sleep.call_count, 1)

  def test_completed_engine_error_is_not_a_success(self):
    report = {"completed": True, "vehicle_tx_enabled": False,
              "models": {"driver": {"engines": {"ort": {}, "CL": {"error": "compile failure"}}}}}
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / "trace.json"
      with self.assertRaisesRegex(RuntimeError, "engine error"):
        collect(Device(report), destination, "trace-fused")
      self.assertEqual(json.loads(destination.read_text()), report)

  def test_qnn_trace_requires_both_reference_and_accelerator_results(self):
    report = {"completed": True, "vehicle_tx_enabled": False,
              "models": {"driver": {"engines": {"ort": {}, "qnn-gpu": {"first_divergence": {"index": 3}}}}}}
    with tempfile.TemporaryDirectory() as directory:
      self.assertEqual(collect(Device(report), Path(directory) / "trace.json", "trace-qnn"), report)

  def test_uploaded_graph_cannot_be_used_by_a_non_diagnostic_profile(self):
    with self.assertRaisesRegex(ValueError, "diagnostic QNN"):
      collect(Device({}), "unused.json", "trace", "unused-directory")


if __name__ == "__main__":
  unittest.main()
