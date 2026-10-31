"""A previous successful report cannot make a new camera attempt pass."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location("validate_cameras", Path(__file__).resolve().parents[1] / "scripts/validate_cameras.py")
cameras = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cameras)


class CameraValidationTest(unittest.TestCase):
  def test_old_pass_is_not_reused_and_missing_stream_fails(self):
    old = {"started_utc": "old", "completed_utc": "old", "attempts": [{"status": "passed"}]}
    failed = {"started_utc": "new", "completed_utc": "new", "vehicle_tx_enabled": False,
              "driving_runtime_ready": False, "attempts": [{"status": "failed", "error": "camera_error:2:2"}]}
    with tempfile.TemporaryDirectory() as directory, patch.object(cameras, "read", side_effect=[old, old, failed]), \
         patch.object(cameras.time, "sleep"), patch("builtins.print"):
      path = Path(directory) / "result.json"
      self.assertFalse(cameras.validate(Mock(package="test"), ["0", "2"], "20", 3, 640, 480, path))
      self.assertEqual(json.loads(path.read_text())["started_utc"], "new")

  def test_transmission_guard_cannot_be_overridden(self):
    bad = {"started_utc": "new", "completed_utc": "new", "vehicle_tx_enabled": True,
           "driving_runtime_ready": False, "attempts": [{"status": "passed"}]}
    with patch.object(cameras, "read", side_effect=[{}, bad]):
      with self.assertRaises(RuntimeError):
        cameras.validate(Mock(package="test"), ["0", "2"], "20", 3, 640, 480, "unused.json")

  def test_composite_transport_does_not_qualify_physical_streams(self):
    composite = {"started_utc": "new", "completed_utc": "new", "vehicle_tx_enabled": False,
                 "driving_runtime_ready": False, "attempts": [{"status": "blocked", "transport_status": "passed",
                 "distinct_sensor_evidence": False}]}
    device = Mock(package="test")
    with tempfile.TemporaryDirectory() as directory, patch.object(cameras, "read", side_effect=[{}, composite]), \
         patch("builtins.print"):
      output = Path(directory) / "result.json"
      self.assertFalse(cameras.validate(device, ["0", "2"], "20", 3, 640, 480, output, samsung_multiview=True))
      self.assertEqual(json.loads(output.read_text())["status"], "blocked")
      self.assertIn("samsung_multiview", device.command.call_args.args)
