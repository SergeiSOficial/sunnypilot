"""A hot-device preflight must not launch camera or runtime workers."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location("validate_vision", Path(__file__).resolve().parents[1] / "scripts/validate_vision.py")
vision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vision)


class VisionValidationTest(unittest.TestCase):
  def test_critical_preflight_stops_without_launch_and_records_cleanup(self):
    device = Mock(spec=vision.Device)
    device.assert_thermal_safe.side_effect = RuntimeError("Unsafe or unavailable Android thermal state: 4")
    device.stop.return_value = {"processes": [], "camera_owned": False}
    def command(args, **kwargs):
      return "0.0 0.0" if "/proc/uptime" in args else "{}"
    with tempfile.TemporaryDirectory() as directory, patch.object(vision, "Device", return_value=device), \
         patch.object(vision.subprocess, "check_output", side_effect=command) as commands:
      path = Path(directory) / "vision.json"
      result = vision.validate("emulator-test", "adb", 15, path)
      self.assertEqual(result["status"], "failed")
      self.assertIn("thermal state: 4", result["error"])
      self.assertEqual(result["cleanup"], device.stop.return_value)
      self.assertEqual(device.stop.call_count, 2)
      self.assertFalse(any("start" in call.args[0] for call in commands.call_args_list))
      self.assertEqual(json.loads(path.read_text())["status"], "failed")
