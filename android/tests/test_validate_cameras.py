"""A previous successful report cannot make a new camera attempt pass."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location("validate_cameras", Path(__file__).resolve().parents[1] / "scripts/validate_cameras.py")
cameras = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cameras)


class CameraValidationTest(unittest.TestCase):
  def test_low_latency_profiles_reject_vendor_routing_before_adb(self):
    script = Path(__file__).resolve().parents[1] / 'scripts/validate_cameras.py'
    for profile in ('low-latency', 'low-latency-preview', 'low-latency-record'):
      result = subprocess.run([sys.executable, str(script), '--serial', 'absent', '--output', 'unused',
                               '--ids', '0,2', '--logical', '20', '--camera-profile', profile],
                              capture_output=True, text=True, check=False)
      self.assertEqual(result.returncode, 2)
      self.assertIn('Low-latency profiles require public independent sessions', result.stderr)

  def test_manual_exposure_rejects_vendor_or_physical_routing_before_adb(self):
    script = Path(__file__).resolve().parents[1] / "scripts/validate_cameras.py"
    for options in (["--camera-profile", "manual-2ms-low-latency", "--ids", "0,2", "--logical", "20"],
                    ["--camera-profile", "manual-8ms-low-latency", "--ids", "0,2", "--logical", "20"],
                    ["--ids", "0,2", "--logical", "20"],
                    ["--ids", "0,2", "--logical", "20", "--samsung-multiview"]):
      result = subprocess.run([sys.executable, str(script), "--serial", "absent", "--output", "unused",
                               "--camera-profile", "manual-8ms", *options], capture_output=True, text=True, check=False)
      self.assertEqual(result.returncode, 2)
      self.assertIn("Manual exposure requires public independent rear cameras", result.stderr)

  def test_high_speed_cli_rejects_unsupported_topologies_before_adb(self):
    script = Path(__file__).resolve().parents[1] / "scripts/validate_cameras.py"
    for options in (["--ids", "0"], ["--ids", "0,1", "--private-gpu"],
                    ["--ids", "0", "--private-gpu", "--logical", "20"],
                    ["--ids", "0", "--private-gpu", "--evidence-directory", "unused"]):
      result = subprocess.run([sys.executable, str(script), "--serial", "absent", "--output", "unused",
                               "--camera-profile", "high-speed-120", *options], capture_output=True, text=True, check=False)
      self.assertEqual(result.returncode, 2)
      self.assertIn("High-speed capture requires one public camera", result.stderr)

  def test_old_pass_is_not_reused_and_missing_stream_fails(self):
    old = {"started_utc": "old", "completed_utc": "old", "attempts": [{"status": "passed"}]}
    failed = {"started_utc": "new", "completed_utc": "new", "vehicle_tx_enabled": False,
              "driving_runtime_ready": False, "attempts": [{"status": "failed", "error": "camera_error:2:2"}]}
    with tempfile.TemporaryDirectory() as directory, patch.object(cameras, "read", side_effect=[old, old, failed]), \
         patch.object(cameras.time, "sleep"), patch("builtins.print"):
      path = Path(directory) / "result.json"
      self.assertFalse(cameras.validate(Mock(spec=cameras.Device, package="test"), ["0", "2"], "20", 3, 640, 480, path))
      self.assertEqual(json.loads(path.read_text())["started_utc"], "new")

  def test_transmission_guard_cannot_be_overridden(self):
    bad = {"started_utc": "new", "completed_utc": "new", "vehicle_tx_enabled": True,
           "driving_runtime_ready": False, "attempts": [{"status": "passed"}]}
    with patch.object(cameras, "read", side_effect=[{}, bad]):
      with self.assertRaises(RuntimeError):
        cameras.validate(Mock(spec=cameras.Device, package="test"), ["0", "2"], "20", 3, 640, 480, "unused.json")

  def test_composite_transport_does_not_qualify_physical_streams(self):
    composite = {"started_utc": "new", "completed_utc": "new", "vehicle_tx_enabled": False,
                 "driving_runtime_ready": False, "attempts": [{"status": "blocked", "transport_status": "passed",
                 "distinct_sensor_evidence": False}]}
    device = Mock(spec=cameras.Device, package="test")
    with tempfile.TemporaryDirectory() as directory, patch.object(cameras, "read", side_effect=[{}, composite]), \
         patch("builtins.print"):
      output = Path(directory) / "result.json"
      self.assertFalse(cameras.validate(device, ["0", "2"], "20", 3, 640, 480, output, samsung_multiview=True))
      self.assertEqual(json.loads(output.read_text())["status"], "blocked")
      self.assertIn("samsung_multiview", device.command.call_args.args)

  def test_thermal_pause_stops_waiting_for_camera_report(self):
    device = Mock(spec=cameras.Device, package="test")
    device.assert_thermal_safe.side_effect = [None, RuntimeError("Unsafe Android thermal state: 2")]
    with tempfile.TemporaryDirectory() as directory, patch.object(cameras, "read", return_value={}), patch("builtins.print"):
      output = Path(directory) / "result.json"
      self.assertFalse(cameras.validate(device, ["0"], None, 3, 640, 480, output))
      self.assertEqual(json.loads(output.read_text())["error"], "Unsafe Android thermal state: 2")

  def test_evidence_pairs_by_capture_time_without_reusing_frames(self):
    from io import BytesIO
    import struct

    def clip(timestamps, value):
      return BytesIO(b"".join(struct.pack(">q", timestamp * 1_000_000) +
                              bytes([value]) * cameras.FRAME_BYTES for timestamp in timestamps))

    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory)
      pairs = cameras.paired_evidence((clip([100, 350, 600], 17), clip([101, 375, 603], 42)), destination)
      self.assertEqual(pairs, [[100_000_000, 101_000_000], [600_000_000, 603_000_000]])
      self.assertEqual((destination / "preview.nv12").read_bytes(), bytes([17]) * cameras.FRAME_BYTES * 2)
      self.assertEqual((destination / "thumbnail.nv12").read_bytes(), bytes([42]) * cameras.FRAME_BYTES * 2)
