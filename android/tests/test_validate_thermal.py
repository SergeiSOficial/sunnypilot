"""Raw capture proves a thermal-test precondition, never perception validity."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_thermal import raw_cameras_started


class RawThermalCaptureTest(unittest.TestCase):
  def test_requires_new_distinct_generations_and_actual_raw_frames(self):
    previous = [{"session_started_elapsed_ns": 100}] * 2
    acquired = [{"sensor_id": str(sensor), "session_started_elapsed_ns": 200,
                 "raw_acquired_frames": 1, "visionipc_valid": False} for sensor in range(2)]
    self.assertTrue(raw_cameras_started(acquired, previous, 150))
    self.assertTrue(raw_cameras_started(acquired, [{}, {}], 150))
    self.assertFalse(raw_cameras_started(acquired[:1], previous, 150))
    self.assertFalse(raw_cameras_started(acquired, [], 150))
    for field, bad in (("session_started_elapsed_ns", 100), ("raw_acquired_frames", 0),
                       ("raw_acquired_frames", True), ("sensor_id", "0"), ("visionipc_valid", True)):
      changed = copy.deepcopy(acquired)
      changed[1][field] = bad
      with self.subTest(field=field, bad=bad):
        self.assertFalse(raw_cameras_started(changed, previous, 150))

  def test_launch_bound_rejects_stale_frames_across_reboot(self):
    previous = [{"session_started_elapsed_ns": 900}] * 2
    acquired = [{"sensor_id": str(sensor), "session_started_elapsed_ns": 200,
                 "raw_acquired_frames": 1, "visionipc_valid": False} for sensor in range(2)]
    self.assertTrue(raw_cameras_started(acquired, previous, 150))
    self.assertFalse(raw_cameras_started(acquired, previous, 201))
    self.assertFalse(raw_cameras_started(acquired, acquired, 150))
    self.assertFalse(raw_cameras_started(acquired, previous, True))
