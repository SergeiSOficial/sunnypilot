"""Native replay diagnostics must use the pinned image and calibration."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from android.scripts.ncnn_replay_inputs import replay_case


class ReplayDiagnosticTest(unittest.TestCase):
  def fixture(self, base):
    folder = base / "sequences"
    folder.mkdir()
    np.savez(base / "dmonitoring_model-inputs.npz", input_img=np.zeros((1, 2), np.uint8), calib=np.zeros((1, 3), np.float32))
    np.savez(folder / "camera-inputs.npz", driver=np.arange(120, dtype=np.uint8).reshape(60, 1, 2))
    np.savez(folder / "dmonitoring_model-camera-replay.npz", outputs=np.arange(60, dtype=np.float16).reshape(60, 1, 1))
    manifest = {"rtol": 1e-3, "atol": 1e-4,
      "assets": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob("*.npz")},
      "models": {"dmonitoring_model": {"camera-replay": {"frames": 60, "seed": 42}}},
      "replay": {"device_from_calib_euler": [1., 2., 3.]}}
    (folder / "manifest.json").write_text(json.dumps(manifest))

  def test_replay_uses_recorded_frame_calibration_and_golden(self):
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      self.fixture(base)
      inputs, golden = replay_case(base, 21)
      np.testing.assert_array_equal(inputs["input_img"], [[42, 43]])
      np.testing.assert_array_equal(inputs["calib"], [[1., 2., 3.]])
      np.testing.assert_array_equal(golden, [[21.]])
      self.assertEqual(inputs["input_img"].dtype, np.uint8)

  def test_changed_recording_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      self.fixture(base)
      (base / "sequences/camera-inputs.npz").write_bytes(b"changed")
      with self.assertRaisesRegex(ValueError, "asset changed"):
        replay_case(base, 21)

  def test_invalid_frame_is_rejected(self):
    for frame in (-1, 60, True, 2.5):
      with self.assertRaisesRegex(ValueError, "frame"):
        replay_case(Path("unused"), frame)
