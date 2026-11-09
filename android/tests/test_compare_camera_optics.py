"""An optical resemblance or timestamp overlap alone cannot certify rear identity."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

try:
  import cv2
except ImportError:
  cv2 = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@unittest.skipUnless(cv2 is not None, "OpenCV is required for optical diagnostics")
class CameraOpticsTest(unittest.TestCase):
  def test_remapped_reference_matches_but_unrelated_or_flat_input_does_not(self):
    from compare_camera_optics import compare, features

    rng = np.random.default_rng(41)
    reference = cv2.GaussianBlur(rng.integers(0, 256, (480, 640), dtype=np.uint8), (5, 5), 1)
    other = cv2.GaussianBlur(rng.integers(0, 256, (480, 640), dtype=np.uint8), (5, 5), 1)
    image = cv2.warpAffine(reference, cv2.getRotationMatrix2D((320, 240), 5, .86), (640, 480))
    matched = compare(features(reference), features(image))
    self.assertGreater(matched["inliers"], 50)
    self.assertEqual(sum(cell["inliers"] for cell in matched["candidate_grid"]), matched["inliers"])
    self.assertLess(compare(features(other), features(image))["inliers"], 12)
    self.assertEqual(compare(features(np.zeros_like(image)), features(image))["inliers"], 0)

  def test_clip_rejects_truncation_and_nonmonotonic_times(self):
    from compare_camera_optics import load_clip

    with tempfile.TemporaryDirectory() as directory:
      clip = Path(directory) / "test.clip"
      good = struct.pack(">q", 42) + bytes(range(6))
      clip.write_bytes(good)
      timestamp, pixels = load_clip(clip, 2, 2)[0]
      self.assertEqual(timestamp, 42)
      np.testing.assert_array_equal(pixels, [[0, 1], [2, 3]])
      for data in (good[:-1], good + good):
        clip.write_bytes(data)
        with self.assertRaises(ValueError):
          load_clip(clip, 2, 2)

  def test_quarter_turn_restores_a_stretched_landscape_raster(self):
    from compare_camera_optics import compare, features, quarter_turn_same_aspect

    rng = np.random.default_rng(42)
    reference = cv2.GaussianBlur(rng.integers(0, 256, (360, 640), dtype=np.uint8), (9, 9), 2)
    stretched = cv2.resize(cv2.rotate(reference, cv2.ROTATE_90_COUNTERCLOCKWISE), (640, 360))
    reference_features = features(reference)
    restored = compare(reference_features, features(quarter_turn_same_aspect(stretched)))
    raw = compare(reference_features, features(stretched))
    self.assertGreater(restored["inliers"], 50)
    self.assertGreater(restored["inliers"], 4 * raw["inliers"])
