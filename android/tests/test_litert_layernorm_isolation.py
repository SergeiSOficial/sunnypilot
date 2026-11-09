import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from isolate_litert_layernorm import capture_half_input


class LayerNormIsolationTest(unittest.TestCase):
  def test_accepts_losslessly_expanded_half_values(self):
    expected = np.array([0.25, -2], dtype=np.float16)
    actual = expected.astype(np.float32)
    np.testing.assert_array_equal(capture_half_input(actual, expected), expected)

  def test_rejects_wrong_boundary_shape_and_nonfinite_values(self):
    expected = np.array([0, 1], dtype=np.float16)
    for actual in (np.array([0.1, 1], dtype=np.float32), np.array([np.nan, 1], dtype=np.float32)):
      with self.assertRaisesRegex(ValueError, 'FP16 boundary'):
        capture_half_input(actual, expected)
    with self.assertRaisesRegex(ValueError, 'shape or dtype'):
      capture_half_input(np.zeros(1, np.float32), expected)
    with self.assertRaisesRegex(ValueError, 'shape or dtype'):
      capture_half_input(np.zeros(2, np.float32), expected.astype(np.float32))
