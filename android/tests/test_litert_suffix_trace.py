import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from trace_litert_suffix_single import compare_axis


class SuffixAxisTraceTest(unittest.TestCase):
  def test_identifies_changed_history_step(self):
    expected = np.zeros((1, 3, 2), np.float32)
    actual = expected.copy()
    actual[0, 1, 0] = 0.1
    result = compare_axis(actual, expected, 1)
    self.assertEqual([entry['mismatched_elements'] for entry in result], [0, 1, 0])
    self.assertEqual([entry['exact'] for entry in result], [True, False, True])

  def test_rejects_invalid_axes_and_shapes(self):
    value = np.zeros((1, 3), np.float32)
    for expected, axis in ((np.zeros((1, 2), np.float32), 1), (value, -1), (value, 2)):
      with self.subTest(axis=axis), self.assertRaises(ValueError):
        compare_axis(value, expected, axis)
    with self.assertRaises(ValueError):
      compare_axis(np.zeros(257), np.zeros(257), 0)


if __name__ == '__main__':
  unittest.main()
