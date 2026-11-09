import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from trace_litert_prefix import align, producer, tap_impact


class PrefixLayoutTest(unittest.TestCase):
  def test_explicit_nhwc_even_with_ambiguous_shape(self):
    value = np.arange(16).reshape(1, 2, 2, 4)
    expected = value.transpose(0, 3, 1, 2)
    actual, layout = align(value, expected, True)
    np.testing.assert_array_equal(actual, expected)
    self.assertIn('metadata', layout)
    cube = np.arange(8).reshape(1, 2, 2, 2)
    actual, _ = align(cube, cube, True)
    np.testing.assert_array_equal(actual, cube.transpose(0, 3, 1, 2))

  def test_unambiguous_shape_and_bad_mapping(self):
    value = np.arange(16).reshape(1, 2, 2, 4)
    actual, layout = align(value, value.transpose(0, 3, 1, 2), False)
    np.testing.assert_array_equal(actual, value.transpose(0, 3, 1, 2))
    self.assertIn('unambiguous', layout)
    with self.assertRaises(ValueError):
      align(value, np.zeros((1, 3, 2, 2)), True)

  def test_producer_uses_actual_output_index(self):
    operations = [{'index': 4, 'outputs': [17]}, {'index': 7, 'outputs': [18]}]
    self.assertEqual(producer(operations, 18)['index'], 7)
    with self.assertRaises(ValueError):
      producer(operations, 19)
    with self.assertRaises(ValueError):
      producer(operations + [{'index': 8, 'outputs': [18]}], 18)

  def test_tap_impact_detects_changed_or_missing_public_output(self):
    original = {'road': np.array([1.0, 0.0], np.float32)}
    same = tap_impact(original, {'road': original['road'].copy(), 'intermediate': np.zeros(1)})
    self.assertTrue(same['tolerance_preserved'])
    self.assertTrue(same['bitwise_preserved'])
    changed = tap_impact(original, {'road': np.array([1.0, 0.1], np.float32)})
    self.assertFalse(changed['tolerance_preserved'])
    self.assertFalse(changed['bitwise_preserved'])
    shape_changed = tap_impact(original, {'road': np.zeros(1, np.float32)})
    self.assertFalse(shape_changed['tolerance_preserved'])
    self.assertFalse(shape_changed['bitwise_preserved'])
    with self.assertRaises(ValueError):
      tap_impact(original, {'intermediate': np.zeros(1)})


if __name__ == '__main__':
  unittest.main()
