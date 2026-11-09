import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from isolate_litert_suffix_gemm import compare_at_reference_precision, normalize_inputs


class SuffixIsolationTest(unittest.TestCase):
  def test_compare_at_original_half_output_boundary(self):
    reference = np.array([1.0, 2.0], dtype=np.float16)
    actual = np.array([1.0001, 2.002], dtype=np.float32)
    result = compare_at_reference_precision(actual, reference)
    self.assertFalse(result['exact'])
    self.assertEqual(result['mismatched_elements'], 0)
    self.assertEqual(result['different_elements'], 1)
    with self.assertRaisesRegex(ValueError, 'shape or floating-point dtype'):
      compare_at_reference_precision(actual, reference[:1])
    with self.assertRaisesRegex(ValueError, 'shape or floating-point dtype'):
      compare_at_reference_precision(actual.astype(np.int32), reference)
    self.assertFalse(compare_at_reference_precision(np.array([1e10], np.float32),
                                                    np.array([1], np.float16))['passed'])

  def test_transpose_and_lossless_half_inputs(self):
    actual = [np.array([[1, 2, 3]], np.float32), np.arange(6, dtype=np.float32).reshape(2, 3),
              np.array([1, 2], np.float32)]
    expected = [actual[0].astype(np.float16), actual[1].T.astype(np.float16), actual[2].astype(np.float16)]
    normalized = normalize_inputs(actual, expected, 0)
    for result, reference in zip(normalized, expected, strict=True):
      np.testing.assert_array_equal(result, reference)
      self.assertEqual(result.dtype, reference.dtype)

  def test_rejects_lossy_cast_and_wrong_layout(self):
    actual = [np.array([[0.1]], np.float32), np.ones((2, 1), np.float32), np.ones(2, np.float32)]
    expected = [actual[0].astype(np.float16), actual[1].astype(np.float16), actual[2].astype(np.float16)]
    with self.assertRaisesRegex(ValueError, 'would change values'):
      normalize_inputs(actual, expected, 1)
    with self.assertRaisesRegex(ValueError, 'layout mismatch'):
      normalize_inputs(actual, expected, 0)


if __name__ == '__main__':
  unittest.main()
