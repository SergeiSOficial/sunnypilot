"""Preserve image bytes and explicit layout contracts at split boundaries."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from decompose_litert_road_dense import adapted_runner
from validate_litert_dense_sections_host import convert_input


class LiteRtDenseSectionsTest(unittest.TestCase):
  def test_packed_history_round_trip_preserves_byte_order(self):
    history = np.arange(2*5*6*128*256, dtype=np.uint32).astype(np.uint8).reshape(2, 5, 6, 128, 256)
    image = np.zeros((2, 6, 128, 256), dtype=np.uint8)
    def packed(values):
      self.assertEqual(values['state_img_q'].shape, (2, 30, 128, 256))
      self.assertEqual(values['state_img_q'].dtype, np.float32)
      return {'next_state_img_q': values['state_img_q']}
    result = adapted_runner(packed)({'new_img': image, 'state_img_q': history})
    np.testing.assert_array_equal(result['next_state_img_q'], history)

  def test_fractional_history_rejected(self):
    inputs = {'new_img': np.zeros((2, 6, 128, 256), np.uint8),
              'state_img_q': np.zeros((2, 5, 6, 128, 256), np.uint8)}
    with self.assertRaisesRegex(ValueError, 'byte values'):
      adapted_runner(lambda values: {'next_state_img_q': values['state_img_q'] + .5})(inputs)

  def test_packed_layout_and_wrong_shape(self):
    packed = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    actual = convert_input('state_img_q', packed, {'dtype': np.float32, 'shape': (2, 2, 2, 3)})
    np.testing.assert_array_equal(actual, packed.transpose(0, 2, 3, 1))
    with self.assertRaisesRegex(ValueError, 'Unexpected converted input shape'):
      convert_input('state_img_q', packed, {'dtype': np.float32, 'shape': (2, 3, 2, 2)})


if __name__ == '__main__':
  unittest.main()
