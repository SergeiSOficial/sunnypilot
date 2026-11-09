import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as h, numpy_helper as nh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_litert_targeted_round import add_rounding


def model():
  x = h.make_tensor_value_info('x', T.FLOAT16, [1, 2])
  y = h.make_tensor_value_info('y', T.FLOAT16, [1, 2])
  w = nh.from_array(np.array([[1., -0.5], [0.25, 2.]], np.float16), 'w')
  graph = h.make_graph([h.make_node('Gemm', ['x', 'w'], ['y'], name='dense')], 'test', [x], [y], [w])
  return h.make_model(graph, opset_imports=[h.make_opsetid('', 18)], ir_version=9)


class TargetedRoundingTest(unittest.TestCase):
  def test_preserves_small_onnx_gemm(self):
    original = model()
    modified = add_rounding(original, ['dense'])
    source = {'x': np.array([[0.8125, -0.15625]], np.float16)}
    expected = ort.InferenceSession(original.SerializeToString(), providers=['CPUExecutionProvider']).run(None, source)[0]
    actual = ort.InferenceSession(modified.SerializeToString(), providers=['CPUExecutionProvider']).run(None, source)[0]
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(len(original.graph.node), 1)

  def test_rejects_missing_or_duplicate_names(self):
    for names in (['missing'], ['dense', 'dense']):
      with self.subTest(names=names), self.assertRaises(ValueError):
        add_rounding(model(), names)


if __name__ == '__main__':
  unittest.main()
