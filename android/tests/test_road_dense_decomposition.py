"""The dense cut must retain public skip outputs and the original half arithmetic."""
from pathlib import Path
import sys
import unittest

import numpy as np
import onnxruntime as ort
from onnx import TensorProto as T, helper as h, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from decompose_road_dense import decompose, OrderedDenseRecompose


def graph(trans_b=1):
  value = lambda name, shape: h.make_tensor_value_info(name, T.FLOAT16, shape)
  weights = numpy_helper.from_array(np.arange(16, dtype=np.float16).reshape(2, 8), 'weight')
  bias = numpy_helper.from_array(np.array([1, -2], np.float16), 'bias')
  nodes = [h.make_node('Identity', ['image'], ['history']),
           h.make_node('Identity', ['history'], ['model/view']),
           h.make_node('Gemm', ['model/view', 'weight', 'bias'], ['model/linear'],
                       name='dense', alpha=1., beta=1., transA=0, transB=trans_b),
           h.make_node('Identity', ['model/linear'], ['output'])]
  g = h.make_graph(nodes, 'dense-test', [value('image', [1, 8])],
                   [value('history', [1, 8]), value('output', [1, 2])], [weights, bias])
  return h.make_model(g, opset_imports=[h.make_opsetid('', 17)], ir_version=10)


class DenseDecompositionTest(unittest.TestCase):
  def test_public_history_and_output_match_original(self):
    model = graph()
    sections, manifest = decompose(model, (2, 8))
    self.assertIn('history', manifest['prefix_crossings'])
    inputs = {'image': np.array([[1, 2, 0, 0, 0, 0, 0, 0]], np.float16)}
    result = OrderedDenseRecompose(sections, manifest['public_outputs'])(inputs)
    original = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    for name, expected in zip(manifest['public_outputs'], original.run(None, inputs), strict=True):
      np.testing.assert_array_equal(result[name], expected)

  def test_wrong_expected_dimensions_are_rejected(self):
    with self.assertRaisesRegex(ValueError, 'dimensions'):
      decompose(graph(), (2, 16))

  def test_unsupported_gemm_arithmetic_is_rejected(self):
    model = graph()
    node = next(n for n in model.graph.node if n.op_type == 'Gemm')
    next(a for a in node.attribute if a.name == 'alpha').f = 0.5
    with self.assertRaisesRegex(ValueError, 'arithmetic'):
      decompose(model)


if __name__ == '__main__':
  unittest.main()
