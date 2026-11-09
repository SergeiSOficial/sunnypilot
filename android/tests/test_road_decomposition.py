"""Dependency cuts must preserve side inputs, skip edges and state outputs."""
from pathlib import Path
import sys
import unittest

import numpy as np
import onnx
from onnx import helper, TensorProto

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from decompose_road import split, Recompose


class RoadDecompositionTest(unittest.TestCase):
  def graph(self):
    value = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph([
      helper.make_node('Add', ['image', 'history'], ['prepared']),
      helper.make_node('Relu', ['prepared'], ['model/vision_outputs']),
      helper.make_node('Mul', ['prepared', 'side'], ['skip']),
      helper.make_node('Add', ['model/vision_outputs', 'skip'], ['outputs']),
      helper.make_node('Identity', ['prepared'], ['next_history'])], 'test',
      [value('image'), value('history'), value('side')], [value('outputs'), value('next_history')])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=10)

  def test_cut_retains_all_skip_edges_and_side_inputs(self):
    sections, manifest = split(self.graph())
    self.assertEqual(manifest['crossing_tensors'], ['model/vision_outputs', 'prepared'])
    runner = Recompose(sections, manifest['public_outputs'])
    inputs = {'image': np.array([[2., -4.]], dtype=np.float32),
              'history': np.array([[1., 2.]], dtype=np.float32),
              'side': np.array([[2., 3.]], dtype=np.float32)}
    import onnxruntime as ort
    original = ort.InferenceSession(self.graph().SerializeToString(), providers=['CPUExecutionProvider'])
    result = runner(inputs)
    for name, expected in zip(manifest['public_outputs'], original.run(None, inputs), strict=True):
      np.testing.assert_array_equal(result[name], expected)
    self.assertEqual([x['section'] for x in runner.timings[-2:]], ['vision', 'policy'])

  def test_missing_boundary_rejected(self):
    with self.assertRaisesRegex(ValueError, 'Missing road vision boundary'):
      split(self.graph(), 'absent')

  def test_public_output_in_vision_is_retained(self):
    model = self.graph()
    model.graph.output.append(helper.make_tensor_value_info('prepared', TensorProto.FLOAT, [1, 2]))
    sections, manifest = split(model)
    self.assertIn('prepared', manifest['public_outputs'])
    self.assertIn('prepared', [x.name for x in sections[0][1].graph.output])

  def test_half_boundary_is_not_promoted(self):
    value = lambda name, dtype: helper.make_tensor_value_info(name, dtype, [1, 2])
    graph = helper.make_graph([
      helper.make_node('Cast', ['input'], ['model/vision_outputs'], to=TensorProto.FLOAT16),
      helper.make_node('Cast', ['model/vision_outputs'], ['outputs'], to=TensorProto.FLOAT)],
      'rounding', [value('input', TensorProto.FLOAT)], [value('outputs', TensorProto.FLOAT)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=10)
    sections, manifest = split(model)
    self.assertEqual(manifest['tensor_contracts']['model/vision_outputs']['onnx_dtype'], TensorProto.FLOAT16)
    x = np.array([[1.0003, -0.333333]], dtype=np.float32)
    actual = Recompose(sections, manifest['public_outputs'])({'input': x})['outputs']
    np.testing.assert_array_equal(actual, x.astype(np.float16).astype(np.float32))
    self.assertFalse(np.array_equal(actual, x))


if __name__ == '__main__':
  unittest.main()
