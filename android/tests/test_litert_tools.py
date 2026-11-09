"""Explicit precision and layout contracts for the diagnostic LiteRT converter."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
try:
  from prepare_litert_probes import lower, s, Interpreter
except ModuleNotFoundError as error:
  if error.name not in ('ai_edge_litert', 'flatbuffers'): raise
  lower = None
from prepare_snpe_probes import prepare


@unittest.skipIf(lower is None, 'Run with Chromebox litert-venv for LiteRT conversion')
class LiteRTToolsTest(unittest.TestCase):
  def test_half_rounding_with_float_transport(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp); prepare(root)
      data = lower(root/'fp16-conv.onnx', explicit_half_conv=True, float_transport=True)
      graph = s.Model.GetRootAsModel(data, 0).Subgraphs(0)
      self.assertEqual(graph.Tensors(graph.Inputs(0)).Type(), s.TensorType.FLOAT32)
      self.assertEqual(graph.Tensors(graph.Outputs(0)).Type(), s.TensorType.FLOAT32)
      codes = s.Model.GetRootAsModel(data, 0)
      operators = [codes.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode()
                   for i in range(graph.OperatorsLength())]
      self.assertEqual(operators, [s.BuiltinOperator.CAST, s.BuiltinOperator.TRANSPOSE,
        s.BuiltinOperator.CAST, s.BuiltinOperator.CONV_2D, s.BuiltinOperator.CAST,
        s.BuiltinOperator.TRANSPOSE, s.BuiltinOperator.CAST])
      runtime = Interpreter(model_content=data, num_threads=1, experimental_delegates=[])
      runtime.allocate_tensors()
      with np.load(root/'fp16-conv.npz') as fixture:
        runtime.set_tensor(runtime.get_input_details()[0]['index'], fixture['x'].astype(np.float32))
        runtime.invoke()
        np.testing.assert_allclose(runtime.get_tensor(runtime.get_output_details()[0]['index']),
                                   fixture['expected'], rtol=1e-3, atol=1e-4)

  def test_unsupported_geometry_is_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp); prepare(root)
      path = root/'fp16-conv.onnx'; model = onnx.load(path)
      model.graph.node[0].attribute.append(onnx.helper.make_attribute('strides', [2, 2]))
      onnx.save(model, path)
      with self.assertRaisesRegex(ValueError, 'Unsupported Conv geometry'):
        lower(path, explicit_half_conv=True, float_transport=True)

  def test_float_transport_cannot_remove_half_boundaries(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp); prepare(root)
      with self.assertRaisesRegex(ValueError, 'requires explicit half Conv'):
        lower(root/'fp16-conv.onnx', float_transport=True)


if __name__ == '__main__': unittest.main()
