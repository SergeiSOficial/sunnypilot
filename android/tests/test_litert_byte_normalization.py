import sys
import unittest
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s
from ai_edge_litert.interpreter import Interpreter, OpResolverType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from patch_litert_byte_normalization import patch


def tiny_model(fused=False, shape=(256,)):
  model = s.ModelT()
  model.version = 3
  model.buffers = [s.BufferT()]
  graph = s.SubGraphT()
  graph.name = 'main'
  def tensor(name, shape, value=None):
    t = s.TensorT()
    t.name, t.shape, t.type, t.buffer = name, np.array(shape, np.int32), s.TensorType.FLOAT32, 0
    if value is not None:
      t.buffer = len(model.buffers)
      buf = s.BufferT()
      buf.data = np.frombuffer(np.asarray([value], dtype='<f4').tobytes(), np.uint8)
      model.buffers.append(buf)
    return t
  graph.tensors = [tensor('model/cat', shape), tensor('offset', [1], 127.5),
                   tensor('InsertedPrecisionFreeCast_model/sub', shape), tensor('scale', [1], 1 / 63.75),
                   tensor('InsertedPrecisionFreeCast_model/div', shape)]
  graph.inputs, graph.outputs = np.array([0], np.int32), np.array([4], np.int32)
  model.operatorCodes = []
  graph.operators = []
  for index, (code, inputs, output) in enumerate(((s.BuiltinOperator.SUB, [0, 1], 2), (s.BuiltinOperator.MUL, [2, 3], 4))):
    opcode = s.OperatorCodeT()
    opcode.builtinCode = opcode.deprecatedBuiltinCode = code
    opcode.version = 1
    model.operatorCodes.append(opcode)
    op = s.OperatorT()
    op.opcodeIndex, op.inputs, op.outputs = index, np.array(inputs, np.int32), np.array([output], np.int32)
    if code == s.BuiltinOperator.SUB:
      op.builtinOptionsType, op.builtinOptions = s.BuiltinOptions.SubOptions, s.SubOptionsT()
      if fused:
        op.builtinOptions.fusedActivationFunction = s.ActivationFunctionType.RELU
    else:
      op.builtinOptionsType, op.builtinOptions = s.BuiltinOptions.MulOptions, s.MulOptionsT()
    graph.operators.append(op)
  model.subgraphs = [graph]
  builder = flatbuffers.Builder(1024)
  builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
  return bytes(builder.Output())


class ByteNormalizationTest(unittest.TestCase):
  def test_exact_all_byte_values_and_source_unchanged(self):
    source = tiny_model()
    modified = patch(source)
    runtime = Interpreter(model_content=modified, num_threads=1,
                          experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    runtime.allocate_tensors()
    values = np.arange(256, dtype=np.float32)
    runtime.set_tensor(runtime.get_input_details()[0]['index'], values)
    runtime.invoke()
    expected = np.divide(values - np.float32(127.5), np.float32(63.75), dtype=np.float32)
    np.testing.assert_array_equal(runtime.get_tensor(runtime.get_output_details()[0]['index']), expected)
    self.assertEqual(len(s.ModelT.InitFromObj(s.Model.GetRootAsModel(source, 0)).subgraphs[0].operators), 2)

  def test_reject_fused_and_repeated_patch(self):
    with self.assertRaises(ValueError):
      patch(tiny_model(fused=True))
    with self.assertRaises(ValueError):
      patch(patch(tiny_model()))

  def test_flattened_lookup_matches_byte_domain(self):
    runtime = Interpreter(model_content=patch(tiny_model(shape=(16, 16)), flatten_indices=True), num_threads=1,
                          experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    runtime.allocate_tensors()
    values = np.arange(256, dtype=np.float32).reshape(16, 16)
    runtime.set_tensor(runtime.get_input_details()[0]['index'], values)
    runtime.invoke()
    np.testing.assert_array_equal(runtime.get_tensor(runtime.get_output_details()[0]['index']),
                                  (values - np.float32(127.5)) / np.float32(63.75))


if __name__ == '__main__':
  unittest.main()
