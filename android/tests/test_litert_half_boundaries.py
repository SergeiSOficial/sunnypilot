import sys
import unittest
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s
from ai_edge_litert.interpreter import Interpreter, OpResolverType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from patch_litert_half_boundaries import patch


def tiny_model():
  model = s.ModelT()
  model.version = 3
  model.buffers = [s.BufferT()]
  graph = s.SubGraphT()
  graph.name = 'main'
  def tensor(name, shape, data=None):
    value = s.TensorT()
    value.name, value.shape, value.type = name, np.array(shape, np.int32), s.TensorType.FLOAT32
    value.buffer = 0
    if data is not None:
      value.buffer = len(model.buffers)
      buffer = s.BufferT()
      buffer.data = np.frombuffer(np.asarray(data, np.float32).tobytes(), np.uint8)
      model.buffers.append(buffer)
    return value
  graph.tensors = [tensor('input', [1, 2]), tensor('weight', [1, 2], [[0.1, 0.2]]),
                   tensor('bias', [1], [0.03]), tensor('dense', [1, 1]), tensor('output', [1, 1])]
  graph.inputs, graph.outputs = np.array([0], np.int32), np.array([4], np.int32)
  fc, relu = s.OperatorCodeT(), s.OperatorCodeT()
  fc.builtinCode = fc.deprecatedBuiltinCode = s.BuiltinOperator.FULLY_CONNECTED
  relu.builtinCode = relu.deprecatedBuiltinCode = s.BuiltinOperator.RELU
  fc.version = relu.version = 1
  model.operatorCodes = [fc, relu]
  dense, activation = s.OperatorT(), s.OperatorT()
  dense.opcodeIndex, dense.inputs, dense.outputs = 0, np.array([0, 1, 2], np.int32), np.array([3], np.int32)
  dense.builtinOptionsType, dense.builtinOptions = s.BuiltinOptions.FullyConnectedOptions, s.FullyConnectedOptionsT()
  activation.opcodeIndex, activation.inputs, activation.outputs = 1, np.array([3], np.int32), np.array([4], np.int32)
  graph.operators = [dense, activation]
  model.subgraphs = [graph]
  builder = flatbuffers.Builder(1024)
  builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
  return bytes(builder.Output())


class LiteRTHalfBoundariesTest(unittest.TestCase):
  def test_rounding_only_selected_consumer(self):
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(tiny_model(), 0))
    graph = model.subgraphs[0]
    second = s.TensorT()
    second.name, second.shape, second.type, second.buffer = 'other_output', np.array([1, 1], np.int32), s.TensorType.FLOAT32, 0
    graph.tensors.append(second)
    op = s.OperatorT()
    op.opcodeIndex, op.inputs, op.outputs = 1, np.array([3], np.int32), np.array([5], np.int32)
    graph.operators.append(op)
    graph.outputs = np.array([4, 5], np.int32)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    source = bytes(builder.Output())
    modified = patch(source, ['dense'], consumer_outputs=['output'])
    runner = Interpreter(model_content=modified, num_threads=1,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    runner.allocate_tensors()
    runner.set_tensor(runner.get_input_details()[0]['index'], np.array([[0.7, 0.8]], np.float32))
    runner.invoke()
    values = {d['name']: runner.get_tensor(d['index']) for d in runner.get_output_details()}
    self.assertNotEqual(values['output'].item(), values['other_output'].item())
    np.testing.assert_array_equal(values['output'], values['other_output'].astype(np.float16).astype(np.float32))
    with self.assertRaisesRegex(ValueError, 'not a direct consumer'):
      patch(source, ['dense'], consumer_outputs=['input'])

  def test_rounding_is_executed_and_source_unchanged(self):
    source = tiny_model()
    modified = patch(source, ['dense'])
    runner = Interpreter(model_content=modified, num_threads=1,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    runner.allocate_tensors()
    runner.set_tensor(runner.get_input_details()[0]['index'], np.array([[0.7, 0.8]], np.float32))
    runner.invoke()
    actual = runner.get_tensor(runner.get_output_details()[0]['index'])
    original = np.float32(np.float32(0.7) * np.float32(0.1) + np.float32(0.8) * np.float32(0.2) + np.float32(0.03))
    np.testing.assert_array_equal(actual, np.array([[original]], np.float16).astype(np.float32))
    graph = s.ModelT.InitFromObj(s.Model.GetRootAsModel(source, 0)).subgraphs[0]
    self.assertEqual(len(graph.operators), 2)

  def test_rejects_invalid_targets_and_repeated_patch(self):
    source = tiny_model()
    for names in ([], ['missing'], ['dense', 'dense'], ['output']):
      with self.subTest(names=names), self.assertRaises(ValueError):
        patch(source, names)
    with self.assertRaises(ValueError):
      patch(patch(source, ['dense']), ['dense'])

  def test_rounding_after_explicit_sqrt(self):
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(tiny_model(), 0))
    graph = model.subgraphs[0]
    graph.tensors[0].shape = np.array([1, 1], np.int32)
    model.operatorCodes[0].builtinCode = model.operatorCodes[0].deprecatedBuiltinCode = s.BuiltinOperator.SQRT
    graph.operators[0].inputs = np.array([0], np.int32)
    graph.operators[0].builtinOptionsType = s.BuiltinOptions.NONE
    graph.operators[0].builtinOptions = None
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    source = bytes(builder.Output())
    with self.assertRaises(ValueError):
      patch(source, ['dense'])
    modified = patch(source, ['dense'], s.BuiltinOperator.SQRT)
    runner = Interpreter(model_content=modified, num_threads=1,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    runner.allocate_tensors()
    runner.set_tensor(runner.get_input_details()[0]['index'], np.array([[2.0]], np.float32))
    runner.invoke()
    expected = np.array([[np.sqrt(np.float32(2.0))]], np.float16).astype(np.float32)
    np.testing.assert_array_equal(runner.get_tensor(runner.get_output_details()[0]['index']), expected)

  def test_rounding_after_explicit_mul(self):
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(tiny_model(), 0))
    graph = model.subgraphs[0]
    graph.tensors[0].shape = np.array([1, 1], np.int32)
    model.operatorCodes[0].builtinCode = model.operatorCodes[0].deprecatedBuiltinCode = s.BuiltinOperator.MUL
    graph.operators[0].inputs = np.array([0, 2], np.int32)
    graph.operators[0].builtinOptionsType = s.BuiltinOptions.MulOptions
    graph.operators[0].builtinOptions = s.MulOptionsT()
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    source = bytes(builder.Output())
    with self.assertRaises(ValueError):
      patch(source, ['dense'])
    runner = Interpreter(model_content=patch(source, ['dense'], s.BuiltinOperator.MUL), num_threads=1,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    runner.allocate_tensors()
    runner.set_tensor(runner.get_input_details()[0]['index'], np.array([[0.7]], np.float32))
    runner.invoke()
    expected = np.array([[np.float32(0.7) * np.float32(0.03)]], np.float16).astype(np.float32)
    np.testing.assert_array_equal(runner.get_tensor(runner.get_output_details()[0]['index']), expected)

  def test_rounding_after_explicit_concat(self):
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(tiny_model(), 0))
    graph = model.subgraphs[0]
    graph.tensors[0].shape = np.array([1, 1], np.int32)
    model.operatorCodes[0].builtinCode = model.operatorCodes[0].deprecatedBuiltinCode = s.BuiltinOperator.CONCATENATION
    graph.operators[0].inputs = np.array([0], np.int32)
    graph.operators[0].builtinOptionsType = s.BuiltinOptions.ConcatenationOptions
    graph.operators[0].builtinOptions = s.ConcatenationOptionsT()
    graph.operators[0].builtinOptions.axis = 1
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    source = bytes(builder.Output())
    runner = Interpreter(model_content=patch(source, ['dense'], s.BuiltinOperator.CONCATENATION), num_threads=1,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    runner.allocate_tensors()
    runner.set_tensor(runner.get_input_details()[0]['index'], np.array([[0.7]], np.float32))
    runner.invoke()
    np.testing.assert_array_equal(runner.get_tensor(runner.get_output_details()[0]['index']),
                                  np.array([[0.7]], np.float16).astype(np.float32))

  def test_rounding_after_explicit_add(self):
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(tiny_model(), 0))
    graph = model.subgraphs[0]
    graph.tensors[0].shape = np.array([1, 1], np.int32)
    model.operatorCodes[0].builtinCode = model.operatorCodes[0].deprecatedBuiltinCode = s.BuiltinOperator.ADD
    graph.operators[0].inputs = np.array([0, 2], np.int32)
    graph.operators[0].builtinOptionsType = s.BuiltinOptions.AddOptions
    graph.operators[0].builtinOptions = s.AddOptionsT()
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    source = bytes(builder.Output())
    runner = Interpreter(model_content=patch(source, ['dense'], s.BuiltinOperator.ADD), num_threads=1,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    runner.allocate_tensors()
    runner.set_tensor(runner.get_input_details()[0]['index'], np.array([[0.7]], np.float32))
    runner.invoke()
    expected = np.array([[np.float32(0.7) + np.float32(0.03)]], np.float16).astype(np.float32)
    np.testing.assert_array_equal(runner.get_tensor(runner.get_output_details()[0]['index']), expected)

  def test_rounding_after_explicit_reshape(self):
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(tiny_model(), 0))
    graph = model.subgraphs[0]
    graph.tensors[0].shape = np.array([1, 1], np.int32)
    model.operatorCodes[0].builtinCode = model.operatorCodes[0].deprecatedBuiltinCode = s.BuiltinOperator.RESHAPE
    graph.operators[0].inputs = np.array([0], np.int32)
    graph.operators[0].builtinOptionsType = s.BuiltinOptions.ReshapeOptions
    graph.operators[0].builtinOptions = s.ReshapeOptionsT()
    graph.operators[0].builtinOptions.newShape = np.array([1, 1], np.int32)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    runner = Interpreter(model_content=patch(bytes(builder.Output()), ['dense'], s.BuiltinOperator.RESHAPE),
                         num_threads=1, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    runner.allocate_tensors()
    runner.set_tensor(runner.get_input_details()[0]['index'], np.array([[0.7]], np.float32))
    runner.invoke()
    np.testing.assert_array_equal(runner.get_tensor(runner.get_output_details()[0]['index']),
                                  np.array([[0.7]], np.float16).astype(np.float32))


if __name__ == '__main__':
  unittest.main()
