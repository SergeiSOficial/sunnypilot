#!/usr/bin/env python3
"""Replace a proven byte-domain normalization with an exact diagnostic lookup."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s


def patch(data, flatten_indices=False):
  model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(data, 0))
  if len(model.subgraphs) != 1:
    raise ValueError('Expected one subgraph')
  graph = model.subgraphs[0]
  def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)
  names = {text(t.name): i for i, t in enumerate(graph.tensors)}
  if len(names) != len(graph.tensors):
    raise ValueError('Duplicate tensor names')
  cat, sub, div = (names[name] for name in ('model/cat', 'InsertedPrecisionFreeCast_model/sub',
                                             'InsertedPrecisionFreeCast_model/div'))
  if any(name in names for name in ('fold3_diagnostic_byte_index', 'fold3_diagnostic_normalization_table')):
    raise ValueError('Already patched')
  def find_producer(index):
    matches = [op for op in graph.operators if index in op.outputs]
    if len(matches) != 1:
      raise ValueError('Tensor must have exactly one producer')
    return matches[0]
  def constant(index):
    tensor = graph.tensors[int(index)]
    if tensor.type != s.TensorType.FLOAT32 or tensor.buffer == 0:
      raise ValueError('Expected float32 constant')
    return np.frombuffer(bytes(model.buffers[tensor.buffer].data), dtype='<f4')
  sub_op, div_op = find_producer(sub), find_producer(div)
  if (sub_op.builtinOptions is None or div_op.builtinOptions is None or
      sub_op.builtinOptions.fusedActivationFunction != s.ActivationFunctionType.NONE or
      div_op.builtinOptions.fusedActivationFunction != s.ActivationFunctionType.NONE):
    raise ValueError('Normalization must not have fused activations')
  if (model.operatorCodes[sub_op.opcodeIndex].builtinCode != s.BuiltinOperator.SUB or
      list(sub_op.inputs[:1]) != [cat] or len(sub_op.inputs) != 2 or
      not np.all(constant(sub_op.inputs[1]) == np.float32(127.5))):
    raise ValueError('Unexpected byte subtraction')
  if (model.operatorCodes[div_op.opcodeIndex].builtinCode != s.BuiltinOperator.MUL or
      list(div_op.inputs[:1]) != [sub] or len(div_op.inputs) != 2 or
      not np.all(constant(div_op.inputs[1]) == np.float32(1.0 / 63.75))):
    raise ValueError('Unexpected reciprocal multiplication')
  if any(graph.tensors[i].type != s.TensorType.FLOAT32 for i in (cat, sub, div)):
    raise ValueError('Unexpected normalization tensor types')
  if not (np.array_equal(graph.tensors[cat].shape, graph.tensors[sub].shape) and
          np.array_equal(graph.tensors[cat].shape, graph.tensors[div].shape)):
    raise ValueError('Normalization shapes differ')
  # Original inputs and histories are uint8. The float transport must carry
  # only integral 0..255 values; this diagnostic is invalid for other inputs.
  values = np.arange(256, dtype=np.float32)
  table_values = np.divide(values - np.float32(127.5), np.float32(63.75), dtype=np.float32)
  buffer = s.BufferT()
  buffer.data = np.frombuffer(table_values.astype('<f4').tobytes(), dtype=np.uint8)
  buffer_id = len(model.buffers)
  model.buffers.append(buffer)
  indices = copy.deepcopy(graph.tensors[cat])
  indices.name, indices.type, indices.buffer = 'fold3_diagnostic_byte_index', s.TensorType.INT32, 0
  table = s.TensorT()
  table.name, table.type, table.buffer = 'fold3_diagnostic_normalization_table', s.TensorType.FLOAT32, buffer_id
  table.shape = np.array([256], np.int32)
  index_id = len(graph.tensors)
  table_id = index_id + 1
  graph.tensors.extend([indices, table])
  def opcode(code):
    result = s.OperatorCodeT()
    result.builtinCode = result.deprecatedBuiltinCode = code
    result.version = 1
    model.operatorCodes.append(result)
    return len(model.operatorCodes) - 1
  cast_input, gather_output = cat, div
  extra_before, extra_after = [], []
  if flatten_indices:
    shape = list(graph.tensors[cat].shape)
    if not shape or any(int(d) <= 0 for d in shape):
      raise ValueError('Static positive dimensions required for diagnostic flattening')
    flattened = copy.deepcopy(graph.tensors[cat])
    flattened.name, flattened.shape = 'fold3_diagnostic_flat_input', np.array([np.prod(shape)], np.int32)
    flattened.shapeSignature = None
    indices.shape, indices.shapeSignature = flattened.shape.copy(), None
    gathered = copy.deepcopy(flattened)
    gathered.name = 'fold3_diagnostic_flat_output'
    flat_id, result_id = len(graph.tensors), len(graph.tensors) + 1
    graph.tensors.extend([flattened, gathered])
    def reshape(input_id, output_id, new_shape):
      op = s.OperatorT()
      op.opcodeIndex = opcode(s.BuiltinOperator.RESHAPE)
      op.inputs, op.outputs = np.array([input_id], np.int32), np.array([output_id], np.int32)
      op.builtinOptionsType, op.builtinOptions = s.BuiltinOptions.ReshapeOptions, s.ReshapeOptionsT()
      op.builtinOptions.newShape = np.array(new_shape, np.int32)
      return op
    extra_before.append(reshape(cat, flat_id, [np.prod(shape)]))
    extra_after.append(reshape(result_id, div, shape))
    cast_input, gather_output = flat_id, result_id
  cast = s.OperatorT()
  cast.opcodeIndex, cast.inputs, cast.outputs = opcode(s.BuiltinOperator.CAST), np.array([cast_input], np.int32), np.array([index_id], np.int32)
  cast.builtinOptionsType, cast.builtinOptions = s.BuiltinOptions.CastOptions, s.CastOptionsT()
  cast.builtinOptions.inDataType, cast.builtinOptions.outDataType = s.TensorType.FLOAT32, s.TensorType.INT32
  gather = s.OperatorT()
  gather.opcodeIndex = opcode(s.BuiltinOperator.GATHER)
  gather.inputs, gather.outputs = np.array([table_id, index_id], np.int32), np.array([gather_output], np.int32)
  gather.builtinOptionsType, gather.builtinOptions = s.BuiltinOptions.GatherOptions, s.GatherOptionsT()
  gather.builtinOptions.axis, gather.builtinOptions.batchDims = 0, 0
  operations = []
  for op in graph.operators:
    operations.extend([*extra_before, cast, gather, *extra_after] if op is div_op else [op])
  graph.operators = operations
  builder = flatbuffers.Builder(1024)
  builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
  return bytes(builder.Output())


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source', type=Path, required=True)
  parser.add_argument('--source-sha256', required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--flatten-indices', action='store_true', help='Diagnostic GPU gather layout')
  args = parser.parse_args()
  if args.output.exists():
    raise ValueError('Preserve prior artifact')
  source = args.source.read_bytes()
  if hashlib.sha256(source).hexdigest() != args.source_sha256:
    raise ValueError('Source hash mismatch')
  result = patch(source, flatten_indices=args.flatten_indices)
  args.output.write_bytes(result)
  print(json.dumps({'source_sha256': args.source_sha256, 'derived_sha256': hashlib.sha256(result).hexdigest(),
                    'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
                    'domain': 'integral byte inputs only'}, indent=2))
