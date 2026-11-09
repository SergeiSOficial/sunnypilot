#!/usr/bin/env python3
"""Insert diagnostic FP16 round trips at selected LiteRT operation outputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s


def digest(data):
  return hashlib.sha256(data).hexdigest()


def patch(data, names, producer=s.BuiltinOperator.FULLY_CONNECTED):
  if not names or len(set(names)) != len(names):
    raise ValueError('Tensor names must be nonempty and unique')
  if producer not in (s.BuiltinOperator.FULLY_CONNECTED, s.BuiltinOperator.SQRT,
                      s.BuiltinOperator.MUL, s.BuiltinOperator.CONCATENATION, s.BuiltinOperator.ADD,
                      s.BuiltinOperator.RESHAPE):
    raise ValueError('Unsupported diagnostic producer')
  model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(data, 0))
  if len(model.subgraphs) != 1:
    raise ValueError('Only a single subgraph is supported')
  graph = model.subgraphs[0]
  def text(name):
    return name.decode() if isinstance(name, bytes) else str(name)
  ids = {text(t.name): i for i, t in enumerate(graph.tensors)}
  if not set(names) <= ids.keys() or len(ids) != len(graph.tensors):
    raise ValueError('Missing or duplicate tensor names')
  if any(name + suffix in ids for name in names for suffix in ('__diagnostic_half', '__diagnostic_rounded')):
    raise ValueError('Diagnostic tensor name already exists')
  target_ids = {ids[name] for name in names}
  if target_ids & set(graph.outputs):
    raise ValueError('Graph output cannot be patched')
  producers = {}
  for index, operator in enumerate(graph.operators):
    for output in operator.outputs:
      if int(output) in target_ids:
        if int(output) in producers:
          raise ValueError('Target has multiple producers')
        code = model.operatorCodes[operator.opcodeIndex].builtinCode
        if code != producer or len(operator.outputs) != 1:
          raise ValueError('Target must have the requested single-output producer')
        producers[int(output)] = index
  if set(producers) != target_ids:
    raise ValueError('Target producer missing')
  code = s.OperatorCodeT()
  code.builtinCode = s.BuiltinOperator.CAST
  code.deprecatedBuiltinCode = min(code.builtinCode, 127)
  code.version = 1
  cast_code = len(model.operatorCodes)
  model.operatorCodes.append(code)
  replacements = {}
  operations = []
  for operator in graph.operators:
    operator.inputs = np.array([replacements.get(int(i), int(i)) for i in operator.inputs], dtype=np.int32)
    operations.append(operator)
    for output in operator.outputs:
      index = int(output)
      if index not in target_ids:
        continue
      original = graph.tensors[index]
      if original.type != s.TensorType.FLOAT32:
        raise ValueError('Target must be float32')
      half = copy.deepcopy(original)
      half.name = text(original.name) + '__diagnostic_half'
      half.type = s.TensorType.FLOAT16
      half.buffer = 0
      rounded = copy.deepcopy(original)
      rounded.name = text(original.name) + '__diagnostic_rounded'
      rounded.buffer = 0
      half_id = len(graph.tensors)
      graph.tensors.extend([half, rounded])
      rounded_id = half_id + 1
      for inp, out, in_type, out_type in ((index, half_id, s.TensorType.FLOAT32, s.TensorType.FLOAT16),
                                           (half_id, rounded_id, s.TensorType.FLOAT16, s.TensorType.FLOAT32)):
        options = s.CastOptionsT()
        options.inDataType, options.outDataType = in_type, out_type
        cast = s.OperatorT()
        cast.opcodeIndex = cast_code
        cast.inputs = np.array([inp], dtype=np.int32)
        cast.outputs = np.array([out], dtype=np.int32)
        cast.builtinOptionsType = s.BuiltinOptions.CastOptions
        cast.builtinOptions = options
        operations.append(cast)
      replacements[index] = rounded_id
  for original, replacement in replacements.items():
    if not any(replacement in op.inputs for op in operations):
      raise ValueError('Target has no downstream consumer')
  graph.operators = operations
  builder = flatbuffers.Builder(1024)
  result = model.Pack(builder)
  builder.Finish(result, file_identifier=b'TFL3')
  return bytes(builder.Output())


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source', type=Path, required=True)
  parser.add_argument('--source-sha256', required=True)
  parser.add_argument('--tensor', action='append', required=True)
  parser.add_argument('--producer', choices=('fully-connected', 'sqrt', 'mul', 'concat', 'add', 'reshape'), default='fully-connected')
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise ValueError('Preserve existing artifact')
  source = args.source.read_bytes()
  if digest(source) != args.source_sha256:
    raise ValueError('Source hash mismatch')
  producer = {'fully-connected': s.BuiltinOperator.FULLY_CONNECTED, 'sqrt': s.BuiltinOperator.SQRT,
              'mul': s.BuiltinOperator.MUL, 'concat': s.BuiltinOperator.CONCATENATION,
              'add': s.BuiltinOperator.ADD, 'reshape': s.BuiltinOperator.RESHAPE}[args.producer]
  result = patch(source, args.tensor, producer)
  args.output.write_bytes(result)
  print(json.dumps({'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
                    'source_sha256': digest(source), 'derived_sha256': digest(result), 'tensors': args.tensor,
                    'producer': args.producer}, indent=2))


if __name__ == '__main__':
  main()
