#!/usr/bin/env python3
"""Compare one LiteRT CPU convolution with ONNX on identical captured inputs."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from onnx import numpy_helper

from decompose_road import ROOT, digest
from fold3_model_trace import compare
from fold3_sequences import initial_inputs
from trace_litert_prefix import align, producer
from validate_litert_dense_sections_host import run_converted, run_onnx, session


def isolate(sections, conversion, model, output, node_name, converted_tensor, seed):
  if output.exists():
    raise ValueError('Use a new output directory')
  receipt = json.loads((conversion / 'receipt.json').read_text())
  manifest = json.loads((sections / 'manifest.json').read_text())
  source = sections / 'prefix.onnx'
  if digest(source) != receipt['source_sha256'] or digest(source) != manifest['sections']['prefix']['sha256']:
    raise ValueError('Source provenance mismatch')
  if digest(model) != receipt['outputs'].get(str(model.relative_to(conversion))):
    raise ValueError('Converted model provenance mismatch')
  refs = ROOT / 'android/app/build/model-reference'
  original = ROOT / 'openpilot/selfdrive/modeld/models/driving_supercombo.onnx'
  original_hash = json.loads((refs / 'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  if manifest['original_sha256'] != original_hash or digest(original) != original_hash:
    raise ValueError('Original model provenance mismatch')
  graph = onnx.shape_inference.infer_shapes(onnx.load(source))
  matches = [node for node in graph.graph.node if node.name == node_name and node.op_type == 'Conv']
  if len(matches) != 1:
    raise ValueError('Expected one named Conv')
  node = matches[0]
  if len(node.input) != 3 or len(node.output) != 1:
    raise ValueError('Expected convolution input, weight and bias')
  initializers = {value.name: value for value in graph.graph.initializer}
  if any(name not in initializers for name in node.input[1:]):
    raise ValueError('Convolution weights must be initializers')
  attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
  group = attrs.get('group', 1)
  weight_shape = tuple(initializers[node.input[1]].dims)
  if group != 1 and not (len(weight_shape) == 4 and weight_shape[0] == group and weight_shape[1] == 1):
    raise ValueError('Only ordinary and multiplier-one depthwise convolution are supported')
  value_info = {v.name: v for v in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
  for name in (node.input[0], node.output[0]):
    if name not in value_info:
      raise ValueError('Convolution shape information missing')
  existing = {v.name for v in graph.graph.output}
  for name in (node.input[0], node.output[0]):
    if name not in existing:
      graph.graph.output.append(value_info[name])
  isolated = copy.deepcopy(graph)
  isolated.graph.ClearField('node')
  isolated.graph.node.append(copy.deepcopy(node))
  isolated.graph.ClearField('input')
  isolated.graph.input.append(copy.deepcopy(value_info[node.input[0]]))
  isolated.graph.ClearField('output')
  isolated.graph.output.append(copy.deepcopy(value_info[node.output[0]]))
  isolated.graph.ClearField('initializer')
  isolated.graph.initializer.extend(copy.deepcopy(initializers[name]) for name in node.input[1:])
  output.mkdir(parents=True)
  tapped_path, isolated_path = output / 'prefix-tapped.onnx', output / 'conv.onnx'
  onnx.save(graph, tapped_path)
  onnx.save(isolated, isolated_path)
  inputs = initial_inputs(refs, 'driving_supercombo', seed)
  if 'rewritten_source_sha256' in manifest:
    inputs['new_img'] = inputs['new_img'].astype(np.float32)
    inputs['state_img_q'] = inputs['state_img_q'].astype(np.float32).reshape(2, 30, 128, 256)
  original_values = run_onnx(session(tapped_path), inputs)
  runtime = Interpreter(model_path=str(model), num_threads=2, experimental_preserve_all_tensors=True,
                        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
  runtime.allocate_tensors()
  run_converted(runtime, {d['name']: inputs[d['name']] for d in runtime.get_input_details()})
  tensors = {v['name']: v for v in runtime.get_tensor_details()}
  if converted_tensor not in tensors:
    raise ValueError('Converted output missing')
  operation = producer(runtime._get_ops_details(), tensors[converted_tensor]['index'])
  expected_operator = 'CONV_2D' if group == 1 else 'DEPTHWISE_CONV_2D'
  if operation['op_name'] != expected_operator or len(operation['inputs']) != 3:
    raise ValueError('Expected a LiteRT convolution with bias')
  raw_model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(model.read_bytes(), 0))
  options = raw_model.subgraphs[0].operators[operation['index']].builtinOptions
  if options.fusedActivationFunction != schema.ActivationFunctionType.NONE:
    raise ValueError('Fused activation is unsupported by this isolation')
  x = original_values[node.input[0]]
  y = original_values[node.output[0]]
  actual_x, _ = align(runtime.get_tensor(int(operation['inputs'][0])), x, True)
  actual_y, _ = align(runtime.get_tensor(tensors[converted_tensor]['index']), y, True)
  raw_weight = runtime.get_tensor(int(operation['inputs'][1]))
  actual_weight = raw_weight.transpose(0, 3, 1, 2) if group == 1 else raw_weight.transpose(3, 0, 1, 2)
  actual_bias = runtime.get_tensor(int(operation['inputs'][2]))
  weight, bias = (numpy_helper.to_array(initializers[name]) for name in node.input[1:])
  if not np.array_equal(weight, actual_weight) or not np.array_equal(bias, actual_bias):
    raise ValueError('Converted weight or bias differs from original')
  conv = session(isolated_path)
  on_original_input = run_onnx(conv, {node.input[0]: x})[node.output[0]]
  on_converted_input = run_onnx(conv, {node.input[0]: actual_x})[node.output[0]]
  report = {'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
            'provider': 'host LiteRT CPU and ONNX CPU', 'seed': seed, 'node': node_name,
            'converted_tensor': converted_tensor, 'actual_operator': operation['op_name'],
            'actual_operator_index': operation['index'], 'group': group,
            'original_sha256': original_hash, 'source_sha256': digest(source), 'model_sha256': digest(model),
            'tapped_sha256': digest(tapped_path), 'isolated_sha256': digest(isolated_path),
            'weights_exact': True, 'bias_exact': True,
            'input_comparison': compare(actual_x, x),
            'tapped_prefix_conv_vs_isolated_original_input': compare(on_original_input, y),
            'onnx_conv_on_converted_vs_original_input': compare(on_converted_input, on_original_input),
            'litert_vs_onnx_on_same_converted_input': compare(actual_y, on_converted_input),
            'litert_vs_original_prefix': compare(actual_y, y),
            'warning': 'Tapped graph and isolated convolution can change fusion and execution; CPU results do not establish GPU behavior.'}
  (output / 'isolation.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('sections', 'conversion', 'model', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
  parser.add_argument('--node', required=True)
  parser.add_argument('--converted-tensor', required=True)
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()
  isolate(args.sections, args.conversion, args.model, args.output, args.node, args.converted_tensor, args.seed)
