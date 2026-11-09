#!/usr/bin/env python3
"""Compare a converted layer normalization with ONNX on identical inputs.

This is a CPU diagnostic, not evidence of accelerator qualification.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from onnx import numpy_helper

from decompose_road import ROOT, digest
from fold3_model_trace import compare
from fold3_sequences import initial_inputs
from isolate_litert_suffix_gemm import compare_at_reference_precision
from prepare_snpe_dense_probe import ordered_half_gemv
from validate_litert_dense_sections_host import run_converted, run_onnx, session


def capture_half_input(actual, expected):
  """Require the converted input to have already crossed the FP16 boundary."""
  if actual.shape != expected.shape or expected.dtype != np.float16 or actual.dtype not in (np.float16, np.float32):
    raise ValueError('Unexpected normalization input shape or dtype')
  with np.errstate(over='ignore', invalid='ignore'):
    cast = actual.astype(np.float16)
  if not np.isfinite(actual).all() or not np.array_equal(cast.astype(actual.dtype), actual):
    raise ValueError('Converted normalization input does not preserve the original FP16 boundary')
  return cast


def isolate(sections, conversion, model, model_hash, output, seed):
  if output.exists():
    raise ValueError('Preserve prior evidence')
  manifest = json.loads((sections / 'manifest.json').read_text())
  receipt = json.loads((conversion / 'receipt.json').read_text())
  source = sections / 'suffix.onnx'
  if digest(source) != manifest['sections']['suffix']['sha256'] or digest(source) != receipt['source_sha256']:
    raise ValueError('Suffix graph provenance mismatch')
  if digest(model) != model_hash:
    raise ValueError('Diagnostic graph hash mismatch')
  refs = ROOT / 'android/app/build/model-reference'
  original = ROOT / 'openpilot/selfdrive/modeld/models/driving_supercombo.onnx'
  original_hash = json.loads((refs / 'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  if manifest['original_sha256'] != original_hash or digest(original) != original_hash:
    raise ValueError('Original graph provenance mismatch')
  for name in ('prefix', 'dense'):
    if digest(sections / (name + '.onnx')) != manifest['sections'][name]['sha256']:
      raise ValueError('Boundary provenance mismatch')
  graph = onnx.shape_inference.infer_shapes(onnx.load(source))
  nodes = [n for n in graph.graph.node if n.name == 'model/p_node_layer_norm_1' and n.op_type == 'LayerNormalization']
  if len(nodes) != 1:
    raise ValueError('Expected one layer normalization')
  node = nodes[0]
  attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
  if len(node.input) != 3 or len(node.output) != 1 or attrs.get('axis') != -1 or attrs.get('stash_type') != 1:
    raise ValueError('Unexpected layer normalization contract')
  infos = {v.name: v for v in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
  if any(name not in infos for name in (node.input[0], node.output[0])):
    raise ValueError('Missing normalization shape or dtype')
  for name in (node.input[0], node.output[0]):
    if name not in {v.name for v in graph.graph.output}:
      graph.graph.output.append(copy.deepcopy(infos[name]))
  isolated = copy.deepcopy(graph)
  isolated.graph.ClearField('node')
  isolated.graph.node.append(copy.deepcopy(node))
  isolated.graph.ClearField('input')
  isolated.graph.input.append(copy.deepcopy(infos[node.input[0]]))
  isolated.graph.ClearField('output')
  isolated.graph.output.append(copy.deepcopy(infos[node.output[0]]))
  parameters = {v.name: v for v in isolated.graph.initializer}
  if any(name not in parameters for name in node.input[1:]):
    raise ValueError('Normalization parameters must be constants')
  isolated.graph.ClearField('initializer')
  isolated.graph.initializer.extend(copy.deepcopy(parameters[name]) for name in node.input[1:])
  output.mkdir(parents=True)
  tapped_path, isolated_path = output / 'suffix-tapped.onnx', output / 'layernorm.onnx'
  onnx.save(graph, tapped_path)
  onnx.save(isolated, isolated_path)
  values = initial_inputs(refs, 'driving_supercombo', seed)
  if 'rewritten_source_sha256' in manifest:
    values['new_img'] = values['new_img'].astype(np.float32)
    values['state_img_q'] = values['state_img_q'].astype(np.float32).reshape(2, 30, 128, 256)
  prefix = run_onnx(session(sections / 'prefix.onnx'), values)
  dense = onnx.load(sections / 'dense.onnx').graph
  weights = {v.name: numpy_helper.to_array(v) for v in dense.initializer}
  values.update(prefix)
  values['model/linear'] = ordered_half_gemv(prefix['model/view'], *(weights[n] for n in dense.node[0].input[1:]))
  untapped = run_onnx(session(source), values)
  expected = run_onnx(session(tapped_path), values)
  runtime = Interpreter(model_path=str(model), num_threads=2, experimental_preserve_all_tensors=True,
                        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
  runtime.allocate_tensors()
  run_converted(runtime, {d['name']: values[d['name']] for d in runtime.get_input_details()})
  tensors = {d['name']: d for d in runtime.get_tensor_details()}
  names = ('model/p_node_layer_norm_1_layernorm_mean', 'model/p_node_layer_norm_1_layernorm_centered')
  operations = runtime._get_ops_details()
  def producer(name):
    matches = [op for op in operations if tensors[name]['index'] in op['outputs']]
    if len(matches) != 1:
      raise ValueError('Unexpected converted graph for ' + name)
    return matches[0]
  mean, centered = (producer(name) for name in names)
  if mean['op_name'] != 'MEAN' or centered['op_name'] != 'SUB' or mean['inputs'][0] != centered['inputs'][0]:
    raise ValueError('Unexpected converted normalization inputs')
  by_index = {d['index']: d for d in tensors.values()}
  input_details = by_index[int(mean['inputs'][0])]
  actual_input = runtime.get_tensor(input_details['index'])
  expected_input = expected[node.input[0]]
  cast_input = capture_half_input(actual_input, expected_input)
  parameter_comparisons = {}
  for name in node.input[1:]:
    parameter_comparisons[name] = compare(runtime.get_tensor(tensors[name]['index']), numpy_helper.to_array(parameters[name]))
    if not parameter_comparisons[name]['exact']:
      raise ValueError('Converted normalization parameters differ from the original')
  reference_session = session(isolated_path)
  isolated_actual = run_onnx(reference_session, {node.input[0]: cast_input})[node.output[0]]
  isolated_original = run_onnx(reference_session, {node.input[0]: expected_input})[node.output[0]]
  actual_output = runtime.get_tensor(tensors[node.output[0]]['index'])
  result = {'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
            'provider': 'host LiteRT CPU and tapped ONNX CPU; not GPU evidence',
            'warning': 'Tapping ONNX outputs may change fusion', 'seed': seed,
            'original_sha256': original_hash, 'source_sha256': digest(source), 'model_sha256': digest(model),
            'tapped_sha256': digest(tapped_path), 'isolated_sha256': digest(isolated_path),
            'onnx_node': node.name, 'onnx_attributes': attrs,
            'litert_input': input_details['name'], 'litert_mean_operator_index': mean['index'],
            'parameters': parameter_comparisons,
            'input_vs_tapped_original': compare(cast_input, expected_input),
            'litert_rounded_vs_onnx_identical_inputs': compare_at_reference_precision(actual_output, isolated_actual),
            'litert_rounded_vs_tapped_original': compare_at_reference_precision(actual_output, expected[node.output[0]]),
            'isolated_vs_tapped_original_inputs': compare(isolated_original, expected[node.output[0]]),
            'tapped_vs_untapped_public_outputs': {name: compare(expected[name], value) for name, value in untapped.items()}}
  (output / 'isolation.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('sections', 'conversion', 'model', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
  parser.add_argument('--model-sha256', required=True)
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()
  isolate(args.sections, args.conversion, args.model, args.model_sha256, args.output, args.seed)
