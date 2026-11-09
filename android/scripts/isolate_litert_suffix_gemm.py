#!/usr/bin/env python3
"""Isolate one converted suffix Gemm on identical captured inputs, host only."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from onnx import numpy_helper

from decompose_road import ROOT, digest
from fold3_model_trace import compare
from fold3_sequences import initial_inputs
from prepare_snpe_dense_probe import ordered_half_gemv
from trace_litert_prefix import producer
from validate_litert_dense_sections_host import run_converted, run_onnx, session


def normalize_inputs(actual_inputs, expected_inputs, trans_b):
  if len(actual_inputs) != 3 or len(expected_inputs) != 3 or trans_b not in (0, 1):
    raise ValueError('Unexpected Gemm inputs')
  result = list(actual_inputs)
  if trans_b == 0:
    # LiteRT fully connected stores [out, in]; ONNX transB=0 expects [in, out].
    result[1] = result[1].T.copy()
  if any(a.shape != b.shape for a, b in zip(result, expected_inputs, strict=True)):
    raise ValueError('Input layout mismatch')
  for index, (actual, expected) in enumerate(zip(result, expected_inputs, strict=True)):
    if actual.dtype != expected.dtype:
      converted = actual.astype(expected.dtype)
      if not np.array_equal(converted.astype(actual.dtype), actual):
        raise ValueError('Input dtype conversion would change values')
      result[index] = converted
  return result


def compare_at_reference_precision(actual, reference):
  """Measure the result after the original output's explicit rounding boundary."""
  if (actual.shape != reference.shape or actual.dtype not in (np.float16, np.float32)
      or reference.dtype not in (np.float16, np.float32)):
    raise ValueError('Unexpected output shape or floating-point dtype')
  with np.errstate(over='ignore', invalid='ignore'):
    rounded = actual.astype(reference.dtype)
  result = compare(rounded, reference)
  result['different_elements'] = int(np.count_nonzero(rounded != reference))
  return result


def isolate(sections, conversion, model, output, node_name, converted_tensor, seed, derived_sha256=None):
  if output.exists():
    raise ValueError('Preserve prior evidence')
  manifest = json.loads((sections / 'manifest.json').read_text())
  receipt = json.loads((conversion / 'receipt.json').read_text())
  source = sections / 'suffix.onnx'
  if digest(source) != manifest['sections']['suffix']['sha256'] or digest(source) != receipt['source_sha256']:
    raise ValueError('Suffix graph provenance mismatch')
  if derived_sha256:
    if digest(model) != derived_sha256:
      raise ValueError('Derived graph hash mismatch')
  elif digest(model) != receipt['outputs'].get(str(model.relative_to(conversion))):
    raise ValueError('Converted graph provenance mismatch')
  refs = ROOT / 'android/app/build/model-reference'
  original = ROOT / 'openpilot/selfdrive/modeld/models/driving_supercombo.onnx'
  expected_hash = json.loads((refs / 'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  if manifest['original_sha256'] != expected_hash or digest(original) != expected_hash:
    raise ValueError('Original graph provenance mismatch')
  for name in ('prefix', 'dense'):
    if digest(sections / (name + '.onnx')) != manifest['sections'][name]['sha256']:
      raise ValueError('Boundary provenance mismatch')
  graph = onnx.shape_inference.infer_shapes(onnx.load(source))
  matches = [n for n in graph.graph.node if n.name == node_name and n.op_type == 'Gemm']
  if len(matches) != 1:
    raise ValueError('Expected a unique named Gemm')
  node = matches[0]
  attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
  if (len(node.input) != 3 or len(node.output) != 1 or attrs.get('transB') not in (0, 1)
      or {k: v for k, v in attrs.items() if k != 'transB'} != {'alpha': 1., 'beta': 1., 'transA': 0}):
    raise ValueError('Unexpected Gemm contract')
  infos = {v.name: v for v in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
  for v in graph.graph.initializer:
    if v.name not in infos:
      infos[v.name] = onnx.helper.make_tensor_value_info(v.name, v.data_type, v.dims)
  tapped_names = [*node.input, *node.output]
  outputs = {v.name for v in graph.graph.output}
  for name in tapped_names:
    if name not in infos:
      raise ValueError('Missing shape/type for ' + name)
    if name not in outputs:
      graph.graph.output.append(copy.deepcopy(infos[name]))
  isolated = copy.deepcopy(graph)
  isolated.graph.ClearField('node')
  isolated.graph.node.append(copy.deepcopy(node))
  isolated.graph.ClearField('input')
  isolated.graph.input.extend(copy.deepcopy(infos[name]) for name in node.input)
  isolated.graph.ClearField('output')
  isolated.graph.output.append(copy.deepcopy(infos[node.output[0]]))
  isolated.graph.ClearField('initializer')
  output.mkdir(parents=True)
  tapped_path, isolated_path = output / 'suffix-tapped.onnx', output / 'gemm.onnx'
  onnx.save(graph, tapped_path)
  onnx.save(isolated, isolated_path)
  values = initial_inputs(refs, 'driving_supercombo', seed)
  if 'rewritten_source_sha256' in manifest:
    values['new_img'] = values['new_img'].astype(np.float32)
    values['state_img_q'] = values['state_img_q'].astype(np.float32).reshape(2, 30, 128, 256)
  prefix = run_onnx(session(sections / 'prefix.onnx'), values)
  dense = onnx.load(sections / 'dense.onnx').graph
  weights = {v.name: numpy_helper.to_array(v) for v in dense.initializer}
  linear = ordered_half_gemv(prefix['model/view'], *(weights[n] for n in dense.node[0].input[1:]))
  values.update(prefix)
  values['model/linear'] = linear
  untapped = run_onnx(session(source), values)
  expected = run_onnx(session(tapped_path), values)
  runtime = Interpreter(model_path=str(model), num_threads=2, experimental_preserve_all_tensors=True,
                        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
  runtime.allocate_tensors()
  run_converted(runtime, {d['name']: values[d['name']] for d in runtime.get_input_details()})
  tensors = {d['name']: d for d in runtime.get_tensor_details()}
  operation = producer(runtime._get_ops_details(), tensors[converted_tensor]['index'])
  if operation['op_name'] != 'FULLY_CONNECTED' or len(operation['inputs']) != 3:
    raise ValueError('Expected fully connected with bias')
  flatbuffer = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(model.read_bytes(), 0))
  options = flatbuffer.subgraphs[0].operators[operation['index']].builtinOptions
  if options.fusedActivationFunction != schema.ActivationFunctionType.NONE:
    raise ValueError('Unexpected fused activation')
  actual_inputs = [runtime.get_tensor(int(index)) for index in operation['inputs']]
  actual_output = runtime.get_tensor(tensors[converted_tensor]['index'])
  expected_inputs = [expected[name] for name in node.input]
  actual_inputs = normalize_inputs(actual_inputs, expected_inputs, attrs['transB'])
  reference_session = session(isolated_path)
  isolated_actual = run_onnx(reference_session, dict(zip(node.input, actual_inputs, strict=True)))[node.output[0]]
  isolated_original = run_onnx(reference_session, dict(zip(node.input, expected_inputs, strict=True)))[node.output[0]]
  result = {'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
            'provider': 'host LiteRT CPU and tapped ONNX CPU; not GPU evidence',
            'warning': 'Tapping ONNX outputs may change fusion', 'seed': seed,
            'original_sha256': expected_hash, 'source_sha256': digest(source), 'model_sha256': digest(model),
            'tapped_sha256': digest(tapped_path), 'isolated_sha256': digest(isolated_path),
            'onnx_node': node.name, 'onnx_inputs': list(node.input), 'onnx_transB': attrs['transB'],
            'litert_operator': operation['op_name'], 'litert_operator_index': operation['index'],
            'litert_inputs': [next(d['name'] for d in tensors.values() if d['index'] == i) for i in operation['inputs']],
            'inputs': [compare(a, b) for a, b in zip(actual_inputs, expected_inputs, strict=True)],
            'litert_vs_onnx_identical_inputs': compare(actual_output, isolated_actual),
            'litert_rounded_vs_onnx_identical_inputs': compare_at_reference_precision(actual_output, isolated_actual),
            'litert_vs_tapped_onnx': compare(actual_output, expected[node.output[0]]),
            'isolated_vs_tapped_original_inputs': compare(isolated_original, expected[node.output[0]]),
            'tapped_vs_untapped_public_outputs': {name: compare(expected[name], value)
                                                  for name, value in untapped.items()}}
  (output / 'isolation.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  for name in ('sections', 'conversion', 'model', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
  p.add_argument('--node-name', required=True)
  p.add_argument('--converted-tensor', required=True)
  p.add_argument('--derived-sha256', help='Hash of a diagnostic derived model; not conversion qualification')
  p.add_argument('--seed', type=int, default=42)
  args = p.parse_args()
  isolate(args.sections, args.conversion, args.model, args.output, args.node_name, args.converted_tensor, args.seed,
          args.derived_sha256)
